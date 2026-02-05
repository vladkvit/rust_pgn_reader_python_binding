use arrow_array::{Array, LargeStringArray, StringArray};
use numpy::{PyArray1, PyArrayMethods};
use pgn_reader::Reader;
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::io::Cursor;

mod board_serialization;
mod comment_parsing;
mod flat_visitor;
mod python_bindings;
mod visitor;

pub use flat_visitor::{parse_game_to_flat, FlatBuffers};
use python_bindings::{ParsedGames, ParsedGamesIter, PositionStatus, PyGameView, PyUciMove};
use visitor::MoveExtractor;

/// Parse games from Arrow chunked array into flat NumPy arrays.
///
/// This implementation uses thread-local FlatBuffers that are merged after
/// parallel parsing, avoiding the overhead of per-game Vec allocations
/// followed by a sequential copy loop.
#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None))]
fn parse_games_flat(
    py: Python<'_>,
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
) -> PyResult<ParsedGames> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);

    // Extract PGN strings from Arrow chunks
    let mut num_elements = 0;
    for chunk in pgn_chunked_array.chunks() {
        num_elements += chunk.len();
    }

    let mut pgn_str_slices: Vec<&str> = Vec::with_capacity(num_elements);
    for chunk in pgn_chunked_array.chunks() {
        if let Some(string_array) = chunk.as_any().downcast_ref::<StringArray>() {
            for i in 0..string_array.len() {
                if string_array.is_valid(i) {
                    pgn_str_slices.push(string_array.value(i));
                }
            }
        } else if let Some(large_string_array) = chunk.as_any().downcast_ref::<LargeStringArray>() {
            for i in 0..large_string_array.len() {
                if large_string_array.is_valid(i) {
                    pgn_str_slices.push(large_string_array.value(i));
                }
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported array type in ChunkedArray: {:?}",
                chunk.data_type()
            )));
        }
    }

    let n_games = pgn_str_slices.len();

    // Build thread pool
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to build thread pool: {}",
                e
            ))
        })?;

    // Estimate capacity: ~70 moves per game
    let games_per_thread = (n_games + num_threads - 1) / num_threads;
    let moves_per_game = 70;

    // Parse in parallel using fold_with for thread-local buffer accumulation.
    // fold_with gives each worker thread its own FlatBuffers instance to accumulate into.
    // This is more efficient than fold() which creates buffers per work-stealing chunk.
    let thread_results: Vec<FlatBuffers> = thread_pool.install(|| {
        pgn_str_slices
            .par_iter()
            .fold_with(
                FlatBuffers::with_capacity(games_per_thread, moves_per_game),
                |mut buffers, &pgn| {
                    let _ = parse_game_to_flat(pgn, &mut buffers);
                    buffers
                },
            )
            .collect()
    });

    // Merge all thread-local buffers with pre-allocation (avoids repeated reallocations)
    let combined_buffers = FlatBuffers::merge_all(thread_results);

    // Convert FlatBuffers to ParsedGames with NumPy arrays
    flat_buffers_to_parsed_games(py, combined_buffers)
}

/// Convert FlatBuffers to ParsedGames with NumPy arrays.
fn flat_buffers_to_parsed_games(py: Python<'_>, buffers: FlatBuffers) -> PyResult<ParsedGames> {
    let n_games = buffers.num_games();
    let total_positions = buffers.total_positions();

    // Compute offsets
    let move_offsets_vec = buffers.compute_move_offsets();
    let position_offsets_vec = buffers.compute_position_offsets();

    // Convert to NumPy arrays
    // Boards: reshape from flat to (N_positions, 8, 8)
    let boards_array = PyArray1::from_vec(py, buffers.boards);
    let boards_reshaped = boards_array
        .reshape([total_positions, 8, 8])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Castling: reshape from flat to (N_positions, 4)
    let castling_array = PyArray1::from_vec(py, buffers.castling);
    let castling_reshaped = castling_array
        .reshape([total_positions, 4])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 1D arrays
    let en_passant_array = PyArray1::from_vec(py, buffers.en_passant);
    let halfmove_clock_array = PyArray1::from_vec(py, buffers.halfmove_clock);
    let turn_array = PyArray1::from_vec(py, buffers.turn);

    let from_squares_array = PyArray1::from_vec(py, buffers.from_squares);
    let to_squares_array = PyArray1::from_vec(py, buffers.to_squares);
    let promotions_array = PyArray1::from_vec(py, buffers.promotions);
    let clocks_array = PyArray1::from_vec(py, buffers.clocks);
    let evals_array = PyArray1::from_vec(py, buffers.evals);

    let move_offsets_array = PyArray1::from_vec(py, move_offsets_vec);
    let position_offsets_array = PyArray1::from_vec(py, position_offsets_vec);

    let is_checkmate_array = PyArray1::from_vec(py, buffers.is_checkmate);
    let is_stalemate_array = PyArray1::from_vec(py, buffers.is_stalemate);

    // is_insufficient: reshape to (N_games, 2)
    let is_insufficient_array = PyArray1::from_vec(py, buffers.is_insufficient);
    let is_insufficient_reshaped = if n_games > 0 {
        is_insufficient_array
            .reshape([n_games, 2])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    } else {
        is_insufficient_array
            .reshape([0, 2])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
    };

    let legal_move_count_array = PyArray1::from_vec(py, buffers.legal_move_count);
    let valid_array = PyArray1::from_vec(py, buffers.valid);

    Ok(ParsedGames {
        boards: boards_reshaped.unbind().into_any(),
        castling: castling_reshaped.unbind().into_any(),
        en_passant: en_passant_array.unbind().into_any(),
        halfmove_clock: halfmove_clock_array.unbind().into_any(),
        turn: turn_array.unbind().into_any(),
        from_squares: from_squares_array.unbind().into_any(),
        to_squares: to_squares_array.unbind().into_any(),
        promotions: promotions_array.unbind().into_any(),
        clocks: clocks_array.unbind().into_any(),
        evals: evals_array.unbind().into_any(),
        move_offsets: move_offsets_array.unbind().into_any(),
        position_offsets: position_offsets_array.unbind().into_any(),
        is_checkmate: is_checkmate_array.unbind().into_any(),
        is_stalemate: is_stalemate_array.unbind().into_any(),
        is_insufficient: is_insufficient_reshaped.unbind().into_any(),
        legal_move_count: legal_move_count_array.unbind().into_any(),
        valid: valid_array.unbind().into_any(),
        headers: buffers.headers,
    })
}

// --- Native Rust versions (no PyResult) ---
pub fn parse_single_game_native(
    pgn: &str,
    store_legal_moves: bool,
) -> Result<MoveExtractor, String> {
    let mut reader = Reader::new(Cursor::new(pgn));
    let mut extractor = MoveExtractor::new(store_legal_moves);
    match reader.read_game(&mut extractor) {
        Ok(Some(_)) => Ok(extractor),
        Ok(None) => Err("No game found in PGN".to_string()),
        Err(err) => Err(format!("Parsing error: {}", err)),
    }
}

pub fn parse_multiple_games_native(
    pgns: &Vec<String>,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> Result<Vec<MoveExtractor>, String> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);

    // Build a custom Rayon thread pool with the desired number of threads
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    thread_pool.install(|| {
        pgns.par_iter()
            .map(|pgn| parse_single_game_native(pgn, store_legal_moves))
            .collect()
    })
}

fn _parse_game_moves_from_arrow_chunks_native(
    pgn_chunked_array: &PyChunkedArray,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> Result<Vec<MoveExtractor>, String> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| format!("Failed to build Rayon thread pool: {}", e))?;

    let mut num_elements = 0;
    for chunk in pgn_chunked_array.chunks() {
        num_elements += chunk.len();
    }
    let mut pgn_str_slices: Vec<&str> = Vec::with_capacity(num_elements);
    for chunk in pgn_chunked_array.chunks() {
        if let Some(string_array) = chunk.as_any().downcast_ref::<StringArray>() {
            for i in 0..string_array.len() {
                if string_array.is_valid(i) {
                    pgn_str_slices.push(string_array.value(i));
                }
            }
        } else if let Some(large_string_array) = chunk.as_any().downcast_ref::<LargeStringArray>() {
            for i in 0..large_string_array.len() {
                if large_string_array.is_valid(i) {
                    pgn_str_slices.push(large_string_array.value(i));
                }
            }
        } else {
            return Err(format!(
                "Unsupported array type in ChunkedArray: {:?}",
                chunk.data_type()
            ));
        }
    }

    thread_pool.install(|| {
        pgn_str_slices
            .par_iter()
            .map(|&pgn_s| parse_single_game_native(pgn_s, store_legal_moves))
            .collect::<Result<Vec<MoveExtractor>, String>>()
    })
}

// --- Python-facing wrappers (PyResult) ---
// TODO check if I can call py.allow_threads and release GIL
// see https://docs.rs/pyo3-arrow/0.10.1/pyo3_arrow/
#[pyfunction]
#[pyo3(signature = (pgn, store_legal_moves = false))]
/// Parses a single PGN game string.
fn parse_game(pgn: &str, store_legal_moves: bool) -> PyResult<MoveExtractor> {
    parse_single_game_native(pgn, store_legal_moves)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// In parallel, parse a set of games
#[pyfunction]
#[pyo3(signature = (pgns, num_threads=None, store_legal_moves=false))]
fn parse_games(
    pgns: Vec<String>,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> PyResult<Vec<MoveExtractor>> {
    parse_multiple_games_native(&pgns, num_threads, store_legal_moves)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None, store_legal_moves=false))]
fn parse_game_moves_arrow_chunked_array(
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> PyResult<Vec<MoveExtractor>> {
    _parse_game_moves_from_arrow_chunks_native(&pgn_chunked_array, num_threads, store_legal_moves)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Parser for chess PGN notation
#[pymodule]
fn rust_pgn_reader_python_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_game, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games, m)?)?;
    m.add_function(wrap_pyfunction!(parse_game_moves_arrow_chunked_array, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games_flat, m)?)?;
    m.add_class::<MoveExtractor>()?;
    m.add_class::<PositionStatus>()?;
    m.add_class::<PyUciMove>()?;
    m.add_class::<ParsedGames>()?;
    m.add_class::<PyGameView>()?;
    m.add_class::<ParsedGamesIter>()?;
    Ok(())
}

#[cfg(test)]
mod pyucimove_tests {
    use super::*;

    #[test]
    fn test_parse_game_without_headers() {
        let pgn = "1. Nf3 d5 2. e4 c5 3. exd5 e5 4. dxe6 0-1";
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert_eq!(extractor.moves.len(), 7);
        assert_eq!(extractor.outcome, Some("Black".to_string()));
    }

    #[test]
    fn test_parse_game_with_standard_fen() {
        // A game starting from a mid-game position
        let pgn = r#"[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

3. Bb5 a6 4. Ba4 Nf6 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(extractor.valid_moves, "Moves should be valid");
        assert_eq!(extractor.moves.len(), 4);
    }

    #[test]
    fn test_parse_chess960_game() {
        // Chess960 game with custom starting position
        let pgn = r#"[Variant "chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1. g3 d5 2. d4 g6 3. b3 Nf6 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Chess960 moves should be valid with proper FEN"
        );
        assert_eq!(extractor.moves.len(), 6);
    }

    #[test]
    fn test_parse_chess960_variant_case_insensitive() {
        // Test that variant detection is case-insensitive
        let pgn = r#"[Variant "Chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1. g3 d5 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Should handle Chess960 case variations"
        );
    }

    #[test]
    fn test_parse_invalid_fen_falls_back() {
        // Invalid FEN should fall back to default and mark invalid
        let pgn = r#"[FEN "invalid fen string"]

1. e4 e5 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            !extractor.valid_moves,
            "Should mark as invalid when FEN parsing fails"
        );
    }

    #[test]
    fn test_fen_header_case_insensitive() {
        // FEN header key should be case-insensitive
        let pgn = r#"[fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

3. Bb5 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Should handle lowercase 'fen' header"
        );
    }

    #[test]
    fn test_parse_game_with_custom_fen_no_variant() {
        // A standard chess game starting from a mid-game position (no Variant header)
        // Position after 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Ruy Lopez)
        let pgn = r#"[Event "Test Game"]
    [FEN "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"]

    3... a6 4. Ba4 Nf6 5. O-O Be7 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Standard game with custom FEN should be valid"
        );
        assert_eq!(extractor.moves.len(), 5); // a6, Ba4, Nf6, O-O, Be7
    }
}
