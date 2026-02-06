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
use python_bindings::{
    ChunkData, ParsedGames, ParsedGamesIter, PositionStatus, PyChunkView, PyGameView, PyUciMove,
};
pub use visitor::MoveExtractor;

/// Parse games from Arrow chunked array into a chunked ParsedGames container.
///
/// This implementation uses explicit chunking with a fixed number of chunks
/// (num_chunks = num_threads * chunk_multiplier) to avoid the allocation storm
/// caused by Rayon's dynamic work-stealing with fold_with.
///
/// Each chunk gets exactly one FlatBuffers instance. Instead of merging all
/// chunks into a single buffer (which was memory-bandwidth-bound), we keep
/// the per-thread buffers and provide virtual indexing across them.
#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None, chunk_multiplier=None))]
fn parse_games_flat(
    py: Python<'_>,
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
    chunk_multiplier: Option<usize>,
) -> PyResult<ParsedGames> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);
    // Default multiplier of 1 means exactly num_threads chunks (one per thread).
    // Higher values (e.g., 4) create more chunks for better load balancing
    // at the cost of slightly more complex indexing.
    let chunk_multiplier = chunk_multiplier.unwrap_or(1);

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
    if n_games == 0 {
        let empty_chunk = flat_buffers_to_chunk_data(py, FlatBuffers::default())?;
        return build_parsed_games(py, vec![empty_chunk]);
    }

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

    // Calculate chunk size for explicit chunking.
    // num_chunks = num_threads * chunk_multiplier (e.g., 16 threads * 1 = 16 chunks)
    let num_chunks = num_threads * chunk_multiplier;
    let chunk_size = (n_games + num_chunks - 1) / num_chunks; // ceiling division
    let chunk_size = chunk_size.max(1); // ensure at least 1 game per chunk

    // Estimate capacity per chunk
    let games_per_chunk = chunk_size;
    let moves_per_game = 70;

    // Parse in parallel using par_chunks for explicit, fixed-size chunking.
    // This creates exactly ceil(n_games / chunk_size) FlatBuffers instances,
    // avoiding the allocation storm from Rayon's dynamic work-stealing.
    let chunk_results: Vec<FlatBuffers> = thread_pool.install(|| {
        pgn_str_slices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut buffers = FlatBuffers::with_capacity(games_per_chunk, moves_per_game);
                for &pgn in chunk {
                    let _ = parse_game_to_flat(pgn, &mut buffers);
                }
                buffers
            })
            .collect()
    });

    // Convert each FlatBuffers to ChunkData (numpy arrays) — no merge needed
    let chunk_data_vec: Vec<ChunkData> = chunk_results
        .into_iter()
        .map(|buf| flat_buffers_to_chunk_data(py, buf))
        .collect::<PyResult<Vec<_>>>()?;

    build_parsed_games(py, chunk_data_vec)
}

/// Convert a single FlatBuffers into a ChunkData with NumPy arrays.
fn flat_buffers_to_chunk_data(py: Python<'_>, buffers: FlatBuffers) -> PyResult<ChunkData> {
    let n_games = buffers.num_games();
    let total_positions = buffers.total_positions();
    let total_moves = buffers.total_moves();

    // Compute local CSR offsets
    let move_offsets_vec = buffers.compute_move_offsets();
    let position_offsets_vec = buffers.compute_position_offsets();

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

    Ok(ChunkData {
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
        num_games: n_games,
        num_moves: total_moves,
        num_positions: total_positions,
    })
}

/// Build a ParsedGames from a Vec of ChunkData.
///
/// Computes prefix-sum boundary arrays and global CSR offsets for
/// position_to_game / move_to_game.
fn build_parsed_games(py: Python<'_>, chunks: Vec<ChunkData>) -> PyResult<ParsedGames> {
    let mut total_games: usize = 0;
    let mut total_moves: usize = 0;
    let mut total_positions: usize = 0;

    // Build boundary arrays (prefix sums)
    let mut game_boundaries = Vec::with_capacity(chunks.len() + 1);
    let mut move_boundaries = Vec::with_capacity(chunks.len() + 1);
    let mut position_boundaries = Vec::with_capacity(chunks.len() + 1);

    game_boundaries.push(0);
    move_boundaries.push(0);
    position_boundaries.push(0);

    for chunk in &chunks {
        total_games += chunk.num_games;
        total_moves += chunk.num_moves;
        total_positions += chunk.num_positions;
        game_boundaries.push(total_games);
        move_boundaries.push(total_moves);
        position_boundaries.push(total_positions);
    }

    // Build global CSR offsets for position_to_game / move_to_game.
    // These are the per-chunk local offsets shifted by the chunk's base offset,
    // concatenated into a single array. Length = total_games + 1.
    let mut global_move_offsets_vec: Vec<u32> = Vec::with_capacity(total_games + 1);
    let mut global_position_offsets_vec: Vec<u32> = Vec::with_capacity(total_games + 1);

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let move_base = move_boundaries[chunk_idx] as u32;
        let pos_base = position_boundaries[chunk_idx] as u32;

        // Read the chunk's local offsets
        let local_move_offsets = chunk.move_offsets.bind(py);
        let local_move_offsets: &Bound<'_, PyArray1<u32>> = local_move_offsets.cast()?;
        let local_move_ro = local_move_offsets.readonly();
        let local_move_slice = local_move_ro.as_slice()?;

        let local_pos_offsets = chunk.position_offsets.bind(py);
        let local_pos_offsets: &Bound<'_, PyArray1<u32>> = local_pos_offsets.cast()?;
        let local_pos_ro = local_pos_offsets.readonly();
        let local_pos_slice = local_pos_ro.as_slice()?;

        // Append all but the last offset (which is the total for this chunk)
        // The last chunk's final offset will be added after the loop.
        for &offset in &local_move_slice[..local_move_slice.len() - 1] {
            global_move_offsets_vec.push(move_base + offset);
        }
        for &offset in &local_pos_slice[..local_pos_slice.len() - 1] {
            global_position_offsets_vec.push(pos_base + offset);
        }
    }

    // Final sentinel value
    global_move_offsets_vec.push(total_moves as u32);
    global_position_offsets_vec.push(total_positions as u32);

    let global_move_offsets = PyArray1::from_vec(py, global_move_offsets_vec);
    let global_position_offsets = PyArray1::from_vec(py, global_position_offsets_vec);

    Ok(ParsedGames {
        chunks,
        game_boundaries,
        move_boundaries,
        position_boundaries,
        total_games,
        total_moves,
        total_positions,
        global_move_offsets: global_move_offsets.unbind().into_any(),
        global_position_offsets: global_position_offsets.unbind().into_any(),
    })
}

// --- Native Rust versions (no PyResult) ---
pub fn parse_single_game_native(
    pgn: &str,
    store_legal_moves: bool,
    store_board_states: bool,
) -> Result<MoveExtractor, String> {
    let mut reader = Reader::new(Cursor::new(pgn));
    let mut extractor = MoveExtractor::new(store_legal_moves, store_board_states);
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
    store_board_states: bool,
) -> Result<Vec<MoveExtractor>, String> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);

    // Build a custom Rayon thread pool with the desired number of threads
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    thread_pool.install(|| {
        pgns.par_iter()
            .map(|pgn| parse_single_game_native(pgn, store_legal_moves, store_board_states))
            .collect()
    })
}

fn _parse_game_moves_from_arrow_chunks_native(
    pgn_chunked_array: &PyChunkedArray,
    num_threads: Option<usize>,
    store_legal_moves: bool,
    store_board_states: bool,
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
            .map(|&pgn_s| parse_single_game_native(pgn_s, store_legal_moves, store_board_states))
            .collect::<Result<Vec<MoveExtractor>, String>>()
    })
}

// --- Python-facing wrappers (PyResult) ---
// TODO check if I can call py.allow_threads and release GIL
// see https://docs.rs/pyo3-arrow/0.10.1/pyo3_arrow/
#[pyfunction]
#[pyo3(signature = (pgn, store_legal_moves = false, store_board_states = false))]
/// Parses a single PGN game string.
fn parse_game(
    pgn: &str,
    store_legal_moves: bool,
    store_board_states: bool,
) -> PyResult<MoveExtractor> {
    parse_single_game_native(pgn, store_legal_moves, store_board_states)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// In parallel, parse a set of games
#[pyfunction]
#[pyo3(signature = (pgns, num_threads=None, store_legal_moves=false, store_board_states=false))]
fn parse_games(
    pgns: Vec<String>,
    num_threads: Option<usize>,
    store_legal_moves: bool,
    store_board_states: bool,
) -> PyResult<Vec<MoveExtractor>> {
    parse_multiple_games_native(&pgns, num_threads, store_legal_moves, store_board_states)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None, store_legal_moves=false, store_board_states=false))]
fn parse_game_moves_arrow_chunked_array(
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
    store_legal_moves: bool,
    store_board_states: bool,
) -> PyResult<Vec<MoveExtractor>> {
    _parse_game_moves_from_arrow_chunks_native(
        &pgn_chunked_array,
        num_threads,
        store_legal_moves,
        store_board_states,
    )
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
    m.add_class::<PyChunkView>()?;
    m.add_class::<ParsedGamesIter>()?;
    Ok(())
}

#[cfg(test)]
mod pyucimove_tests {
    use super::*;

    #[test]
    fn test_parse_game_without_headers() {
        let pgn = "1. Nf3 d5 2. e4 c5 3. exd5 e5 4. dxe6 0-1";
        let result = parse_single_game_native(pgn, false, false);
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
        let result = parse_single_game_native(pgn, false, false);
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
        let result = parse_single_game_native(pgn, false, false);
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
        let result = parse_single_game_native(pgn, false, false);
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
        let result = parse_single_game_native(pgn, false, false);
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
        let result = parse_single_game_native(pgn, false, false);
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
        let result = parse_single_game_native(pgn, false, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Standard game with custom FEN should be valid"
        );
        assert_eq!(extractor.moves.len(), 5); // a6, Ba4, Nf6, O-O, Be7
    }

    #[test]
    fn test_parse_game_with_board_states() {
        // Test that board states are populated when enabled
        let pgn = "1. e4 e5 2. Nf3 Nc6 1-0";
        let result = parse_single_game_native(pgn, false, true);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert_eq!(extractor.moves.len(), 4);
        // 5 positions: initial + 4 moves
        let data = extractor
            .board_state_data
            .as_ref()
            .expect("board_state_data should be Some");
        assert_eq!(data.board_states.len(), 5 * 64);
        assert_eq!(data.en_passant_states.len(), 5);
        assert_eq!(data.halfmove_clocks.len(), 5);
        assert_eq!(data.turn_states.len(), 5);
        assert_eq!(data.castling_states.len(), 5 * 4);
    }

    #[test]
    fn test_parse_game_without_board_states() {
        // Test that board states are NOT populated when disabled
        let pgn = "1. e4 e5 2. Nf3 Nc6 1-0";
        let result = parse_single_game_native(pgn, false, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert_eq!(extractor.moves.len(), 4);
        // Board state data should be None when disabled
        assert!(extractor.board_state_data.is_none());
    }
}
