use arrow_array::{Array, LargeStringArray, StringArray};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;

mod board_serialization;
mod comment_parsing;
mod flat;
mod python_bindings;
mod tokenizer;

pub use flat::{FlatOutput, ParseConfig, parse_games_flat};
use python_bindings::{ParsedGames, ParsedGamesIter, PyGameView};

/// Shared parallel parsing logic for a slice of PGN strings.
///
/// Both `parse_games` and `parse_games_from_strings` delegate here after
/// extracting their `&str` slices from different input types. The GIL is
/// released for the duration of both parsing passes.
fn parse_str_slices(
    py: Python<'_>,
    slices: &[&str],
    num_threads: usize,
    config: &ParseConfig,
) -> PyResult<ParsedGames> {
    let flat = py
        .detach(|| parse_games_flat(slices, num_threads, config))
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
    build_parsed_games(py, flat)
}

/// Parse games from an Arrow chunked array into a flat ParsedGames container.
///
/// Two-pass implementation: a parallel counting pass computes exact output
/// sizes and per-game offsets, then a parallel parsing pass writes every
/// game into its precomputed disjoint range of single flat output arrays
/// (rayon work stealing over small tasks; output order == input order).
#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None, store_comments=false, store_legal_moves=false))]
fn parse_games(
    py: Python<'_>,
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
    store_comments: bool,
    store_legal_moves: bool,
) -> PyResult<ParsedGames> {
    let config = ParseConfig {
        store_comments,
        store_legal_moves,
    };
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);

    // Extract PGN strings from Arrow chunks
    let num_elements: usize = pgn_chunked_array.chunks().iter().map(|c| c.len()).sum();
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

    parse_str_slices(py, &pgn_str_slices, num_threads, &config)
}

/// Convert a FlatOutput into a ParsedGames with NumPy arrays (zero-copy).
fn build_parsed_games(py: Python<'_>, flat: FlatOutput) -> PyResult<ParsedGames> {
    let num_games = flat.num_games;
    let num_moves = flat.total_moves;
    let num_positions = flat.total_positions;
    let num_legal_moves = flat.legal_move_from_squares.len();

    // Rust-side offset copies for fast PyGameView construction.
    let move_offsets_rs = flat.move_offsets.clone();
    let position_offsets_rs = flat.position_offsets.clone();
    let parsed_move_counts_rs = flat.parsed_move_counts.clone();

    // Boards: reshape from flat to (N_positions, 8, 8)
    let boards_array = PyArray1::from_vec(py, flat.boards);
    let boards_reshaped = boards_array
        .reshape([num_positions, 8, 8])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Castling: reshape from flat to (N_positions, 4)
    let castling_array = PyArray1::from_vec(py, flat.castling);
    let castling_reshaped = castling_array
        .reshape([num_positions, 4])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Insufficient material: reshape from flat to (N_games, 2)
    let is_insufficient_array = PyArray1::from_vec(py, flat.is_insufficient);
    let is_insufficient_reshaped = is_insufficient_array
        .reshape([num_games, 2])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(ParsedGames {
        boards: boards_reshaped.unbind().into_any(),
        castling: castling_reshaped.unbind().into_any(),
        en_passant: PyArray1::from_vec(py, flat.en_passant).unbind().into_any(),
        halfmove_clock: PyArray1::from_vec(py, flat.halfmove_clock)
            .unbind()
            .into_any(),
        turn: PyArray1::from_vec(py, flat.turn).unbind().into_any(),
        from_squares: PyArray1::from_vec(py, flat.from_squares)
            .unbind()
            .into_any(),
        to_squares: PyArray1::from_vec(py, flat.to_squares).unbind().into_any(),
        promotions: PyArray1::from_vec(py, flat.promotions).unbind().into_any(),
        clocks: PyArray1::from_vec(py, flat.clocks).unbind().into_any(),
        evals: PyArray1::from_vec(py, flat.evals).unbind().into_any(),
        move_offsets: PyArray1::from_vec(py, flat.move_offsets)
            .unbind()
            .into_any(),
        position_offsets: PyArray1::from_vec(py, flat.position_offsets)
            .unbind()
            .into_any(),
        parsed_move_counts: PyArray1::from_vec(py, flat.parsed_move_counts)
            .unbind()
            .into_any(),
        is_checkmate: PyArray1::from_vec(py, flat.is_checkmate)
            .unbind()
            .into_any(),
        is_stalemate: PyArray1::from_vec(py, flat.is_stalemate)
            .unbind()
            .into_any(),
        is_insufficient: is_insufficient_reshaped.unbind().into_any(),
        legal_move_count: PyArray1::from_vec(py, flat.legal_move_count)
            .unbind()
            .into_any(),
        valid: PyArray1::from_vec(py, flat.valid).unbind().into_any(),
        headers: flat.headers,
        outcome: flat.outcome,
        parse_errors: flat.parse_errors,
        comments: flat.comments,
        legal_move_from_squares: PyArray1::from_vec(py, flat.legal_move_from_squares)
            .unbind()
            .into_any(),
        legal_move_to_squares: PyArray1::from_vec(py, flat.legal_move_to_squares)
            .unbind()
            .into_any(),
        legal_move_promotions: PyArray1::from_vec(py, flat.legal_move_promotions)
            .unbind()
            .into_any(),
        legal_move_offsets: PyArray1::from_vec(py, flat.legal_move_offsets)
            .unbind()
            .into_any(),
        move_offsets_rs,
        position_offsets_rs,
        parsed_move_counts_rs,
        num_games,
        num_moves,
        num_positions,
        num_legal_moves,
    })
}

/// Parse a single PGN game string.
#[pyfunction]
#[pyo3(signature = (pgn, store_comments=false, store_legal_moves=false))]
fn parse_game(
    py: Python<'_>,
    pgn: &str,
    store_comments: bool,
    store_legal_moves: bool,
) -> PyResult<ParsedGames> {
    if pgn.trim().is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No game found in PGN",
        ));
    }
    let config = ParseConfig {
        store_comments,
        store_legal_moves,
    };
    parse_str_slices(py, &[pgn], 1, &config)
}

/// Parse multiple PGN game strings in parallel.
///
/// Convenience wrapper for when you have a list of strings rather than an Arrow array.
#[pyfunction]
#[pyo3(signature = (pgns, num_threads=None, store_comments=false, store_legal_moves=false))]
fn parse_games_from_strings(
    py: Python<'_>,
    pgns: Vec<String>,
    num_threads: Option<usize>,
    store_comments: bool,
    store_legal_moves: bool,
) -> PyResult<ParsedGames> {
    let config = ParseConfig {
        store_comments,
        store_legal_moves,
    };
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);
    let str_slices: Vec<&str> = pgns.iter().map(|s| s.as_str()).collect();
    parse_str_slices(py, &str_slices, num_threads, &config)
}

/// Parser for chess PGN notation
#[pymodule(gil_used = true)]
fn rust_pgn_reader_python_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_game, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games_from_strings, m)?)?;
    m.add_class::<ParsedGames>()?;
    m.add_class::<PyGameView>()?;
    m.add_class::<ParsedGamesIter>()?;
    Ok(())
}
