use arrow_array::{Array, LargeStringArray, StringArray};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;

mod board_serialization;
mod comment_parsing;
mod python_bindings;
mod visitor;

use python_bindings::{ChunkData, ParsedGames, ParsedGamesIter, PyChunkView, PyGameView};
pub use visitor::{Buffers, ParseConfig, parse_game_to_buffers};

/// Parse games from Arrow chunked array into a chunked ParsedGames container.
///
/// This implementation uses explicit chunking with a fixed number of chunks
/// (num_chunks = num_threads * chunk_multiplier) to avoid the allocation storm
/// caused by Rayon's dynamic work-stealing with fold_with.
///
/// Each chunk gets exactly one Buffers instance. Instead of merging all
/// chunks into a single buffer (which was memory-bandwidth-bound), we keep
/// the per-thread buffers and provide virtual indexing across them.
#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None, chunk_multiplier=None, store_comments=false, store_legal_moves=false))]
fn parse_games(
    py: Python<'_>,
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
    chunk_multiplier: Option<usize>,
    store_comments: bool,
    store_legal_moves: bool,
) -> PyResult<ParsedGames> {
    let config = ParseConfig {
        store_comments,
        store_legal_moves,
    };
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
        let empty_chunk = buffers_to_chunk_data(py, Buffers::default())?;
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
    // This creates exactly ceil(n_games / chunk_size) Buffers instances,
    // avoiding the allocation storm from Rayon's dynamic work-stealing.
    let chunk_results: Vec<Buffers> = thread_pool.install(|| {
        pgn_str_slices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut buffers = Buffers::with_capacity(games_per_chunk, moves_per_game, &config);
                for &pgn in chunk {
                    let _ = parse_game_to_buffers(pgn, &mut buffers, &config);
                }
                buffers
            })
            .collect()
    });

    // Convert each Buffers to ChunkData (numpy arrays) — no merge needed
    let chunk_data_vec: Vec<ChunkData> = chunk_results
        .into_iter()
        .map(|buf| buffers_to_chunk_data(py, buf))
        .collect::<PyResult<Vec<_>>>()?;

    build_parsed_games(py, chunk_data_vec)
}

/// Convert a single Buffers into a ChunkData with NumPy arrays.
fn buffers_to_chunk_data(py: Python<'_>, buffers: Buffers) -> PyResult<ChunkData> {
    let n_games = buffers.num_games();
    let total_positions = buffers.total_positions();
    let total_moves = buffers.total_moves();

    // Compute all CSR offsets BEFORE any from_vec calls consume buffer fields
    let move_offsets_vec = buffers.compute_move_offsets();
    let position_offsets_vec = buffers.compute_position_offsets();
    let has_legal_moves = !buffers.legal_move_counts.is_empty();
    let legal_move_offsets_vec = if has_legal_moves {
        buffers.compute_legal_move_offsets()
    } else {
        Vec::new()
    };
    let total_legal_moves_count = buffers.total_legal_moves();

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

    let legal_move_from_squares_array = PyArray1::from_vec(py, buffers.legal_move_from_squares);
    let legal_move_to_squares_array = PyArray1::from_vec(py, buffers.legal_move_to_squares);
    let legal_move_promotions_array = PyArray1::from_vec(py, buffers.legal_move_promotions);
    let legal_move_offsets_array = PyArray1::from_vec(py, legal_move_offsets_vec);

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
        outcome: buffers.outcome,
        comments: buffers.comments,
        legal_move_from_squares: legal_move_from_squares_array.unbind().into_any(),
        legal_move_to_squares: legal_move_to_squares_array.unbind().into_any(),
        legal_move_promotions: legal_move_promotions_array.unbind().into_any(),
        legal_move_offsets: legal_move_offsets_array.unbind().into_any(),
        num_games: n_games,
        num_moves: total_moves,
        num_positions: total_positions,
        num_legal_moves: total_legal_moves_count,
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

/// Parse a single PGN game string.
///
/// Convenience wrapper that creates a single-element Arrow array internally.
#[pyfunction]
#[pyo3(signature = (pgn, store_comments=false, store_legal_moves=false))]
fn parse_game(
    py: Python<'_>,
    pgn: &str,
    store_comments: bool,
    store_legal_moves: bool,
) -> PyResult<ParsedGames> {
    let config = ParseConfig {
        store_comments,
        store_legal_moves,
    };
    let mut buffers = Buffers::with_capacity(1, 70, &config);
    parse_game_to_buffers(pgn, &mut buffers, &config)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let chunk = buffers_to_chunk_data(py, buffers)?;
    build_parsed_games(py, vec![chunk])
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

    let n_games = pgns.len();
    if n_games == 0 {
        let empty_chunk = buffers_to_chunk_data(py, Buffers::default())?;
        return build_parsed_games(py, vec![empty_chunk]);
    }

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to build thread pool: {}",
                e
            ))
        })?;

    let num_chunks = num_threads;
    let chunk_size = (n_games + num_chunks - 1) / num_chunks;
    let chunk_size = chunk_size.max(1);
    let games_per_chunk = chunk_size;
    let moves_per_game = 70;

    let chunk_results: Vec<Buffers> = thread_pool.install(|| {
        pgns.par_chunks(chunk_size)
            .map(|chunk| {
                let mut buffers = Buffers::with_capacity(games_per_chunk, moves_per_game, &config);
                for pgn in chunk {
                    let _ = parse_game_to_buffers(pgn, &mut buffers, &config);
                }
                buffers
            })
            .collect()
    });

    let chunk_data_vec: Vec<ChunkData> = chunk_results
        .into_iter()
        .map(|buf| buffers_to_chunk_data(py, buf))
        .collect::<PyResult<Vec<_>>>()?;

    build_parsed_games(py, chunk_data_vec)
}

/// Parser for chess PGN notation
#[pymodule]
fn rust_pgn_reader_python_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_game, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games_from_strings, m)?)?;
    m.add_class::<ParsedGames>()?;
    m.add_class::<PyGameView>()?;
    m.add_class::<PyChunkView>()?;
    m.add_class::<ParsedGamesIter>()?;
    Ok(())
}
