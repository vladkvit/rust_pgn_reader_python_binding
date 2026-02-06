//! Benchmark for PGN parsing APIs, designed to mirror the Python workflow.
//!
//! This benchmark emulates the call graph of:
//! - `parse_game_moves_arrow_chunked_array()` (Arrow API → Vec<MoveExtractor>)
//! - `parse_games_flat()` (Flat API → FlatBuffers with NumPy-like arrays)
//!
//! Both use zero-copy &str slices from Arrow StringArrays, matching Python's behavior.

use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use rust_pgn_reader_python_binding::{parse_game_to_flat, parse_single_game_native, FlatBuffers};

const FILE_PATH: &str = "2013-07-train-00000-of-00001.parquet";

/// Chunk multiplier for explicit chunking in Flat API.
/// 1 = exactly num_threads chunks (minimal merge overhead)
/// Higher values provide better load balancing at cost of more buffers to merge.
const CHUNK_MULTIPLIER: usize = 1;

/// Read parquet file and return the raw Arrow StringArrays.
/// This preserves Arrow's memory layout for zero-copy string access.
fn read_parquet_to_string_arrays(file_path: &str) -> Vec<StringArray> {
    let file = File::open(Path::new(file_path)).expect("Unable to open file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("Failed to create ParquetRecordBatchReaderBuilder");
    let mut reader = builder
        .build()
        .expect("Failed to build ParquetRecordBatchReader");

    let mut arrays = Vec::new();
    while let Some(batch) = reader
        .next()
        .transpose()
        .expect("Error reading record batch")
    {
        if let Some(array) = batch
            .column_by_name("movetext")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
        {
            // Clone the StringArray to own it (Arrow uses Arc internally, so this is cheap)
            arrays.push(array.clone());
        } else {
            panic!("movetext column not found or not a StringArray");
        }
    }
    arrays
}

/// Extract &str slices from Arrow StringArrays (zero-copy).
/// This mirrors the extraction logic in `_parse_game_moves_from_arrow_chunks_native`
/// and `parse_games_flat` in lib.rs.
fn extract_str_slices<'a>(arrays: &'a [StringArray]) -> Vec<&'a str> {
    let total_len: usize = arrays.iter().map(|a| a.len()).sum();
    let mut slices = Vec::with_capacity(total_len);

    for array in arrays {
        for i in 0..array.len() {
            if array.is_valid(i) {
                slices.push(array.value(i));
            }
        }
    }
    slices
}

/// Benchmark the Arrow API workflow.
///
/// This mirrors `parse_game_moves_arrow_chunked_array()` from Python:
/// 1. Read parquet to Arrow arrays
/// 2. Extract &str slices from StringArray (like the Python-bound function does)
/// 3. Parse each game in parallel → Vec<MoveExtractor>
///
/// Args:
/// - store_board_states: Whether to populate board state vectors (for benchmarking overhead)
pub fn bench_arrow_api(store_board_states: bool) {
    // Step 1: Read parquet to Arrow StringArrays
    let arrays = read_parquet_to_string_arrays(FILE_PATH);

    // Step 2: Extract &str slices (zero-copy, mirrors Arrow chunk iteration)
    let pgn_slices = extract_str_slices(&arrays);
    println!("Read {} games from parquet.", pgn_slices.len());
    println!("store_board_states: {}", store_board_states);

    // Step 3: Build thread pool (same pattern as lib.rs)
    let num_threads = num_cpus::get();
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    // Step 4: Parse in parallel → Vec<MoveExtractor>
    // This mirrors _parse_game_moves_from_arrow_chunks_native
    let start = Instant::now();

    let extractors: Vec<_> = thread_pool
        .install(|| {
            pgn_slices
                .par_iter()
                .map(|&pgn| parse_single_game_native(pgn, false, store_board_states))
                .collect::<Result<Vec<_>, _>>()
        })
        .expect("Parsing failed");

    let parsing_duration = start.elapsed();

    // Report results
    let total_moves: usize = extractors.iter().map(|e| e.moves.len()).sum();
    let num_games = extractors.len();
    println!("Parsing time: {:?}", parsing_duration);
    println!("Parsed {} games, {} total moves.", num_games, total_moves);

    // Explicitly drop to measure cleanup time
    // This is the cost Python will pay when the list goes out of scope
    let drop_start = Instant::now();
    drop(extractors);
    let drop_duration = drop_start.elapsed();

    let total_duration = start.elapsed();
    println!("Cleanup time (drop): {:?}", drop_duration);
    println!("Total time (parsing + cleanup): {:?}", total_duration);
}

/// Benchmark the Flat API workflow.
///
/// This mirrors `parse_games_flat()` from Python:
/// 1. Read parquet to Arrow arrays
/// 2. Extract &str slices from StringArray
/// 3. Parse in parallel with explicit chunking (par_chunks) → fixed number of FlatBuffers
/// 4. Merge all FlatBuffers into one
pub fn bench_flat_api() {
    // Step 1: Read parquet to Arrow StringArrays
    let arrays = read_parquet_to_string_arrays(FILE_PATH);

    // Step 2: Extract &str slices (zero-copy)
    let pgn_slices = extract_str_slices(&arrays);
    println!("Read {} games from parquet.", pgn_slices.len());

    // Step 3: Build thread pool and compute capacity estimates
    let num_threads = num_cpus::get();
    let n_games = pgn_slices.len();
    let moves_per_game = 70; // Estimate ~70 moves per game

    // Calculate chunk size for explicit chunking
    let num_chunks = num_threads * CHUNK_MULTIPLIER;
    let chunk_size = (n_games + num_chunks - 1) / num_chunks; // ceiling division
    let chunk_size = chunk_size.max(1);
    let games_per_chunk = chunk_size;

    println!(
        "Using {} threads, {} chunks, {} games/chunk",
        num_threads, num_chunks, games_per_chunk
    );

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    // Step 4: Parse in parallel using par_chunks for explicit, fixed-size chunking
    // This creates exactly ceil(n_games / chunk_size) FlatBuffers instances,
    // avoiding the allocation storm from Rayon's dynamic work-stealing.
    let start = Instant::now();

    let chunk_results: Vec<FlatBuffers> = thread_pool.install(|| {
        pgn_slices
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

    let duration_parallel = start.elapsed();
    println!("Parallel parsing time: {:?}", duration_parallel);
    println!("Created {} FlatBuffers to merge", chunk_results.len());

    // Step 5: Merge all chunk buffers (mirrors parse_games_flat)
    let combined_buffers = FlatBuffers::merge_all(chunk_results);

    let duration_with_merge = start.elapsed();
    println!("Total time (parsing + merge): {:?}", duration_with_merge);
    println!(
        "Parsed {} games, {} total positions.",
        combined_buffers.num_games(),
        combined_buffers.total_positions()
    );

    // Measure cleanup time for fair comparison with Arrow API
    let drop_start = Instant::now();
    drop(combined_buffers);
    let drop_duration = drop_start.elapsed();

    let total_duration = start.elapsed();
    println!("Cleanup time (drop): {:?}", drop_duration);
    println!(
        "Total time (parsing + merge + cleanup): {:?}",
        total_duration
    );
}

fn main() {
    println!("=== Arrow API (MoveExtractor, store_board_states=false) ===\n");
    bench_arrow_api(false);

    println!("\n=== Arrow API (MoveExtractor, store_board_states=true) ===\n");
    bench_arrow_api(true);

    println!("\n=== Flat API (FlatBuffers) ===\n");
    bench_flat_api();
}
