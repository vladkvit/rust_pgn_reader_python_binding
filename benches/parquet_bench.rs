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
pub fn bench_arrow_api() {
    // Step 1: Read parquet to Arrow StringArrays
    let arrays = read_parquet_to_string_arrays(FILE_PATH);

    // Step 2: Extract &str slices (zero-copy, mirrors Arrow chunk iteration)
    let pgn_slices = extract_str_slices(&arrays);
    println!("Read {} games from parquet.", pgn_slices.len());

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
                .map(|&pgn| parse_single_game_native(pgn, false))
                .collect::<Result<Vec<_>, _>>()
        })
        .expect("Parsing failed");

    let duration = start.elapsed();

    // Report results
    let total_moves: usize = extractors.iter().map(|e| e.moves.len()).sum();
    println!("Parsing time: {:?}", duration);
    println!(
        "Parsed {} games, {} total moves.",
        extractors.len(),
        total_moves
    );
}

/// Benchmark the Flat API workflow.
///
/// This mirrors `parse_games_flat()` from Python:
/// 1. Read parquet to Arrow arrays
/// 2. Extract &str slices from StringArray
/// 3. Parse in parallel with fold_with → thread-local FlatBuffers
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
    let games_per_thread = (n_games + num_threads - 1) / num_threads;
    let moves_per_game = 70; // Estimate ~70 moves per game

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    // Step 4: Parse in parallel using fold_with for thread-local buffer accumulation
    // This is exactly the pattern used in parse_games_flat() in lib.rs
    let start = Instant::now();

    let thread_results: Vec<FlatBuffers> = thread_pool.install(|| {
        pgn_slices
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

    let duration_parallel = start.elapsed();
    println!("Parallel parsing time: {:?}", duration_parallel);

    // Step 5: Merge all thread-local buffers (mirrors parse_games_flat)
    let combined_buffers = FlatBuffers::merge_all(thread_results);

    let duration_total = start.elapsed();
    println!("Total time (including merge): {:?}", duration_total);
    println!(
        "Parsed {} games, {} total positions.",
        combined_buffers.num_games(),
        combined_buffers.total_positions()
    );
}

fn main() {
    println!("=== Arrow API (MoveExtractor) ===\n");
    bench_arrow_api();

    println!("\n=== Flat API (FlatBuffers) ===\n");
    bench_flat_api();
}
