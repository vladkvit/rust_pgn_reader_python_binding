//! Benchmark for PGN parsing API, designed to mirror the Python workflow.
//!
//! `cargo bench --bench parquet_bench`
//! `samply record --rate 10000 cargo bench --bench parquet_bench`

use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use rust_pgn_reader_python_binding::{parse_game_to_buffers, Buffers, ParseConfig};

const FILE_PATH: &str = "2013-07-train-00000-of-00001.parquet";

/// Chunk multiplier for explicit chunking.
/// 1 = exactly num_threads chunks (minimal overhead)
/// Higher values provide better load balancing at cost of more buffers.
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
            arrays.push(array.clone());
        } else {
            panic!("movetext column not found or not a StringArray");
        }
    }
    arrays
}

/// Extract &str slices from Arrow StringArrays (zero-copy).
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

/// Benchmark the parsing API workflow.
///
/// 1. Read parquet to Arrow arrays
/// 2. Extract &str slices from StringArray
/// 3. Parse in parallel with explicit chunking (par_chunks) -> fixed number of Buffers
///
/// No merge step - the chunked architecture keeps per-thread buffers as-is.
pub fn bench_parse_api() {
    let config = ParseConfig {
        store_comments: false,
        store_legal_moves: false,
    };

    // Step 1: Read parquet to Arrow StringArrays
    let arrays = read_parquet_to_string_arrays(FILE_PATH);

    // Step 2: Extract &str slices (zero-copy)
    let pgn_slices = extract_str_slices(&arrays);
    println!("Read {} games from parquet.", pgn_slices.len());

    // Step 3: Build thread pool and compute capacity estimates
    let num_threads = num_cpus::get();
    let n_games = pgn_slices.len();
    let moves_per_game = 70;

    // Calculate chunk size for explicit chunking
    let num_chunks = num_threads * CHUNK_MULTIPLIER;
    let chunk_size = (n_games + num_chunks - 1) / num_chunks;
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

    // Step 4: Parse in parallel using par_chunks
    let start = Instant::now();

    let chunk_results: Vec<Buffers> = thread_pool.install(|| {
        pgn_slices
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

    let duration_parallel = start.elapsed();
    println!("Parallel parsing time: {:?}", duration_parallel);
    println!(
        "Created {} Buffers chunks (no merge needed)",
        chunk_results.len()
    );

    // Compute totals from chunks
    let total_games: usize = chunk_results.iter().map(|b| b.num_games()).sum();
    let total_positions: usize = chunk_results.iter().map(|b| b.total_positions()).sum();

    let duration_total = start.elapsed();
    println!("Total time (parsing, no merge): {:?}", duration_total);
    println!(
        "Parsed {} games, {} total positions.",
        total_games, total_positions
    );

    // Measure cleanup time
    let drop_start = Instant::now();
    drop(chunk_results);
    let drop_duration = drop_start.elapsed();

    let total_duration = start.elapsed();
    println!("Cleanup time (drop): {:?}", drop_duration);
    println!("Total time (parsing + cleanup): {:?}", total_duration);
}

fn main() {
    println!("=== Parse API (Buffers) ===\n");
    bench_parse_api();
}
