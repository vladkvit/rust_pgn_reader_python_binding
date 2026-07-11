//! Benchmark for PGN parsing API, designed to mirror the Python workflow.
//!
//! `cargo bench --bench parquet_bench`
//! `samply record --rate 10000 cargo bench --bench parquet_bench`

use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use rust_pgn_reader_python_binding::{ParseConfig, parse_games_flat};

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
            arrays.push(array.clone());
        } else {
            panic!("movetext column not found or not a StringArray");
        }
    }
    arrays
}

/// Extract &str slices from Arrow StringArrays (zero-copy).
fn extract_str_slices(arrays: &[StringArray]) -> Vec<&str> {
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
/// 3. Two-pass parallel parse into exact-size flat arrays
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

    let num_threads = num_cpus::get();
    println!("Using {} threads (two-pass flat output)", num_threads);

    // Step 3: Parse with the two-pass flat driver
    let start = Instant::now();

    let flat = parse_games_flat(&pgn_slices, num_threads, &config).expect("parse failed");

    let duration_parallel = start.elapsed();
    println!("Parallel parsing time: {:?}", duration_parallel);
    println!(
        "Parsed {} games, {} total positions, {} total moves.",
        flat.num_games, flat.total_positions, flat.total_moves
    );

    // Measure cleanup time
    let drop_start = Instant::now();
    drop(flat);
    let drop_duration = drop_start.elapsed();

    let total_duration = start.elapsed();
    println!("Cleanup time (drop): {:?}", drop_duration);
    println!("Total time (parsing + cleanup): {:?}", total_duration);
}

fn main() {
    println!("=== Parse API (flat two-pass) ===\n");
    bench_parse_api();
}
