use arrow::array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use rust_pgn_reader_python_binding::{
    parse_game_to_flat, parse_multiple_games_native, FlatBuffers,
};

pub fn bench_parquet() {
    let file_path = "2013-07-train-00000-of-00001.parquet";

    // Open the Parquet file
    let file = File::open(Path::new(file_path)).expect("Unable to open file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("Failed to create ParquetRecordBatchReaderBuilder");
    let mut reader = builder
        .build()
        .expect("Failed to build ParquetRecordBatchReader");

    // Process record batches
    let mut movetexts = Vec::new();
    while let Some(batch) = reader
        .next()
        .transpose()
        .expect("Error reading record batch")
    {
        // Extract "movetext" column from the record batch
        if let Some(array) = batch
            .column_by_name("movetext")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
        {
            for i in 0..array.len() {
                if array.is_valid(i) {
                    movetexts.push(array.value(i).to_string());
                }
            }
        } else {
            panic!("movetext column not found or not a StringArray");
        }
    }

    println!("Read {} rows.", movetexts.len());
    // Measure start time
    let start = Instant::now();

    let result = parse_multiple_games_native(&movetexts, None, false);

    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    match result {
        Ok(parsed) => println!("Parsed {} games.", parsed.len()),
        Err(err) => eprintln!("Error parsing games: {}", err),
    }

    let duration2 = start.elapsed();

    println!("Time after checks: {:?}", duration2);
}

/// Read movetexts from a parquet file.
fn read_movetexts_from_parquet(file_path: &str) -> Vec<String> {
    let file = File::open(Path::new(file_path)).expect("Unable to open file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .expect("Failed to create ParquetRecordBatchReaderBuilder");
    let mut reader = builder
        .build()
        .expect("Failed to build ParquetRecordBatchReader");

    let mut movetexts = Vec::new();
    while let Some(batch) = reader
        .next()
        .transpose()
        .expect("Error reading record batch")
    {
        if let Some(array) = batch
            .column_by_name("movetext")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
        {
            for i in 0..array.len() {
                if array.is_valid(i) {
                    movetexts.push(array.value(i).to_string());
                }
            }
        } else {
            panic!("movetext column not found or not a StringArray");
        }
    }
    movetexts
}

/// Benchmark using the flat API (FlatBuffers + parallel fold_with).
pub fn bench_parquet_flat() {
    let file_path = "2013-07-train-00000-of-00001.parquet";

    let movetexts = read_movetexts_from_parquet(file_path);
    println!("Read {} rows.", movetexts.len());

    let num_threads = num_cpus::get();
    let n_games = movetexts.len();
    let games_per_thread = (n_games + num_threads - 1) / num_threads;
    let moves_per_game = 70;

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    let start = Instant::now();

    // Parse in parallel using fold_with for thread-local buffer accumulation
    let thread_results: Vec<FlatBuffers> = thread_pool.install(|| {
        movetexts
            .par_iter()
            .fold_with(
                FlatBuffers::with_capacity(games_per_thread, moves_per_game),
                |mut buffers, pgn| {
                    let _ = parse_game_to_flat(pgn, &mut buffers);
                    buffers
                },
            )
            .collect()
    });

    let duration_parallel = start.elapsed();
    println!("Parallel parsing time: {:?}", duration_parallel);

    // Merge all thread-local buffers with pre-allocation
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
    println!("=== MoveExtractor API ===\n");
    bench_parquet();

    println!("\n=== Flat API ===\n");
    bench_parquet_flat();
}
