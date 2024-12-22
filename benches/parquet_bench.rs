use arrow::array::{Array, StringArray};

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

// use parquet::file::reader::{FileReader, SerializedFileReader};
// use parquet::record::RowAccessor;

use std::fs::File;
use std::path::Path;
use std::time::Instant;

// use criterion::{criterion_group, criterion_main, Criterion};

use rust_pgn_reader_python_binding::parse_multiple_games_native;

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

    let result = parse_multiple_games_native(&movetexts, None);

    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);

    match result {
        Ok(parsed) => println!("Parsed {} games.", parsed.len()),
        Err(err) => eprintln!("Error parsing games: {}", err),
    }

    let duration2 = start.elapsed();

    println!("Time after checks: {:?}", duration2);
}

fn main() {
    bench_parquet();
}
