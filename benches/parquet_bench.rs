use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};

use rust_pgn_reader_python_binding::{parse_multiple_games_native, parse_single_game_native};

pub fn bench_parquet() {
    let file_path = "2013-07-train-00000-of-00001.parquet";

    // Measure start time
    let start = Instant::now();

    // Open the Parquet file
    let file = File::open(Path::new(file_path)).expect("Unable to open file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

    let mut reader = builder.build().unwrap();

    let record_batch = reader.next().unwrap().unwrap();

    println!("Read {} records.", record_batch.num_rows());

    // Iterate through row groups

    // Measure end time
    let duration = start.elapsed();
    println!("Time taken: {:?}", duration);
}

fn main() {
    bench_parquet();
}
