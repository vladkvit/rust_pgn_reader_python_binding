//! Read a parquet file directly from Rust and parse it.
//!
//! Backs `parse_games_from_parquet`, letting Python hand over a path instead
//! of a decoded Arrow array.
//!
//! Decoding runs *in* the parse tasks rather than up front: the file's row
//! groups are partitioned into one contiguous range per chunk, and each
//! chunk independently opens the file, decodes only its row groups (and only
//! the requested column) and parses them into a [`Buffers`]. That way parquet
//! decompression is parallelized across the same pool as PGN parsing — the
//! eager single-threaded reader is roughly as expensive as the whole parse,
//! so a serial decode would dominate.

use arrow_array::{Array, LargeStringArray, StringArray};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::ops::Range;
use std::path::Path;

use crate::visitor::{Buffers, ParseConfig, parse_game_to_buffers};

/// Estimated moves per game used for buffer preallocation.
const MOVES_PER_GAME_ESTIMATE: usize = 70;

/// Parse every game in `path`'s `column` using `num_threads` workers spread
/// over `num_chunks` chunks.
///
/// Returns one `Buffers` per chunk in file order. Errors (missing file,
/// unknown column, decode failure) are returned as strings.
pub fn parse_parquet_parallel(
    path: &Path,
    column: &str,
    num_threads: usize,
    num_chunks: usize,
    config: &ParseConfig,
) -> Result<Vec<Buffers>, String> {
    // Load the footer once; every task reuses this metadata.
    let file = File::open(path)
        .map_err(|e| format!("Failed to open parquet file '{}': {}", path.display(), e))?;
    let metadata = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())
        .map_err(|e| format!("Failed to read parquet metadata: {}", e))?;

    let schema = metadata.metadata().file_metadata().schema_descr();
    if !schema.columns().iter().any(|c| c.name() == column) {
        return Err(format!("Column '{}' not found in parquet file", column));
    }

    let num_row_groups = metadata.metadata().num_row_groups();
    if num_row_groups == 0 {
        return Ok(Vec::new());
    }

    let rg_rows: Vec<usize> = (0..num_row_groups)
        .map(|i| metadata.metadata().row_group(i).num_rows() as usize)
        .collect();
    let ranges = partition_row_groups(&rg_rows, num_chunks.max(1));

    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| format!("Failed to build thread pool: {}", e))?;

    let results: Vec<Result<Buffers, String>> = thread_pool.install(|| {
        ranges
            .par_iter()
            .map(|range| {
                let rows: usize = rg_rows[range.clone()].iter().sum();
                parse_row_groups(path, &metadata, column, range.clone(), rows, config)
            })
            .collect()
    });

    results.into_iter().collect()
}

/// Decode one contiguous range of row groups and parse its games.
fn parse_row_groups(
    path: &Path,
    metadata: &ArrowReaderMetadata,
    column: &str,
    row_groups: Range<usize>,
    rows: usize,
    config: &ParseConfig,
) -> Result<Buffers, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open parquet file '{}': {}", path.display(), e))?;
    let schema = metadata.metadata().file_metadata().schema_descr();
    let mask = ProjectionMask::columns(schema, [column]);
    let reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, metadata.clone())
        .with_projection(mask)
        .with_row_groups(row_groups.collect())
        .build()
        .map_err(|e| format!("Failed to build parquet reader: {}", e))?;

    let mut buffers = Buffers::with_capacity(rows, MOVES_PER_GAME_ESTIMATE, config);
    for batch in reader {
        let batch = batch.map_err(|e| format!("Failed to read parquet batch: {}", e))?;
        let array = batch.column(0);
        if let Some(strings) = array.as_any().downcast_ref::<StringArray>() {
            for i in 0..strings.len() {
                if strings.is_valid(i) {
                    let _ = parse_game_to_buffers(strings.value(i), &mut buffers, config);
                }
            }
        } else if let Some(strings) = array.as_any().downcast_ref::<LargeStringArray>() {
            for i in 0..strings.len() {
                if strings.is_valid(i) {
                    let _ = parse_game_to_buffers(strings.value(i), &mut buffers, config);
                }
            }
        } else {
            return Err(format!(
                "Unsupported parquet column type: {:?}",
                array.data_type()
            ));
        }
    }
    Ok(buffers)
}

/// Split row groups into at most `num_chunks` contiguous ranges, balancing
/// total rows. Returns ranges over row-group indices.
fn partition_row_groups(rg_rows: &[usize], num_chunks: usize) -> Vec<Range<usize>> {
    let total: usize = rg_rows.iter().sum();
    let num_chunks = num_chunks.max(1);
    if total == 0 {
        return Vec::new();
    }
    let target = total.div_ceil(num_chunks);
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut acc = 0;
    for (i, &rows) in rg_rows.iter().enumerate() {
        acc += rows;
        if acc >= target && ranges.len() + 1 < num_chunks {
            ranges.push(start..i + 1);
            start = i + 1;
            acc = 0;
        }
    }
    if start < rg_rows.len() {
        ranges.push(start..rg_rows.len());
    }
    ranges
}
