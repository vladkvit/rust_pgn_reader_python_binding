// use std::time::Instant;
// use std::{
//     env,
//     fs::File,
//     io, mem,
//     sync::{
//         atomic::{AtomicBool, Ordering},
//         Arc,
//     },
// };

// use pgn_reader::{BufferedReader, RawHeader, San, SanPlus, Skip, Visitor};
// use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn my_own_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
