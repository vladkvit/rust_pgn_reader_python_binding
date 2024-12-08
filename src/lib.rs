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

// use shakmaty::{fen::Fen, CastlingMode, Chess, Position};

use pgn_reader::{BufferedReader, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use std::io::Cursor;

/// A Visitor to extract SAN moves from PGN movetext
struct MoveExtractor {
    moves: Vec<String>,
}

impl MoveExtractor {
    fn new() -> MoveExtractor {
        MoveExtractor { moves: Vec::new() }
    }
}

impl Visitor for MoveExtractor {
    type Result = Vec<String>;

    fn begin_game(&mut self) {
        self.moves.clear();
    }

    fn san(&mut self, san_plus: SanPlus) {
        self.moves.push(san_plus.san.to_string());
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {
        self.moves.clone()
    }
}

/// Parses PGN movetext and returns a list of SAN moves
#[pyfunction]
fn parse_moves(pgn: &str) -> PyResult<Vec<String>> {
    let mut reader = BufferedReader::new(Cursor::new(pgn));
    let mut extractor = MoveExtractor::new();

    match reader.read_game(&mut extractor) {
        Ok(Some(moves)) => Ok(moves),
        Ok(None) => Err(pyo3::exceptions::PyValueError::new_err(
            "No game found in PGN",
        )),
        Err(err) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Parsing error: {}",
            err
        ))),
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn my_own_parser(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_moves, m)?)?;
    Ok(())
}
