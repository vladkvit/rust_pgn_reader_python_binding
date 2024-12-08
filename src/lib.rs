use shakmaty::{uci::UciMove, Chess, Position};

use pgn_reader::{BufferedReader, RawComment, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use std::io::Cursor;

#[pyclass]
/// A Visitor to extract SAN moves from PGN movetext
struct MoveExtractor {
    #[pyo3(get)]
    moves: Vec<String>,

    #[pyo3(get)]
    valid_moves: bool,

    #[pyo3(get)]
    comments: Vec<String>,

    pos: Chess,
}

#[pymethods]
impl MoveExtractor {
    #[new]
    fn new() -> MoveExtractor {
        MoveExtractor {
            moves: Vec::new(),
            pos: Chess::default(),
            valid_moves: true,
            comments: Vec::new(),
        }
    }
}

impl Visitor for MoveExtractor {
    type Result = Vec<String>;

    fn begin_game(&mut self) {
        self.moves.clear();
        self.pos = Chess::default();
        self.valid_moves = true;
    }

    fn san(&mut self, san_plus: SanPlus) {
        if self.valid_moves {
            match san_plus.san.to_move(&self.pos) {
                Ok(m) => {
                    self.pos.play_unchecked(&m);
                    let uci = UciMove::from_standard(&m);
                    self.moves.push(uci.to_string());
                }
                Err(err) => {
                    eprintln!("error in game: {} {}", err, san_plus);
                    self.valid_moves = false;
                }
            }
        }
    }

    fn comment(&mut self, _comment: RawComment<'_>) {
        self.comments
            .push(String::from_utf8_lossy(_comment.as_bytes()).into_owned());
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
fn parse_moves(pgn: &str) -> PyResult<MoveExtractor> {
    let mut reader = BufferedReader::new(Cursor::new(pgn));
    let mut extractor = MoveExtractor::new();

    match reader.read_game(&mut extractor) {
        Ok(Some(_)) => Ok(extractor),
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
fn rust_pgn_reader_python_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_moves, m)?)?;
    Ok(())
}
