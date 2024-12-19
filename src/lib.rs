use shakmaty::{uci::UciMove, Chess, Position};

use crate::comment_parsing::comments;
use crate::comment_parsing::CommentContent;
use pgn_reader::{BufferedReader, RawComment, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use std::io::Cursor;

mod comment_parsing;

#[pyclass]
/// A Visitor to extract SAN moves and comments from PGN movetext
struct MoveExtractor {
    #[pyo3(get)]
    moves: Vec<String>,

    #[pyo3(get)]
    valid_moves: bool,

    #[pyo3(get)]
    comments: Vec<String>,

    #[pyo3(get)]
    evals: Vec<f64>,

    #[pyo3(get)]
    clock_times: Vec<(u32, u32, u32)>,

    pos: Chess,
}

#[pymethods]
impl MoveExtractor {
    #[new]
    fn new() -> MoveExtractor {
        MoveExtractor {
            moves: Vec::with_capacity(100),
            pos: Chess::default(),
            valid_moves: true,
            comments: Vec::with_capacity(100),
            evals: Vec::with_capacity(100),
            clock_times: Vec::with_capacity(100),
        }
    }
}

impl Visitor for MoveExtractor {
    type Result = bool;

    fn begin_game(&mut self) {
        self.moves.clear();
        self.pos = Chess::default();
        self.valid_moves = true;
        self.comments.clear();
        self.evals.clear();
        self.clock_times.clear();
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
        let comment = String::from_utf8_lossy(_comment.as_bytes()).into_owned();
        match comments(&comment) {
            Ok((remaining_input, parsed_comments)) => {
                for content in parsed_comments {
                    match content {
                        CommentContent::Text(text) => self.comments.push(text),
                        CommentContent::Eval(eval_value) => self.evals.push(eval_value),
                        CommentContent::ClkTime(clk_time) => self.clock_times.push(clk_time),
                    }
                }
            }
            Err(e) => {
                eprintln!("Error parsing comment: {:?}", e);
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {
        self.valid_moves
    }
}

/// Parses PGN movetext and returns a list of SAN moves and parsed comments
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
