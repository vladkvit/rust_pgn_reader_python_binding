use crate::comment_parsing::parse_comments;
use crate::comment_parsing::CommentContent;
use pgn_reader::{BufferedReader, RawComment, RawHeader, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use shakmaty::Color;
use shakmaty::{uci::UciMove, Chess, Outcome, Position};
use std::io::Cursor;

mod comment_parsing;

#[pyclass]
/// Holds the status of a chess position.
#[derive(Clone)]
pub struct PositionStatus {
    #[pyo3(get)]
    is_checkmate: bool,

    #[pyo3(get)]
    is_stalemate: bool,

    #[pyo3(get)]
    legal_move_count: usize,

    #[pyo3(get)]
    is_game_over: bool,

    #[pyo3(get)]
    insufficient_material: (bool, bool),

    #[pyo3(get)]
    turn: bool,
}
#[pyclass]
/// A Visitor to extract SAN moves and comments from PGN movetext
pub struct MoveExtractor {
    #[pyo3(get)]
    moves: Vec<String>,

    #[pyo3(get)]
    valid_moves: bool,

    #[pyo3(get)]
    comments: Vec<String>,

    #[pyo3(get)]
    evals: Vec<f64>,

    #[pyo3(get)]
    clock_times: Vec<(u32, u8, f64)>,

    #[pyo3(get)]
    outcome: Option<String>,

    #[pyo3(get)]
    headers: Vec<(String, String)>,

    #[pyo3(get)]
    castling_rights: Vec<(bool, bool, bool, bool)>,

    #[pyo3(get)]
    position_status: Option<PositionStatus>,

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
            outcome: None,
            headers: Vec::with_capacity(10),
            castling_rights: Vec::with_capacity(100),
            position_status: None,
        }
    }

    fn turn(&self) -> bool {
        match self.pos.turn() {
            Color::White => true,
            Color::Black => false,
        }
    }

    fn push_castling_bitboards(&mut self) {
        let castling_bitboard = self.pos.castles().castling_rights();
        let castling_rights = (
            castling_bitboard.contains(shakmaty::Square::A1),
            castling_bitboard.contains(shakmaty::Square::H1),
            castling_bitboard.contains(shakmaty::Square::A8),
            castling_bitboard.contains(shakmaty::Square::H8),
        );

        self.castling_rights.push(castling_rights);
    }

    fn update_position_status(&mut self) {
        // TODO this checks legal_moves() a bunch of times
        self.position_status = Some(PositionStatus {
            is_checkmate: self.pos.is_checkmate(),
            is_stalemate: self.pos.is_stalemate(),
            legal_move_count: self.pos.legal_moves().len(),
            is_game_over: self.pos.is_game_over(),
            insufficient_material: (
                self.pos.has_insufficient_material(Color::White),
                self.pos.has_insufficient_material(Color::Black),
            ),
            turn: match self.pos.turn() {
                Color::White => true,
                Color::Black => false,
            },
        });
    }
}

impl Visitor for MoveExtractor {
    type Result = bool;

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        let key_str = String::from_utf8_lossy(key).into_owned();
        let value_str = String::from_utf8_lossy(value.as_bytes()).into_owned();
        self.headers.push((key_str, value_str));
    }

    fn begin_game(&mut self) {
        self.moves.clear();
        self.pos = Chess::default();
        self.valid_moves = true;
        self.comments.clear();
        self.evals.clear();
        self.clock_times.clear();

        self.push_castling_bitboards();
    }

    fn san(&mut self, san_plus: SanPlus) {
        if self.valid_moves {
            match san_plus.san.to_move(&self.pos) {
                Ok(m) => {
                    self.pos.play_unchecked(&m);
                    let uci = UciMove::from_standard(&m);
                    self.moves.push(uci.to_string());
                    self.push_castling_bitboards();
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
        match parse_comments(&comment) {
            Ok((remaining_input, parsed_comments)) => {
                if !remaining_input.is_empty() {
                    eprintln!("Unparsed remaining input: {:?}", remaining_input);
                    return;
                }
                let mut eval_encountered = false;
                let mut clk_time_encountered = false;
                let mut move_comments = String::new();

                for content in parsed_comments {
                    match content {
                        CommentContent::Text(text) => {
                            if !text.trim().is_empty() {
                                move_comments.push_str(&text);
                            }
                        }
                        CommentContent::Eval(eval_value) => {
                            if eval_encountered {
                                eprintln!("Multiple Eval values found in comment: {:?}", _comment);
                                return;
                            }
                            eval_encountered = true;
                            self.evals.push(eval_value);
                        }
                        CommentContent::ClkTime(clk_time) => {
                            if clk_time_encountered {
                                eprintln!(
                                    "Multiple ClkTime values found in comment: {:?}",
                                    _comment
                                );
                                return;
                            }
                            clk_time_encountered = true;
                            self.clock_times.push(clk_time);
                        }
                    }
                }
                self.comments.push(move_comments);
            }
            Err(e) => {
                eprintln!("Error parsing comment: {:?}", e);
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn outcome(&mut self, _outcome: Option<Outcome>) {
        self.outcome = _outcome.map(|o| match o {
            Outcome::Decisive { winner } => format!("{:?}", winner),
            Outcome::Draw => "Draw".to_string(),
        });
        self.update_position_status();
    }

    fn end_game(&mut self) -> Self::Result {
        self.valid_moves
    }
}

// --- Native Rust versions (no PyResult) ---
pub fn parse_single_game_native(pgn: &str) -> Result<MoveExtractor, String> {
    let mut reader = BufferedReader::new(Cursor::new(pgn));
    let mut extractor = MoveExtractor::new();
    match reader.read_game(&mut extractor) {
        Ok(Some(_)) => Ok(extractor),
        Ok(None) => Err("No game found in PGN".to_string()),
        Err(err) => Err(format!("Parsing error: {}", err)),
    }
}

pub fn parse_multiple_games_native(
    pgns: &Vec<String>,
    num_threads: Option<usize>,
) -> Result<Vec<MoveExtractor>, String> {
    let num_threads = num_threads.unwrap_or_else(|| num_cpus::get());

    // Build a custom Rayon thread pool with the desired number of threads
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    thread_pool.install(|| {
        let results: Vec<Result<MoveExtractor, String>> = pgns
            .par_iter()
            .map(|pgn| parse_single_game_native(pgn))
            .collect();

        let mut extractors: Vec<MoveExtractor> = Vec::with_capacity(results.len());
        for res in results {
            match res {
                Ok(extractor) => extractors.push(extractor),
                Err(e) => return Err(e),
            }
        }
        Ok(extractors)
    })
}

// --- Python-facing wrappers (PyResult) ---
#[pyfunction]
/// Parses a single PGN game string.
fn parse_game(pgn: &str) -> PyResult<MoveExtractor> {
    parse_single_game_native(pgn).map_err(|err| pyo3::exceptions::PyValueError::new_err(err))
}

/// In parallel, parse a set of games
#[pyfunction]
#[pyo3(signature = (pgns, num_threads=None))]
fn parse_games(pgns: Vec<String>, num_threads: Option<usize>) -> PyResult<Vec<MoveExtractor>> {
    parse_multiple_games_native(&pgns, num_threads)
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err))
}

/// Parser for chess PGN notation
#[pymodule]
fn rust_pgn_reader_python_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_game, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games, m)?)?;
    m.add_class::<MoveExtractor>()?;
    m.add_class::<PositionStatus>()?;
    Ok(())
}
