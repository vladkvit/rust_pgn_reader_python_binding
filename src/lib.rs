use crate::comment_parsing::{CommentContent, ParsedTag, parse_comments};
use arrow_array::{Array, LargeStringArray, StringArray};
use pgn_reader::{KnownOutcome, Outcome, RawComment, RawTag, Reader, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use shakmaty::Color;
use shakmaty::{Chess, Position, Role, Square, uci::UciMove};
use std::io::Cursor;
use std::ops::ControlFlow;

mod comment_parsing;

// Definition of PyUciMove
#[pyclass(get_all, set_all, module = "rust_pgn_reader_python_binding")]
#[derive(Clone, Debug)]
pub struct PyUciMove {
    pub from_square: u8,
    pub to_square: u8,
    pub promotion: Option<u8>,
}

#[pymethods]
impl PyUciMove {
    #[new]
    fn new(from_square: u8, to_square: u8, promotion: Option<u8>) -> Self {
        PyUciMove {
            from_square,
            to_square,
            promotion,
        }
    }

    #[getter]
    fn get_from_square_name(&self) -> String {
        Square::new(self.from_square as u32).to_string()
    }

    #[getter]
    fn get_to_square_name(&self) -> String {
        Square::new(self.to_square as u32).to_string()
    }

    #[getter]
    fn get_promotion_name(&self) -> Option<String> {
        self.promotion.and_then(|p_u8| {
            Role::try_from(p_u8)
                .map(|role| format!("{:?}", role)) // Get the debug representation (e.g., "Queen")
                .ok()
        })
    }

    // __str__ method for Python representation
    fn __str__(&self) -> String {
        let promo_str = self.promotion.map_or("".to_string(), |p_u8| {
            Role::try_from(p_u8)
                .map(|role| role.char().to_string())
                .unwrap_or_else(|_| "".to_string()) // Handle potential error if u8 is not a valid Role
        });
        format!(
            "{}{}{}",
            Square::new(self.from_square as u32),
            Square::new(self.to_square as u32),
            promo_str
        )
    }

    // __repr__ for a more developer-friendly representation
    fn __repr__(&self) -> String {
        let promo_repr = self.promotion.map_or("None".to_string(), |p_u8| {
            Role::try_from(p_u8)
                .map(|role| format!("Some('{}')", role.char()))
                .unwrap_or_else(|_| format!("Some(InvalidRole({}))", p_u8))
        });
        format!(
            "PyUciMove(from_square={}, to_square={}, promotion={})",
            Square::new(self.from_square as u32),
            Square::new(self.to_square as u32),
            promo_repr
        )
    }
}

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
    moves: Vec<PyUciMove>,

    store_legal_moves: bool,
    flat_legal_moves: Vec<PyUciMove>,
    legal_moves_offsets: Vec<usize>,

    #[pyo3(get)]
    valid_moves: bool,

    #[pyo3(get)]
    comments: Vec<Option<String>>,

    #[pyo3(get)]
    evals: Vec<Option<f64>>,

    #[pyo3(get)]
    clock_times: Vec<Option<(u32, u8, f64)>>,

    #[pyo3(get)]
    outcome: Option<String>,

    #[pyo3(get)]
    headers: Vec<(String, String)>,

    #[pyo3(get)]
    castling_rights: Vec<Option<(bool, bool, bool, bool)>>,

    #[pyo3(get)]
    position_status: Option<PositionStatus>,

    pos: Chess,
}

#[pymethods]
impl MoveExtractor {
    #[new]
    #[pyo3(signature = (store_legal_moves = false))]
    fn new(store_legal_moves: bool) -> MoveExtractor {
        MoveExtractor {
            moves: Vec::with_capacity(100),
            store_legal_moves,
            flat_legal_moves: Vec::with_capacity(if store_legal_moves { 100 * 30 } else { 0 }), // Pre-allocate for moves
            legal_moves_offsets: Vec::with_capacity(if store_legal_moves { 100 } else { 0 }), // Pre-allocate for offsets
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

        self.castling_rights.push(Some(castling_rights));
    }

    fn push_legal_moves(&mut self) {
        // Record the starting offset for the current position's legal moves.
        self.legal_moves_offsets.push(self.flat_legal_moves.len());

        let legal_moves_for_pos = self.pos.legal_moves();
        self.flat_legal_moves.reserve(legal_moves_for_pos.len());

        for m in legal_moves_for_pos {
            let uci_move_obj = UciMove::from_standard(m);
            if let UciMove::Normal {
                from,
                to,
                promotion: promo_opt,
            } = uci_move_obj
            {
                self.flat_legal_moves.push(PyUciMove {
                    from_square: from as u8,
                    to_square: to as u8,
                    promotion: promo_opt.map(|p_role| p_role as u8),
                });
            }
        }
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

    #[getter]
    fn legal_moves(&self) -> Vec<Vec<PyUciMove>> {
        let mut result = Vec::with_capacity(self.legal_moves_offsets.len());
        if self.legal_moves_offsets.is_empty() {
            return result;
        }

        for i in 0..self.legal_moves_offsets.len() - 1 {
            let start = self.legal_moves_offsets[i];
            let end = self.legal_moves_offsets[i + 1];
            result.push(self.flat_legal_moves[start..end].to_vec());
        }

        // Handle the last chunk
        if let Some(&start) = self.legal_moves_offsets.last() {
            result.push(self.flat_legal_moves[start..].to_vec());
        }

        result
    }
}

impl Visitor for MoveExtractor {
    type Tags = Vec<(String, String)>;
    type Movetext = ();
    type Output = bool;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        self.headers.clear();
        ControlFlow::Continue(Vec::with_capacity(10))
    }

    fn tag(
        &mut self,
        tags: &mut Self::Tags,
        key: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        let key_str = String::from_utf8_lossy(key).into_owned();
        let value_str = String::from_utf8_lossy(value.as_bytes()).into_owned();
        tags.push((key_str, value_str));
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        self.headers = tags;
        self.moves.clear();
        self.flat_legal_moves.clear();
        self.legal_moves_offsets.clear();
        self.pos = Chess::default();
        self.valid_moves = true;
        self.comments.clear();
        self.evals.clear();
        self.clock_times.clear();
        self.castling_rights.clear();

        self.push_castling_bitboards();
        if self.store_legal_moves {
            self.push_legal_moves();
        }
        ControlFlow::Continue(())
    }

    // Roughly half the time during parsing is spent here in san()
    fn san(
        &mut self,
        _movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        if self.valid_moves {
            // Most of the function time is spent calculating to_move()
            match san_plus.san.to_move(&self.pos) {
                Ok(m) => {
                    self.pos.play_unchecked(m);
                    if self.store_legal_moves {
                        self.push_legal_moves();
                    }
                    let uci_move_obj = UciMove::from_standard(m);

                    match uci_move_obj {
                        UciMove::Normal {
                            from,
                            to,
                            promotion: promo_opt,
                        } => {
                            let py_uci_move = PyUciMove {
                                from_square: from as u8,
                                to_square: to as u8,
                                promotion: promo_opt.map(|p_role| p_role as u8),
                            };
                            self.moves.push(py_uci_move);
                            self.push_castling_bitboards();

                            // Push placeholders to keep vectors in sync
                            self.comments.push(None);
                            self.evals.push(None);
                            self.clock_times.push(None);
                        }
                        _ => {
                            // This case handles UciMove::Put and UciMove::Null,
                            // which are not expected from standard PGN moves
                            // that PyUciMove is designed to represent.
                            eprintln!(
                                "Unexpected UCI move type from standard PGN move: {:?}. Game moves might be invalid.",
                                uci_move_obj
                            );
                            self.valid_moves = false;
                        }
                    }
                }
                Err(err) => {
                    eprintln!("error in game: {} {}", err, san_plus);
                    self.valid_moves = false;
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn comment(
        &mut self,
        _movetext: &mut Self::Movetext,
        _comment: RawComment<'_>,
    ) -> ControlFlow<Self::Output> {
        match parse_comments(_comment.as_bytes()) {
            Ok((remaining_input, parsed_comments)) => {
                if !remaining_input.is_empty() {
                    eprintln!("Unparsed remaining input: {:?}", remaining_input);
                    return ControlFlow::Continue(());
                }

                let mut move_comments = String::new();

                for content in parsed_comments {
                    match content {
                        CommentContent::Text(text) => {
                            if !text.trim().is_empty() {
                                if !move_comments.is_empty() {
                                    move_comments.push(' ');
                                }
                                move_comments.push_str(&text);
                            }
                        }
                        CommentContent::Tag(tag_content) => match tag_content {
                            ParsedTag::Eval(eval_value) => {
                                if let Some(last_eval) = self.evals.last_mut() {
                                    *last_eval = Some(eval_value);
                                }
                            }
                            ParsedTag::Mate(mate_value) => {
                                if !move_comments.is_empty() && !move_comments.ends_with(' ') {
                                    move_comments.push(' ');
                                }
                                move_comments.push_str(&format!("[Mate {}]", mate_value));
                            }
                            ParsedTag::ClkTime {
                                hours,
                                minutes,
                                seconds,
                            } => {
                                if let Some(last_clk) = self.clock_times.last_mut() {
                                    *last_clk = Some((hours, minutes, seconds));
                                }
                            }
                        },
                    }
                }

                if let Some(last_comment) = self.comments.last_mut() {
                    *last_comment = Some(move_comments);
                }
            }
            Err(e) => {
                eprintln!("Error parsing comment: {:?}", e);
            }
        }
        ControlFlow::Continue(())
    }

    fn begin_variation(
        &mut self,
        _movetext: &mut Self::Movetext,
    ) -> ControlFlow<Self::Output, Skip> {
        ControlFlow::Continue(Skip(true)) // stay in the mainline
    }

    fn outcome(
        &mut self,
        _movetext: &mut Self::Movetext,
        _outcome: Outcome,
    ) -> ControlFlow<Self::Output> {
        self.outcome = Some(match _outcome {
            Outcome::Known(known) => match known {
                KnownOutcome::Decisive { winner } => format!("{:?}", winner),
                KnownOutcome::Draw => "Draw".to_string(),
            },
            Outcome::Unknown => "Unknown".to_string(),
        });
        self.update_position_status();
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {
        self.valid_moves
    }
}

// --- Native Rust versions (no PyResult) ---
pub fn parse_single_game_native(
    pgn: &str,
    store_legal_moves: bool,
) -> Result<MoveExtractor, String> {
    let mut reader = Reader::new(Cursor::new(pgn));
    let mut extractor = MoveExtractor::new(store_legal_moves);
    match reader.read_game(&mut extractor) {
        Ok(Some(_)) => Ok(extractor),
        Ok(None) => Err("No game found in PGN".to_string()),
        Err(err) => Err(format!("Parsing error: {}", err)),
    }
}

pub fn parse_multiple_games_native(
    pgns: &Vec<String>,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> Result<Vec<MoveExtractor>, String> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);

    // Build a custom Rayon thread pool with the desired number of threads
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build Rayon thread pool");

    thread_pool.install(|| {
        pgns.par_iter()
            .map(|pgn| parse_single_game_native(pgn, store_legal_moves))
            .collect()
    })
}

fn _parse_game_moves_from_arrow_chunks_native(
    pgn_chunked_array: &PyChunkedArray,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> Result<Vec<MoveExtractor>, String> {
    let num_threads = num_threads.unwrap_or_else(num_cpus::get);
    let thread_pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| format!("Failed to build Rayon thread pool: {}", e))?;

    let mut num_elements = 0;
    for chunk in pgn_chunked_array.chunks() {
        num_elements += chunk.len();
    }
    let mut pgn_str_slices: Vec<&str> = Vec::with_capacity(num_elements);
    for chunk in pgn_chunked_array.chunks() {
        if let Some(string_array) = chunk.as_any().downcast_ref::<StringArray>() {
            for i in 0..string_array.len() {
                if string_array.is_valid(i) {
                    pgn_str_slices.push(string_array.value(i));
                }
            }
        } else if let Some(large_string_array) = chunk.as_any().downcast_ref::<LargeStringArray>() {
            for i in 0..large_string_array.len() {
                if large_string_array.is_valid(i) {
                    pgn_str_slices.push(large_string_array.value(i));
                }
            }
        } else {
            return Err(format!(
                "Unsupported array type in ChunkedArray: {:?}",
                chunk.data_type()
            ));
        }
    }

    thread_pool.install(|| {
        pgn_str_slices
            .par_iter()
            .map(|&pgn_s| parse_single_game_native(pgn_s, store_legal_moves))
            .collect::<Result<Vec<MoveExtractor>, String>>()
    })
}

// --- Python-facing wrappers (PyResult) ---
// TODO check if I can call py.allow_threads and release GIL
// see https://docs.rs/pyo3-arrow/0.10.1/pyo3_arrow/
#[pyfunction]
#[pyo3(signature = (pgn, store_legal_moves = false))]
/// Parses a single PGN game string.
fn parse_game(pgn: &str, store_legal_moves: bool) -> PyResult<MoveExtractor> {
    parse_single_game_native(pgn, store_legal_moves)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// In parallel, parse a set of games
#[pyfunction]
#[pyo3(signature = (pgns, num_threads=None, store_legal_moves=false))]
fn parse_games(
    pgns: Vec<String>,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> PyResult<Vec<MoveExtractor>> {
    parse_multiple_games_native(&pgns, num_threads, store_legal_moves)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None, store_legal_moves=false))]
fn parse_game_moves_arrow_chunked_array(
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
    store_legal_moves: bool,
) -> PyResult<Vec<MoveExtractor>> {
    _parse_game_moves_from_arrow_chunks_native(&pgn_chunked_array, num_threads, store_legal_moves)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Parser for chess PGN notation
#[pymodule]
fn rust_pgn_reader_python_binding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_game, m)?)?;
    m.add_function(wrap_pyfunction!(parse_games, m)?)?;
    m.add_function(wrap_pyfunction!(parse_game_moves_arrow_chunked_array, m)?)?;
    m.add_class::<MoveExtractor>()?;
    m.add_class::<PositionStatus>()?;
    m.add_class::<PyUciMove>()?;
    Ok(())
}

#[cfg(test)]
mod pyucimove_tests {
    use super::*;
    use shakmaty::{Role, Square};

    #[test]
    fn test_py_uci_move_no_promotion() {
        let uci_move = PyUciMove::new(Square::E2 as u8, Square::E4 as u8, None);
        assert_eq!(uci_move.from_square, Square::E2 as u8);
        assert_eq!(uci_move.to_square, Square::E4 as u8);
        assert_eq!(uci_move.promotion, None);
        assert_eq!(uci_move.get_from_square_name(), "e2");
        assert_eq!(uci_move.get_to_square_name(), "e4");
        assert_eq!(uci_move.get_promotion_name(), None);
        assert_eq!(uci_move.__str__(), "e2e4");
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=e2, to_square=e4, promotion=None)"
        );
    }

    #[test]
    fn test_py_uci_move_with_queen_promotion() {
        let uci_move = PyUciMove::new(Square::E7 as u8, Square::E8 as u8, Some(Role::Queen as u8));
        assert_eq!(uci_move.from_square, Square::E7 as u8);
        assert_eq!(uci_move.to_square, Square::E8 as u8);
        assert_eq!(uci_move.promotion, Some(Role::Queen as u8));
        assert_eq!(uci_move.get_from_square_name(), "e7");
        assert_eq!(uci_move.get_to_square_name(), "e8");
        assert_eq!(uci_move.get_promotion_name(), Some("Queen".to_string()));
        assert_eq!(uci_move.__str__(), "e7e8q");
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=e7, to_square=e8, promotion=Some('q'))"
        );
    }

    #[test]
    fn test_py_uci_move_with_rook_promotion() {
        let uci_move = PyUciMove::new(Square::A7 as u8, Square::A8 as u8, Some(Role::Rook as u8));
        assert_eq!(uci_move.from_square, Square::A7 as u8);
        assert_eq!(uci_move.to_square, Square::A8 as u8);
        assert_eq!(uci_move.promotion, Some(Role::Rook as u8));
        assert_eq!(uci_move.get_from_square_name(), "a7");
        assert_eq!(uci_move.get_to_square_name(), "a8");
        assert_eq!(uci_move.get_promotion_name(), Some("Rook".to_string()));
        assert_eq!(uci_move.__str__(), "a7a8r");
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=a7, to_square=a8, promotion=Some('r'))"
        );
    }

    #[test]
    fn test_py_uci_move_invalid_promotion_val() {
        // Test with a u8 value that doesn't correspond to a valid Role
        let uci_move = PyUciMove::new(Square::B7 as u8, Square::B8 as u8, Some(99)); // 99 is not a valid Role
        assert_eq!(uci_move.from_square, Square::B7 as u8);
        assert_eq!(uci_move.to_square, Square::B8 as u8);
        assert_eq!(uci_move.promotion, Some(99));
        assert_eq!(uci_move.get_from_square_name(), "b7");
        assert_eq!(uci_move.get_to_square_name(), "b8");
        assert_eq!(uci_move.get_promotion_name(), None); // Should be None as 99 is invalid
        assert_eq!(uci_move.__str__(), "b7b8"); // Should produce no promotion char
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=b7, to_square=b8, promotion=Some(InvalidRole(99)))"
        );
    }
}
