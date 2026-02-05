use crate::comment_parsing::{parse_comments, CommentContent, ParsedTag};
use arrow_array::{Array, LargeStringArray, StringArray};
use numpy::{PyArray1, PyArrayMethods};
use pgn_reader::{KnownOutcome, Outcome, RawComment, RawTag, Reader, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use pyo3_arrow::PyChunkedArray;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use shakmaty::fen::Fen;
use shakmaty::CastlingMode;
use shakmaty::Color;
use shakmaty::{uci::UciMove, Chess, Position, Role, Square};
use std::collections::HashMap;
use std::io::Cursor;
use std::ops::ControlFlow;

mod board_serialization;
mod comment_parsing;

use board_serialization::{
    get_castling_rights, get_en_passant_file, get_halfmove_clock, get_turn, serialize_board,
};

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

    // Board state tracking for flat output (not directly exposed to Python)
    board_states: Vec<u8>,      // Flattened: 64 bytes per position
    en_passant_states: Vec<i8>, // Per position: -1 or file 0-7
    halfmove_clocks: Vec<u8>,   // Per position
    turn_states: Vec<bool>,     // Per position: true=white
    castling_states: Vec<bool>, // Flattened: 4 bools per position [K,Q,k,q]
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
            board_states: Vec::with_capacity(100 * 64),
            en_passant_states: Vec::with_capacity(100),
            halfmove_clocks: Vec::with_capacity(100),
            turn_states: Vec::with_capacity(100),
            castling_states: Vec::with_capacity(100 * 4),
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

    /// Record current board state to flat arrays for ParsedGames output.
    fn push_board_state(&mut self) {
        self.board_states
            .extend_from_slice(&serialize_board(&self.pos));
        self.en_passant_states.push(get_en_passant_file(&self.pos));
        self.halfmove_clocks.push(get_halfmove_clock(&self.pos));
        self.turn_states.push(get_turn(&self.pos));
        let castling = get_castling_rights(&self.pos);
        self.castling_states.extend_from_slice(&castling);
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
        self.valid_moves = true;
        self.comments.clear();
        self.evals.clear();
        self.clock_times.clear();
        self.castling_rights.clear();
        self.board_states.clear();
        self.en_passant_states.clear();
        self.halfmove_clocks.clear();
        self.turn_states.clear();
        self.castling_states.clear();

        // Determine castling mode from Variant header (case-insensitive)
        let castling_mode = self
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("Variant"))
            .and_then(|(_, v)| {
                let v_lower = v.to_lowercase();
                if v_lower == "chess960" {
                    Some(CastlingMode::Chess960)
                } else {
                    None
                }
            })
            .unwrap_or(CastlingMode::Standard);

        // Try to parse FEN from headers, fall back to default position
        let fen_header = self
            .headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("FEN"))
            .map(|(_, v)| v.as_str());

        if let Some(fen_str) = fen_header {
            match fen_str.parse::<Fen>() {
                Ok(fen) => match fen.into_position(castling_mode) {
                    Ok(pos) => self.pos = pos,
                    Err(e) => {
                        eprintln!("invalid FEN position: {}", e);
                        self.pos = Chess::default();
                        self.valid_moves = false;
                    }
                },
                Err(e) => {
                    eprintln!("failed to parse FEN: {}", e);
                    self.pos = Chess::default();
                    self.valid_moves = false;
                }
            }
        } else {
            self.pos = Chess::default();
        }

        self.push_castling_bitboards();
        if self.store_legal_moves {
            self.push_legal_moves();
        }
        // Record initial board state for flat output
        self.push_board_state();
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
                    // Record board state after move for flat output
                    self.push_board_state();
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

/// Flat array container for parsed chess games, optimized for ML training.
///
/// # Indexing
/// - `N_games`: Number of games
/// - `N_moves`: Total moves across all games  
/// - `N_positions`: Total board positions recorded (varies per game due to initial position + moves)
///
/// # Board layout
/// Boards use square indexing: a1=0, b1=1, ..., h8=63
/// Piece encoding: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
#[pyclass]
pub struct ParsedGames {
    // === Board state arrays (N_positions) ===
    /// Board positions, shape (N_positions, 8, 8), dtype uint8
    #[pyo3(get)]
    boards: Py<PyAny>,

    /// Castling rights [K,Q,k,q], shape (N_positions, 4), dtype bool
    #[pyo3(get)]
    castling: Py<PyAny>,

    /// En passant file (-1 if none), shape (N_positions,), dtype int8
    #[pyo3(get)]
    en_passant: Py<PyAny>,

    /// Halfmove clock, shape (N_positions,), dtype uint8
    #[pyo3(get)]
    halfmove_clock: Py<PyAny>,

    /// Side to move (true=white), shape (N_positions,), dtype bool
    #[pyo3(get)]
    turn: Py<PyAny>,

    // === Move arrays (N_moves) ===
    /// From squares, shape (N_moves,), dtype uint8
    #[pyo3(get)]
    from_squares: Py<PyAny>,

    /// To squares, shape (N_moves,), dtype uint8
    #[pyo3(get)]
    to_squares: Py<PyAny>,

    /// Promotions (-1=none, 2=N, 3=B, 4=R, 5=Q), shape (N_moves,), dtype int8
    #[pyo3(get)]
    promotions: Py<PyAny>,

    /// Clock times in seconds (NaN if missing), shape (N_moves,), dtype float32
    #[pyo3(get)]
    clocks: Py<PyAny>,

    /// Engine evals (NaN if missing), shape (N_moves,), dtype float32
    #[pyo3(get)]
    evals: Py<PyAny>,

    // === Offsets ===
    /// Move offsets for CSR-style indexing, shape (N_games + 1,), dtype uint32
    /// Game i's moves: move_offsets[i]..move_offsets[i+1]
    #[pyo3(get)]
    move_offsets: Py<PyAny>,

    /// Position offsets for CSR-style indexing, shape (N_games + 1,), dtype uint32
    /// Game i's positions: position_offsets[i]..position_offsets[i+1]
    #[pyo3(get)]
    position_offsets: Py<PyAny>,

    // === Final position status (N_games) ===
    /// Final position is checkmate, shape (N_games,), dtype bool
    #[pyo3(get)]
    is_checkmate: Py<PyAny>,

    /// Final position is stalemate, shape (N_games,), dtype bool
    #[pyo3(get)]
    is_stalemate: Py<PyAny>,

    /// Insufficient material (white, black), shape (N_games, 2), dtype bool
    #[pyo3(get)]
    is_insufficient: Py<PyAny>,

    /// Legal move count in final position, shape (N_games,), dtype uint16
    #[pyo3(get)]
    legal_move_count: Py<PyAny>,

    // === Parse status (N_games) ===
    /// Whether game parsed successfully, shape (N_games,), dtype bool
    #[pyo3(get)]
    valid: Py<PyAny>,

    // === Raw headers (N_games) ===
    /// Raw PGN headers as list of dicts
    #[pyo3(get)]
    headers: Vec<HashMap<String, String>>,
}

#[pyfunction]
#[pyo3(signature = (pgn_chunked_array, num_threads=None))]
fn parse_games_flat(
    py: Python<'_>,
    pgn_chunked_array: PyChunkedArray,
    num_threads: Option<usize>,
) -> PyResult<ParsedGames> {
    // 1. Parse all games using existing logic
    let extractors = _parse_game_moves_from_arrow_chunks_native(
        &pgn_chunked_array,
        num_threads,
        false, // store_legal_moves = false for performance
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let n_games = extractors.len();

    // 2. Compute move counts and position counts from actual recorded data
    let move_counts: Vec<u32> = extractors.iter().map(|e| e.moves.len() as u32).collect();

    // Position counts derived from actual board_states data
    let position_counts: Vec<u32> = extractors
        .iter()
        .map(|e| (e.board_states.len() / 64) as u32)
        .collect();

    // Build move offsets
    let mut move_offsets_vec: Vec<u32> = Vec::with_capacity(n_games + 1);
    move_offsets_vec.push(0);
    for &count in &move_counts {
        move_offsets_vec.push(move_offsets_vec.last().unwrap() + count);
    }

    // Build position offsets
    let mut position_offsets_vec: Vec<u32> = Vec::with_capacity(n_games + 1);
    position_offsets_vec.push(0);
    for &count in &position_counts {
        position_offsets_vec.push(position_offsets_vec.last().unwrap() + count);
    }

    let total_moves = *move_offsets_vec.last().unwrap() as usize;
    let total_positions = *position_offsets_vec.last().unwrap() as usize;

    // 3. Pre-allocate flat vectors
    let mut boards_vec: Vec<u8> = Vec::with_capacity(total_positions * 64);
    let mut castling_vec: Vec<bool> = Vec::with_capacity(total_positions * 4);
    let mut en_passant_vec: Vec<i8> = Vec::with_capacity(total_positions);
    let mut halfmove_clock_vec: Vec<u8> = Vec::with_capacity(total_positions);
    let mut turn_vec: Vec<bool> = Vec::with_capacity(total_positions);

    let mut from_squares_vec: Vec<u8> = Vec::with_capacity(total_moves);
    let mut to_squares_vec: Vec<u8> = Vec::with_capacity(total_moves);
    let mut promotions_vec: Vec<i8> = Vec::with_capacity(total_moves);
    let mut clocks_vec: Vec<f32> = Vec::with_capacity(total_moves);
    let mut evals_vec: Vec<f32> = Vec::with_capacity(total_moves);

    let mut is_checkmate_vec: Vec<bool> = Vec::with_capacity(n_games);
    let mut is_stalemate_vec: Vec<bool> = Vec::with_capacity(n_games);
    let mut is_insufficient_vec: Vec<bool> = Vec::with_capacity(n_games * 2);
    let mut legal_move_count_vec: Vec<u16> = Vec::with_capacity(n_games);
    let mut valid_vec: Vec<bool> = Vec::with_capacity(n_games);
    let mut headers_vec: Vec<HashMap<String, String>> = Vec::with_capacity(n_games);

    // 4. Copy data from each extractor
    for extractor in &extractors {
        // Board states
        boards_vec.extend_from_slice(&extractor.board_states);
        castling_vec.extend(extractor.castling_states.iter().copied());
        en_passant_vec.extend_from_slice(&extractor.en_passant_states);
        halfmove_clock_vec.extend_from_slice(&extractor.halfmove_clocks);
        turn_vec.extend(extractor.turn_states.iter().copied());

        // Moves
        for m in &extractor.moves {
            from_squares_vec.push(m.from_square);
            to_squares_vec.push(m.to_square);
            promotions_vec.push(m.promotion.map(|p| p as i8).unwrap_or(-1));
        }

        // Clocks (convert to seconds)
        for clock in &extractor.clock_times {
            clocks_vec.push(
                clock
                    .map(|(h, m, s)| h as f32 * 3600.0 + m as f32 * 60.0 + s as f32)
                    .unwrap_or(f32::NAN),
            );
        }

        // Evals (convert mate values to large numbers)
        for eval in &extractor.evals {
            evals_vec.push(eval.map(|e| e as f32).unwrap_or(f32::NAN));
        }

        // Final position status
        if let Some(ref status) = extractor.position_status {
            is_checkmate_vec.push(status.is_checkmate);
            is_stalemate_vec.push(status.is_stalemate);
            is_insufficient_vec.push(status.insufficient_material.0);
            is_insufficient_vec.push(status.insufficient_material.1);
            legal_move_count_vec.push(status.legal_move_count as u16);
        } else {
            // No status computed - use defaults
            is_checkmate_vec.push(false);
            is_stalemate_vec.push(false);
            is_insufficient_vec.push(false);
            is_insufficient_vec.push(false);
            legal_move_count_vec.push(0);
        }

        // Valid flag
        valid_vec.push(extractor.valid_moves);

        // Headers as HashMap
        let header_map: HashMap<String, String> = extractor
            .headers
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        headers_vec.push(header_map);
    }

    // 5. Convert to numpy arrays
    // Boards: reshape from flat to (N_positions, 8, 8)
    let boards_array = PyArray1::from_vec(py, boards_vec);
    let boards_reshaped = boards_array
        .reshape([total_positions, 8, 8])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Castling: reshape from flat to (N_positions, 4)
    let castling_array = PyArray1::from_vec(py, castling_vec);
    let castling_reshaped = castling_array
        .reshape([total_positions, 4])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // 1D arrays
    let en_passant_array = PyArray1::from_vec(py, en_passant_vec);
    let halfmove_clock_array = PyArray1::from_vec(py, halfmove_clock_vec);
    let turn_array = PyArray1::from_vec(py, turn_vec);

    let from_squares_array = PyArray1::from_vec(py, from_squares_vec);
    let to_squares_array = PyArray1::from_vec(py, to_squares_vec);
    let promotions_array = PyArray1::from_vec(py, promotions_vec);
    let clocks_array = PyArray1::from_vec(py, clocks_vec);
    let evals_array = PyArray1::from_vec(py, evals_vec);

    let move_offsets_array = PyArray1::from_vec(py, move_offsets_vec);
    let position_offsets_array = PyArray1::from_vec(py, position_offsets_vec);

    let is_checkmate_array = PyArray1::from_vec(py, is_checkmate_vec);
    let is_stalemate_array = PyArray1::from_vec(py, is_stalemate_vec);

    // is_insufficient: reshape to (N_games, 2)
    let is_insufficient_array = PyArray1::from_vec(py, is_insufficient_vec);
    let is_insufficient_reshaped = is_insufficient_array
        .reshape([n_games, 2])
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    let legal_move_count_array = PyArray1::from_vec(py, legal_move_count_vec);
    let valid_array = PyArray1::from_vec(py, valid_vec);

    Ok(ParsedGames {
        boards: boards_reshaped.unbind().into_any(),
        castling: castling_reshaped.unbind().into_any(),
        en_passant: en_passant_array.unbind().into_any(),
        halfmove_clock: halfmove_clock_array.unbind().into_any(),
        turn: turn_array.unbind().into_any(),
        from_squares: from_squares_array.unbind().into_any(),
        to_squares: to_squares_array.unbind().into_any(),
        promotions: promotions_array.unbind().into_any(),
        clocks: clocks_array.unbind().into_any(),
        evals: evals_array.unbind().into_any(),
        move_offsets: move_offsets_array.unbind().into_any(),
        position_offsets: position_offsets_array.unbind().into_any(),
        is_checkmate: is_checkmate_array.unbind().into_any(),
        is_stalemate: is_stalemate_array.unbind().into_any(),
        is_insufficient: is_insufficient_reshaped.unbind().into_any(),
        legal_move_count: legal_move_count_array.unbind().into_any(),
        valid: valid_array.unbind().into_any(),
        headers: headers_vec,
    })
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
    m.add_function(wrap_pyfunction!(parse_games_flat, m)?)?;
    m.add_class::<MoveExtractor>()?;
    m.add_class::<PositionStatus>()?;
    m.add_class::<PyUciMove>()?;
    m.add_class::<ParsedGames>()?;
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

    #[test]
    fn test_parse_game_without_headers() {
        let pgn = "1. Nf3 d5 2. e4 c5 3. exd5 e5 4. dxe6 0-1";
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert_eq!(extractor.moves.len(), 7);
        assert_eq!(extractor.outcome, Some("Black".to_string()));
    }

    #[test]
    fn test_parse_game_with_standard_fen() {
        // A game starting from a mid-game position
        let pgn = r#"[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

3. Bb5 a6 4. Ba4 Nf6 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(extractor.valid_moves, "Moves should be valid");
        assert_eq!(extractor.moves.len(), 4);
    }

    #[test]
    fn test_parse_chess960_game() {
        // Chess960 game with custom starting position
        let pgn = r#"[Variant "chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1. g3 d5 2. d4 g6 3. b3 Nf6 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Chess960 moves should be valid with proper FEN"
        );
        assert_eq!(extractor.moves.len(), 6);
    }

    #[test]
    fn test_parse_chess960_variant_case_insensitive() {
        // Test that variant detection is case-insensitive
        let pgn = r#"[Variant "Chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1. g3 d5 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Should handle Chess960 case variations"
        );
    }

    #[test]
    fn test_parse_invalid_fen_falls_back() {
        // Invalid FEN should fall back to default and mark invalid
        let pgn = r#"[FEN "invalid fen string"]

1. e4 e5 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            !extractor.valid_moves,
            "Should mark as invalid when FEN parsing fails"
        );
    }

    #[test]
    fn test_fen_header_case_insensitive() {
        // FEN header key should be case-insensitive
        let pgn = r#"[fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

3. Bb5 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Should handle lowercase 'fen' header"
        );
    }

    #[test]
    fn test_parse_game_with_custom_fen_no_variant() {
        // A standard chess game starting from a mid-game position (no Variant header)
        // Position after 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Ruy Lopez)
        let pgn = r#"[Event "Test Game"]
    [FEN "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"]

    3... a6 4. Ba4 Nf6 5. O-O Be7 1-0"#;
        let result = parse_single_game_native(pgn, false);
        assert!(result.is_ok());
        let extractor = result.unwrap();
        assert!(
            extractor.valid_moves,
            "Standard game with custom FEN should be valid"
        );
        assert_eq!(extractor.moves.len(), 5); // a6, Ba4, Nf6, O-O, Be7
    }
}
