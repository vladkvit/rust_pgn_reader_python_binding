use crate::board_serialization::{
    get_castling_rights, get_en_passant_file, get_halfmove_clock, get_turn, serialize_board,
};
use crate::comment_parsing::{parse_comments, CommentContent, ParsedTag};
use crate::python_bindings::{PositionStatus, PyUciMove};
use pgn_reader::{KnownOutcome, Outcome, RawComment, RawTag, SanPlus, Skip, Visitor};
use pyo3::prelude::*;
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Color, Position};
use std::ops::ControlFlow;

#[pyclass]
/// A Visitor to extract SAN moves and comments from PGN movetext
pub struct MoveExtractor {
    #[pyo3(get)]
    pub moves: Vec<PyUciMove>,

    pub store_legal_moves: bool,
    pub store_board_states: bool,
    pub flat_legal_moves: Vec<PyUciMove>,
    pub legal_moves_offsets: Vec<usize>,

    #[pyo3(get)]
    pub valid_moves: bool,

    #[pyo3(get)]
    pub comments: Vec<Option<String>>,

    #[pyo3(get)]
    pub evals: Vec<Option<f64>>,

    #[pyo3(get)]
    pub clock_times: Vec<Option<(u32, u8, f64)>>,

    #[pyo3(get)]
    pub outcome: Option<String>,

    #[pyo3(get)]
    pub headers: Vec<(String, String)>,

    #[pyo3(get)]
    pub castling_rights: Vec<Option<(bool, bool, bool, bool)>>,

    #[pyo3(get)]
    pub position_status: Option<PositionStatus>,

    pub pos: Chess,

    // Board state tracking for flat output (not directly exposed to Python)
    // Only populated if store_board_states is true
    pub board_states: Vec<u8>,      // Flattened: 64 bytes per position
    pub en_passant_states: Vec<i8>, // Per position: -1 or file 0-7
    pub halfmove_clocks: Vec<u8>,   // Per position
    pub turn_states: Vec<bool>,     // Per position: true=white
    pub castling_states: Vec<bool>, // Flattened: 4 bools per position [K,Q,k,q]
}

#[pymethods]
impl MoveExtractor {
    #[new]
    #[pyo3(signature = (store_legal_moves = false, store_board_states = false))]
    pub fn new(store_legal_moves: bool, store_board_states: bool) -> MoveExtractor {
        MoveExtractor {
            moves: Vec::with_capacity(100),
            store_legal_moves,
            store_board_states,
            flat_legal_moves: Vec::with_capacity(if store_legal_moves { 100 * 30 } else { 0 }),
            legal_moves_offsets: Vec::with_capacity(if store_legal_moves { 100 } else { 0 }),
            pos: Chess::default(),
            valid_moves: true,
            comments: Vec::with_capacity(100),
            evals: Vec::with_capacity(100),
            clock_times: Vec::with_capacity(100),
            outcome: None,
            headers: Vec::with_capacity(10),
            castling_rights: Vec::with_capacity(100),
            position_status: None,
            // Only pre-allocate if storing board states
            board_states: Vec::with_capacity(if store_board_states { 100 * 64 } else { 0 }),
            en_passant_states: Vec::with_capacity(if store_board_states { 100 } else { 0 }),
            halfmove_clocks: Vec::with_capacity(if store_board_states { 100 } else { 0 }),
            turn_states: Vec::with_capacity(if store_board_states { 100 } else { 0 }),
            castling_states: Vec::with_capacity(if store_board_states { 100 * 4 } else { 0 }),
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
        // Record initial board state for flat output (only if enabled)
        if self.store_board_states {
            self.push_board_state();
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
                    // Record board state after move for flat output (only if enabled)
                    if self.store_board_states {
                        self.push_board_state();
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
