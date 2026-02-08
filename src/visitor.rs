//! SoA (Struct-of-Arrays) visitor for PGN parsing.
//!
//! This module provides a memory-efficient parsing approach that writes
//! directly to shared buffers instead of allocating per-game Vec structures.
//! Used by `parse_games` for optimal performance.

use crate::board_serialization::{
    get_castling_rights, get_en_passant_file, get_halfmove_clock, get_turn, serialize_board,
};
use crate::comment_parsing::{parse_comments, CommentContent, ParsedTag};
use pgn_reader::{KnownOutcome, Outcome, RawComment, RawTag, SanPlus, Skip, Visitor};
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Color, Position};
use std::collections::HashMap;
use std::ops::ControlFlow;

/// Configuration for what optional data to store during parsing.
#[derive(Clone, Debug)]
pub struct ParseConfig {
    pub store_comments: bool,
    pub store_legal_moves: bool,
}

/// Compute CSR-style prefix-sum offsets from a slice of counts.
/// Returns a Vec of length `counts.len() + 1`, starting with 0.
fn prefix_sum(counts: &[u32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut acc: u32 = 0;
    offsets.push(0);
    for &count in counts {
        acc += count;
        offsets.push(acc);
    }
    offsets
}

/// Accumulated buffers for multiple parsed games.
///
/// This struct holds all data in a struct-of-arrays layout, optimized for:
/// - Efficient thread-local accumulation during parallel parsing
/// - Fast merging of thread-local buffers via `extend_from_slice`
/// - Direct conversion to NumPy arrays without intermediate allocations
#[derive(Default, Clone)]
pub struct Buffers {
    // Board state arrays (one entry per position)
    pub boards: Vec<u8>,         // Flattened: 64 bytes per position
    pub castling: Vec<bool>,     // Flattened: 4 bools per position [K,Q,k,q]
    pub en_passant: Vec<i8>,     // Per position: -1 or file 0-7
    pub halfmove_clock: Vec<u8>, // Per position
    pub turn: Vec<bool>,         // Per position: true=white

    // Move arrays (one entry per move)
    pub from_squares: Vec<u8>,
    pub to_squares: Vec<u8>,
    pub promotions: Vec<i8>, // -1 for no promotion
    pub clocks: Vec<f32>,    // NaN for missing
    pub evals: Vec<f32>,     // NaN for missing

    // Per-game data
    pub move_counts: Vec<u32>,     // Number of moves per game
    pub position_counts: Vec<u32>, // Number of positions per game
    pub is_checkmate: Vec<bool>,
    pub is_stalemate: Vec<bool>,
    pub is_insufficient: Vec<bool>, // Flattened: 2 bools per game [white, black]
    pub legal_move_count: Vec<u16>,
    pub valid: Vec<bool>,
    pub headers: Vec<HashMap<String, String>>,
    pub outcome: Vec<Option<String>>, // "White", "Black", "Draw", "Unknown", or None
    pub parse_errors: Vec<Option<String>>, // Per-game: None if valid, Some(msg) if not

    // Optional: raw text comments (per-move), only populated when store_comments=true
    pub comments: Vec<Option<String>>,

    // Optional: legal moves at each position, only populated when store_legal_moves=true
    // Stored as flat arrays with CSR-style offsets
    pub legal_move_from_squares: Vec<u8>,
    pub legal_move_to_squares: Vec<u8>,
    pub legal_move_promotions: Vec<i8>,
    pub legal_move_counts: Vec<u32>, // Number of legal moves per position
}

impl Buffers {
    /// Create a new Buffers with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `estimated_games` - Expected number of games
    /// * `moves_per_game` - Expected average moves per game (default: 70)
    /// * `config` - Configuration for optional features
    pub fn with_capacity(
        estimated_games: usize,
        moves_per_game: usize,
        config: &ParseConfig,
    ) -> Self {
        let estimated_moves = estimated_games * moves_per_game;
        let estimated_positions = estimated_moves + estimated_games; // +1 initial position per game

        Buffers {
            // Board state arrays
            boards: Vec::with_capacity(estimated_positions * 64),
            castling: Vec::with_capacity(estimated_positions * 4),
            en_passant: Vec::with_capacity(estimated_positions),
            halfmove_clock: Vec::with_capacity(estimated_positions),
            turn: Vec::with_capacity(estimated_positions),

            // Move arrays
            from_squares: Vec::with_capacity(estimated_moves),
            to_squares: Vec::with_capacity(estimated_moves),
            promotions: Vec::with_capacity(estimated_moves),
            clocks: Vec::with_capacity(estimated_moves),
            evals: Vec::with_capacity(estimated_moves),

            // Per-game data
            move_counts: Vec::with_capacity(estimated_games),
            position_counts: Vec::with_capacity(estimated_games),
            is_checkmate: Vec::with_capacity(estimated_games),
            is_stalemate: Vec::with_capacity(estimated_games),
            is_insufficient: Vec::with_capacity(estimated_games * 2),
            legal_move_count: Vec::with_capacity(estimated_games),
            valid: Vec::with_capacity(estimated_games),
            headers: Vec::with_capacity(estimated_games),
            outcome: Vec::with_capacity(estimated_games),
            parse_errors: Vec::with_capacity(estimated_games),

            // Optional comments
            comments: if config.store_comments {
                Vec::with_capacity(estimated_moves)
            } else {
                Vec::new()
            },

            // Optional legal moves
            legal_move_from_squares: if config.store_legal_moves {
                Vec::with_capacity(estimated_positions * 30)
            } else {
                Vec::new()
            },
            legal_move_to_squares: if config.store_legal_moves {
                Vec::with_capacity(estimated_positions * 30)
            } else {
                Vec::new()
            },
            legal_move_promotions: if config.store_legal_moves {
                Vec::with_capacity(estimated_positions * 30)
            } else {
                Vec::new()
            },
            legal_move_counts: if config.store_legal_moves {
                Vec::with_capacity(estimated_positions)
            } else {
                Vec::new()
            },
        }
    }

    /// Number of games in this buffer.
    pub fn num_games(&self) -> usize {
        self.headers.len()
    }

    /// Total number of moves across all games.
    #[allow(dead_code)]
    pub fn total_moves(&self) -> usize {
        self.from_squares.len()
    }

    /// Total number of positions across all games.
    pub fn total_positions(&self) -> usize {
        self.boards.len() / 64
    }

    /// Compute CSR-style offsets from move counts.
    pub fn compute_move_offsets(&self) -> Vec<u32> {
        prefix_sum(&self.move_counts)
    }

    /// Compute CSR-style offsets from position counts.
    pub fn compute_position_offsets(&self) -> Vec<u32> {
        prefix_sum(&self.position_counts)
    }

    /// Compute CSR-style offsets from legal move counts (per position).
    pub fn compute_legal_move_offsets(&self) -> Vec<u32> {
        prefix_sum(&self.legal_move_counts)
    }

    /// Total number of legal moves stored across all positions.
    pub fn total_legal_moves(&self) -> usize {
        self.legal_move_from_squares.len()
    }
}

/// Visitor that writes directly to shared Buffers.
///
/// This visitor does not allocate any per-game Vec structures.
/// All data is appended directly to the shared Buffers.
pub struct GameVisitor<'a> {
    buffers: &'a mut Buffers,
    config: ParseConfig,
    pos: Chess,
    valid_moves: bool,
    current_headers: Vec<(String, String)>,
    current_outcome: Option<String>,
    current_error: Option<String>,
    // Track counts for current game
    current_move_count: u32,
    current_position_count: u32,
}

impl<'a> GameVisitor<'a> {
    pub fn new(buffers: &'a mut Buffers, config: &ParseConfig) -> Self {
        GameVisitor {
            buffers,
            config: config.clone(),
            pos: Chess::default(),
            valid_moves: true,
            current_headers: Vec::with_capacity(10),
            current_outcome: None,
            current_error: None,
            current_move_count: 0,
            current_position_count: 0,
        }
    }

    /// Record current board state to buffers.
    fn push_board_state(&mut self) {
        self.buffers
            .boards
            .extend_from_slice(&serialize_board(&self.pos));
        let castling = get_castling_rights(&self.pos);
        self.buffers.castling.extend_from_slice(&castling);
        self.buffers.en_passant.push(get_en_passant_file(&self.pos));
        self.buffers
            .halfmove_clock
            .push(get_halfmove_clock(&self.pos));
        self.buffers.turn.push(get_turn(&self.pos));
        self.current_position_count += 1;

        // Store legal moves if enabled
        if self.config.store_legal_moves {
            self.push_legal_moves();
        }
    }

    /// Record legal moves at current position to buffers.
    fn push_legal_moves(&mut self) {
        let legal_moves = self.pos.legal_moves();
        let mut count: u32 = 0;
        for m in legal_moves {
            let uci_move_obj = UciMove::from_standard(m);
            if let UciMove::Normal {
                from,
                to,
                promotion,
            } = uci_move_obj
            {
                self.buffers.legal_move_from_squares.push(from as u8);
                self.buffers.legal_move_to_squares.push(to as u8);
                self.buffers
                    .legal_move_promotions
                    .push(promotion.map(|p| p as i8).unwrap_or(-1));
                count += 1;
            }
        }
        self.buffers.legal_move_counts.push(count);
    }

    /// Record move data to buffers.
    fn push_move(&mut self, from: u8, to: u8, promotion: Option<u8>) {
        self.buffers.from_squares.push(from);
        self.buffers.to_squares.push(to);
        self.buffers
            .promotions
            .push(promotion.map(|p| p as i8).unwrap_or(-1));
        // Push placeholders for clock and eval (will be overwritten by comment())
        self.buffers.clocks.push(f32::NAN);
        self.buffers.evals.push(f32::NAN);
        // Push comment placeholder if enabled (will be overwritten by comment())
        if self.config.store_comments {
            self.buffers.comments.push(None);
        }
        self.current_move_count += 1;
    }

    /// Record final position status.
    fn update_position_status(&mut self) {
        self.buffers.is_checkmate.push(self.pos.is_checkmate());
        self.buffers.is_stalemate.push(self.pos.is_stalemate());
        self.buffers
            .is_insufficient
            .push(self.pos.has_insufficient_material(Color::White));
        self.buffers
            .is_insufficient
            .push(self.pos.has_insufficient_material(Color::Black));
        self.buffers
            .legal_move_count
            .push(self.pos.legal_moves().len() as u16);
    }

    /// Record a parse error for the current game.
    fn set_error(&mut self, msg: String) {
        self.valid_moves = false;
        self.current_error = Some(msg);
    }

    /// Finalize current game - record per-game data.
    fn finalize_game(&mut self) {
        self.buffers.move_counts.push(self.current_move_count);
        self.buffers
            .position_counts
            .push(self.current_position_count);
        self.buffers.valid.push(self.valid_moves);
        self.buffers.outcome.push(self.current_outcome.take());
        self.buffers.parse_errors.push(self.current_error.take());

        // Convert headers to HashMap
        let header_map: HashMap<String, String> = self.current_headers.drain(..).collect();
        self.buffers.headers.push(header_map);
    }
}

impl Visitor for GameVisitor<'_> {
    type Tags = Vec<(String, String)>;
    type Movetext = ();
    type Output = bool;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        self.current_headers.clear();
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
        self.current_headers = tags;
        self.valid_moves = true;
        self.current_outcome = None;
        self.current_error = None;
        self.current_move_count = 0;
        self.current_position_count = 0;

        // Determine castling mode from Variant header (case-insensitive)
        let castling_mode = self
            .current_headers
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
            .current_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("FEN"))
            .map(|(_, v)| v.as_str());

        if let Some(fen_str) = fen_header {
            match fen_str.parse::<Fen>() {
                Ok(fen) => match fen.into_position(castling_mode) {
                    Ok(pos) => self.pos = pos,
                    Err(e) => {
                        self.set_error(format!("invalid FEN position: {}", e));
                        self.pos = Chess::default();
                    }
                },
                Err(e) => {
                    self.set_error(format!("failed to parse FEN: {}", e));
                    self.pos = Chess::default();
                }
            }
        } else {
            self.pos = Chess::default();
        }

        // Record initial board state
        self.push_board_state();
        ControlFlow::Continue(())
    }

    fn san(
        &mut self,
        _movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        if self.valid_moves {
            match san_plus.san.to_move(&self.pos) {
                Ok(m) => {
                    self.pos.play_unchecked(m);

                    // Record board state after move
                    self.push_board_state();

                    let uci_move_obj = UciMove::from_standard(m);
                    match uci_move_obj {
                        UciMove::Normal {
                            from,
                            to,
                            promotion,
                        } => {
                            self.push_move(from as u8, to as u8, promotion.map(|p| p as u8));
                        }
                        _ => {
                            self.set_error(format!("unexpected UCI move type: {:?}", uci_move_obj));
                        }
                    }
                }
                Err(err) => {
                    self.set_error(format!("illegal move: {} {}", err, san_plus));
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn comment(
        &mut self,
        _movetext: &mut Self::Movetext,
        comment: RawComment<'_>,
    ) -> ControlFlow<Self::Output> {
        if let Ok((_, parsed_comments)) = parse_comments(comment.as_bytes()) {
            let mut move_comments = if self.config.store_comments {
                Some(String::new())
            } else {
                None
            };

            for content in parsed_comments {
                match content {
                    CommentContent::Tag(tag_content) => match tag_content {
                        ParsedTag::Eval(eval_value) => {
                            // Update the last eval entry
                            if let Some(last_eval) = self.buffers.evals.last_mut() {
                                *last_eval = eval_value as f32;
                            }
                        }
                        ParsedTag::ClkTime {
                            hours,
                            minutes,
                            seconds,
                        } => {
                            // Convert to seconds and update the last clock entry
                            if let Some(last_clk) = self.buffers.clocks.last_mut() {
                                *last_clk =
                                    hours as f32 * 3600.0 + minutes as f32 * 60.0 + seconds as f32;
                            }
                        }
                        ParsedTag::Mate(mate_value) => {
                            // Mate scores stored as text in comments (matching old API behavior)
                            if let Some(ref mut comments) = move_comments {
                                if !comments.is_empty() && !comments.ends_with(' ') {
                                    comments.push(' ');
                                }
                                comments.push_str(&format!("[Mate {}]", mate_value));
                            }
                        }
                    },
                    CommentContent::Text(text) => {
                        if let Some(ref mut comments) = move_comments {
                            if !text.trim().is_empty() {
                                if !comments.is_empty() {
                                    comments.push(' ');
                                }
                                comments.push_str(&text);
                            }
                        }
                    }
                }
            }

            // Update the last comment entry if comments are enabled
            if let Some(comment_text) = move_comments {
                if let Some(last_comment) = self.buffers.comments.last_mut() {
                    *last_comment = Some(comment_text);
                }
            }
        }
        ControlFlow::Continue(())
    }

    fn begin_variation(
        &mut self,
        _movetext: &mut Self::Movetext,
    ) -> ControlFlow<Self::Output, Skip> {
        ControlFlow::Continue(Skip(true)) // Skip variations, stay in mainline
    }

    fn outcome(
        &mut self,
        _movetext: &mut Self::Movetext,
        _outcome: Outcome,
    ) -> ControlFlow<Self::Output> {
        self.current_outcome = Some(match _outcome {
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
        // Handle case where outcome() was not called (e.g., incomplete game)
        if self.buffers.is_checkmate.len() < self.buffers.headers.len() + 1 {
            self.update_position_status();
        }
        self.finalize_game();
        self.valid_moves
    }
}

/// Parse a single game directly into Buffers.
pub fn parse_game_to_buffers(
    pgn: &str,
    buffers: &mut Buffers,
    config: &ParseConfig,
) -> Result<bool, String> {
    use pgn_reader::Reader;
    use std::io::Cursor;

    let mut reader = Reader::new(Cursor::new(pgn));
    let mut visitor = GameVisitor::new(buffers, config);

    match reader.read_game(&mut visitor) {
        Ok(Some(valid)) => Ok(valid),
        Ok(None) => Err("No game found in PGN".to_string()),
        Err(err) => Err(format!("Parsing error: {}", err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ParseConfig {
        ParseConfig {
            store_comments: false,
            store_legal_moves: false,
        }
    }

    #[test]
    fn test_parse_simple_game() {
        let pgn = r#"[Event "Test"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(result.unwrap()); // valid game
        assert_eq!(buffers.num_games(), 1);
        assert_eq!(buffers.move_counts[0], 4); // 4 moves
        assert_eq!(buffers.position_counts[0], 5); // 5 positions (initial + 4 moves)
        assert_eq!(buffers.total_moves(), 4);
        assert_eq!(buffers.total_positions(), 5);
        assert_eq!(buffers.outcome[0], Some("White".to_string()));
    }

    #[test]
    fn test_parse_game_with_annotations() {
        let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:02:58] } 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert_eq!(buffers.total_moves(), 2);

        // Check that evals were parsed
        assert!(!buffers.evals[0].is_nan());
        assert!((buffers.evals[0] - 0.17).abs() < 0.01);
        assert!(!buffers.evals[1].is_nan());
        assert!((buffers.evals[1] - 0.19).abs() < 0.01);

        // Check that clocks were parsed (3 minutes = 180 seconds)
        assert!(!buffers.clocks[0].is_nan());
        assert!((buffers.clocks[0] - 180.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_games_in_one_buffer() {
        let pgn1 = r#"[Event "Game1"]
[Result "1-0"]

1. e4 e5 1-0"#;

        let pgn2 = r#"[Event "Game2"]
[Result "0-1"]

1. d4 d5 2. c4 0-1"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(2, 70, &config);

        parse_game_to_buffers(pgn1, &mut buffers, &config).unwrap();
        parse_game_to_buffers(pgn2, &mut buffers, &config).unwrap();

        assert_eq!(buffers.num_games(), 2);
        assert_eq!(buffers.total_moves(), 5); // 2 + 3 moves
        assert_eq!(buffers.move_counts, vec![2, 3]);
        assert_eq!(buffers.outcome[0], Some("White".to_string()));
        assert_eq!(buffers.outcome[1], Some("Black".to_string()));
    }

    #[test]
    fn test_compute_offsets() {
        let mut buffers = Buffers::default();
        buffers.move_counts = vec![4, 6, 3];
        buffers.position_counts = vec![5, 7, 4];

        let move_offsets = buffers.compute_move_offsets();
        assert_eq!(move_offsets, vec![0, 4, 10, 13]);

        let pos_offsets = buffers.compute_position_offsets();
        assert_eq!(pos_offsets, vec![0, 5, 12, 16]);
    }

    #[test]
    fn test_outcome_without_headers() {
        // PGN without Result header - outcome comes from movetext
        let pgn = "1. e4 e5 2. Nf3 Nc6 0-1";

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        parse_game_to_buffers(pgn, &mut buffers, &config).unwrap();

        assert_eq!(buffers.outcome[0], Some("Black".to_string()));
    }

    #[test]
    fn test_outcome_draw() {
        let pgn = "1. e4 e5 1/2-1/2";

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        parse_game_to_buffers(pgn, &mut buffers, &config).unwrap();

        assert_eq!(buffers.outcome[0], Some("Draw".to_string()));
    }

    #[test]
    fn test_comments_disabled() {
        let pgn = r#"1. e4 { a comment } e5 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        parse_game_to_buffers(pgn, &mut buffers, &config).unwrap();

        assert!(buffers.comments.is_empty());
    }

    #[test]
    fn test_comments_enabled() {
        let pgn = r#"1. e4 { a comment } 1... e5 { [%eval 0.19] } 1-0"#;

        let config = ParseConfig {
            store_comments: true,
            store_legal_moves: false,
        };
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        parse_game_to_buffers(pgn, &mut buffers, &config).unwrap();

        assert_eq!(buffers.comments.len(), 2);
        // Raw text from PGN includes surrounding spaces from the parser
        assert_eq!(buffers.comments[0], Some(" a comment ".to_string()));
        // The second comment only has an eval tag, so text portion is empty
        assert_eq!(buffers.comments[1], Some("".to_string()));
    }

    #[test]
    fn test_legal_moves_disabled() {
        let pgn = "1. e4 e5 1-0";

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        parse_game_to_buffers(pgn, &mut buffers, &config).unwrap();

        assert!(buffers.legal_move_from_squares.is_empty());
        assert!(buffers.legal_move_counts.is_empty());
    }

    #[test]
    fn test_legal_moves_enabled() {
        let pgn = "1. e4 1-0";

        let config = ParseConfig {
            store_comments: false,
            store_legal_moves: true,
        };
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        parse_game_to_buffers(pgn, &mut buffers, &config).unwrap();

        // 2 positions: initial + after e4
        assert_eq!(buffers.legal_move_counts.len(), 2);
        // Initial position has 20 legal moves
        assert_eq!(buffers.legal_move_counts[0], 20);
        // After e4, black has 20 legal moves
        assert_eq!(buffers.legal_move_counts[1], 20);
        // Total legal moves stored
        assert_eq!(buffers.legal_move_from_squares.len(), 40);
    }

    #[test]
    fn test_parse_game_without_headers() {
        let pgn = "1. Nf3 d5 2. e4 c5 3. exd5 e5 4. dxe6 0-1";

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(result.unwrap()); // valid game
        assert_eq!(buffers.num_games(), 1);
        assert_eq!(buffers.total_moves(), 7);
        assert_eq!(buffers.outcome[0], Some("Black".to_string()));
    }

    #[test]
    fn test_parse_game_with_standard_fen() {
        // A game starting from a mid-game position
        let pgn = r#"[FEN "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

3. Bb5 a6 4. Ba4 Nf6 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(result.unwrap()); // valid game
        assert_eq!(buffers.total_moves(), 4);
    }

    #[test]
    fn test_parse_chess960_game() {
        // Chess960 game with custom starting position
        let pgn = r#"[Variant "chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1. g3 d5 2. d4 g6 3. b3 Nf6 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(
            result.unwrap(),
            "Chess960 moves should be valid with proper FEN"
        );
        assert_eq!(buffers.total_moves(), 6);
    }

    #[test]
    fn test_parse_chess960_variant_case_insensitive() {
        // Test that variant detection is case-insensitive
        let pgn = r#"[Variant "Chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1. g3 d5 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(result.unwrap(), "Should handle Chess960 case variations");
    }

    #[test]
    fn test_parse_invalid_fen_falls_back() {
        // Invalid FEN should fall back to default and mark invalid
        let pgn = r#"[FEN "invalid fen string"]

1. e4 e5 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(
            !result.unwrap(),
            "Should mark as invalid when FEN parsing fails"
        );
    }

    #[test]
    fn test_fen_header_case_insensitive() {
        // FEN header key should be case-insensitive
        let pgn = r#"[fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"]

3. Bb5 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(result.unwrap(), "Should handle lowercase 'fen' header");
    }

    #[test]
    fn test_parse_game_with_custom_fen_no_variant() {
        // Standard chess from a mid-game position (no Variant header)
        // Position after 1.e4 e5 2.Nf3 Nc6 3.Bb5 (Ruy Lopez)
        let pgn = r#"[Event "Test Game"]
[FEN "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"]

3... a6 4. Ba4 Nf6 5. O-O Be7 1-0"#;

        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        let result = parse_game_to_buffers(pgn, &mut buffers, &config);

        assert!(result.is_ok());
        assert!(
            result.unwrap(),
            "Standard game with custom FEN should be valid"
        );
        assert_eq!(buffers.total_moves(), 5); // a6, Ba4, Nf6, O-O, Be7
    }
}
