//! Flat buffer visitor for direct SoA output.
//!
//! This module provides a memory-efficient parsing approach that writes
//! directly to flat buffers instead of allocating per-game Vec structures.
//! Used by `parse_games_flat` for optimal performance.

use crate::board_serialization::{
    get_castling_rights, get_en_passant_file, get_halfmove_clock, get_turn, serialize_board,
};
use crate::comment_parsing::{parse_comments, CommentContent, ParsedTag};
use pgn_reader::{Outcome, RawComment, RawTag, SanPlus, Skip, Visitor};
use shakmaty::{fen::Fen, uci::UciMove, CastlingMode, Chess, Color, Position};
use std::collections::HashMap;
use std::ops::ControlFlow;

/// Accumulated flat buffers for multiple parsed games.
///
/// This struct holds all data in a struct-of-arrays layout, optimized for:
/// - Efficient thread-local accumulation during parallel parsing
/// - Fast merging of thread-local buffers via `extend_from_slice`
/// - Direct conversion to NumPy arrays without intermediate allocations
#[derive(Default, Clone)]
pub struct FlatBuffers {
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
}

impl FlatBuffers {
    /// Create a new FlatBuffers with pre-allocated capacity.
    ///
    /// # Arguments
    /// * `estimated_games` - Expected number of games
    /// * `moves_per_game` - Expected average moves per game (default: 70)
    pub fn with_capacity(estimated_games: usize, moves_per_game: usize) -> Self {
        let estimated_moves = estimated_games * moves_per_game;
        let estimated_positions = estimated_moves + estimated_games; // +1 initial position per game

        FlatBuffers {
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

    /// Compute CSR-style offsets from counts.
    pub fn compute_move_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.move_counts.len() + 1);
        offsets.push(0);
        for &count in &self.move_counts {
            offsets.push(offsets.last().unwrap() + count);
        }
        offsets
    }

    /// Compute CSR-style offsets from position counts.
    pub fn compute_position_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.position_counts.len() + 1);
        offsets.push(0);
        for &count in &self.position_counts {
            offsets.push(offsets.last().unwrap() + count);
        }
        offsets
    }
}

/// Visitor that writes directly to FlatBuffers.
///
/// This visitor does not allocate any per-game Vec structures.
/// All data is appended directly to the shared FlatBuffers.
pub struct FlatVisitor<'a> {
    buffers: &'a mut FlatBuffers,
    pos: Chess,
    valid_moves: bool,
    current_headers: Vec<(String, String)>,
    // Track counts for current game
    current_move_count: u32,
    current_position_count: u32,
}

impl<'a> FlatVisitor<'a> {
    pub fn new(buffers: &'a mut FlatBuffers) -> Self {
        FlatVisitor {
            buffers,
            pos: Chess::default(),
            valid_moves: true,
            current_headers: Vec::with_capacity(10),
            current_move_count: 0,
            current_position_count: 0,
        }
    }

    /// Record current board state to flat buffers.
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
    }

    /// Record move data to flat buffers.
    fn push_move(&mut self, from: u8, to: u8, promotion: Option<u8>) {
        self.buffers.from_squares.push(from);
        self.buffers.to_squares.push(to);
        self.buffers
            .promotions
            .push(promotion.map(|p| p as i8).unwrap_or(-1));
        // Push placeholders for clock and eval (will be overwritten by comment())
        self.buffers.clocks.push(f32::NAN);
        self.buffers.evals.push(f32::NAN);
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

    /// Finalize current game - record per-game data.
    fn finalize_game(&mut self) {
        self.buffers.move_counts.push(self.current_move_count);
        self.buffers
            .position_counts
            .push(self.current_position_count);
        self.buffers.valid.push(self.valid_moves);

        // Convert headers to HashMap
        let header_map: HashMap<String, String> = self.current_headers.drain(..).collect();
        self.buffers.headers.push(header_map);
    }
}

impl Visitor for FlatVisitor<'_> {
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
                            eprintln!(
                                "Unexpected UCI move type: {:?}. Game moves might be invalid.",
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
        comment: RawComment<'_>,
    ) -> ControlFlow<Self::Output> {
        if let Ok((_, parsed_comments)) = parse_comments(comment.as_bytes()) {
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
                        ParsedTag::Mate(_) => {
                            // Mate scores are handled as comments, not numeric evals
                        }
                    },
                    CommentContent::Text(_) => {
                        // Text comments are not stored in flat output
                    }
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

/// Parse a single game directly into FlatBuffers.
pub fn parse_game_to_flat(pgn: &str, buffers: &mut FlatBuffers) -> Result<bool, String> {
    use pgn_reader::Reader;
    use std::io::Cursor;

    let mut reader = Reader::new(Cursor::new(pgn));
    let mut visitor = FlatVisitor::new(buffers);

    match reader.read_game(&mut visitor) {
        Ok(Some(valid)) => Ok(valid),
        Ok(None) => Err("No game found in PGN".to_string()),
        Err(err) => Err(format!("Parsing error: {}", err)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_game() {
        let pgn = r#"[Event "Test"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0"#;

        let mut buffers = FlatBuffers::with_capacity(1, 70);
        let result = parse_game_to_flat(pgn, &mut buffers);

        assert!(result.is_ok());
        assert!(result.unwrap()); // valid game
        assert_eq!(buffers.num_games(), 1);
        assert_eq!(buffers.move_counts[0], 4); // 4 moves
        assert_eq!(buffers.position_counts[0], 5); // 5 positions (initial + 4 moves)
        assert_eq!(buffers.total_moves(), 4);
        assert_eq!(buffers.total_positions(), 5);
    }

    #[test]
    fn test_parse_game_with_annotations() {
        let pgn = r#"[Event "Test"]
[Result "1-0"]

1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:02:58] } 1-0"#;

        let mut buffers = FlatBuffers::with_capacity(1, 70);
        let result = parse_game_to_flat(pgn, &mut buffers);

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

        let mut buffers = FlatBuffers::with_capacity(2, 70);

        parse_game_to_flat(pgn1, &mut buffers).unwrap();
        parse_game_to_flat(pgn2, &mut buffers).unwrap();

        assert_eq!(buffers.num_games(), 2);
        assert_eq!(buffers.total_moves(), 5); // 2 + 3 moves
        assert_eq!(buffers.move_counts, vec![2, 3]);
    }

    #[test]
    fn test_compute_offsets() {
        let mut buffers = FlatBuffers::default();
        buffers.move_counts = vec![4, 6, 3];
        buffers.position_counts = vec![5, 7, 4];

        let move_offsets = buffers.compute_move_offsets();
        assert_eq!(move_offsets, vec![0, 4, 10, 13]);

        let pos_offsets = buffers.compute_position_offsets();
        assert_eq!(pos_offsets, vec![0, 5, 12, 16]);
    }
}
