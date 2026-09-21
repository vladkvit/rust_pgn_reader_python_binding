//! SoA (Struct-of-Arrays) visitor for PGN parsing.
//!
//! Single-pass architecture: one [`tokenizer::parse_game`] call per game,
//! writing directly into shared [`Buffers`] (no per-game Vec structures,
//! no sizing pass — games are independent and appended in order, so
//! per-chunk buffers concatenate trivially and deterministically).
//!
//! Performance notes:
//! - SAN resolution uses the [`san_resolver`] fast path, falling back to
//!   shakmaty's `San::to_move` so error strings and edge-case semantics
//!   stay identical.
//! - The serialized board is maintained incrementally (`apply_move`) and
//!   differentially checked against `serialize_board` in debug builds.
//! - Once a game goes invalid, remaining SAN tokens are skipped without
//!   any SAN parsing (early abort).

use crate::board_serialization::{
    apply_move, get_castling_rights, get_en_passant_file, get_halfmove_clock, get_turn,
    serialize_board,
};
use crate::comment_parsing::{Tag, scan_tags};
use crate::tokenizer::{self, Outcome};
use shakmaty::{
    CastlingMode, CastlingSide, Chess, Color, Position,
    fen::Fen,
    san::{San, SanPlus, Suffix},
    uci::UciMove,
};
use std::collections::HashMap;

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

/// Support castling notated with zeros ("0-0", "0-0-0"), with optional
/// check/checkmate suffix, matching pgn-reader's explicit handling.
fn castling_with_zeros(token: &[u8]) -> Option<SanPlus> {
    let (body, suffix) = match token.last() {
        Some(b'+') => (&token[..token.len() - 1], Some(Suffix::Check)),
        Some(b'#') => (&token[..token.len() - 1], Some(Suffix::Checkmate)),
        _ => (token, None),
    };
    let side = match body {
        b"0-0" => CastlingSide::KingSide,
        b"0-0-0" => CastlingSide::QueenSide,
        _ => return None,
    };
    Some(SanPlus {
        san: San::Castle(side),
        suffix,
    })
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

/// Visitor that parses one game and appends directly to shared Buffers.
///
/// Instances are cheap and stateless between games (`begin_headers`
/// resets per-game state): the same visitor is reused for every game a
/// thread/chunk parses.
struct GameVisitor<'a> {
    buffers: &'a mut Buffers,
    config: &'a ParseConfig,
    pos: Chess,
    /// Serialized form of `pos.board()`, maintained incrementally:
    /// re-initialized per game in `end_headers` (covers FEN/960 starts) and
    /// updated per move in `san_token` via `apply_move`. Only read after
    /// `end_headers` initialized it: an invalid game makes `san_token`
    /// early-return before any board is pushed. Differentially checked
    /// against `serialize_board` in debug builds.
    board: [u8; 64],
    valid_moves: bool,
    current_headers: Vec<(String, String)>,
    current_outcome: Option<String>,
    current_error: Option<String>,
    // Per-game tracking
    current_position_count: u32,
    /// Index into the move arrays where the current game started:
    /// comments, evals and clocks annotate moves of the current game only.
    game_move_base: usize,
    /// Legal-move count of the most recently pushed position (only
    /// maintained when store_legal_moves is set), reused at game end so
    /// the final position's move list is generated at most once.
    last_legal_count: u32,
}

impl<'a> GameVisitor<'a> {
    fn new(buffers: &'a mut Buffers, config: &'a ParseConfig) -> Self {
        GameVisitor {
            buffers,
            config,
            pos: Chess::default(),
            board: [0; 64],
            valid_moves: true,
            current_headers: Vec::with_capacity(10),
            current_outcome: None,
            current_error: None,
            current_position_count: 0,
            game_move_base: 0,
            last_legal_count: 0,
        }
    }

    /// Record current board state to buffers.
    fn push_board_state(&mut self) {
        // Differential safety net for the incremental board: free in
        // release, so every corpus-driven test doubles as a differential
        // test against the bitboard scan.
        debug_assert_eq!(
            self.board,
            serialize_board(&self.pos),
            "incremental board out of sync"
        );
        self.buffers.boards.extend_from_slice(&self.board);
        self.buffers
            .castling
            .extend_from_slice(&get_castling_rights(&self.pos));
        self.buffers.en_passant.push(get_en_passant_file(&self.pos));
        self.buffers
            .halfmove_clock
            .push(get_halfmove_clock(&self.pos));
        self.buffers.turn.push(get_turn(&self.pos));
        self.current_position_count += 1;

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
        self.last_legal_count = count;
    }

    /// Record move data to buffers.
    fn push_move(&mut self, from: u8, to: u8, promotion: Option<u8>) {
        self.buffers.from_squares.push(from);
        self.buffers.to_squares.push(to);
        self.buffers
            .promotions
            .push(promotion.map(|p| p as i8).unwrap_or(-1));
        // Push placeholders for clock and eval (overwritten by comment())
        self.buffers.clocks.push(f32::NAN);
        self.buffers.evals.push(f32::NAN);
        // Push comment placeholder if enabled (overwritten by comment())
        if self.config.store_comments {
            self.buffers.comments.push(None);
        }
    }

    /// Record final position status for the current game.
    fn update_position_status(&mut self) {
        // FEN-error games push no board state: there is no final position
        // to describe. All-false entries and a zero legal-move count carry
        // exactly that meaning ("no position recorded").
        if self.current_position_count == 0 {
            self.buffers.is_checkmate.push(false);
            self.buffers.is_stalemate.push(false);
            self.buffers.is_insufficient.push(false);
            self.buffers.is_insufficient.push(false);
            self.buffers.legal_move_count.push(0);
            return;
        }
        // Reuse the count from the last legal-move generation of the final
        // position when available, instead of generating the list twice.
        // Checkmate/stalemate are definitional from the count plus the
        // checkers bitboard, so the move list is generated at most once.
        let legal_count = if self.config.store_legal_moves {
            self.last_legal_count
        } else {
            self.pos.legal_moves().len() as u32
        };
        let in_check = self.pos.checkers().any();
        self.buffers
            .is_checkmate
            .push(legal_count == 0 && in_check);
        self.buffers
            .is_stalemate
            .push(legal_count == 0 && !in_check);
        self.buffers
            .is_insufficient
            .push(self.pos.has_insufficient_material(Color::White));
        self.buffers
            .is_insufficient
            .push(self.pos.has_insufficient_material(Color::Black));
        self.buffers.legal_move_count.push(legal_count as u16);
    }

    /// Record a parse error for the current game.
    fn set_error(&mut self, msg: String) {
        self.valid_moves = false;
        self.current_error = Some(msg);
    }

    /// 1-based half-move ordinal of the token currently being processed
    /// (i.e. the failing token in error messages), counted from the start
    /// of the game — not from a FEN header's fullmove counter.
    fn current_ply(&self) -> usize {
        self.buffers.from_squares.len() - self.game_move_base + 1
    }

    /// Finalize current game - record per-game data.
    fn finalize_game(&mut self) {
        self.buffers
            .move_counts
            .push((self.buffers.from_squares.len() - self.game_move_base) as u32);
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

impl tokenizer::Visitor for GameVisitor<'_> {
    fn begin_headers(&mut self) {
        self.current_headers.clear();
        self.valid_moves = true;
        self.current_outcome = None;
        self.current_error = None;
        self.current_position_count = 0;
        self.game_move_base = self.buffers.from_squares.len();
    }

    fn header(&mut self, key: &[u8], value: &[u8]) {
        let key_str = String::from_utf8_lossy(key).into_owned();
        let value_str = String::from_utf8_lossy(value).into_owned();
        self.current_headers.push((key_str, value_str));
    }

    fn end_headers(&mut self) {
        // Determine castling mode from Variant header (case-insensitive)
        let castling_mode = self
            .current_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("Variant"))
            .and_then(|(_, v)| {
                if v.eq_ignore_ascii_case("chess960") {
                    Some(CastlingMode::Chess960)
                } else {
                    None
                }
            })
            .unwrap_or(CastlingMode::Standard);

        // Try to parse FEN from headers, fall back to the default position.
        // On FEN error the game is invalid from the start, so no board
        // state is pushed: recording a fake default-position board would
        // silently fabricate data for a game whose start position is
        // unknown.
        let fen_header = self
            .current_headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("FEN"))
            .map(|(_, v)| v.as_str());
        let parsed_pos = match fen_header {
            Some(fen_str) => fen_str
                .parse::<Fen>()
                .map_err(|e| format!("failed to parse FEN: {}", e))
                .and_then(|fen| {
                    fen.into_position(castling_mode)
                        .map_err(|e| format!("invalid FEN position: {}", e))
                }),
            None => Ok(Chess::default()),
        };
        match parsed_pos {
            Ok(pos) => {
                self.pos = pos;
                self.board = serialize_board(&self.pos);
                self.push_board_state();
            }
            Err(msg) => {
                self.set_error(msg);
                // Deterministic position for the (unused — invalid game)
                // status computation in end_game.
                self.pos = Chess::default();
            }
        }
    }

    fn san_token(&mut self, token: &[u8]) {
        // Early abort: once the game is invalid, skip SAN parsing entirely.
        if !self.valid_moves {
            return;
        }

        let san_plus = match castling_with_zeros(token) {
            Some(san_plus) => san_plus,
            None => match SanPlus::from_ascii(token) {
                Ok(san_plus) => san_plus,
                Err(error) => {
                    let token_str = String::from_utf8_lossy(token);
                    self.set_error(format!(
                        "failed to parse SAN at ply {}: {} ({})",
                        self.current_ply(),
                        token_str,
                        error
                    ));
                    return;
                }
            },
        };

        // Fast path (src/san_resolver.rs): resolves the common case without
        // materializing a MoveList. `None` => fall back to shakmaty's
        // `to_move` so error strings and edge-case semantics stay identical.
        // (`resolve` differentially asserts agreement with `to_move` in
        // debug builds, so corpus-driven tests double as differential tests.)
        let resolved = crate::san_resolver::resolve(&self.pos, &san_plus.san);

        let move_result = match resolved {
            Some(m) => Ok(m),
            None => san_plus.san.to_move(&self.pos),
        };

        match move_result {
            Ok(m) => {
                let mover = self.pos.turn();
                apply_move(&mut self.board, m, mover);
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
                        self.set_error(format!(
                            "unexpected UCI move type at ply {}: {:?}",
                            self.current_ply(),
                            uci_move_obj
                        ));
                    }
                }
            }
            Err(err) => {
                self.set_error(format!(
                    "illegal move at ply {}: {} ({})",
                    self.current_ply(),
                    san_plus,
                    err
                ));
            }
        }
    }

    fn comment(&mut self, comment: &[u8]) {
        // Comments annotate the last-written move of the CURRENT game; a
        // comment before the game's first move has no slot to update.
        // (The pre-4.x code could touch the previous game's last slot in
        // that case — a game-boundary quirk, deliberately fixed here.)
        // Once the game is invalid, annotations are ignored entirely,
        // consistent with san_token's early abort.
        let move_count = self.buffers.from_squares.len() - self.game_move_base;
        if !self.valid_moves || move_count == 0 {
            return;
        }
        let last_move = self.buffers.from_squares.len() - 1;

        // Raw comment text, verbatim (tags included), trimmed.
        if self.config.store_comments {
            self.buffers.comments[last_move] =
                Some(String::from_utf8_lossy(comment).trim().to_string());
        }

        scan_tags(comment, |tag| match tag {
            Tag::Eval(eval) => {
                self.buffers.evals[last_move] = eval as f32;
            }
            Tag::Clk(seconds) => {
                self.buffers.clocks[last_move] = seconds as f32;
            }
            // TODO: mate scores ([%eval #N]) are currently dropped from the
            // numeric evals (they remain in the raw comment text when
            // store_comments is on). Decide an encoding — a separate
            // per-move i8 `mates` array (-128 = none) vs. a
            // ±large-magnitude sentinel in `evals` — and store them.
            Tag::Mate(_) => {}
        });
    }

    fn outcome(&mut self, outcome: Outcome) {
        self.current_outcome = Some(match outcome {
            Outcome::WhiteWins => "White".to_string(),
            Outcome::BlackWins => "Black".to_string(),
            Outcome::Draw => "Draw".to_string(),
            Outcome::Unknown => "Unknown".to_string(),
        });
    }

    fn end_game(&mut self) {
        // The tokenizer always calls end_game, and the position cannot
        // change between the outcome token and here, so the final-position
        // status is written exactly once, unconditionally.
        self.update_position_status();
        self.finalize_game();
    }
}

/// Parse a single PGN game string into `buffers`.
///
/// Contract: one game per input slice. Content after the first game (a
/// blank line followed by anything, or a new tag section) is ignored.
///
/// Returns `Ok(valid)` with `valid = false` for games that failed to parse
/// fully (illegal/unparsable move, invalid FEN); the game entry is still
/// recorded. Returns `Err` only for empty/whitespace-only input, which
/// produces no game entry (parity with the pre-4.x "No game found in PGN"
/// behavior, whose callers silently skipped such inputs).
pub fn parse_game_to_buffers(
    pgn: &str,
    buffers: &mut Buffers,
    config: &ParseConfig,
) -> Result<bool, String> {
    if pgn.trim().is_empty() {
        return Err("No game found in PGN".to_string());
    }
    let mut visitor = GameVisitor::new(buffers, config);
    tokenizer::parse_game(pgn.as_bytes(), &mut visitor);
    Ok(visitor.valid_moves)
}

/// Parse a batch of games (byte spans into a caller-owned buffer) into
/// `buffers`. This is the shared worker for all drivers: the eager path
/// slices Arrow strings into one-span-per-game batches; the streaming path
/// (future) supplies spans produced by [`crate::splitter::split_games`].
pub fn parse_batch(
    bytes: &[u8],
    spans: &[std::ops::Range<usize>],
    buffers: &mut Buffers,
    config: &ParseConfig,
) {
    let mut visitor = GameVisitor::new(buffers, config);
    for span in spans {
        tokenizer::parse_game(&bytes[span.clone()], &mut visitor);
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

    fn full_config() -> ParseConfig {
        ParseConfig {
            store_comments: true,
            store_legal_moves: true,
        }
    }

    fn parse_one(pgn: &str, config: &ParseConfig) -> Buffers {
        let mut buffers = Buffers::with_capacity(1, 70, config);
        parse_game_to_buffers(pgn, &mut buffers, config).unwrap();
        buffers
    }

    #[test]
    fn test_parse_simple_game() {
        let pgn = "[Event \"Test\"]\n[White \"Player1\"]\n[Black \"Player2\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0";
        let buffers = parse_one(pgn, &default_config());

        assert_eq!(buffers.num_games(), 1);
        assert!(buffers.valid[0]);
        assert_eq!(buffers.move_counts[0], 4);
        assert_eq!(buffers.position_counts[0], 5); // initial + 4 moves
        assert_eq!(buffers.total_moves(), 4);
        assert_eq!(buffers.total_positions(), 5);
        assert_eq!(buffers.outcome[0], Some("White".to_string()));
        assert_eq!(
            buffers.headers[0].get("Event"),
            Some(&"Test".to_string())
        );
    }

    #[test]
    fn test_parse_game_with_annotations() {
        let pgn = "[Event \"Test\"]\n[Result \"1-0\"]\n\n1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:02:58] } 1-0";
        let buffers = parse_one(pgn, &default_config());

        assert_eq!(buffers.total_moves(), 2);
        assert!((buffers.evals[0] - 0.17).abs() < 0.01);
        assert!((buffers.evals[1] - 0.19).abs() < 0.01);
        assert!((buffers.clocks[0] - 180.0).abs() < 0.01);
        assert!((buffers.clocks[1] - 178.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_games_in_one_buffer() {
        let pgn1 = "[Event \"Game1\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0";
        let pgn2 = "[Event \"Game2\"]\n[Result \"0-1\"]\n\n1. d4 d5 2. c4 0-1";

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

        assert_eq!(buffers.compute_move_offsets(), vec![0, 4, 10, 13]);
        assert_eq!(buffers.compute_position_offsets(), vec![0, 5, 12, 16]);
    }

    #[test]
    fn test_outcome_without_headers() {
        let buffers = parse_one("1. e4 e5 2. Nf3 Nc6 0-1", &default_config());
        assert_eq!(buffers.outcome[0], Some("Black".to_string()));
    }

    #[test]
    fn test_outcome_draw() {
        let buffers = parse_one("1. e4 e5 1/2-1/2", &default_config());
        assert_eq!(buffers.outcome[0], Some("Draw".to_string()));
    }

    #[test]
    fn test_outcome_unknown() {
        let buffers = parse_one("1. e4 e5 *", &default_config());
        assert_eq!(buffers.outcome[0], Some("Unknown".to_string()));
    }

    #[test]
    fn test_comments_disabled() {
        let buffers = parse_one("1. e4 { a comment } e5 1-0", &default_config());
        assert!(buffers.comments.is_empty());
    }

    #[test]
    fn test_comments_enabled() {
        let config = ParseConfig {
            store_comments: true,
            store_legal_moves: false,
        };
        let buffers = parse_one(
            "1. e4 { a comment } 1... e5 { [%eval 0.19] } 1-0",
            &config,
        );

        assert_eq!(buffers.comments.len(), 2);
        // Comments are stored raw, verbatim (tags included), trimmed.
        assert_eq!(buffers.comments[0], Some("a comment".to_string()));
        assert_eq!(buffers.comments[1], Some("[%eval 0.19]".to_string()));
    }

    #[test]
    fn test_comment_before_first_move_has_no_slot() {
        // A pre-game comment has no move to annotate: eval/clk/comment are
        // all ignored (and must not leak into any previous game's slots).
        let config = ParseConfig {
            store_comments: true,
            store_legal_moves: false,
        };
        let mut buffers = Buffers::with_capacity(2, 70, &config);
        parse_game_to_buffers("1. e4 e5 1-0", &mut buffers, &config).unwrap();
        parse_game_to_buffers("{ [%clk 1:00:00] } 1. d4 d5 *", &mut buffers, &config).unwrap();

        assert!(buffers.clocks[1].is_nan(), "no leak into previous game");
        assert!(buffers.clocks[2].is_nan(), "pre-game comment ignored");
        assert_eq!(buffers.comments[2], None);
    }

    #[test]
    fn test_legal_moves_disabled() {
        let buffers = parse_one("1. e4 e5 1-0", &default_config());
        assert!(buffers.legal_move_from_squares.is_empty());
        assert!(buffers.legal_move_counts.is_empty());
    }

    #[test]
    fn test_legal_moves_enabled() {
        let config = ParseConfig {
            store_comments: false,
            store_legal_moves: true,
        };
        let buffers = parse_one("1. e4 1-0", &config);

        // 2 positions: initial + after e4; offsets len = positions + 1
        assert_eq!(buffers.legal_move_counts, vec![20, 20]);
        assert_eq!(buffers.compute_legal_move_offsets(), vec![0, 20, 40]);
        assert_eq!(buffers.total_legal_moves(), 40);
    }

    #[test]
    fn test_headers_only_game() {
        // A game with headers but no movetext at all: the initial position
        // (from the FEN header) must still be recorded.
        let pgn = "[Event \"Test\"]\n[FEN \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n";
        let buffers = parse_one(pgn, &default_config());

        assert_eq!(buffers.num_games(), 1);
        assert!(buffers.valid[0]);
        assert_eq!(buffers.move_counts[0], 0);
        assert_eq!(buffers.position_counts[0], 1);
        assert_eq!(buffers.parse_errors[0], None);
        assert_eq!(buffers.outcome[0], None);
    }

    #[test]
    fn test_parse_game_without_headers() {
        let buffers = parse_one("1. Nf3 d5 2. e4 c5 3. exd5 e5 4. dxe6 0-1", &default_config());
        assert!(buffers.valid[0]);
        assert_eq!(buffers.total_moves(), 7);
        assert_eq!(buffers.outcome[0], Some("Black".to_string()));
    }

    #[test]
    fn test_parse_game_with_standard_fen() {
        let pgn = "[FEN \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n\n3. Bb5 a6 4. Ba4 Nf6 1-0";
        let buffers = parse_one(pgn, &default_config());
        assert!(buffers.valid[0]);
        assert_eq!(buffers.total_moves(), 4);
    }

    #[test]
    fn test_parse_chess960_game() {
        let pgn = "[Variant \"chess960\"]\n[FEN \"brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1\"]\n\n1. g3 d5 2. d4 g6 3. b3 Nf6 1-0";
        let buffers = parse_one(pgn, &default_config());
        assert!(buffers.valid[0], "Chess960 moves should be valid with FEN");
        assert_eq!(buffers.total_moves(), 6);
    }

    #[test]
    fn test_parse_chess960_variant_case_insensitive() {
        let pgn = "[Variant \"Chess960\"]\n[FEN \"brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1\"]\n\n1. g3 d5 1-0";
        let buffers = parse_one(pgn, &default_config());
        assert!(buffers.valid[0], "Should handle Chess960 case variations");
    }

    #[test]
    fn test_parse_invalid_fen_pushes_no_board() {
        // Invalid FEN: the game is invalid from the start, so no board
        // state is pushed (no fabricated default-position board).
        let pgn = "[FEN \"invalid fen string\"]\n\n1. e4 e5 1-0";
        let buffers = parse_one(pgn, &default_config());

        assert!(!buffers.valid[0]);
        assert!(
            buffers.parse_errors[0]
                .as_deref()
                .unwrap()
                .starts_with("failed to parse FEN:")
        );
        assert_eq!(buffers.move_counts[0], 0);
        assert_eq!(buffers.position_counts[0], 0);
        assert_eq!(buffers.legal_move_count[0], 0);
        assert!(buffers.outcome[0].is_some(), "outcome still recorded");
    }

    #[test]
    fn test_fen_header_case_insensitive() {
        let pgn = "[fen \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n\n3. Bb5 1-0";
        let buffers = parse_one(pgn, &default_config());
        assert!(buffers.valid[0], "Should handle lowercase 'fen' header");
    }

    #[test]
    fn test_parse_game_with_custom_fen_no_variant() {
        let pgn = "[Event \"Test Game\"]\n[FEN \"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3\"]\n\n3... a6 4. Ba4 Nf6 5. O-O Be7 1-0";
        let buffers = parse_one(pgn, &default_config());
        assert!(buffers.valid[0]);
        assert_eq!(buffers.total_moves(), 5); // a6, Ba4, Nf6, O-O, Be7
    }

    #[test]
    fn test_castling_with_zeros_normalized() {
        let pgn = "1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. 0-0 0-0 1-0";
        let buffers = parse_one(pgn, &default_config());
        assert!(buffers.valid[0], "zeros castling should parse as O-O");
        assert_eq!(buffers.total_moves(), 8);
        assert_eq!(buffers.parse_errors[0], None);
    }

    #[test]
    fn test_invalid_san_keeps_prefix_only() {
        let buffers = parse_one("1. e4 xyzzy9 e5 Nf3 1-0", &default_config());

        assert!(!buffers.valid[0]);
        let err = buffers.parse_errors[0].as_deref().unwrap();
        assert!(
            err.starts_with("failed to parse SAN at ply 2:") && err.contains("xyzzy9"),
            "unexpected error message: {err}"
        );
        // Valid prefix is recorded; the rest of the game is dropped
        // (single-pass: no sentinel tail).
        assert_eq!(buffers.move_counts[0], 1);
        assert_eq!(buffers.position_counts[0], 2);
    }

    #[test]
    fn test_illegal_move_sets_parse_error() {
        let buffers = parse_one("1. e4 e4 2. Nf3 1-0", &default_config());

        assert!(!buffers.valid[0]);
        let err = buffers.parse_errors[0].as_deref().unwrap();
        assert!(
            err.starts_with("illegal move at ply 2:") && err.contains("e4"),
            "unexpected error message: {err}"
        );
        assert_eq!(buffers.move_counts[0], 1);
        assert_eq!(buffers.position_counts[0], 2);
    }

    #[test]
    fn test_checkmate_status() {
        let buffers = parse_one("1. f3 e5 2. g4 Qh4# 0-1", &default_config());
        assert!(buffers.valid[0]);
        assert!(buffers.is_checkmate[0]);
        assert!(!buffers.is_stalemate[0]);
        assert_eq!(buffers.legal_move_count[0], 0);
    }

    #[test]
    fn test_empty_input_errors() {
        let config = default_config();
        let mut buffers = Buffers::with_capacity(1, 70, &config);
        assert!(parse_game_to_buffers("", &mut buffers, &config).is_err());
        assert!(parse_game_to_buffers("  \n\t ", &mut buffers, &config).is_err());
        assert_eq!(buffers.num_games(), 0);
    }

    /// Corpus exercising every tricky path, parsed both game-by-game into
    /// one Buffers and in two halves into separate Buffers: concatenation
    /// of per-chunk buffers must equal the single-buffer result (games are
    /// independent; chunk boundaries are invisible). This is the
    /// single-pass replacement for the two-pass thread-invariance tests.
    #[test]
    fn test_chunk_boundaries_are_invisible() {
        let pgns = [
            "[Event \"A\"]\n[White \"x\"]\n[Black \"y\"]\n\n1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:02:58] } 2. Nf3 { [%eval -0.2] } Nc6 { some text } 1-0",
            "1. e4 e5 2. Nf3 Nc6 1-0",
            "[Event \"HeadersOnly\"]\n[Site \"nowhere\"]\n",
            "[FEN \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n\n3. Bb5 a6 4. Ba4 Nf6 1-0",
            "[Variant \"chess960\"]\n[FEN \"brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1\"]\n\n1. g3 d5 2. d4 g6 1-0",
            "[Variant \"chess960\"]\n[FEN \"r2k3r/8/8/8/8/8/8/R4K1R w KQkq - 0 1\"]\n\n1. O-O O-O-O *",
            "1. e4 xyzzy9 e5 Nf3 1-0",
            "1. e4 e4 2. Nf3 1-0",
            "[FEN \"invalid fen string\"]\n\n1. e4 e5 1-0",
            "1. e4 (1. d4 {inner} d5 (1... Nf6 2. c4) 2. c4) e5 2. Nf3 1-0",
            "1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. 0-0 0-0 { [%clk 0:01:00] } 1/2-1/2",
            "% escape line\n[Event \"esc\"]\n\n1. e4 c5 { [%eval 0.3] } *",
            "1. e4 {unterminated comment",
            "1. f3 e5 2. g4 Qh4# 0-1",
            "[Event \"Mate tag\"]\n\n1. e4 e5 { [%eval #12] } *",
            "{ [%clk 9:59:59] } 1. e4 e5 *",
        ];

        for config in [default_config(), full_config()] {
            let mut all = Buffers::with_capacity(pgns.len(), 70, &config);
            for pgn in &pgns {
                let _ = parse_game_to_buffers(pgn, &mut all, &config);
            }

            let mid = 7;
            let mut first = Buffers::with_capacity(mid, 70, &config);
            for pgn in &pgns[..mid] {
                let _ = parse_game_to_buffers(pgn, &mut first, &config);
            }
            let mut second = Buffers::with_capacity(pgns.len() - mid, 70, &config);
            for pgn in &pgns[mid..] {
                let _ = parse_game_to_buffers(pgn, &mut second, &config);
            }

            let concat = concat_buffers(&first, &second);
            assert_buffers_equal(&all, &concat);
        }
    }

    /// Concatenate two Buffers (what the chunked driver effectively does).
    fn concat_buffers(a: &Buffers, b: &Buffers) -> Buffers {
        let mut out = a.clone();
        macro_rules! cat {
            ($($f:ident),+) => { $(out.$f.extend_from_slice(&b.$f);)+ };
        }
        cat!(
            boards, castling, en_passant, halfmove_clock, turn,
            from_squares, to_squares, promotions, clocks, evals,
            move_counts, position_counts, is_checkmate, is_stalemate,
            is_insufficient, legal_move_count, valid, headers, outcome,
            parse_errors, comments,
            legal_move_from_squares, legal_move_to_squares,
            legal_move_promotions, legal_move_counts
        );
        out
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    fn assert_buffers_equal(a: &Buffers, b: &Buffers) {
        assert_eq!(a.boards, b.boards);
        assert_eq!(a.castling, b.castling);
        assert_eq!(a.en_passant, b.en_passant);
        assert_eq!(a.halfmove_clock, b.halfmove_clock);
        assert_eq!(a.turn, b.turn);
        assert_eq!(a.from_squares, b.from_squares);
        assert_eq!(a.to_squares, b.to_squares);
        assert_eq!(a.promotions, b.promotions);
        assert_eq!(bits(&a.clocks), bits(&b.clocks), "clocks differ");
        assert_eq!(bits(&a.evals), bits(&b.evals), "evals differ");
        assert_eq!(a.move_counts, b.move_counts);
        assert_eq!(a.position_counts, b.position_counts);
        assert_eq!(a.is_checkmate, b.is_checkmate);
        assert_eq!(a.is_stalemate, b.is_stalemate);
        assert_eq!(a.is_insufficient, b.is_insufficient);
        assert_eq!(a.legal_move_count, b.legal_move_count);
        assert_eq!(a.valid, b.valid);
        assert_eq!(a.headers, b.headers);
        assert_eq!(a.outcome, b.outcome);
        assert_eq!(a.parse_errors, b.parse_errors);
        assert_eq!(a.comments, b.comments);
        assert_eq!(a.legal_move_from_squares, b.legal_move_from_squares);
        assert_eq!(a.legal_move_to_squares, b.legal_move_to_squares);
        assert_eq!(a.legal_move_promotions, b.legal_move_promotions);
        assert_eq!(a.legal_move_counts, b.legal_move_counts);
    }

    /// The shared worker: same result as parsing the games individually.
    #[test]
    fn test_parse_batch_matches_individual() {
        let games = [
            "1. e4 e5 2. Nf3 Nc6 1-0",
            "[FEN \"invalid fen string\"]\n\n1. e4 *",
            "1. e4 xyzzy9 e5 *",
            "1. f3 e5 2. g4 Qh4# 0-1",
        ];
        for config in [default_config(), full_config()] {
            let mut batched = Buffers::with_capacity(games.len(), 70, &config);
            {
                let joined: String = games.join("\n\n");
                let (spans, rest) = crate::splitter::split_games(joined.as_bytes());
                assert_eq!(rest, joined.len());
                assert_eq!(spans.len(), games.len());
                parse_batch(joined.as_bytes(), &spans, &mut batched, &config);
            }

            let mut individual = Buffers::with_capacity(games.len(), 70, &config);
            for game in &games {
                let _ = parse_game_to_buffers(game, &mut individual, &config);
            }

            assert_buffers_equal(&batched, &individual);
        }
    }
}
