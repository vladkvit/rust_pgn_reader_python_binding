//! Two-pass parallel parsing into single exact-size flat output arrays.
//!
//! Pass 1 (parallel, per game): tokenizer-only scan counting mainline SAN
//! tokens per game. Prefix sums give exact total array sizes and per-game
//! write offsets. No SAN parsing, no board state.
//!
//! Pass 2 (parallel, work-stealing over tasks of contiguous games): full
//! parse; every game writes into its precomputed disjoint `&mut` sub-slice
//! of the final output arrays (partitioned up front with `split_at_mut`,
//! no unsafe, no atomics). Output order == input order by construction.
//!
//! Invalid games (illegal/unparsable move mid-game): pass-1 counts remain
//! authoritative for the CSR offsets; the unwritten tail of the game's
//! range is filled with sentinels (promotions/en_passant = -1, clocks/evals
//! = NaN, everything else zero) and the actually-parsed move count is
//! recorded in `parsed_move_counts`.

use crate::board_serialization::{
    get_castling_rights, get_en_passant_file, get_halfmove_clock, get_turn, serialize_board,
};
use crate::comment_parsing::{CommentContent, ParsedTag, parse_comments};
use crate::tokenizer::{self, Outcome, Visitor};
use crate::visitor::{ParseConfig, castling_with_zeros};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use shakmaty::{CastlingMode, Chess, Color, Position, fen::Fen, san::SanPlus, uci::UciMove};
use std::collections::HashMap;

/// Number of games per pass-2 task. Small enough that rayon work stealing
/// evens out length imbalance between tasks (no straggler tail), large
/// enough that per-task overhead is negligible.
const GAMES_PER_TASK: usize = 256;

/// Final flat output of a parse run. All arrays are exact-size,
/// allocated once, and indexed by the CSR offset arrays.
#[derive(Default)]
pub struct FlatOutput {
    // Per-position arrays (total_positions entries)
    pub boards: Vec<u8>,         // 64 bytes per position
    pub castling: Vec<bool>,     // 4 bools per position [K,Q,k,q]
    pub en_passant: Vec<i8>,     // -1 or file 0-7
    pub halfmove_clock: Vec<u8>, // per position
    pub turn: Vec<bool>,         // true = white

    // Per-move arrays (total_moves entries)
    pub from_squares: Vec<u8>,
    pub to_squares: Vec<u8>,
    pub promotions: Vec<i8>, // -1 for no promotion
    pub clocks: Vec<f32>,    // NaN for missing
    pub evals: Vec<f32>,     // NaN for missing

    // CSR offsets (num_games + 1 entries), from pass-1 counts
    pub move_offsets: Vec<u32>,
    pub position_offsets: Vec<u32>,

    // Per-game arrays (num_games entries)
    pub parsed_move_counts: Vec<u32>, // == offsets diff for valid games
    pub is_checkmate: Vec<bool>,
    pub is_stalemate: Vec<bool>,
    pub is_insufficient: Vec<bool>, // 2 bools per game [white, black]
    pub legal_move_count: Vec<u16>,
    pub valid: Vec<bool>,
    pub headers: Vec<HashMap<String, String>>,
    pub outcome: Vec<Option<String>>,
    pub parse_errors: Vec<Option<String>>,

    // Optional: per-move comments (only when store_comments)
    pub comments: Vec<Option<String>>,

    // Optional: legal moves per position (only when store_legal_moves)
    pub legal_move_from_squares: Vec<u8>,
    pub legal_move_to_squares: Vec<u8>,
    pub legal_move_promotions: Vec<i8>,
    pub legal_move_offsets: Vec<u32>, // total_positions + 1 entries

    pub num_games: usize,
    pub total_moves: usize,
    pub total_positions: usize,
}

// ---------------------------------------------------------------------------
// Pass 1: counting
// ---------------------------------------------------------------------------

struct CountingVisitor {
    moves: u32,
}

impl Visitor for CountingVisitor {
    fn begin_headers(&mut self) {}
    fn header(&mut self, _key: &[u8], _value: &[u8]) {}
    fn end_headers(&mut self) {}
    fn san_token(&mut self, _token: &[u8]) {
        self.moves += 1;
    }
    fn comment(&mut self, _comment: &[u8]) {}
    fn outcome(&mut self, _outcome: Outcome) {}
    fn end_game(&mut self) {}
}

/// Count mainline SAN tokens in one game (upper bound on parsed moves;
/// exact for games without parse errors).
fn count_san_tokens(pgn: &str) -> u32 {
    let mut counter = CountingVisitor { moves: 0 };
    tokenizer::parse_game(pgn.as_bytes(), &mut counter);
    counter.moves
}

// ---------------------------------------------------------------------------
// Pass 2: slice-writing visitor
// ---------------------------------------------------------------------------

/// Disjoint mutable sub-slices of the output arrays for one task
/// (a contiguous run of games).
struct TaskOutput<'a> {
    /// Pass-1 allocated move counts for this task's games.
    move_counts_alloc: &'a [u32],

    boards: &'a mut [u8],
    castling: &'a mut [bool],
    en_passant: &'a mut [i8],
    halfmove_clock: &'a mut [u8],
    turn: &'a mut [bool],

    from_squares: &'a mut [u8],
    to_squares: &'a mut [u8],
    promotions: &'a mut [i8],
    clocks: &'a mut [f32],
    evals: &'a mut [f32],
    comments: &'a mut [Option<String>], // empty when store_comments=false

    parsed_move_counts: &'a mut [u32],
    is_checkmate: &'a mut [bool],
    is_stalemate: &'a mut [bool],
    is_insufficient: &'a mut [bool],
    legal_move_count: &'a mut [u16],
    valid: &'a mut [bool],
    headers: &'a mut [HashMap<String, String>],
    outcome: &'a mut [Option<String>],
    parse_errors: &'a mut [Option<String>],
}

struct Task<'a> {
    out: TaskOutput<'a>,
    games: &'a [&'a str],
}

/// Task-local legal-move accumulation (sizes unknowable in pass 1;
/// concatenated after pass 2). `counts` has one entry per allocated
/// position in the task, including zero entries for unwritten tails.
#[derive(Default)]
struct TaskLegal {
    counts: Vec<u32>,
    from: Vec<u8>,
    to: Vec<u8>,
    promotions: Vec<i8>,
}

/// Visitor writing one task's games into its output slices.
struct FlatVisitor<'a> {
    out: TaskOutput<'a>,
    store_comments: bool,
    store_legal_moves: bool,
    pos: Chess,
    valid_moves: bool,
    current_headers: Vec<(String, String)>,
    current_outcome: Option<String>,
    current_error: Option<String>,
    status_written: bool,
    /// Index of the current game within the task.
    game_idx: usize,
    /// Task-local start of the current game's allocated ranges.
    move_base: usize,
    pos_base: usize,
    /// Task-local write cursors.
    move_cursor: usize,
    pos_cursor: usize,
    legal: TaskLegal,
}

impl<'a> FlatVisitor<'a> {
    fn new(out: TaskOutput<'a>, config: &ParseConfig) -> Self {
        let legal = if config.store_legal_moves {
            TaskLegal {
                counts: Vec::with_capacity(out.en_passant.len()),
                from: Vec::with_capacity(out.en_passant.len() * 30),
                to: Vec::with_capacity(out.en_passant.len() * 30),
                promotions: Vec::with_capacity(out.en_passant.len() * 30),
            }
        } else {
            TaskLegal::default()
        };
        FlatVisitor {
            out,
            store_comments: config.store_comments,
            store_legal_moves: config.store_legal_moves,
            pos: Chess::default(),
            valid_moves: true,
            current_headers: Vec::with_capacity(10),
            current_outcome: None,
            current_error: None,
            status_written: false,
            game_idx: 0,
            move_base: 0,
            pos_base: 0,
            move_cursor: 0,
            pos_cursor: 0,
            legal,
        }
    }

    fn finish(self) -> TaskLegal {
        debug_assert_eq!(self.game_idx, self.out.move_counts_alloc.len());
        debug_assert_eq!(self.move_cursor, self.out.from_squares.len());
        debug_assert_eq!(self.pos_cursor, self.out.en_passant.len());
        self.legal
    }

    fn push_board_state(&mut self) {
        let p = self.pos_cursor;
        self.out.boards[p * 64..(p + 1) * 64].copy_from_slice(&serialize_board(&self.pos));
        self.out.castling[p * 4..(p + 1) * 4].copy_from_slice(&get_castling_rights(&self.pos));
        self.out.en_passant[p] = get_en_passant_file(&self.pos);
        self.out.halfmove_clock[p] = get_halfmove_clock(&self.pos);
        self.out.turn[p] = get_turn(&self.pos);
        self.pos_cursor += 1;

        if self.store_legal_moves {
            self.push_legal_moves();
        }
    }

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
                self.legal.from.push(from as u8);
                self.legal.to.push(to as u8);
                self.legal
                    .promotions
                    .push(promotion.map(|p| p as i8).unwrap_or(-1));
                count += 1;
            }
        }
        self.legal.counts.push(count);
    }

    fn push_move(&mut self, from: u8, to: u8, promotion: Option<u8>) {
        let m = self.move_cursor;
        self.out.from_squares[m] = from;
        self.out.to_squares[m] = to;
        self.out.promotions[m] = promotion.map(|p| p as i8).unwrap_or(-1);
        self.out.clocks[m] = f32::NAN;
        self.out.evals[m] = f32::NAN;
        // comments[m] is already None from allocation
        self.move_cursor += 1;
    }

    fn update_position_status(&mut self) {
        let gi = self.game_idx;
        self.out.is_checkmate[gi] = self.pos.is_checkmate();
        self.out.is_stalemate[gi] = self.pos.is_stalemate();
        self.out.is_insufficient[gi * 2] = self.pos.has_insufficient_material(Color::White);
        self.out.is_insufficient[gi * 2 + 1] = self.pos.has_insufficient_material(Color::Black);
        self.out.legal_move_count[gi] = self.pos.legal_moves().len() as u16;
        self.status_written = true;
    }

    fn set_error(&mut self, msg: String) {
        self.valid_moves = false;
        self.current_error = Some(msg);
    }
}

impl Visitor for FlatVisitor<'_> {
    fn begin_headers(&mut self) {
        self.current_headers.clear();
        self.valid_moves = true;
        self.current_outcome = None;
        self.current_error = None;
        self.status_written = false;
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
                    self.set_error(format!("failed to parse SAN: {} ({})", error, token_str));
                    return;
                }
            },
        };

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

    fn comment(&mut self, comment: &[u8]) {
        // Comments annotate the last-written move of the CURRENT game; a
        // comment before the game's first move has no slot to update.
        // (The old Buffers code could touch the previous game's last slot
        // in that case — a chunk-boundary-dependent quirk, deliberately
        // fixed here.)
        if self.move_cursor == self.move_base {
            return;
        }
        let last_move = self.move_cursor - 1;

        if let Ok((_, parsed_comments)) = parse_comments(comment) {
            let mut move_comments = if self.store_comments {
                Some(String::new())
            } else {
                None
            };

            for content in parsed_comments {
                match content {
                    CommentContent::Tag(tag_content) => match tag_content {
                        ParsedTag::Eval(eval_value) => {
                            self.out.evals[last_move] = eval_value as f32;
                        }
                        ParsedTag::ClkTime {
                            hours,
                            minutes,
                            seconds,
                        } => {
                            self.out.clocks[last_move] =
                                hours as f32 * 3600.0 + minutes as f32 * 60.0 + seconds as f32;
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
                        if let Some(ref mut comments) = move_comments
                            && !text.trim().is_empty()
                        {
                            if !comments.is_empty() {
                                comments.push(' ');
                            }
                            comments.push_str(&text);
                        }
                    }
                }
            }

            if let Some(comment_text) = move_comments {
                self.out.comments[last_move] = Some(comment_text);
            }
        }
    }

    fn outcome(&mut self, outcome: Outcome) {
        self.current_outcome = Some(match outcome {
            Outcome::WhiteWins => "White".to_string(),
            Outcome::BlackWins => "Black".to_string(),
            Outcome::Draw => "Draw".to_string(),
            Outcome::Unknown => "Unknown".to_string(),
        });
        self.update_position_status();
    }

    fn end_game(&mut self) {
        if !self.status_written {
            self.update_position_status();
        }

        let gi = self.game_idx;
        let parsed_moves = (self.move_cursor - self.move_base) as u32;
        self.out.parsed_move_counts[gi] = parsed_moves;
        self.out.valid[gi] = self.valid_moves;
        self.out.outcome[gi] = self.current_outcome.take();
        self.out.parse_errors[gi] = self.current_error.take();
        self.out.headers[gi] = self.current_headers.drain(..).collect();

        // Fill the unwritten tail of the allocated ranges with sentinels
        // (only non-empty for games that errored mid-parse).
        let alloc_moves = self.out.move_counts_alloc[gi] as usize;
        let alloc_positions = alloc_moves + 1;
        for m in self.move_cursor..self.move_base + alloc_moves {
            self.out.promotions[m] = -1;
            self.out.clocks[m] = f32::NAN;
            self.out.evals[m] = f32::NAN;
        }
        for p in self.pos_cursor..self.pos_base + alloc_positions {
            self.out.en_passant[p] = -1;
        }
        if self.store_legal_moves {
            for _ in self.pos_cursor..self.pos_base + alloc_positions {
                self.legal.counts.push(0);
            }
        }

        // Advance to the next game's allocated ranges.
        self.move_base += alloc_moves;
        self.pos_base += alloc_positions;
        self.move_cursor = self.move_base;
        self.pos_cursor = self.pos_base;
        self.game_idx += 1;
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn prefix_sum_u32(counts: impl Iterator<Item = u32>) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(counts.size_hint().0 + 1);
    let mut acc: u32 = 0;
    offsets.push(0);
    for count in counts {
        acc += count;
        offsets.push(acc);
    }
    offsets
}

/// Split `head_len` elements off the front of a mutable rest-slice.
macro_rules! split_mut {
    ($rest:expr, $len:expr) => {{
        let (head, tail) = std::mem::take(&mut $rest).split_at_mut($len);
        $rest = tail;
        head
    }};
}

/// Split `head_len` elements off the front of a shared rest-slice.
macro_rules! split_ref {
    ($rest:expr, $len:expr) => {{
        let (head, tail) = $rest.split_at($len);
        $rest = tail;
        head
    }};
}

/// Parse all games in `slices` (one game per string; empty/whitespace-only
/// strings are skipped) into exact-size flat arrays using two passes.
pub fn parse_games_flat(
    slices: &[&str],
    num_threads: usize,
    config: &ParseConfig,
) -> Result<FlatOutput, String> {
    // Skip empty inputs (parity with the previous per-game "No game found
    // in PGN" behavior, where such inputs produced no game entry).
    let games: Vec<&str> = slices
        .iter()
        .copied()
        .filter(|s| !s.trim().is_empty())
        .collect();
    let n = games.len();

    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| format!("Failed to build thread pool: {}", e))?;

    // ---- Pass 1: count mainline SAN tokens per game ----
    let move_counts: Vec<u32> =
        pool.install(|| games.par_iter().map(|g| count_san_tokens(g)).collect());

    let move_offsets = prefix_sum_u32(move_counts.iter().copied());
    let position_offsets = prefix_sum_u32(move_counts.iter().map(|&c| c + 1));
    let total_moves = move_offsets[n] as usize;
    let total_positions = position_offsets[n] as usize;

    // ---- Allocate final arrays once, at exact size ----
    // vec![0; n] uses alloc_zeroed: no memset for large allocations; the
    // first-touch page faults happen on the worker threads during pass 2.
    let mut out = FlatOutput {
        boards: vec![0; total_positions * 64],
        castling: vec![false; total_positions * 4],
        en_passant: vec![0; total_positions],
        halfmove_clock: vec![0; total_positions],
        turn: vec![false; total_positions],
        from_squares: vec![0; total_moves],
        to_squares: vec![0; total_moves],
        promotions: vec![0; total_moves],
        clocks: vec![0.0; total_moves],
        evals: vec![0.0; total_moves],
        move_offsets,
        position_offsets,
        parsed_move_counts: vec![0; n],
        is_checkmate: vec![false; n],
        is_stalemate: vec![false; n],
        is_insufficient: vec![false; n * 2],
        legal_move_count: vec![0; n],
        valid: vec![false; n],
        headers: (0..n).map(|_| HashMap::new()).collect(),
        outcome: vec![None; n],
        parse_errors: vec![None; n],
        comments: if config.store_comments {
            vec![None; total_moves]
        } else {
            Vec::new()
        },
        legal_move_from_squares: Vec::new(),
        legal_move_to_squares: Vec::new(),
        legal_move_promotions: Vec::new(),
        legal_move_offsets: Vec::new(),
        num_games: n,
        total_moves,
        total_positions,
    };

    // ---- Partition the output arrays into per-task disjoint slices ----
    let mut tasks: Vec<Task> = Vec::with_capacity(n.div_ceil(GAMES_PER_TASK.max(1)));
    {
        let mut boards_rest: &mut [u8] = &mut out.boards;
        let mut castling_rest: &mut [bool] = &mut out.castling;
        let mut en_passant_rest: &mut [i8] = &mut out.en_passant;
        let mut halfmove_rest: &mut [u8] = &mut out.halfmove_clock;
        let mut turn_rest: &mut [bool] = &mut out.turn;
        let mut from_rest: &mut [u8] = &mut out.from_squares;
        let mut to_rest: &mut [u8] = &mut out.to_squares;
        let mut promo_rest: &mut [i8] = &mut out.promotions;
        let mut clocks_rest: &mut [f32] = &mut out.clocks;
        let mut evals_rest: &mut [f32] = &mut out.evals;
        let mut comments_rest: &mut [Option<String>] = &mut out.comments;
        let mut parsed_rest: &mut [u32] = &mut out.parsed_move_counts;
        let mut checkmate_rest: &mut [bool] = &mut out.is_checkmate;
        let mut stalemate_rest: &mut [bool] = &mut out.is_stalemate;
        let mut insufficient_rest: &mut [bool] = &mut out.is_insufficient;
        let mut legal_count_rest: &mut [u16] = &mut out.legal_move_count;
        let mut valid_rest: &mut [bool] = &mut out.valid;
        let mut headers_rest: &mut [HashMap<String, String>] = &mut out.headers;
        let mut outcome_rest: &mut [Option<String>] = &mut out.outcome;
        let mut errors_rest: &mut [Option<String>] = &mut out.parse_errors;
        let mut counts_rest: &[u32] = &move_counts;
        let mut games_rest: &[&str] = &games;

        let mut g = 0;
        while g < n {
            let g_end = (g + GAMES_PER_TASK).min(n);
            let task_games = g_end - g;
            let task_moves = (out.move_offsets[g_end] - out.move_offsets[g]) as usize;
            let task_positions = task_moves + task_games;

            let task = Task {
                out: TaskOutput {
                    move_counts_alloc: split_ref!(counts_rest, task_games),
                    boards: split_mut!(boards_rest, task_positions * 64),
                    castling: split_mut!(castling_rest, task_positions * 4),
                    en_passant: split_mut!(en_passant_rest, task_positions),
                    halfmove_clock: split_mut!(halfmove_rest, task_positions),
                    turn: split_mut!(turn_rest, task_positions),
                    from_squares: split_mut!(from_rest, task_moves),
                    to_squares: split_mut!(to_rest, task_moves),
                    promotions: split_mut!(promo_rest, task_moves),
                    clocks: split_mut!(clocks_rest, task_moves),
                    evals: split_mut!(evals_rest, task_moves),
                    comments: if config.store_comments {
                        split_mut!(comments_rest, task_moves)
                    } else {
                        &mut []
                    },
                    parsed_move_counts: split_mut!(parsed_rest, task_games),
                    is_checkmate: split_mut!(checkmate_rest, task_games),
                    is_stalemate: split_mut!(stalemate_rest, task_games),
                    is_insufficient: split_mut!(insufficient_rest, task_games * 2),
                    legal_move_count: split_mut!(legal_count_rest, task_games),
                    valid: split_mut!(valid_rest, task_games),
                    headers: split_mut!(headers_rest, task_games),
                    outcome: split_mut!(outcome_rest, task_games),
                    parse_errors: split_mut!(errors_rest, task_games),
                },
                games: split_ref!(games_rest, task_games),
            };
            tasks.push(task);
            g = g_end;
        }

        // ---- Pass 2: full parse, work-stealing over tasks ----
        let legal_parts: Vec<TaskLegal> = pool.install(|| {
            tasks
                .into_par_iter()
                .map(|task| {
                    let mut visitor = FlatVisitor::new(task.out, config);
                    for &pgn in task.games {
                        tokenizer::parse_game(pgn.as_bytes(), &mut visitor);
                    }
                    visitor.finish()
                })
                .collect()
        });

        // ---- Concatenate task-local legal-move arrays (optional path) ----
        if config.store_legal_moves {
            let total_legal: usize = legal_parts.iter().map(|p| p.from.len()).sum();
            let total_counts: usize = legal_parts.iter().map(|p| p.counts.len()).sum();
            debug_assert_eq!(total_counts, total_positions);

            out.legal_move_from_squares = Vec::with_capacity(total_legal);
            out.legal_move_to_squares = Vec::with_capacity(total_legal);
            out.legal_move_promotions = Vec::with_capacity(total_legal);
            let mut legal_counts: Vec<u32> = Vec::with_capacity(total_counts);
            for part in &legal_parts {
                out.legal_move_from_squares.extend_from_slice(&part.from);
                out.legal_move_to_squares.extend_from_slice(&part.to);
                out.legal_move_promotions
                    .extend_from_slice(&part.promotions);
                legal_counts.extend_from_slice(&part.counts);
            }
            out.legal_move_offsets = prefix_sum_u32(legal_counts.into_iter());
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visitor::{Buffers, parse_game_to_buffers};

    /// Differential check: the flat two-pass path must produce the same
    /// per-game data as the Buffers path for every game. For invalid games
    /// the flat arrays are compared over the actually-parsed prefix and the
    /// allocated tail is checked for sentinels.
    fn assert_flat_matches_buffers(pgns: &[&str], config: &ParseConfig, num_threads: usize) {
        let flat = parse_games_flat(pgns, num_threads, config).unwrap();

        let mut buffers = Buffers::with_capacity(pgns.len().max(1), 70, config);
        for &p in pgns {
            let _ = parse_game_to_buffers(p, &mut buffers, config);
        }

        assert_eq!(flat.num_games, buffers.num_games(), "game count");
        let n = flat.num_games;
        assert_eq!(flat.move_offsets.len(), n + 1);
        assert_eq!(flat.position_offsets.len(), n + 1);

        let b_move_off = buffers.compute_move_offsets();
        let b_pos_off = buffers.compute_position_offsets();
        let b_legal_off = if config.store_legal_moves {
            buffers.compute_legal_move_offsets()
        } else {
            Vec::new()
        };

        for i in 0..n {
            let parsed = flat.parsed_move_counts[i] as usize;
            assert_eq!(
                parsed as u32, buffers.move_counts[i],
                "game {i}: parsed moves"
            );
            assert_eq!(flat.valid[i], buffers.valid[i], "game {i}: valid");
            assert_eq!(flat.headers[i], buffers.headers[i], "game {i}: headers");
            assert_eq!(flat.outcome[i], buffers.outcome[i], "game {i}: outcome");
            assert_eq!(
                flat.parse_errors[i], buffers.parse_errors[i],
                "game {i}: parse_errors"
            );
            assert_eq!(flat.is_checkmate[i], buffers.is_checkmate[i], "game {i}");
            assert_eq!(flat.is_stalemate[i], buffers.is_stalemate[i], "game {i}");
            assert_eq!(
                flat.is_insufficient[i * 2..i * 2 + 2],
                buffers.is_insufficient[i * 2..i * 2 + 2],
                "game {i}: is_insufficient"
            );
            assert_eq!(
                flat.legal_move_count[i], buffers.legal_move_count[i],
                "game {i}: legal_move_count"
            );

            // --- Moves: parsed prefix identical, tail sentinel-filled ---
            let fm = flat.move_offsets[i] as usize;
            let bm = b_move_off[i] as usize;
            let alloc = (flat.move_offsets[i + 1] - flat.move_offsets[i]) as usize;
            assert!(parsed <= alloc, "game {i}: parsed > allocated");

            assert_eq!(
                &flat.from_squares[fm..fm + parsed],
                &buffers.from_squares[bm..bm + parsed],
                "game {i}: from_squares"
            );
            assert_eq!(
                &flat.to_squares[fm..fm + parsed],
                &buffers.to_squares[bm..bm + parsed],
                "game {i}: to_squares"
            );
            assert_eq!(
                &flat.promotions[fm..fm + parsed],
                &buffers.promotions[bm..bm + parsed],
                "game {i}: promotions"
            );
            for k in 0..parsed {
                assert_eq!(
                    flat.clocks[fm + k].to_bits(),
                    buffers.clocks[bm + k].to_bits(),
                    "game {i} move {k}: clocks"
                );
                assert_eq!(
                    flat.evals[fm + k].to_bits(),
                    buffers.evals[bm + k].to_bits(),
                    "game {i} move {k}: evals"
                );
            }
            if config.store_comments {
                assert_eq!(
                    &flat.comments[fm..fm + parsed],
                    &buffers.comments[bm..bm + parsed],
                    "game {i}: comments"
                );
            }
            for k in parsed..alloc {
                assert_eq!(flat.from_squares[fm + k], 0, "game {i}: tail from");
                assert_eq!(flat.to_squares[fm + k], 0, "game {i}: tail to");
                assert_eq!(flat.promotions[fm + k], -1, "game {i}: tail promo");
                assert!(flat.clocks[fm + k].is_nan(), "game {i}: tail clock");
                assert!(flat.evals[fm + k].is_nan(), "game {i}: tail eval");
                if config.store_comments {
                    assert_eq!(flat.comments[fm + k], None, "game {i}: tail comment");
                }
            }

            // --- Positions: parsed prefix identical, tail sentinel-filled ---
            let parsed_pos = parsed + 1;
            assert_eq!(
                parsed_pos as u32, buffers.position_counts[i],
                "game {i}: position count"
            );
            let fp = flat.position_offsets[i] as usize;
            let bp = b_pos_off[i] as usize;
            let alloc_pos = (flat.position_offsets[i + 1] - flat.position_offsets[i]) as usize;
            assert_eq!(alloc_pos, alloc + 1);

            assert_eq!(
                &flat.boards[fp * 64..(fp + parsed_pos) * 64],
                &buffers.boards[bp * 64..(bp + parsed_pos) * 64],
                "game {i}: boards"
            );
            assert_eq!(
                &flat.castling[fp * 4..(fp + parsed_pos) * 4],
                &buffers.castling[bp * 4..(bp + parsed_pos) * 4],
                "game {i}: castling"
            );
            assert_eq!(
                &flat.en_passant[fp..fp + parsed_pos],
                &buffers.en_passant[bp..bp + parsed_pos],
                "game {i}: en_passant"
            );
            assert_eq!(
                &flat.halfmove_clock[fp..fp + parsed_pos],
                &buffers.halfmove_clock[bp..bp + parsed_pos],
                "game {i}: halfmove_clock"
            );
            assert_eq!(
                &flat.turn[fp..fp + parsed_pos],
                &buffers.turn[bp..bp + parsed_pos],
                "game {i}: turn"
            );
            for k in parsed_pos..alloc_pos {
                assert_eq!(flat.en_passant[fp + k], -1, "game {i}: tail en_passant");
                assert_eq!(
                    &flat.boards[(fp + k) * 64..(fp + k + 1) * 64],
                    &[0u8; 64],
                    "game {i}: tail board"
                );
            }

            // --- Legal moves per position (optional) ---
            if config.store_legal_moves {
                for k in 0..parsed_pos {
                    let fs = flat.legal_move_offsets[fp + k] as usize;
                    let fe = flat.legal_move_offsets[fp + k + 1] as usize;
                    let bs = b_legal_off[bp + k] as usize;
                    let be = b_legal_off[bp + k + 1] as usize;
                    assert_eq!(fe - fs, be - bs, "game {i} pos {k}: legal count");
                    assert_eq!(
                        &flat.legal_move_from_squares[fs..fe],
                        &buffers.legal_move_from_squares[bs..be],
                        "game {i} pos {k}: legal from"
                    );
                    assert_eq!(
                        &flat.legal_move_to_squares[fs..fe],
                        &buffers.legal_move_to_squares[bs..be],
                        "game {i} pos {k}: legal to"
                    );
                    assert_eq!(
                        &flat.legal_move_promotions[fs..fe],
                        &buffers.legal_move_promotions[bs..be],
                        "game {i} pos {k}: legal promo"
                    );
                }
                for k in parsed_pos..alloc_pos {
                    assert_eq!(
                        flat.legal_move_offsets[fp + k],
                        flat.legal_move_offsets[fp + k + 1],
                        "game {i}: tail position must have zero legal moves"
                    );
                }
            }
        }
    }

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

    /// Corpus exercising every tricky path: comments with eval/clk/mate
    /// tags, empty inputs, FEN starts, Chess960, invalid SAN, illegal
    /// moves, invalid FEN, variations, zeros castling, trailing games,
    /// escape lines, unterminated comments, headers-only, no headers.
    const TRICKY: &[&str] = &[
        "[Event \"A\"]\n[White \"x\"]\n[Black \"y\"]\n\n1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:02:58] } 2. Nf3 { [%eval -0.2] } Nc6 { some text } 1-0",
        "1. e4 e5 2. Nf3 Nc6 1-0",
        "",
        "   \n\t  ",
        "[Event \"HeadersOnly\"]\n[Site \"nowhere\"]\n",
        "[FEN \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n\n3. Bb5 a6 4. Ba4 Nf6 1-0",
        "[Variant \"chess960\"]\n[FEN \"brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1\"]\n\n1. g3 d5 2. d4 g6 1-0",
        "1. e4 xyzzy9 e5 Nf3 1-0",
        "1. e4 e4 2. Nf3 1-0",
        "[FEN \"invalid fen string\"]\n\n1. e4 e5 1-0",
        "1. e4 (1. d4 {inner} d5 (1... Nf6 2. c4) 2. c4) e5 2. Nf3 1-0",
        "1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. 0-0 0-0 { [%clk 0:01:00] } 1/2-1/2",
        "[Event \"a\"]\n\n1. e4 1-0\n\n[Event \"b\"]\n\n1. d4 0-1",
        "% escape line\n[Event \"esc\"]\n\n1. e4 c5 { [%eval 0.3] } *",
        "1. e4 {unterminated comment",
        "1. f3 e5 2. g4 Qh4# 0-1",
        "[Event \"Mate tag\"]\n\n1. e4 e5 { [%eval #12] } *",
    ];

    #[test]
    fn test_flat_matches_buffers_tricky() {
        assert_flat_matches_buffers(TRICKY, &default_config(), 2);
    }

    #[test]
    fn test_flat_matches_buffers_comments_and_legal_moves() {
        assert_flat_matches_buffers(TRICKY, &full_config(), 2);
    }

    #[test]
    fn test_flat_matches_buffers_single_threaded() {
        assert_flat_matches_buffers(TRICKY, &default_config(), 1);
    }

    #[test]
    fn test_flat_matches_buffers_many_games_multi_task() {
        // More games than GAMES_PER_TASK so pass 2 spans multiple tasks,
        // with invalid games sprinkled across task boundaries.
        let mut pgns: Vec<&str> = Vec::new();
        for i in 0..(3 * GAMES_PER_TASK + 17) {
            pgns.push(TRICKY[i % TRICKY.len()]);
        }
        assert_flat_matches_buffers(&pgns, &default_config(), 4);
        assert_flat_matches_buffers(&pgns, &full_config(), 4);
    }

    #[test]
    fn test_pass1_count_is_upper_bound() {
        let config = default_config();
        // Invalid SAN mid-game: 4 tokens counted, 1 move parsed.
        let flat = parse_games_flat(&["1. e4 xyzzy9 e5 Nf3 1-0"], 1, &config).unwrap();
        assert_eq!(flat.num_games, 1);
        assert_eq!(flat.move_offsets[1] - flat.move_offsets[0], 4);
        assert_eq!(flat.parsed_move_counts[0], 1);
        assert!(!flat.valid[0]);

        // Valid game: count == parsed.
        let flat = parse_games_flat(&["1. e4 e5 2. Nf3 Nc6 1-0"], 1, &config).unwrap();
        assert_eq!(flat.move_offsets[1] - flat.move_offsets[0], 4);
        assert_eq!(flat.parsed_move_counts[0], 4);
        assert!(flat.valid[0]);
    }

    #[test]
    fn test_empty_input() {
        let flat = parse_games_flat(&[], 1, &default_config()).unwrap();
        assert_eq!(flat.num_games, 0);
        assert_eq!(flat.move_offsets, vec![0]);
        assert_eq!(flat.position_offsets, vec![0]);
        assert!(flat.boards.is_empty());
    }

    #[test]
    fn test_empty_strings_skipped() {
        let flat = parse_games_flat(&["", "1. e4 e5 1-0", "  \n "], 1, &default_config()).unwrap();
        assert_eq!(flat.num_games, 1);
        assert_eq!(flat.parsed_move_counts[0], 2);
    }
}
