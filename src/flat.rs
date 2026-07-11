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
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
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

    fn parse_one(pgn: &str, config: &ParseConfig) -> FlatOutput {
        parse_games_flat(&[pgn], 1, config).unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// Full structural equality between two runs (f32 compared bitwise).
    fn assert_outputs_equal(a: &FlatOutput, b: &FlatOutput) {
        assert_eq!(a.num_games, b.num_games);
        assert_eq!(a.total_moves, b.total_moves);
        assert_eq!(a.total_positions, b.total_positions);
        assert_eq!(a.boards, b.boards);
        assert_eq!(a.castling, b.castling);
        assert_eq!(a.en_passant, b.en_passant);
        assert_eq!(a.halfmove_clock, b.halfmove_clock);
        assert_eq!(a.turn, b.turn);
        assert_eq!(a.from_squares, b.from_squares);
        assert_eq!(a.to_squares, b.to_squares);
        assert_eq!(a.promotions, b.promotions);
        assert_eq!(bits(&a.clocks), bits(&b.clocks));
        assert_eq!(bits(&a.evals), bits(&b.evals));
        assert_eq!(a.move_offsets, b.move_offsets);
        assert_eq!(a.position_offsets, b.position_offsets);
        assert_eq!(a.parsed_move_counts, b.parsed_move_counts);
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
        assert_eq!(a.legal_move_offsets, b.legal_move_offsets);
    }

    /// Assert game `i` of `batch` has identical content to game 0 of
    /// `single` (a one-game parse of the same PGN).
    fn assert_game_matches_single(batch: &FlatOutput, i: usize, single: &FlatOutput) {
        assert_eq!(single.num_games, 1);
        let parsed = batch.parsed_move_counts[i] as usize;
        assert_eq!(parsed, single.parsed_move_counts[0] as usize, "game {i}");

        let alloc = (batch.move_offsets[i + 1] - batch.move_offsets[i]) as usize;
        assert_eq!(alloc, single.move_offsets[1] as usize, "game {i}: alloc");

        let bm = batch.move_offsets[i] as usize;
        let bp = batch.position_offsets[i] as usize;
        let alloc_pos = alloc + 1;

        assert_eq!(
            &batch.boards[bp * 64..(bp + alloc_pos) * 64],
            &single.boards[..],
            "game {i}: boards"
        );
        assert_eq!(
            &batch.castling[bp * 4..(bp + alloc_pos) * 4],
            &single.castling[..],
            "game {i}: castling"
        );
        assert_eq!(
            &batch.en_passant[bp..bp + alloc_pos],
            &single.en_passant[..],
            "game {i}: en_passant"
        );
        assert_eq!(
            &batch.halfmove_clock[bp..bp + alloc_pos],
            &single.halfmove_clock[..],
            "game {i}: halfmove_clock"
        );
        assert_eq!(
            &batch.turn[bp..bp + alloc_pos],
            &single.turn[..],
            "game {i}: turn"
        );
        assert_eq!(
            &batch.from_squares[bm..bm + alloc],
            &single.from_squares[..],
            "game {i}: from_squares"
        );
        assert_eq!(
            &batch.to_squares[bm..bm + alloc],
            &single.to_squares[..],
            "game {i}: to_squares"
        );
        assert_eq!(
            &batch.promotions[bm..bm + alloc],
            &single.promotions[..],
            "game {i}: promotions"
        );
        assert_eq!(
            bits(&batch.clocks[bm..bm + alloc]),
            bits(&single.clocks),
            "game {i}: clocks"
        );
        assert_eq!(
            bits(&batch.evals[bm..bm + alloc]),
            bits(&single.evals),
            "game {i}: evals"
        );

        assert_eq!(batch.valid[i], single.valid[0], "game {i}: valid");
        assert_eq!(batch.headers[i], single.headers[0], "game {i}: headers");
        assert_eq!(batch.outcome[i], single.outcome[0], "game {i}: outcome");
        assert_eq!(
            batch.parse_errors[i], single.parse_errors[0],
            "game {i}: parse_errors"
        );
        assert_eq!(batch.is_checkmate[i], single.is_checkmate[0], "game {i}");
        assert_eq!(batch.is_stalemate[i], single.is_stalemate[0], "game {i}");
        assert_eq!(
            batch.is_insufficient[i * 2..i * 2 + 2],
            single.is_insufficient[..],
            "game {i}: is_insufficient"
        );
        assert_eq!(
            batch.legal_move_count[i], single.legal_move_count[0],
            "game {i}: legal_move_count"
        );

        if !batch.comments.is_empty() {
            assert_eq!(
                &batch.comments[bm..bm + alloc],
                &single.comments[..],
                "game {i}: comments"
            );
        }
        if !batch.legal_move_offsets.is_empty() {
            for k in 0..alloc_pos {
                let bs = batch.legal_move_offsets[bp + k] as usize;
                let be = batch.legal_move_offsets[bp + k + 1] as usize;
                let ss = single.legal_move_offsets[k] as usize;
                let se = single.legal_move_offsets[k + 1] as usize;
                assert_eq!(be - bs, se - ss, "game {i} pos {k}: legal count");
                assert_eq!(
                    &batch.legal_move_from_squares[bs..be],
                    &single.legal_move_from_squares[ss..se],
                    "game {i} pos {k}: legal from"
                );
                assert_eq!(
                    &batch.legal_move_to_squares[bs..be],
                    &single.legal_move_to_squares[ss..se],
                    "game {i} pos {k}: legal to"
                );
                assert_eq!(
                    &batch.legal_move_promotions[bs..be],
                    &single.legal_move_promotions[ss..se],
                    "game {i} pos {k}: legal promo"
                );
            }
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
    fn test_batch_matches_individual_parses() {
        for config in [default_config(), full_config()] {
            let batch = parse_games_flat(TRICKY, 2, &config).unwrap();
            let non_empty: Vec<&str> = TRICKY
                .iter()
                .copied()
                .filter(|s| !s.trim().is_empty())
                .collect();
            assert_eq!(batch.num_games, non_empty.len());
            for (i, pgn) in non_empty.iter().enumerate() {
                let single = parse_one(pgn, &config);
                assert_game_matches_single(&batch, i, &single);
            }
        }
    }

    #[test]
    fn test_thread_count_invariance_multi_task() {
        // More games than GAMES_PER_TASK so pass 2 spans multiple tasks,
        // with invalid games sprinkled across task boundaries. Output
        // must be identical regardless of thread/task scheduling.
        let mut pgns: Vec<&str> = Vec::new();
        for i in 0..(3 * GAMES_PER_TASK + 17) {
            pgns.push(TRICKY[i % TRICKY.len()]);
        }
        for config in [default_config(), full_config()] {
            let a = parse_games_flat(&pgns, 1, &config).unwrap();
            let b = parse_games_flat(&pgns, 4, &config).unwrap();
            assert_outputs_equal(&a, &b);
        }
    }

    #[test]
    fn test_parse_simple_game() {
        let pgn = "[Event \"Test\"]\n[White \"Player1\"]\n[Black \"Player2\"]\n[Result \"1-0\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0";
        let flat = parse_one(pgn, &default_config());

        assert_eq!(flat.num_games, 1);
        assert!(flat.valid[0]);
        assert_eq!(flat.parsed_move_counts[0], 4);
        assert_eq!(flat.move_offsets, vec![0, 4]);
        assert_eq!(flat.position_offsets, vec![0, 5]);
        assert_eq!(flat.total_moves, 4);
        assert_eq!(flat.total_positions, 5);
        assert_eq!(flat.outcome[0], Some("White".to_string()));
        assert_eq!(flat.headers[0].get("Event"), Some(&"Test".to_string()));
    }

    #[test]
    fn test_parse_game_with_annotations() {
        let pgn = "[Event \"Test\"]\n[Result \"1-0\"]\n\n1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... e5 { [%eval 0.19] [%clk 0:02:58] } 1-0";
        let flat = parse_one(pgn, &default_config());

        assert_eq!(flat.total_moves, 2);
        assert!((flat.evals[0] - 0.17).abs() < 0.01);
        assert!((flat.evals[1] - 0.19).abs() < 0.01);
        assert!((flat.clocks[0] - 180.0).abs() < 0.01);
        assert!((flat.clocks[1] - 178.0).abs() < 0.01);
    }

    #[test]
    fn test_multiple_games() {
        let pgns = &[
            "[Event \"Game1\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0",
            "[Event \"Game2\"]\n[Result \"0-1\"]\n\n1. d4 d5 2. c4 0-1",
        ];
        let flat = parse_games_flat(pgns, 1, &default_config()).unwrap();

        assert_eq!(flat.num_games, 2);
        assert_eq!(flat.total_moves, 5);
        assert_eq!(flat.move_offsets, vec![0, 2, 5]);
        assert_eq!(flat.position_offsets, vec![0, 3, 7]);
        assert_eq!(flat.outcome[0], Some("White".to_string()));
        assert_eq!(flat.outcome[1], Some("Black".to_string()));
    }

    #[test]
    fn test_outcome_without_headers() {
        let flat = parse_one("1. e4 e5 2. Nf3 Nc6 0-1", &default_config());
        assert_eq!(flat.outcome[0], Some("Black".to_string()));
    }

    #[test]
    fn test_outcome_draw() {
        let flat = parse_one("1. e4 e5 1/2-1/2", &default_config());
        assert_eq!(flat.outcome[0], Some("Draw".to_string()));
    }

    #[test]
    fn test_comments_disabled() {
        let flat = parse_one("1. e4 { a comment } e5 1-0", &default_config());
        assert!(flat.comments.is_empty());
    }

    #[test]
    fn test_comments_enabled() {
        let config = ParseConfig {
            store_comments: true,
            store_legal_moves: false,
        };
        let flat = parse_one("1. e4 { a comment } 1... e5 { [%eval 0.19] } 1-0", &config);

        assert_eq!(flat.comments.len(), 2);
        // Raw text from PGN includes surrounding spaces from the parser
        assert_eq!(flat.comments[0], Some(" a comment ".to_string()));
        // The second comment only has an eval tag, so text portion is empty
        assert_eq!(flat.comments[1], Some("".to_string()));
    }

    #[test]
    fn test_legal_moves_disabled() {
        let flat = parse_one("1. e4 e5 1-0", &default_config());
        assert!(flat.legal_move_from_squares.is_empty());
        assert!(flat.legal_move_offsets.is_empty());
    }

    #[test]
    fn test_legal_moves_enabled() {
        let config = ParseConfig {
            store_comments: false,
            store_legal_moves: true,
        };
        let flat = parse_one("1. e4 1-0", &config);

        // 2 positions: initial + after e4, offsets len = positions + 1
        assert_eq!(flat.legal_move_offsets.len(), 3);
        // Initial position has 20 legal moves; after e4, black has 20
        assert_eq!(flat.legal_move_offsets, vec![0, 20, 40]);
        assert_eq!(flat.legal_move_from_squares.len(), 40);
    }

    #[test]
    fn test_headers_only_game() {
        // A game with headers but no movetext at all: the initial position
        // (from the FEN header) must still be recorded.
        let pgn = "[Event \"Test\"]\n[FEN \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n";
        let flat = parse_one(pgn, &default_config());

        assert_eq!(flat.num_games, 1);
        assert!(flat.valid[0]);
        assert_eq!(flat.parsed_move_counts[0], 0);
        assert_eq!(flat.position_offsets, vec![0, 1]);
        assert_eq!(flat.headers[0].get("Event"), Some(&"Test".to_string()));
        assert_eq!(flat.parse_errors[0], None);
        assert_eq!(flat.outcome[0], None);
    }

    #[test]
    fn test_parse_game_without_headers() {
        let flat = parse_one(
            "1. Nf3 d5 2. e4 c5 3. exd5 e5 4. dxe6 0-1",
            &default_config(),
        );
        assert!(flat.valid[0]);
        assert_eq!(flat.total_moves, 7);
        assert_eq!(flat.outcome[0], Some("Black".to_string()));
    }

    #[test]
    fn test_parse_game_with_standard_fen() {
        let pgn = "[FEN \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n\n3. Bb5 a6 4. Ba4 Nf6 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(flat.valid[0]);
        assert_eq!(flat.total_moves, 4);
    }

    #[test]
    fn test_parse_chess960_game() {
        let pgn = "[Variant \"chess960\"]\n[FEN \"brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1\"]\n\n1. g3 d5 2. d4 g6 3. b3 Nf6 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(flat.valid[0], "Chess960 moves should be valid with FEN");
        assert_eq!(flat.total_moves, 6);
    }

    #[test]
    fn test_parse_chess960_variant_case_insensitive() {
        let pgn = "[Variant \"Chess960\"]\n[FEN \"brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1\"]\n\n1. g3 d5 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(flat.valid[0], "Should handle Chess960 case variations");
    }

    #[test]
    fn test_parse_invalid_fen_falls_back() {
        let pgn = "[FEN \"invalid fen string\"]\n\n1. e4 e5 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(!flat.valid[0], "invalid FEN should mark game invalid");
        assert!(
            flat.parse_errors[0]
                .as_deref()
                .unwrap()
                .starts_with("failed to parse FEN:")
        );
        // No moves parsed, but initial (default) position recorded.
        assert_eq!(flat.parsed_move_counts[0], 0);
    }

    #[test]
    fn test_fen_header_case_insensitive() {
        let pgn = "[fen \"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3\"]\n\n3. Bb5 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(flat.valid[0], "Should handle lowercase 'fen' header");
    }

    #[test]
    fn test_parse_game_with_custom_fen_no_variant() {
        let pgn = "[Event \"Test Game\"]\n[FEN \"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3\"]\n\n3... a6 4. Ba4 Nf6 5. O-O Be7 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(flat.valid[0]);
        assert_eq!(flat.total_moves, 5); // a6, Ba4, Nf6, O-O, Be7
    }

    #[test]
    fn test_castling_with_zeros_normalized() {
        let pgn = "1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. 0-0 0-0 1-0";
        let flat = parse_one(pgn, &default_config());
        assert!(flat.valid[0], "zeros castling should parse as O-O");
        assert_eq!(flat.total_moves, 8);
        assert_eq!(flat.parse_errors[0], None);
    }

    #[test]
    fn test_invalid_san_sets_parse_error_and_fills_tail() {
        let flat = parse_one("1. e4 xyzzy9 e5 1-0", &default_config());

        assert!(!flat.valid[0]);
        let err = flat.parse_errors[0].as_deref().unwrap();
        assert!(
            err.starts_with("failed to parse SAN:") && err.contains("xyzzy9"),
            "unexpected error message: {err}"
        );
        // Pass 1 counted 3 tokens; only 1 move parsed.
        assert_eq!(flat.move_offsets, vec![0, 3]);
        assert_eq!(flat.parsed_move_counts[0], 1);
        // Tail move slots are sentinel-filled.
        for m in 1..3 {
            assert_eq!(flat.from_squares[m], 0);
            assert_eq!(flat.promotions[m], -1);
            assert!(flat.clocks[m].is_nan());
            assert!(flat.evals[m].is_nan());
        }
        // Tail position slots: zero board, en_passant sentinel.
        for p in 2..4 {
            assert_eq!(&flat.boards[p * 64..(p + 1) * 64], &[0u8; 64]);
            assert_eq!(flat.en_passant[p], -1);
        }
    }

    #[test]
    fn test_illegal_move_sets_parse_error() {
        let flat = parse_one("1. e4 e4 2. Nf3 1-0", &default_config());
        assert!(!flat.valid[0]);
        assert!(
            flat.parse_errors[0]
                .as_deref()
                .unwrap()
                .starts_with("illegal move:")
        );
        assert_eq!(flat.parsed_move_counts[0], 1);
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

    #[test]
    fn test_checkmate_status() {
        let flat = parse_one("1. f3 e5 2. g4 Qh4# 0-1", &default_config());
        assert!(flat.valid[0]);
        assert!(flat.is_checkmate[0]);
        assert!(!flat.is_stalemate[0]);
        assert_eq!(flat.legal_move_count[0], 0);
    }
}
