//! Phase 0 SAN resolver: candidate bitboards + scratch-copy legality.
//!
//! Fast path replacing `San::to_move` for the common case. shakmaty's
//! `San::to_move` materializes a `MoveList` (`ArrayVec<Move, 270>`, ~2.2 KB)
//! per SAN token and runs the full legal-move machinery (`checkers`,
//! `evasions`, `slider_blockers`, `gen_pawn_moves`); the by-value return
//! memcpy survives even fat LTO. This resolver keeps shakmaty's
//! generate-then-filter philosophy but on candidate bitboards (nearly always
//! 1-2 squares) and never builds a move list.
//!
//! Contract: `resolve` returns `Some(m)` iff shakmaty's
//! `san.to_move(pos) == Ok(m)`. It returns `None` whenever the fast path
//! does not apply (castling, drops, null moves) OR the outcome is not a
//! unique legal candidate (0 => `IllegalSan`, 2+ => `AmbiguousSan`); the
//! caller MUST then fall back to `san.to_move(pos)` so error semantics stay
//! byte-identical.
//!
//! Set-equality argument for correctness: shakmaty's post-retain candidate
//! set (san.rs `to_move`) equals {legal moves matching role, to, from-file/
//! rank, capture flag, promotion}. Ours equals {pseudo-legal candidates
//! matching the same filters} ∩ {king not attacked after the move}, and for
//! standard chess "pseudo-legal + own king safe afterwards" == legal. Hence
//! both sets are identical and the unique-candidate outcomes agree.
//!
//! Legality is tested without materializing the resulting position
//! toggle the move's squares in
//! the occupancy (`occ' = occupied ^ from ^ ep_captured | to`) and check
//! whether the mover's king square (`to` itself for king moves) is attacked
//! under `occ'` via `Board::attacks_to`, masking `to`/`ep_captured` out of
//! the attacker set. `occ'` equals the real post-move occupancy, and the
//! enemy piece bitboards differ from the post-move board only at those
//! masked squares, so this is exactly the attack state of the resulting
//! position: pins, check evasions and en-passant-reveals-check all fall out
//! uniformly. (An earlier version cloned the position and ran
//! `play_unchecked` on a scratch copy; profiling showed clone+play as the
//! resolver's main cost, hence this upgrade.)

use shakmaty::{
    Bitboard, Chess, Color, EnPassantMode, Move, Position, Rank, Role, Square, attacks, san::San,
};

/// Tries to resolve `san` to the unique legal move of `pos`.
///
/// `None` means "fall back to `san.to_move(pos)`" — either the fast path
/// does not cover this SAN shape, or the result would be an error.
/// Performance critical
/// About 1/4 of san_token time is spent here
pub fn resolve(pos: &Chess, san: &San) -> Option<Move> {
    let &San::Normal {
        role,
        file,
        rank,
        capture,
        to,
        promotion,
    } = san
    else {
        // Castle / Put / Null: rare (~1/40 tokens); always fall back.
        return None;
    };

    let board = pos.board();
    let us = pos.us();
    let them = pos.them();
    let occupied = board.occupied();
    let turn = pos.turn();

    // Destination occupied by one of our own pieces: never a candidate.
    if us.contains(to) {
        return None;
    }

    // Promotion sanity. shakmaty generates promotions only for pawn moves to
    // the backrank and only to N/B/R/Q; any mismatch retains zero candidates
    // (=> IllegalSan), which the fallback reproduces.
    match promotion {
        None => {}
        Some(Role::Knight | Role::Bishop | Role::Rook | Role::Queen) if role == Role::Pawn => {}
        Some(_) => return None,
    }
    if role == Role::Pawn {
        let promo_rank = match turn {
            Color::White => Rank::Eighth,
            Color::Black => Rank::First,
        };
        if (to.rank() == promo_rank) != promotion.is_some() {
            return None;
        }
    } else if capture != them.contains(to) {
        // Non-pawn: SAN capture flag must match board occupancy (`us` at `to`
        // was already excluded above).
        return None;
    }

    // SAN disambiguation masks, applied to candidate `from` squares.
    let mut disambig = Bitboard::FULL;
    if let Some(f) = file {
        disambig &= Bitboard::from_file(f);
    }
    if let Some(r) = rank {
        disambig &= Bitboard::from_rank(r);
    }

    // Pseudo-legal candidate `from` squares (mirrors the bitboard shortcuts
    // in `Chess::san_candidates`), plus whether en passant applies.
    let mut froms = Bitboard::EMPTY;
    let mut ep = false;

    match role {
        Role::Knight => froms = attacks::knight_attacks(to) & pos.our(Role::Knight),
        Role::Bishop => froms = attacks::bishop_attacks(to, occupied) & pos.our(Role::Bishop),
        Role::Rook => froms = attacks::rook_attacks(to, occupied) & pos.our(Role::Rook),
        Role::Queen => froms = attacks::queen_attacks(to, occupied) & pos.our(Role::Queen),
        Role::King => froms = attacks::king_attacks(to) & pos.our(Role::King),
        Role::Pawn => {
            if capture {
                if them.contains(to) {
                    froms = attacks::pawn_attacks(!turn, to) & pos.our(Role::Pawn);
                }
                // En passant. `EnPassantMode::Always` is exactly the raw
                // `self.ep_square` condition used by `Chess::san_candidates`.
                // The emptiness guard is defensive for corrupt setups: never
                // fast-path an ep capture onto an occupied target so outcomes
                // stay aligned with the `to_move` fallback.
                if pos.ep_square(EnPassantMode::Always) == Some(to) && !occupied.contains(to) {
                    ep = true;
                }
            } else if !occupied.contains(to) {
                // Pushes: at most one origin square.
                froms = pawn_push_froms(pos, to, turn, occupied);
            }
        }
    }

    froms &= disambig;

    // Same `capture` field value shakmaty puts on generated moves.
    let capture_role = board.role_at(to);

    let mut unique: Option<Move> = None;
    for from in froms {
        let m = Move::Normal {
            role,
            from,
            capture: capture_role,
            to,
            promotion,
        };
        if is_legal_after(pos, m) {
            if unique.is_some() {
                return None; // ambiguous => fall back (AmbiguousSan)
            }
            unique = Some(m);
        }
    }

    if ep {
        for from in attacks::pawn_attacks(!turn, to) & pos.our(Role::Pawn) & disambig {
            let m = Move::EnPassant { from, to };
            if is_legal_after(pos, m) {
                if unique.is_some() {
                    return None;
                }
                unique = Some(m);
            }
        }
    }

    // None here == zero legal candidates => fall back (IllegalSan).
    //
    // Differential check: whenever the fast path answers, it must agree with
    // shakmaty exactly. Free in release; every corpus-driven test doubles as
    // a differential test in `cargo test`.
    #[cfg(debug_assertions)]
    if let Some(m) = unique {
        debug_assert_eq!(
            san.to_move(pos).ok(),
            Some(m),
            "san_resolver disagrees with to_move for {san:?}"
        );
    }
    unique
}

/// Pseudo-legal origin squares for a pawn push (single or double) to `to`.
fn pawn_push_froms(pos: &Chess, to: Square, turn: Color, occupied: Bitboard) -> Bitboard {
    let back: i32 = match turn {
        Color::White => -8,
        Color::Black => 8,
    };
    let Some(single) = to.offset(back) else {
        return Bitboard::EMPTY;
    };
    if pos.our(Role::Pawn).contains(single) {
        return Bitboard::from_square(single);
    }
    if occupied.contains(single) {
        return Bitboard::EMPTY; // double push can't jump a piece
    }
    let double_rank = match turn {
        Color::White => Rank::Fourth,
        Color::Black => Rank::Fifth,
    };
    if to.rank() != double_rank {
        return Bitboard::EMPTY;
    }
    match single.offset(back) {
        Some(double) if pos.our(Role::Pawn).contains(double) => Bitboard::from_square(double),
        _ => Bitboard::EMPTY,
    }
}

/// Uniform legality test on XORed occupancy: the mover's king must not be
/// attacked once the move's squares are toggled in the occupancy and the
/// captured square (if any) is masked out of the attacker set.
///
/// Equivalence to playing the move out on a scratch copy: `occ_after` is
/// exactly the post-move occupancy, and the enemy piece bitboards of the
/// pre-move board differ from the post-move board only at `to` (normal
/// capture) or the ep-captured square — both masked out of the attacker
/// set. Our own moved/promoted piece never contributes to enemy attacks.
/// Hence pins, check evasions and ep-reveals-check are all covered.
fn is_legal_after(pos: &Chess, m: Move) -> bool {
    let us = pos.turn();
    let board = pos.board();

    let (from, to, role, captured) = match m {
        Move::Normal { role, from, to, .. } => (from, to, role, Bitboard::EMPTY),
        Move::EnPassant { from, to } => (
            from,
            to,
            Role::Pawn,
            Bitboard::from_square(Square::from_coords(to.file(), from.rank())),
        ),
        // `resolve` only constructs Normal and EnPassant candidates.
        _ => return false,
    };

    // King square in the resulting position.
    let king = if role == Role::King {
        to
    } else {
        match board.king_of(us) {
            Some(king) => king,
            None => return false,
        }
    };

    let to_bb = Bitboard::from_square(to);
    let occ_after = (board.occupied() ^ Bitboard::from_square(from) ^ captured) | to_bb;

    (board.attacks_to(king, !us, occ_after) & !(to_bb | captured)).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{CastlingMode, fen::Fen, san::SanPlus};

    fn pos(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("valid FEN")
            .into_position(CastlingMode::Standard)
            .expect("legal position")
    }

    fn san(s: &str) -> San {
        SanPlus::from_ascii(s.as_bytes()).expect("valid SAN").san
    }

    /// Resolver returned Some => must equal to_move's result exactly.
    /// `expect_fast` additionally asserts the fast path did not fall back.
    fn check(p: &Chess, s: &San, expect_fast: bool) {
        let reference = s.to_move(p);
        match resolve(p, s) {
            Some(m) => assert_eq!(
                reference.as_ref().ok(),
                Some(&m),
                "resolver disagrees with to_move"
            ),
            None => assert!(
                !expect_fast,
                "unexpected fallback for a SAN the fast path should handle"
            ),
        }
    }

    fn assert_resolves(fen: &str, san_str: &str) {
        let p = pos(fen);
        let s = san(san_str);
        check(&p, &s, true);
        assert!(
            resolve(&p, &s).is_some(),
            "expected fast-path resolution for {san_str} in {fen}"
        );
    }

    fn assert_falls_back(fen: &str, san_str: &str) {
        let p = pos(fen);
        let s = san(san_str);
        assert_eq!(
            resolve(&p, &s),
            None,
            "expected fallback for {san_str} in {fen}"
        );
    }

    /// Differential sweep: every legal move of the position, converted to its
    /// canonical (disambiguated) SAN, must resolve on the fast path to the
    /// exact same move — except castling, which always falls back.
    fn differential(p: &Chess) {
        for m in p.legal_moves() {
            let s = San::from_move(p, m);
            match s {
                San::Normal { .. } => {
                    assert_eq!(
                        resolve(p, &s),
                        Some(m),
                        "fast path failed for {s} in {}",
                        Fen::from_position(p, EnPassantMode::Always)
                    );
                }
                _ => assert_eq!(resolve(p, &s), None),
            }
        }
    }

    /// Walk every line of `depth` plies from `p`, running the differential
    /// sweep at every node (exercises ep rights, castling rights decay,
    /// checks, pins as they naturally arise).
    fn differential_walk(p: &Chess, depth: u32) {
        differential(p);
        if depth == 0 {
            return;
        }
        for m in p.legal_moves() {
            let mut next = p.clone();
            next.play_unchecked(m);
            differential_walk(&next, depth - 1);
        }
    }

    #[test]
    fn startpos_all_moves() {
        differential_walk(&Chess::default(), 2);
    }

    #[test]
    fn tricky_middlegame_walk() {
        // Position with castling rights, pins, captures and pawn structure.
        differential_walk(
            &pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
            1,
        );
    }

    #[test]
    fn ep_position_walk() {
        differential_walk(
            &pos("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
            1,
        );
    }

    #[test]
    fn promotion_position_walk() {
        differential_walk(&pos("3r1r2/4P3/8/8/8/8/8/k3K3 w - - 0 1"), 1);
    }

    #[test]
    fn pin_disambiguates() {
        // Knights b1 and e2 both reach c3, but e2 is pinned by the e4 rook:
        // "Nc3" is unambiguous and must resolve on the fast path.
        assert_resolves("4k3/8/8/8/4r3/8/4N3/1N2K3 w - - 0 1", "Nc3");
    }

    #[test]
    fn ambiguous_falls_back() {
        // Same but no pin: two legal knights => AmbiguousSan via fallback.
        let fen = "4k3/8/8/8/8/8/4N3/1N2K3 w - - 0 1";
        assert_falls_back(fen, "Nc3");
        assert!(san("Nc3").to_move(&pos(fen)).is_err());
        // With file/rank disambiguators both resolve on the fast path.
        assert_resolves(fen, "Nbc3");
        assert_resolves(fen, "Nec3");
        assert_resolves(fen, "N1c3");
        assert_resolves(fen, "N2c3");
    }

    #[test]
    fn pinned_piece_move_falls_back_as_illegal() {
        // Only knight e2 attacks d4 but it is pinned: 0 candidates.
        let fen = "4k3/8/8/8/4r3/8/4N3/4K3 w - - 0 1";
        assert_falls_back(fen, "Nd4");
        assert!(san("Nd4").to_move(&pos(fen)).is_err());
    }

    #[test]
    fn en_passant_resolves() {
        assert_resolves(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "exf6",
        );
        assert_resolves(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "exd6",
        );
    }

    #[test]
    fn ep_reveals_check_is_illegal() {
        // exd3 e.p. removes both pawns from the 4th rank, exposing the black
        // king on a4 to the h4 rook: illegal, 0 candidates, falls back.
        let fen = "8/8/8/8/k2Pp2R/8/8/4K3 b - d3 0 1";
        assert_falls_back(fen, "exd3");
        assert!(san("exd3").to_move(&pos(fen)).is_err());
    }

    #[test]
    fn promotions_resolve() {
        let fen = "3r1r2/4P3/8/8/8/8/8/k3K3 w - - 0 1";
        assert_resolves(fen, "exd8=Q");
        assert_resolves(fen, "exf8=N");
        assert_resolves(fen, "e8=Q");
        assert_resolves(fen, "e8=B");
        // Promotion move without '=X' or to a non-backrank square: fallback.
        assert_falls_back(fen, "e8");
        assert!(san("e8").to_move(&pos(fen)).is_err());
    }

    #[test]
    fn check_evasions_resolve() {
        // White king e1 in check from the d3 knight.
        let fen = "4k3/8/8/8/8/2Rn4/8/4K3 w - - 0 1";
        assert_resolves(fen, "Rxd3"); // capture the checker
        assert_resolves(fen, "Kd1"); // step out of check
        // Moves that do not address the check: 0 candidates => fallback.
        assert_falls_back(fen, "Rc8");
        assert!(san("Rc8").to_move(&pos(fen)).is_err());
    }

    #[test]
    fn king_cannot_retreat_along_checking_ray() {
        // Rook a1 checks the king along rank 1. Kf1 stays on the ray: the
        // vacated e1 square must not block the rook in the legality test
        // (i.e. occupancy must be updated, not the pre-move one).
        let fen = "4k3/8/8/8/8/8/8/r3K3 w - - 0 1";
        assert_falls_back(fen, "Kf1");
        assert!(san("Kf1").to_move(&pos(fen)).is_err());
        assert_resolves(fen, "Kd2"); // off the ray: legal
    }

    #[test]
    fn king_captures_adjacent_checker() {
        // Undefended rook e2 checking Ke1: Kxe2 is legal; the captured rook
        // must be excluded from the attacker set.
        assert_resolves("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", "Kxe2");
        // Same but the rook is defended by the g3 knight: illegal.
        let fen = "4k3/8/8/8/8/6n1/4r3/4K3 w - - 0 1";
        assert_falls_back(fen, "Kxe2");
        assert!(san("Kxe2").to_move(&pos(fen)).is_err());
    }

    #[test]
    fn double_push_rules() {
        assert_resolves(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e4",
        );
        // Double push over an occupied square: illegal => fallback.
        let blocked = "4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1";
        assert_falls_back(blocked, "e4");
        assert!(san("e4").to_move(&pos(blocked)).is_err());
    }

    #[test]
    fn castling_falls_back() {
        let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
        assert_falls_back(fen, "O-O");
        assert_falls_back(fen, "O-O-O");
        // ... but to_move succeeds, i.e. the fallback path covers it.
        assert!(san("O-O").to_move(&pos(fen)).is_ok());
    }

    #[test]
    fn capture_flag_mismatch_falls_back() {
        let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        assert_falls_back(start, "Nxf3"); // capture flag, empty square
        assert!(san("Nxf3").to_move(&pos(start)).is_err());
    }

    #[test]
    fn chess960_normal_moves_resolve() {
        // Chess960 start (NRKBBQNR); castling always falls back, normal
        // moves must use the fast path.
        let p: Chess = "nrkbbqnr/pppppppp/8/8/8/8/PPPPPPPP/NRKBBQNR w KQkq - 0 1"
            .parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Chess960)
            .unwrap();
        differential_walk(&p, 1);
    }
}
