//! Board state serialization helpers for ParsedGames output.
//!
//! Board encoding: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
//! This matches python-chess piece_type (1-6) with color offset (+6 for black).
//!
//! Square indexing: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
//! Note: This differs from some Python code that uses [7 - rank, file] indexing.
//! The Python wrapper can transpose if needed.

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role, Square};

/// Byte value of a piece in the serialized board encoding
/// (1-6 white PNBRQK, 7-12 black pnbrqk; Role enum values are 1-6).
fn piece_byte(role: Role, color: Color) -> u8 {
    role as u8 + if color == Color::White { 0 } else { 6 }
}

/// Serialize board position to 64-byte array.
/// Index mapping: square index (a1=0, h8=63) -> piece value (0-12)
///
/// Optimized to iterate by piece type using bitboards directly,
/// avoiding expensive piece_at() lookups for all 64 squares.
pub fn serialize_board(pos: &Chess) -> [u8; 64] {
    let mut board = [0u8; 64];
    let b = pos.board();

    for role in Role::ALL {
        for sq in b.by_role(role) & b.white() {
            board[sq as usize] = piece_byte(role, Color::White);
        }
        for sq in b.by_role(role) & b.black() {
            board[sq as usize] = piece_byte(role, Color::Black);
        }
    }

    board
}

/// Apply the board delta of `m` (played by `color`) to a serialized board.
///
/// Mirrors shakmaty's `do_move`: captures are implicit overwrites (including
/// promotion captures); castling clears both origin squares before setting
/// the destinations, which keeps the degenerate Chess960 cases correct —
/// a from-square can coincide with a to-square (e.g. Kg1+Rf1 O-O leaves the
/// king in place, Kf1+Rh1 O-O lands the rook on the king's origin).
pub fn apply_move(board: &mut [u8; 64], m: Move, color: Color) {
    match m {
        Move::Normal {
            role,
            from,
            to,
            promotion,
            ..
        } => {
            board[from as usize] = 0;
            board[to as usize] = piece_byte(promotion.unwrap_or(role), color);
        }
        Move::EnPassant { from, to } => {
            board[Square::from_coords(to.file(), from.rank()) as usize] = 0;
            board[from as usize] = 0;
            board[to as usize] = piece_byte(Role::Pawn, color);
        }
        Move::Castle { king, rook } => {
            let side = CastlingSide::from_queen_side(rook < king);
            board[king as usize] = 0;
            board[rook as usize] = 0;
            board[Square::from_coords(side.rook_to_file(), rook.rank()) as usize] =
                piece_byte(Role::Rook, color);
            board[Square::from_coords(side.king_to_file(), king.rank()) as usize] =
                piece_byte(Role::King, color);
        }
        // `San::to_move` can never return a drop move for standard Chess
        // (`San::Put` finds zero candidates, `San::Null` errors
        // unconditionally), so the parser never produces these.
        Move::Put { .. } => unreachable!("drop moves do not occur in Chess"),
    }
}

/// Get en passant file (0-7) or -1 if none.
/// Uses Always mode to report the e.p. square whenever a double pawn push occurred,
/// regardless of whether a legal capture is available.
pub fn get_en_passant_file(pos: &Chess) -> i8 {
    pos.ep_square(EnPassantMode::Always)
        .map(|sq| sq.file() as i8)
        .unwrap_or(-1)
}

/// Get halfmove clock (for 50-move rule).
pub fn get_halfmove_clock(pos: &Chess) -> u8 {
    pos.halfmoves().min(255) as u8
}

/// Get side to move: true = white, false = black.
pub fn get_turn(pos: &Chess) -> bool {
    pos.turn() == Color::White
}

/// Get castling rights as [K, Q, k, q] (white kingside, white queenside, black kingside, black queenside).
pub fn get_castling_rights(pos: &Chess) -> [bool; 4] {
    let rights = pos.castles().castling_rights();
    [
        rights.contains(Square::H1), // White kingside (rook on h1)
        rights.contains(Square::A1), // White queenside (rook on a1)
        rights.contains(Square::H8), // Black kingside (rook on h8)
        rights.contains(Square::A8), // Black queenside (rook on a8)
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{CastlingMode, fen::Fen, san::SanPlus};

    fn pos(fen: &str, mode: CastlingMode) -> Chess {
        fen.parse::<Fen>()
            .expect("valid FEN")
            .into_position(mode)
            .expect("legal position")
    }

    /// Differential: `apply_move` on the serialized board must match
    /// `serialize_board` of the played-out position.
    fn check_delta(p: &Chess, m: Move) {
        let mut board = serialize_board(p);
        apply_move(&mut board, m, p.turn());
        let mut after = p.clone();
        after.play_unchecked(m);
        assert_eq!(board, serialize_board(&after), "delta mismatch for {m:?}");
    }

    fn check_san(fen: &str, san: &str) {
        let p = pos(fen, CastlingMode::Standard);
        let m = SanPlus::from_ascii(san.as_bytes())
            .expect("valid SAN")
            .san
            .to_move(&p)
            .expect("legal move");
        check_delta(&p, m);
    }

    /// Chess960 castle applied from a directly constructed move (castle
    /// rights are irrelevant: `play_unchecked` does not verify them).
    fn check_960_castle(fen: &str, king: Square, rook: Square) {
        let p = pos(fen, CastlingMode::Chess960);
        check_delta(&p, Move::Castle { king, rook });
    }

    #[test]
    fn test_serialize_initial_board() {
        let pos = Chess::default();
        let board = serialize_board(&pos);

        // White pieces on rank 1 (indices 0-7)
        assert_eq!(board[0], 4); // a1 = white rook
        assert_eq!(board[1], 2); // b1 = white knight
        assert_eq!(board[2], 3); // c1 = white bishop
        assert_eq!(board[3], 5); // d1 = white queen
        assert_eq!(board[4], 6); // e1 = white king
        assert_eq!(board[5], 3); // f1 = white bishop
        assert_eq!(board[6], 2); // g1 = white knight
        assert_eq!(board[7], 4); // h1 = white rook

        // White pawns on rank 2 (indices 8-15)
        for i in 8..16 {
            assert_eq!(board[i], 1); // white pawn
        }

        // Empty squares (ranks 3-6, indices 16-47)
        for i in 16..48 {
            assert_eq!(board[i], 0);
        }

        // Black pawns on rank 7 (indices 48-55)
        for i in 48..56 {
            assert_eq!(board[i], 7); // black pawn (1 + 6)
        }

        // Black pieces on rank 8 (indices 56-63)
        assert_eq!(board[56], 10); // a8 = black rook (4 + 6)
        assert_eq!(board[57], 8); // b8 = black knight (2 + 6)
        assert_eq!(board[58], 9); // c8 = black bishop (3 + 6)
        assert_eq!(board[59], 11); // d8 = black queen (5 + 6)
        assert_eq!(board[60], 12); // e8 = black king (6 + 6)
        assert_eq!(board[61], 9); // f8 = black bishop (3 + 6)
        assert_eq!(board[62], 8); // g8 = black knight (2 + 6)
        assert_eq!(board[63], 10); // h8 = black rook (4 + 6)
    }

    #[test]
    fn test_initial_castling_rights() {
        let pos = Chess::default();
        let rights = get_castling_rights(&pos);
        assert_eq!(rights, [true, true, true, true]); // [K, Q, k, q]
    }

    #[test]
    fn test_initial_en_passant() {
        let pos = Chess::default();
        assert_eq!(get_en_passant_file(&pos), -1);
    }

    #[test]
    fn test_initial_halfmove_clock() {
        let pos = Chess::default();
        assert_eq!(get_halfmove_clock(&pos), 0);
    }

    #[test]
    fn test_initial_turn() {
        let pos = Chess::default();
        assert!(get_turn(&pos)); // White to move
    }

    #[test]
    fn test_apply_quiet_and_double_push() {
        check_san(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Nf3",
        );
        check_san(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "e4",
        );
    }

    #[test]
    fn test_apply_capture() {
        check_san(
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "exd5",
        );
    }

    #[test]
    fn test_apply_promotion() {
        let fen = "3r1r2/4P3/8/8/8/8/8/k3K3 w - - 0 1";
        check_san(fen, "e8=Q"); // quiet
        check_san(fen, "exd8=Q"); // capture
    }

    #[test]
    fn test_apply_en_passant() {
        // White: the e.p.-captured pawn sits beside the mover, not on `to`.
        check_san(
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "exf6",
        );
        // Black (opposite color offset in the encoding).
        check_san(
            "rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 2",
            "fxe3",
        );
    }

    #[test]
    fn test_apply_castle_standard() {
        check_san("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O");
        check_san("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O-O");
        check_san("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "O-O");
        check_san("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "O-O-O");
    }

    #[test]
    fn test_apply_castle_960_king_stationary() {
        // Kg1 + Rh1, O-O: king_to == king's origin.
        check_960_castle("4k3/8/8/8/8/8/8/6KR w - - 0 1", Square::G1, Square::H1);
    }

    #[test]
    fn test_apply_castle_960_rook_lands_on_king_origin() {
        // Kf1 + Rh1, O-O: rook_to (f1) == king's origin.
        check_960_castle("4k3/8/8/8/8/8/8/5K1R w - - 0 1", Square::F1, Square::H1);
    }

    #[test]
    fn test_apply_castle_960_swap() {
        // Kf1 + Rg1, O-O: king and rook swap squares (each destination is
        // the other's origin).
        check_960_castle("4k3/8/8/8/8/8/8/5KR1 w - - 0 1", Square::F1, Square::G1);
        // Same pattern for black: exercises the +6 color offset in castles.
        check_960_castle("5kr1/8/8/8/8/8/8/4K3 b - - 0 1", Square::F8, Square::G8);
    }

    #[test]
    fn test_apply_castle_960_queenside() {
        // Kb1 + Ra1, O-O-O: both pieces move, rook crosses over the king.
        check_960_castle("4k3/8/8/8/8/8/8/RK6 w - - 0 1", Square::B1, Square::A1);
    }
}
