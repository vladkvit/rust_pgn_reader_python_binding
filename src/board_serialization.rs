//! Board state serialization helpers for ParsedGames output.
//!
//! Board encoding: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
//! This matches python-chess piece_type (1-6) with color offset (+6 for black).
//!
//! Square indexing: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
//! Note: This differs from some Python code that uses [7 - rank, file] indexing.
//! The Python wrapper can transpose if needed.

use shakmaty::{Chess, Color, EnPassantMode, Position, Role, Square};

/// TODO this is a bottleneck for the multithreaded part of the parser
/// Serialize board position to 64-byte array.
/// Index mapping: square index (a1=0, h8=63) -> piece value (0-12)
pub fn serialize_board(pos: &Chess) -> [u8; 64] {
    let mut board = [0u8; 64];
    let b = pos.board();

    for sq in Square::ALL {
        if let Some(piece) = b.piece_at(sq) {
            let piece_val = match piece.role {
                Role::Pawn => 1,
                Role::Knight => 2,
                Role::Bishop => 3,
                Role::Rook => 4,
                Role::Queen => 5,
                Role::King => 6,
            };
            let color_offset = if piece.color == Color::White { 0 } else { 6 };
            board[sq as usize] = piece_val + color_offset;
        }
    }
    board
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
}
