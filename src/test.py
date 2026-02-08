import unittest
import numpy as np

import rust_pgn_reader_python_binding
from rust_pgn_reader_python_binding import PyGameView  # for a typing check

import pyarrow as pa


class TestParsedGames(unittest.TestCase):
    def test_basic_structure(self):
        """Test basic parsing returns correct structure."""
        pgns = [
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0",
            "1. d4 d5 2. c4 e6 0-1",
        ]
        chunked = pa.chunked_array([pa.array(pgns)])
        # Use 1 thread to get a single chunk for predictable array shapes
        result = rust_pgn_reader_python_binding.parse_games(chunked, num_threads=1)

        # Check game count
        self.assertEqual(len(result), 2)
        self.assertEqual(result.num_games, 2)
        self.assertEqual(result.num_moves, 9)
        self.assertEqual(result.num_positions, 11)  # 9 moves + 2 initial positions

        # Check per-game structure via game views
        game0 = result[0]
        self.assertEqual(len(game0), 5)  # Game 1: 5 half-moves
        self.assertEqual(game0.num_positions, 6)

        game1 = result[1]
        self.assertEqual(len(game1), 4)  # Game 2: 4 half-moves
        self.assertEqual(game1.num_positions, 5)

        # With 1 thread, all games are in a single chunk
        self.assertEqual(result.num_chunks, 1)
        chunk = result.chunks[0]
        total_moves = 9
        total_positions = 9 + 2
        self.assertEqual(chunk.boards.shape, (total_positions, 8, 8))
        self.assertEqual(chunk.castling.shape, (total_positions, 4))
        self.assertEqual(chunk.en_passant.shape, (total_positions,))
        self.assertEqual(chunk.from_squares.shape, (total_moves,))
        self.assertEqual(chunk.valid.shape, (2,))

    def test_initial_board_encoding(self):
        """Test initial board state encoding."""
        pgns = ["1. e4 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        initial = result[0].initial_board  # First position

        # Encoding: 0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K, +6 for black
        # Square indexing: a1=0, b1=1, ..., h1=7, a2=8, ...

        # a1 (index 0) = white rook = 4
        self.assertEqual(initial.flat[0], 4)
        # b1 (index 1) = white knight = 2
        self.assertEqual(initial.flat[1], 2)
        # e1 (index 4) = white king = 6
        self.assertEqual(initial.flat[4], 6)
        # e2 (index 12) = white pawn = 1
        self.assertEqual(initial.flat[12], 1)
        # e4 (index 28) = empty = 0
        self.assertEqual(initial.flat[28], 0)
        # e7 (index 52) = black pawn = 7
        self.assertEqual(initial.flat[52], 7)
        # e8 (index 60) = black king = 12
        self.assertEqual(initial.flat[60], 12)

    def test_board_after_move(self):
        """Test board state updates correctly after move."""
        pgns = ["1. e4 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        # Position 0: initial, Position 1: after e4
        after_e4 = result[0].boards[1]

        # e2 (index 12) should be empty
        self.assertEqual(after_e4.flat[12], 0)
        # e4 (index 28) should have white pawn
        self.assertEqual(after_e4.flat[28], 1)

    def test_en_passant_tracking(self):
        """Test en passant square tracking."""
        pgns = ["1. e4 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        # Initial: no en passant
        self.assertEqual(game.en_passant[0], -1)
        # After e4: en passant on e-file (file index 4)
        self.assertEqual(game.en_passant[1], 4)

    def test_castling_rights(self):
        """Test castling rights tracking."""
        # White moves rook, losing kingside castling
        pgns = ["1. e4 e5 2. Nf3 Nc6 3. Rg1 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        # Initial: all castling [K, Q, k, q] = [True, True, True, True]
        self.assertTrue(all(game.castling[0]))

        # After Rg1 (position 5): white kingside lost
        # Castling order: [K, Q, k, q]
        self.assertFalse(game.castling[5, 0])  # White K
        self.assertTrue(game.castling[5, 1])  # White Q
        self.assertTrue(game.castling[5, 2])  # Black k
        self.assertTrue(game.castling[5, 3])  # Black q

    def test_turn_tracking(self):
        """Test side-to-move tracking."""
        pgns = ["1. e4 e5 2. Nf3 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        # Initial: white to move
        self.assertTrue(game.turn[0])
        # After e4: black to move
        self.assertFalse(game.turn[1])
        # After e5: white to move
        self.assertTrue(game.turn[2])

    def test_game_view_access(self):
        """Test GameView provides correct slices."""
        pgns = ["1. e4 e5 2. Nf3 1-0", "1. d4 d5 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game0 = result[0]
        self.assertEqual(len(game0), 3)
        self.assertEqual(game0.num_positions, 4)
        self.assertEqual(game0.boards.shape, (4, 8, 8))
        self.assertTrue(game0.is_valid)

        game1 = result[1]
        self.assertEqual(len(game1), 2)
        self.assertEqual(game1.num_positions, 3)

    def test_game_view_move_uci(self):
        """Test GameView UCI move conversion."""
        pgns = ["1. e4 e5 2. Nf3 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        self.assertEqual(game.move_uci(0), "e2e4")
        self.assertEqual(game.move_uci(1), "e7e5")
        self.assertEqual(game.move_uci(2), "g1f3")

        self.assertEqual(game.moves_uci(), ["e2e4", "e7e5", "g1f3"])

    def test_iteration(self):
        """Test iteration over games."""
        pgns = ["1. e4 1-0", "1. d4 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        games = list(result)
        self.assertEqual(len(games), 3)
        self.assertIsInstance(games[0], PyGameView)

    def test_slicing(self):
        """Test slicing returns list of game views."""
        pgns = ["1. e4 1-0", "1. d4 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        sliced = result[1:3]
        self.assertEqual(len(sliced), 2)
        games = list(sliced)
        self.assertEqual(len(games[0]), 1)  # d4 game

    def test_position_to_game_mapping(self):
        """Test position to game index mapping."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]  # 2 moves (3 pos), 1 move (2 pos)
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        # Positions: 0,1,2 (game 0), 3,4 (game 1)
        pos_indices = np.array([0, 1, 2, 3, 4])
        game_indices = result.position_to_game(pos_indices)

        np.testing.assert_array_equal(game_indices, [0, 0, 0, 1, 1])

    def test_move_to_game_mapping(self):
        """Test move to game index mapping."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]  # 2 moves, 1 move
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        move_indices = np.array([0, 1, 2])
        game_indices = result.move_to_game(move_indices)

        np.testing.assert_array_equal(game_indices, [0, 0, 1])

    def test_position_to_game_accepts_various_dtypes(self):
        """Test position_to_game accepts various integer dtypes."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        for dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            pos_indices = np.array([0, 1, 2, 3, 4], dtype=dtype)
            game_indices = result.position_to_game(pos_indices)
            np.testing.assert_array_equal(game_indices, [0, 0, 0, 1, 1])

    def test_move_to_game_accepts_various_dtypes(self):
        """Test move_to_game accepts various integer dtypes."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        for dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            move_indices = np.array([0, 1, 2], dtype=dtype)
            game_indices = result.move_to_game(move_indices)
            np.testing.assert_array_equal(game_indices, [0, 0, 1])

    def test_clocks_and_evals(self):
        """Test clock and eval parsing."""
        pgn = """1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... e5 { [%eval 0.19] [%clk 0:00:29] } 1-0"""
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        self.assertAlmostEqual(game.evals[0], 0.17, places=2)
        self.assertAlmostEqual(game.evals[1], 0.19, places=2)
        self.assertAlmostEqual(game.clocks[0], 30.0, places=1)
        self.assertAlmostEqual(game.clocks[1], 29.0, places=1)

    def test_missing_clocks_evals_are_nan(self):
        """Test missing clocks/evals are NaN."""
        pgns = ["1. e4 e5 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        self.assertTrue(np.isnan(game.clocks[0]))
        self.assertTrue(np.isnan(game.evals[0]))

    def test_headers_preserved(self):
        """Test headers are preserved as dicts."""
        pgn = """[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]

1. e4 1-0"""
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        self.assertEqual(game.headers["White"], "Player1")
        self.assertEqual(game.headers["Black"], "Player2")
        self.assertEqual(game.headers["WhiteElo"], "1500")

    def test_invalid_game_flagged(self):
        """Test invalid games are flagged but don't break structure."""
        pgns = [
            "1. e4 e5 1-0",  # Valid
            "1. e4 Qxd7 1-0",  # Invalid move
            "1. d4 d5 0-1",  # Valid
        ]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        self.assertEqual(len(result), 3)
        self.assertTrue(result[0].is_valid)
        self.assertFalse(result[1].is_valid)
        self.assertTrue(result[2].is_valid)

    def test_checkmate_detection(self):
        """Test checkmate is detected."""
        # Scholar's mate
        pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        self.assertTrue(game.is_checkmate)
        self.assertFalse(game.is_stalemate)
        self.assertEqual(game.legal_move_count, 0)

    def test_promotion(self):
        """Test promotion encoding."""
        pgn = """[FEN "8/P7/8/8/8/8/8/4K2k w - - 0 1"]

1. a8=Q 1-0"""
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        game = result[0]
        # Promotion to queen = 5
        self.assertEqual(game.promotions[0], 5)
        self.assertEqual(game.move_uci(0), "a7a8q")

    def test_num_properties(self):
        """Test num_games, num_moves, num_positions properties."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]  # 2 + 1 = 3 moves, 3 + 2 = 5 positions
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        self.assertEqual(result.num_games, 2)
        self.assertEqual(result.num_moves, 3)
        self.assertEqual(result.num_positions, 5)

    def test_negative_indexing(self):
        """Test negative index access."""
        pgns = ["1. e4 1-0", "1. d4 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        # -1 should be the last game
        last_game = result[-1]
        self.assertEqual(len(last_game), 1)

        # Check that from_squares for c4 is c2
        # c2 = file 2 (c) + rank 1 (2nd rank) * 8 = 2 + 8 = 10
        self.assertEqual(last_game.from_squares[0], 10)  # c2

    def test_outcome(self):
        """Test outcome is parsed from movetext."""
        pgns = ["1. e4 e5 1-0", "1. d4 d5 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked)

        self.assertEqual(result[0].outcome, "White")
        self.assertEqual(result[1].outcome, "Black")
        self.assertEqual(result[2].outcome, "Draw")

    def test_outcome_without_headers(self):
        """Test outcome works for PGNs without Result header."""
        pgn = "1. e4 e5 0-1"
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)
        self.assertEqual(result[0].outcome, "Black")

    def test_is_game_over(self):
        """Test is_game_over derived property."""
        # Scholar's mate
        pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)
        self.assertTrue(result[0].is_game_over)

        # Not game over
        pgn2 = "1. e4 e5 1-0"
        chunked2 = pa.chunked_array([pa.array([pgn2])])
        result2 = rust_pgn_reader_python_binding.parse_games(chunked2)
        self.assertFalse(result2[0].is_game_over)

    def test_comments_disabled_by_default(self):
        """Test comments are empty by default."""
        pgn = "1. e4 { a comment } e5 1-0"
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(chunked)
        self.assertEqual(result[0].comments, [])

    def test_comments_enabled(self):
        """Test comments are stored when enabled."""
        pgn = "1. e4 {asdf} e5 { [%eval 0.19] } 1-0"
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games(
            chunked, store_comments=True
        )
        comments = result[0].comments
        self.assertEqual(len(comments), 2)
        self.assertEqual(comments[0], "asdf")
        self.assertIsNotNone(comments[1])  # eval-only comment

    def test_parse_game_string(self):
        """Test parse_game convenience function for single string."""
        pgn = "1. e4 e5 2. Nf3 Nc6 1-0"
        result = rust_pgn_reader_python_binding.parse_game(pgn)
        self.assertEqual(result.num_games, 1)
        self.assertEqual(result[0].moves_uci(), ["e2e4", "e7e5", "g1f3", "b8c6"])
        self.assertEqual(result[0].outcome, "White")

    def test_parse_games_from_strings(self):
        """Test parse_games_from_strings convenience function."""
        pgns = [
            "1. e4 e5 1-0",
            "1. d4 d5 0-1",
        ]
        result = rust_pgn_reader_python_binding.parse_games_from_strings(pgns)
        self.assertEqual(result.num_games, 2)
        self.assertTrue(result[0].is_valid)
        self.assertTrue(result[1].is_valid)

    def test_long_game_with_comments(self):
        """Test a long game with inline comments, verifying all moves."""
        pgn = "1. e4 {asdf} e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O {hello} Bc5 5. d3 d6 6. h3 h6 7. c3 O-O 8. Be3 a6 9. Ba4 Bd7 10. Bxc5 dxc5 11. Bxc6 Bxc6 12. Nxe5 Bb5 13. Re1 Re8 14. c4 Rxe5 15. cxb5 axb5 16. Nc3 b4 17. Nd5 Nxd5 18. exd5 Rxe1+ 19. Qxe1 Qxd5 20. Qe3 b6 21. b3 c6 22. Qe2 b5 23. Rd1 Qd4 24. Qe7 Rxa2 25. Qe8+ Kh7 26. Qxf7 c4 27. Qf5+ Kh8 28. Qf8+ Kh7 29. Qf5+ Kh8 30. Qf8+ Kh7 31. Qf5+ Kh8 32. Qf8+ Kh7 1/2-1/2"

        result = rust_pgn_reader_python_binding.parse_game(pgn, store_comments=True)
        game = result[0]

        moves_reference = [
            "e2e4",
            "e7e5",
            "g1f3",
            "b8c6",
            "f1b5",
            "g8f6",
            "e1g1",
            "f8c5",
            "d2d3",
            "d7d6",
            "h2h3",
            "h7h6",
            "c2c3",
            "e8g8",
            "c1e3",
            "a7a6",
            "b5a4",
            "c8d7",
            "e3c5",
            "d6c5",
            "a4c6",
            "d7c6",
            "f3e5",
            "c6b5",
            "f1e1",
            "f8e8",
            "c3c4",
            "e8e5",
            "c4b5",
            "a6b5",
            "b1c3",
            "b5b4",
            "c3d5",
            "f6d5",
            "e4d5",
            "e5e1",
            "d1e1",
            "d8d5",
            "e1e3",
            "b7b6",
            "b2b3",
            "c7c6",
            "e3e2",
            "b6b5",
            "a1d1",
            "d5d4",
            "e2e7",
            "a8a2",
            "e7e8",
            "g8h7",
            "e8f7",
            "c5c4",
            "f7f5",
            "h7h8",
            "f5f8",
            "h8h7",
            "f8f5",
            "h7h8",
            "f5f8",
            "h8h7",
            "f8f5",
            "h7h8",
            "f5f8",
            "h8h7",
        ]

        self.assertEqual(game.moves_uci(), moves_reference)
        self.assertTrue(game.is_valid)
        self.assertEqual(game.outcome, "Draw")
        self.assertFalse(game.is_checkmate)
        self.assertFalse(game.is_stalemate)
        self.assertFalse(game.is_game_over)

        # Comments: "asdf" on move 0, "hello" on move 6, rest should be non-None
        # but empty (comments are enabled so placeholders exist)
        comments = game.comments
        self.assertEqual(len(comments), 64)
        self.assertIn("asdf", comments[0])
        self.assertIn("hello", comments[6])

    def test_full_game_with_headers(self):
        """Test a full game with many headers, verifying headers and moves."""
        pgn = """[Event "Rated Classical game"]
[Site "https://lichess.org/lhy6ehiv"]
[White "goerch"]
[Black "niltonrosao001"]
[Result "0-1"]
[UTCDate "2013.06.30"]
[UTCTime "22:10:02"]
[WhiteElo "1702"]
[BlackElo "2011"]
[WhiteRatingDiff "-3"]
[BlackRatingDiff "+5"]
[ECO "A46"]
[Opening "Indian Game: Spielmann-Indian"]
[TimeControl "600+8"]
[Termination "Normal"]

1. d4 Nf6 2. Nf3 c5 3. e3 b6 4. Nc3 e6 5. Bb5 a6 6. Bd3 Bb7 7. O-O b5 8. b3 d5 9. Bb2 Nbd7 10. a4 b4 11. Ne2 Bd6 12. c4 bxc3 13. Bxc3 O-O 14. Ng3 Rc8 15. dxc5 Nxc5 16. Nd4 Nxd3 17. Qxd3 Qb6 18. Rab1 Bb4 19. Nge2 Ne4 20. Rfc1 Nxc3 21. Nxc3 Rc7 22. Na2 Rfc8 23. Rc2 g6 24. Nxb4 Qxb4 25. Rbc1 Rxc2 26. Nxc2 Qc3 27. Qxc3 Rxc3 28. Kf1 d4 29. exd4 Be4 30. Ke1 Rxc2 31. Rxc2 Bxc2 32. Kd2 Bxb3 33. a5 Bd5 0-1
"""

        result = rust_pgn_reader_python_binding.parse_game(pgn)
        game = result[0]

        moves_reference = [
            "d2d4",
            "g8f6",
            "g1f3",
            "c7c5",
            "e2e3",
            "b7b6",
            "b1c3",
            "e7e6",
            "f1b5",
            "a7a6",
            "b5d3",
            "c8b7",
            "e1g1",
            "b6b5",
            "b2b3",
            "d7d5",
            "c1b2",
            "b8d7",
            "a2a4",
            "b5b4",
            "c3e2",
            "f8d6",
            "c2c4",
            "b4c3",
            "b2c3",
            "e8g8",
            "e2g3",
            "a8c8",
            "d4c5",
            "d7c5",
            "f3d4",
            "c5d3",
            "d1d3",
            "d8b6",
            "a1b1",
            "d6b4",
            "g3e2",
            "f6e4",
            "f1c1",
            "e4c3",
            "e2c3",
            "c8c7",
            "c3a2",
            "f8c8",
            "c1c2",
            "g7g6",
            "a2b4",
            "b6b4",
            "b1c1",
            "c7c2",
            "d4c2",
            "b4c3",
            "d3c3",
            "c8c3",
            "g1f1",
            "d5d4",
            "e3d4",
            "b7e4",
            "f1e1",
            "c3c2",
            "c1c2",
            "e4c2",
            "e1d2",
            "c2b3",
            "a4a5",
            "b3d5",
        ]

        self.assertEqual(game.moves_uci(), moves_reference)
        self.assertTrue(game.is_valid)
        self.assertEqual(game.outcome, "Black")

        # Verify headers
        self.assertEqual(game.headers["Event"], "Rated Classical game")
        self.assertEqual(game.headers["Site"], "https://lichess.org/lhy6ehiv")
        self.assertEqual(game.headers["White"], "goerch")
        self.assertEqual(game.headers["Black"], "niltonrosao001")
        self.assertEqual(game.headers["Result"], "0-1")
        self.assertEqual(game.headers["WhiteElo"], "1702")
        self.assertEqual(game.headers["BlackElo"], "2011")
        self.assertEqual(game.headers["ECO"], "A46")
        self.assertEqual(game.headers["Opening"], "Indian Game: Spielmann-Indian")
        self.assertEqual(game.headers["TimeControl"], "600+8")
        self.assertEqual(game.headers["Termination"], "Normal")

    def test_annotated_game_all_evals_and_clocks(self):
        """Test a fully annotated game with eval and clock on every move."""
        pgn = """1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... c5 { [%eval 0.19] [%clk 0:00:30] }
2. Nf3 { [%eval 0.25] [%clk 0:00:29] } 2... Nc6 { [%eval 0.33] [%clk 0:00:30] }
3. Bc4 { [%eval -0.13] [%clk 0:00:28] } 3... e6 { [%eval -0.04] [%clk 0:00:30] }
4. c3 { [%eval -0.4] [%clk 0:00:27] } 4... b5 { [%eval 1.18] [%clk 0:00:30] }
5. Bb3 { [%eval 0.21] [%clk 0:00:26] } 5... c4 { [%eval 0.32] [%clk 0:00:29] }
6. Bc2 { [%eval 0.2] [%clk 0:00:25] } 6... a5 { [%eval 0.6] [%clk 0:00:29] }
7. d4 { [%eval 0.29] [%clk 0:00:23] } 7... cxd3 { [%eval 0.6] [%clk 0:00:27] }
8. Qxd3 { [%eval 0.12] [%clk 0:00:22] } 8... Nf6 { [%eval 0.52] [%clk 0:00:26] }
9. e5 { [%eval 0.39] [%clk 0:00:21] } 9... Nd5 { [%eval 0.45] [%clk 0:00:25] }
10. Bg5 { [%eval -0.44] [%clk 0:00:18] } 10... Qc7 { [%eval -0.12] [%clk 0:00:23] }
11. Nbd2 { [%eval -3.15] [%clk 0:00:14] } 11... h6 { [%eval -2.99] [%clk 0:00:23] }
12. Bh4 { [%eval -3.0] [%clk 0:00:11] } 12... Ba6 { [%eval -0.12] [%clk 0:00:23] }
13. b3 { [%eval -4.14] [%clk 0:00:02] } 13... Nf4 { [%eval -2.73] [%clk 0:00:21] } 0-1"""

        result = rust_pgn_reader_python_binding.parse_game(pgn)
        game = result[0]

        # Verify move count
        self.assertEqual(len(game), 26)
        self.assertTrue(game.is_valid)
        self.assertEqual(game.outcome, "Black")

        # Verify all 26 evals
        evals_reference = [
            0.17,
            0.19,
            0.25,
            0.33,
            -0.13,
            -0.04,
            -0.4,
            1.18,
            0.21,
            0.32,
            0.2,
            0.6,
            0.29,
            0.6,
            0.12,
            0.52,
            0.39,
            0.45,
            -0.44,
            -0.12,
            -3.15,
            -2.99,
            -3.0,
            -0.12,
            -4.14,
            -2.73,
        ]
        for i, expected_eval in enumerate(evals_reference):
            self.assertAlmostEqual(
                game.evals[i], expected_eval, places=2, msg=f"Eval mismatch at move {i}"
            )

        # Verify all 26 clocks (stored as seconds)
        clock_seconds_reference = [
            30,
            30,
            29,
            30,
            28,
            30,
            27,
            30,
            26,
            29,
            25,
            29,
            23,
            27,
            22,
            26,
            21,
            25,
            18,
            23,
            14,
            23,
            11,
            23,
            2,
            21,
        ]
        for i, expected_seconds in enumerate(clock_seconds_reference):
            self.assertAlmostEqual(
                game.clocks[i],
                float(expected_seconds),
                places=1,
                msg=f"Clock mismatch at move {i}",
            )

    def test_castling_rights_through_game(self):
        """Test castling rights through multiple positions including king and rook moves."""
        pgn = "1. e4 e5 2. Bc4 c6 3. Nf3 d6 4. Rg1 f6 5. Rh1 g6 6. Ke2 b6 7. Ke1 g5 1-0"

        result = rust_pgn_reader_python_binding.parse_game(pgn)
        game = result[0]

        # 14 moves + 1 initial = 15 positions
        self.assertEqual(game.num_positions, 15)

        # Castling order: [K, Q, k, q]
        # Positions 0-6 (initial through 3. Nf3): all castling rights intact
        for pos in range(7):
            self.assertTrue(
                all(game.castling[pos]),
                f"Position {pos}: all castling should be intact",
            )

        # Position 7 (after 4. Rg1): white kingside lost
        self.assertFalse(game.castling[7, 0])  # White K gone
        self.assertTrue(game.castling[7, 1])  # White Q intact
        self.assertTrue(game.castling[7, 2])  # Black k intact
        self.assertTrue(game.castling[7, 3])  # Black q intact

        # Positions 8-10: same (Rh1 restores rook but not castling rights)
        for pos in [8, 9, 10]:
            self.assertFalse(game.castling[pos, 0])  # White K still gone
            self.assertTrue(game.castling[pos, 1])  # White Q intact

        # Position 11 (after 6. Ke2): white both castling lost
        self.assertFalse(game.castling[11, 0])  # White K
        self.assertFalse(game.castling[11, 1])  # White Q
        self.assertTrue(game.castling[11, 2])  # Black k
        self.assertTrue(game.castling[11, 3])  # Black q

        # Position 12-14: white castling remains lost (Ke1 doesn't restore it)
        for pos in [12, 13, 14]:
            self.assertFalse(game.castling[pos, 0])
            self.assertFalse(game.castling[pos, 1])

    def test_parse_games_from_strings_with_status(self):
        """Test parse_games_from_strings with checkmate, move verification, and status."""
        pgns = [
            # Game 0: ends in checkmate
            "1. Nf3 g6 2. b3 Bg7 3. Nc3 e5 4. Bb2 e4 5. Ng1 d6 6. Rb1 a5 7. Nxe4 Bxb2 8. Rxb2 Nf6 9. Nxf6+ Qxf6 10. Rb1 Ra6 11. e3 Rb6 12. d4 a4 13. bxa4 Nc6 14. Rxb6 cxb6 15. Bb5 Bd7 16. Bxc6 Bxc6 17. Nf3 Bxa4 18. O-O O-O 19. Re1 Rc8 20. Re2 d5 21. Ne5 Qf5 22. Qd3 Bxc2 23. Qxf5 Bxf5 24. h3 b5 25. Rb2 f6 26. Ng4 Rc6 27. Nh6+ Kg7 28. Nxf5+ gxf5 29. Rxb5 Rc7 30. Rxd5 Kg6 31. f4 Kh5 32. Rxf5+ Kh4 33. Rxf6 Kg3 34. d5 Rc1# 0-1",
            # Game 1: no checkmate, game abandoned mid-play
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O Bc5 5. d3 d6 6. h3 h6 7. c3 O-O 1-0",
        ]
        result = rust_pgn_reader_python_binding.parse_games_from_strings(pgns)

        self.assertEqual(result.num_games, 2)

        # Game 0: checkmate
        game0 = result[0]
        self.assertTrue(game0.is_valid)
        self.assertTrue(game0.is_checkmate)
        self.assertFalse(game0.is_stalemate)
        self.assertTrue(game0.is_game_over)
        self.assertEqual(game0.legal_move_count, 0)
        self.assertEqual(game0.outcome, "Black")
        # Verify some key moves
        moves0 = game0.moves_uci()
        self.assertEqual(len(moves0), 68)
        self.assertEqual(moves0[0], "g1f3")
        self.assertEqual(moves0[-1], "c7c1")

        # Game 1: no checkmate
        game1 = result[1]
        self.assertTrue(game1.is_valid)
        self.assertFalse(game1.is_checkmate)
        self.assertFalse(game1.is_stalemate)
        self.assertFalse(game1.is_game_over)
        self.assertEqual(game1.outcome, "White")
        moves1 = game1.moves_uci()
        self.assertEqual(len(moves1), 14)
        self.assertEqual(moves1[0], "e2e4")

    def test_multithreaded_correctness(self):
        """Test that multithreaded parsing produces correct results across chunk boundaries."""
        # Generate enough games to force multiple chunks with 4 threads
        pgns = []
        for i in range(40):
            if i % 3 == 0:
                pgns.append("1. e4 e5 2. Nf3 Nc6 1-0")
            elif i % 3 == 1:
                pgns.append("1. d4 d5 0-1")
            else:
                pgns.append("1. c4 e5 2. Nc3 1/2-1/2")

        # Parse single-threaded as reference
        result_1t = rust_pgn_reader_python_binding.parse_games_from_strings(
            pgns, num_threads=1
        )

        # Parse multi-threaded
        result_4t = rust_pgn_reader_python_binding.parse_games_from_strings(
            pgns, num_threads=4
        )

        # Same totals
        self.assertEqual(result_1t.num_games, result_4t.num_games)
        self.assertEqual(result_1t.num_moves, result_4t.num_moves)
        self.assertEqual(result_1t.num_positions, result_4t.num_positions)
        self.assertEqual(result_4t.num_games, 40)

        # Multiple chunks were actually created
        self.assertGreater(result_4t.num_chunks, 1)

        # Every game matches: moves, outcome, validity
        for i in range(40):
            g1 = result_1t[i]
            g4 = result_4t[i]
            self.assertEqual(g1.moves_uci(), g4.moves_uci(), f"Game {i} moves differ")
            self.assertEqual(g1.outcome, g4.outcome, f"Game {i} outcome differs")
            self.assertEqual(g1.is_valid, g4.is_valid, f"Game {i} validity differs")

        # position_to_game works across chunk boundaries
        all_pos = np.arange(result_4t.num_positions)
        game_ids = result_4t.position_to_game(all_pos)
        game_ids_1t = result_1t.position_to_game(all_pos)
        np.testing.assert_array_equal(game_ids, game_ids_1t)

        # move_to_game works across chunk boundaries
        all_moves = np.arange(result_4t.num_moves)
        move_game_ids = result_4t.move_to_game(all_moves)
        move_game_ids_1t = result_1t.move_to_game(all_moves)
        np.testing.assert_array_equal(move_game_ids, move_game_ids_1t)

    def test_empty_inputs(self):
        """Test empty inputs return valid empty results."""
        # parse_games_from_strings with empty list
        result = rust_pgn_reader_python_binding.parse_games_from_strings([])
        self.assertEqual(result.num_games, 0)
        self.assertEqual(result.num_moves, 0)
        self.assertEqual(result.num_positions, 0)
        self.assertEqual(len(result), 0)

        # parse_games with empty chunked array
        chunked = pa.chunked_array([pa.array([], type=pa.string())])
        result2 = rust_pgn_reader_python_binding.parse_games(chunked)
        self.assertEqual(result2.num_games, 0)
        self.assertEqual(len(result2), 0)

    def test_chunk_view_getters(self):
        """Test PyChunkView exposes correct array shapes and values."""
        pgns = ["1. e4 e5 1-0", "1. d4 d5 2. c4 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games(chunked, num_threads=1)

        self.assertEqual(result.num_chunks, 1)
        chunk = result.chunks[0]

        # Scalar getters
        self.assertEqual(chunk.num_games, 2)
        self.assertEqual(chunk.num_moves, 5)
        self.assertEqual(chunk.num_positions, 7)

        # Array shapes
        self.assertEqual(chunk.boards.shape, (7, 8, 8))
        self.assertEqual(chunk.castling.shape, (7, 4))
        self.assertEqual(chunk.en_passant.shape, (7,))
        self.assertEqual(chunk.halfmove_clock.shape, (7,))
        self.assertEqual(chunk.turn.shape, (7,))
        self.assertEqual(chunk.from_squares.shape, (5,))
        self.assertEqual(chunk.to_squares.shape, (5,))
        self.assertEqual(chunk.promotions.shape, (5,))
        self.assertEqual(chunk.clocks.shape, (5,))
        self.assertEqual(chunk.evals.shape, (5,))
        self.assertEqual(chunk.is_checkmate.shape, (2,))
        self.assertEqual(chunk.is_stalemate.shape, (2,))
        self.assertEqual(chunk.is_insufficient.shape, (2, 2))
        self.assertEqual(chunk.legal_move_count.shape, (2,))
        self.assertEqual(chunk.valid.shape, (2,))

        # CSR offset shapes
        self.assertEqual(chunk.move_offsets.shape, (3,))  # 2 games + 1
        self.assertEqual(chunk.position_offsets.shape, (3,))

        # Vec getters
        self.assertEqual(len(chunk.headers), 2)
        self.assertEqual(len(chunk.outcome), 2)
        self.assertEqual(chunk.outcome[0], "White")
        self.assertEqual(chunk.outcome[1], "Black")

        # Chunk values match per-game views
        game0 = result[0]
        game1 = result[1]
        np.testing.assert_array_equal(game0.boards, chunk.boards[:3])
        np.testing.assert_array_equal(game1.boards, chunk.boards[3:])

    def test_legal_moves_from_python(self):
        """Test legal_moves property returns correct moves for known positions."""
        pgn = "1. e4 1-0"
        result = rust_pgn_reader_python_binding.parse_game(pgn, store_legal_moves=True)
        game = result[0]

        legal = game.legal_moves
        # 2 positions: initial + after e4
        self.assertEqual(len(legal), 2)

        # Initial position: 20 legal moves
        self.assertEqual(len(legal[0]), 20)

        # After e4: black has 20 legal moves
        self.assertEqual(len(legal[1]), 20)

        # Verify e2e4 is among initial legal moves (from=12, to=28)
        initial_from_to = [(m[0], m[1]) for m in legal[0]]
        self.assertIn((12, 28), initial_from_to)  # e2e4

        # Verify d7d5 is among black's legal moves (from=51, to=35)
        black_from_to = [(m[0], m[1]) for m in legal[1]]
        self.assertIn((51, 35), black_from_to)  # d7d5

    def test_parse_error_surfaced(self):
        """Test that parse errors are stored and accessible."""
        pgns = [
            "1. e4 e5 1-0",  # Valid
            "1. e4 Qxd7 1-0",  # Invalid move
        ]
        result = rust_pgn_reader_python_binding.parse_games_from_strings(pgns)

        # Valid game: no error
        self.assertTrue(result[0].is_valid)
        self.assertIsNone(result[0].parse_error)

        # Invalid game: error message stored
        self.assertFalse(result[1].is_valid)
        self.assertIsNotNone(result[1].parse_error)
        self.assertIn("illegal move", result[1].parse_error)

    def test_parse_error_invalid_fen(self):
        """Test that invalid FEN produces a parse error."""
        pgn = '[FEN "invalid fen string"]\n\n1. e4 e5 1-0'
        result = rust_pgn_reader_python_binding.parse_game(pgn)

        self.assertFalse(result[0].is_valid)
        self.assertIsNotNone(result[0].parse_error)
        self.assertIn("FEN", result[0].parse_error)

    def test_repr(self):
        """Test __repr__ on ParsedGames, PyGameView, PyChunkView."""
        pgns = ["1. e4 e5 1-0"]
        result = rust_pgn_reader_python_binding.parse_games_from_strings(pgns)

        # ParsedGames repr
        r = repr(result)
        self.assertIn("ParsedGames", r)
        self.assertIn("1 games", r)

        # PyGameView repr
        game_repr = repr(result[0])
        self.assertIn("PyGameView", game_repr)

        # PyChunkView repr
        chunk_repr = repr(result.chunks[0])
        self.assertIn("PyChunkView", chunk_repr)


if __name__ == "__main__":
    unittest.main()
