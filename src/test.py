import unittest
import numpy as np

import rust_pgn_reader_python_binding
from rust_pgn_reader_python_binding import PyGameView  # for a typing check

import pyarrow as pa


class TestPgnExtraction(unittest.TestCase):
    def run_extractor(self, pgn_string):
        extractor = rust_pgn_reader_python_binding.parse_game(pgn_string)
        return extractor

    def test_short_pgn(self):
        pgn_moves = "1. e4 {asdf} e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O {hello} Bc5 5. d3 d6 6. h3 h6 7. c3 O-O 8. Be3 a6 9. Ba4 Bd7 10. Bxc5 dxc5 11. Bxc6 Bxc6 12. Nxe5 Bb5 13. Re1 Re8 14. c4 Rxe5 15. cxb5 axb5 16. Nc3 b4 17. Nd5 Nxd5 18. exd5 Rxe1+ 19. Qxe1 Qxd5 20. Qe3 b6 21. b3 c6 22. Qe2 b5 23. Rd1 Qd4 24. Qe7 Rxa2 25. Qe8+ Kh7 26. Qxf7 c4 27. Qf5+ Kh8 28. Qf8+ Kh7 29. Qf5+ Kh8 30. Qf8+ Kh7 31. Qf5+ Kh8 32. Qf8+ Kh7 1/2-1/2"
        extractor = self.run_extractor(pgn_moves)

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

        comments_reference = [
            "asdf",
            None,
            None,
            None,
            None,
            None,
            "hello",
        ] + [None for _ in range(57)]

        valid_reference = True
        evals_reference = [None for _ in range(len(moves_reference))]
        clock_times_reference = [None for _ in range(len(moves_reference))]

        self.assertTrue([str(move) for move in extractor.moves] == moves_reference)
        self.assertTrue(extractor.comments == comments_reference)
        self.assertTrue(extractor.valid_moves == valid_reference)
        self.assertTrue(extractor.evals == evals_reference)
        self.assertTrue(extractor.clock_times == clock_times_reference)

        assert extractor.position_status is not None  # appease the type checker
        self.assertFalse(extractor.position_status.is_checkmate)
        self.assertFalse(extractor.position_status.is_stalemate)
        self.assertFalse(extractor.position_status.is_game_over)

    def test_full_pgn(self):
        pgn_moves = """
[Event "Rated Classical game"]
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
        extractor = self.run_extractor(pgn_moves)

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
        comments_reference = [None for _ in range(len(moves_reference))]
        valid_reference = True
        evals_reference = [None for _ in range(len(moves_reference))]
        clock_times_reference = [None for _ in range(len(moves_reference))]
        headers_reference = [
            ("Event", "Rated Classical game"),
            ("Site", "https://lichess.org/lhy6ehiv"),
            ("White", "goerch"),
            ("Black", "niltonrosao001"),
            ("Result", "0-1"),
            ("UTCDate", "2013.06.30"),
            ("UTCTime", "22:10:02"),
            ("WhiteElo", "1702"),
            ("BlackElo", "2011"),
            ("WhiteRatingDiff", "-3"),
            ("BlackRatingDiff", "+5"),
            ("ECO", "A46"),
            ("Opening", "Indian Game: Spielmann-Indian"),
            ("TimeControl", "600+8"),
            ("Termination", "Normal"),
        ]

        self.assertTrue([str(move) for move in extractor.moves] == moves_reference)
        self.assertTrue(extractor.comments == comments_reference)
        self.assertTrue(extractor.valid_moves == valid_reference)
        self.assertTrue(extractor.evals == evals_reference)
        self.assertTrue(extractor.clock_times == clock_times_reference)
        self.assertTrue(extractor.headers == headers_reference)

        assert extractor.position_status is not None  # appease the type checker
        self.assertFalse(extractor.position_status.is_checkmate)
        self.assertFalse(extractor.position_status.is_stalemate)
        self.assertFalse(extractor.position_status.is_game_over)

    def test_full_pgn_annotated(self):
        pgn_moves = """
    1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... c5 { [%eval 0.19] [%clk 0:00:30] }
    2. Nf3 { [%eval 0.25] [%clk 0:00:29] } 2... Nc6 { [%eval 0.33] [%clk 0:00:30] }
    3. Bc4 { [%eval -0.13] [%clk 0:00:28] } 3... e6 { [%eval -0.04] [%clk 0:00:30] }
    4. c3 { [%eval -0.4] [%clk 0:00:27] } 4... b5? { [%eval 1.18] [%clk 0:00:30] }
    5. Bb3?! { [%eval 0.21] [%clk 0:00:26] } 5... c4 { [%eval 0.32] [%clk 0:00:29] }
    6. Bc2 { [%eval 0.2] [%clk 0:00:25] } 6... a5 { [%eval 0.6] [%clk 0:00:29] }
    7. d4 { [%eval 0.29] [%clk 0:00:23] } 7... cxd3 { [%eval 0.6] [%clk 0:00:27] }
    8. Qxd3 { [%eval 0.12] [%clk 0:00:22] } 8... Nf6 { [%eval 0.52] [%clk 0:00:26] }
    9. e5 { [%eval 0.39] [%clk 0:00:21] } 9... Nd5 { [%eval 0.45] [%clk 0:00:25] }
    10. Bg5?! { [%eval -0.44] [%clk 0:00:18] } 10... Qc7 { [%eval -0.12] [%clk 0:00:23] }
    11. Nbd2?? { [%eval -3.15] [%clk 0:00:14] } 11... h6 { [%eval -2.99] [%clk 0:00:23] }
    12. Bh4 { [%eval -3.0] [%clk 0:00:11] } 12... Ba6? { [%eval -0.12] [%clk 0:00:23] }
    13. b3?? { [%eval -4.14] [%clk 0:00:02] } 13... Nf4? { [%eval -2.73] [%clk 0:00:21] } 0-1
    """
        extractor = self.run_extractor(pgn_moves)

        moves_reference = [
            "e2e4",
            "c7c5",
            "g1f3",
            "b8c6",
            "f1c4",
            "e7e6",
            "c2c3",
            "b7b5",
            "c4b3",
            "c5c4",
            "b3c2",
            "a7a5",
            "d2d4",
            "c4d3",
            "d1d3",
            "g8f6",
            "e4e5",
            "f6d5",
            "c1g5",
            "d8c7",
            "b1d2",
            "h7h6",
            "g5h4",
            "c8a6",
            "b2b3",
            "d5f4",
        ]
        comments_reference = [
            "",
        ] * 26
        valid_reference = True
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
        clock_times_reference = [
            (0, 0, 30),
            (0, 0, 30),
            (0, 0, 29),
            (0, 0, 30),
            (0, 0, 28),
            (0, 0, 30),
            (0, 0, 27),
            (0, 0, 30),
            (0, 0, 26),
            (0, 0, 29),
            (0, 0, 25),
            (0, 0, 29),
            (0, 0, 23),
            (0, 0, 27),
            (0, 0, 22),
            (0, 0, 26),
            (0, 0, 21),
            (0, 0, 25),
            (0, 0, 18),
            (0, 0, 23),
            (0, 0, 14),
            (0, 0, 23),
            (0, 0, 11),
            (0, 0, 23),
            (0, 0, 2),
            (0, 0, 21),
        ]
        self.assertTrue([str(move) for move in extractor.moves] == moves_reference)
        self.assertTrue(extractor.comments == comments_reference)
        self.assertTrue(extractor.valid_moves == valid_reference)
        self.assertTrue(extractor.evals == evals_reference)
        self.assertTrue(extractor.clock_times == clock_times_reference)

        assert extractor.position_status is not None  # appease the type checker
        self.assertFalse(extractor.position_status.is_checkmate)
        self.assertFalse(extractor.position_status.is_stalemate)
        self.assertFalse(extractor.position_status.is_game_over)

    def test_multithreaded(self):
        pgns = [
            "1. Nf3 g6 2. b3 Bg7 3. Nc3 e5 4. Bb2 e4 5. Ng1 d6 6. Rb1 a5 7. Nxe4 Bxb2 8. Rxb2 Nf6 9. Nxf6+ Qxf6 10. Rb1 Ra6 11. e3 Rb6 12. d4 a4 13. bxa4 Nc6 14. Rxb6 cxb6 15. Bb5 Bd7 16. Bxc6 Bxc6 17. Nf3 Bxa4 18. O-O O-O 19. Re1 Rc8 20. Re2 d5 21. Ne5 Qf5 22. Qd3 Bxc2 23. Qxf5 Bxf5 24. h3 b5 25. Rb2 f6 26. Ng4 Rc6 27. Nh6+ Kg7 28. Nxf5+ gxf5 29. Rxb5 Rc7 30. Rxd5 Kg6 31. f4 Kh5 32. Rxf5+ Kh4 33. Rxf6 Kg3 34. d5 Rc1# 0-1",
            "1. e4 {asdf} e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O {hello} Bc5 5. d3 d6 6. h3 h6 7. c3 O-O",
        ]
        extractor = rust_pgn_reader_python_binding.parse_games(pgns)

        moves_reference = [
            [
                "g1f3",
                "g7g6",
                "b2b3",
                "f8g7",
                "b1c3",
                "e7e5",
                "c1b2",
                "e5e4",
                "f3g1",
                "d7d6",
                "a1b1",
                "a7a5",
                "c3e4",
                "g7b2",
                "b1b2",
                "g8f6",
                "e4f6",
                "d8f6",
                "b2b1",
                "a8a6",
                "e2e3",
                "a6b6",
                "d2d4",
                "a5a4",
                "b3a4",
                "b8c6",
                "b1b6",
                "c7b6",
                "f1b5",
                "c8d7",
                "b5c6",
                "d7c6",
                "g1f3",
                "c6a4",
                "e1g1",
                "e8g8",
                "f1e1",
                "f8c8",
                "e1e2",
                "d6d5",
                "f3e5",
                "f6f5",
                "d1d3",
                "a4c2",
                "d3f5",
                "c2f5",
                "h2h3",
                "b6b5",
                "e2b2",
                "f7f6",
                "e5g4",
                "c8c6",
                "g4h6",
                "g8g7",
                "h6f5",
                "g6f5",
                "b2b5",
                "c6c7",
                "b5d5",
                "g7g6",
                "f2f4",
                "g6h5",
                "d5f5",
                "h5h4",
                "f5f6",
                "h4g3",
                "d4d5",
                "c7c1",
            ],
            [
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
            ],
        ]

        comments_reference = [
            [None for _ in range(len(moves_reference[0]))],
            [
                "asdf",
                None,
                None,
                None,
                None,
                None,
                "hello",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        ]

        self.assertTrue(extractor[0].comments == comments_reference[0])
        self.assertTrue(extractor[1].comments == comments_reference[1])

        self.assertTrue(
            [str(move) for move in extractor[0].moves] == moves_reference[0]
        )
        self.assertTrue(
            [str(move) for move in extractor[1].moves] == moves_reference[1]
        )
        assert extractor[0].position_status is not None  # appease the type checker

        self.assertTrue(extractor[0].position_status.is_checkmate)
        self.assertFalse(extractor[0].position_status.is_stalemate)
        self.assertTrue(extractor[0].position_status.is_game_over)
        self.assertTrue(extractor[0].position_status.legal_move_count == 0)
        self.assertTrue(extractor[0].position_status.turn == 1)
        self.assertTrue(extractor[0].position_status.insufficient_material == (0, 0))

        self.assertTrue(extractor[1].position_status is None)
        extractor[1].update_position_status()
        assert extractor[1].position_status is not None  # appease the type checker
        self.assertFalse(extractor[1].position_status.is_checkmate)
        self.assertFalse(extractor[1].position_status.is_stalemate)
        self.assertFalse(extractor[1].position_status.is_game_over)
        self.assertTrue(extractor[1].position_status.legal_move_count == 36)
        self.assertTrue(extractor[1].position_status.turn == 1)
        self.assertTrue(extractor[1].position_status.insufficient_material == (0, 0))

    def test_castling(self):
        pgn_moves = """
        1. e4 e5 2. Bc4 c6 3. Nf3 d6 4. Rg1 f6 5. Rh1 g6 6. Ke2 b6 7. Ke1 g5
        """

        extractor = self.run_extractor(pgn_moves)

        castling_reference = [
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
            (True, False, True, True),
            (True, False, True, True),
            (True, False, True, True),
            (True, False, True, True),
            (False, False, True, True),
            (False, False, True, True),
            (False, False, True, True),
            (False, False, True, True),
        ]

        self.assertTrue(extractor.castling_rights == castling_reference)

    def test_parse_game_moves_arrow_chunked_array(self):
        pgns = [
            "1. Nf3 g6 2. b3 Bg7 3. Nc3 e5 4. Bb2 e4 5. Ng1 d6 6. Rb1 a5 7. Nxe4 Bxb2 8. Rxb2 Nf6 9. Nxf6+ Qxf6 10. Rb1 Ra6 11. e3 Rb6 12. d4 a4 13. bxa4 Nc6 14. Rxb6 cxb6 15. Bb5 Bd7 16. Bxc6 Bxc6 17. Nf3 Bxa4 18. O-O O-O 19. Re1 Rc8 20. Re2 d5 21. Ne5 Qf5 22. Qd3 Bxc2 23. Qxf5 Bxf5 24. h3 b5 25. Rb2 f6 26. Ng4 Rc6 27. Nh6+ Kg7 28. Nxf5+ gxf5 29. Rxb5 Rc7 30. Rxd5 Kg6 31. f4 Kh5 32. Rxf5+ Kh4 33. Rxf6 Kg3 34. d5 Rc1# 0-1",
            "1. e4 {asdf} e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O {hello} Bc5 5. d3 d6 6. h3 h6 7. c3 O-O",
        ]

        # Create a PyArrow ChunkedArray
        arrow_array = pa.array(pgns, type=pa.string())
        chunked_array = pa.chunked_array([arrow_array])

        extractors = (
            rust_pgn_reader_python_binding.parse_game_moves_arrow_chunked_array(
                chunked_array
            )
        )

        moves_reference = [
            [
                "g1f3",
                "g7g6",
                "b2b3",
                "f8g7",
                "b1c3",
                "e7e5",
                "c1b2",
                "e5e4",
                "f3g1",
                "d7d6",
                "a1b1",
                "a7a5",
                "c3e4",
                "g7b2",
                "b1b2",
                "g8f6",
                "e4f6",
                "d8f6",
                "b2b1",
                "a8a6",
                "e2e3",
                "a6b6",
                "d2d4",
                "a5a4",
                "b3a4",
                "b8c6",
                "b1b6",
                "c7b6",
                "f1b5",
                "c8d7",
                "b5c6",
                "d7c6",
                "g1f3",
                "c6a4",
                "e1g1",
                "e8g8",
                "f1e1",
                "f8c8",
                "e1e2",
                "d6d5",
                "f3e5",
                "f6f5",
                "d1d3",
                "a4c2",
                "d3f5",
                "c2f5",
                "h2h3",
                "b6b5",
                "e2b2",
                "f7f6",
                "e5g4",
                "c8c6",
                "g4h6",
                "g8g7",
                "h6f5",
                "g6f5",
                "b2b5",
                "c6c7",
                "b5d5",
                "g7g6",
                "f2f4",
                "g6h5",
                "d5f5",
                "h5h4",
                "f5f6",
                "h4g3",
                "d4d5",
                "c7c1",
            ],
            [
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
            ],
        ]

        self.assertTrue(
            [str(move) for move in extractors[0].moves] == moves_reference[0]
        )
        self.assertTrue(
            [str(move) for move in extractors[1].moves] == moves_reference[1]
        )

        comments_reference = [
            [None for _ in range(len(moves_reference[0]))],
            [
                "asdf",
                None,
                None,
                None,
                None,
                None,
                "hello",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        ]

        self.assertTrue(extractors[0].comments == comments_reference[0])
        self.assertTrue(extractors[1].comments == comments_reference[1])

        extractors[0].update_position_status()  # Ensure status is calculated
        assert extractors[0].position_status is not None  # appease the type checker
        self.assertTrue(extractors[0].position_status.is_checkmate)
        self.assertFalse(extractors[0].position_status.is_stalemate)
        self.assertTrue(extractors[0].position_status.is_game_over)
        self.assertTrue(extractors[0].position_status.legal_move_count == 0)
        self.assertTrue(
            extractors[0].position_status.turn == True
        )  # White's turn, but Black delivered checkmate
        self.assertTrue(
            extractors[0].position_status.insufficient_material == (False, False)
        )

        self.assertTrue(
            extractors[1].position_status is None
        )  # Not set by default for parse_game_moves_arrow_chunked_array
        extractors[1].update_position_status()
        assert extractors[1].position_status is not None  # appease the type checker
        self.assertFalse(extractors[1].position_status.is_checkmate)
        self.assertFalse(extractors[1].position_status.is_stalemate)
        self.assertFalse(extractors[1].position_status.is_game_over)
        self.assertTrue(extractors[1].position_status.legal_move_count == 36)
        self.assertTrue(extractors[1].position_status.turn == True)  # White's turn
        self.assertTrue(
            extractors[1].position_status.insufficient_material == (False, False)
        )


class TestParsedGamesFlat(unittest.TestCase):
    def test_basic_structure(self):
        """Test basic flat parsing returns correct structure."""
        pgns = [
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0",
            "1. d4 d5 2. c4 e6 0-1",
        ]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Check game count
        self.assertEqual(len(result), 2)

        # Check move offsets
        self.assertEqual(len(result.move_offsets), 3)
        self.assertEqual(result.move_offsets[0], 0)
        self.assertEqual(result.move_offsets[1], 5)  # Game 1: 5 half-moves
        self.assertEqual(result.move_offsets[2], 9)  # Game 2: 4 half-moves

        # Check shapes
        total_moves = 9
        total_positions = 9 + 2  # moves + initial positions

        self.assertEqual(result.boards.shape, (total_positions, 8, 8))
        self.assertEqual(result.castling.shape, (total_positions, 4))
        self.assertEqual(result.en_passant.shape, (total_positions,))
        self.assertEqual(result.from_squares.shape, (total_moves,))
        self.assertEqual(result.valid.shape, (2,))

    def test_initial_board_encoding(self):
        """Test initial board state encoding."""
        pgns = ["1. e4 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        initial = result.boards[0]  # First position

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
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Position 0: initial, Position 1: after e4
        after_e4 = result.boards[1]

        # e2 (index 12) should be empty
        self.assertEqual(after_e4.flat[12], 0)
        # e4 (index 28) should have white pawn
        self.assertEqual(after_e4.flat[28], 1)

    def test_en_passant_tracking(self):
        """Test en passant square tracking."""
        pgns = ["1. e4 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Initial: no en passant
        self.assertEqual(result.en_passant[0], -1)
        # After e4: en passant on e-file (file index 4)
        self.assertEqual(result.en_passant[1], 4)

    def test_castling_rights(self):
        """Test castling rights tracking."""
        # White moves rook, losing kingside castling
        pgns = ["1. e4 e5 2. Nf3 Nc6 3. Rg1 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Initial: all castling [K, Q, k, q] = [True, True, True, True]
        self.assertTrue(all(result.castling[0]))

        # After Rg1 (position 5): white kingside lost
        # Castling order: [K, Q, k, q]
        self.assertFalse(result.castling[5, 0])  # White K
        self.assertTrue(result.castling[5, 1])  # White Q
        self.assertTrue(result.castling[5, 2])  # Black k
        self.assertTrue(result.castling[5, 3])  # Black q

    def test_turn_tracking(self):
        """Test side-to-move tracking."""
        pgns = ["1. e4 e5 2. Nf3 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Initial: white to move
        self.assertTrue(result.turn[0])
        # After e4: black to move
        self.assertFalse(result.turn[1])
        # After e5: white to move
        self.assertTrue(result.turn[2])

    def test_game_view_access(self):
        """Test GameView provides correct slices."""
        pgns = ["1. e4 e5 2. Nf3 1-0", "1. d4 d5 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

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
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        game = result[0]
        self.assertEqual(game.move_uci(0), "e2e4")
        self.assertEqual(game.move_uci(1), "e7e5")
        self.assertEqual(game.move_uci(2), "g1f3")

        self.assertEqual(game.moves_uci(), ["e2e4", "e7e5", "g1f3"])

    def test_iteration(self):
        """Test iteration over games."""
        pgns = ["1. e4 1-0", "1. d4 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        games = list(result)
        self.assertEqual(len(games), 3)
        self.assertIsInstance(games[0], PyGameView)

    def test_slicing(self):
        """Test slicing returns BatchSlice."""
        pgns = ["1. e4 1-0", "1. d4 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        sliced = result[1:3]
        self.assertEqual(len(sliced), 2)
        games = list(sliced)
        self.assertEqual(len(games[0]), 1)  # d4 game

    def test_position_to_game_mapping(self):
        """Test position to game index mapping."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]  # 2 moves (3 pos), 1 move (2 pos)
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Positions: 0,1,2 (game 0), 3,4 (game 1)
        pos_indices = np.array([0, 1, 2, 3, 4])
        game_indices = result.position_to_game(pos_indices)

        np.testing.assert_array_equal(game_indices, [0, 0, 0, 1, 1])

    def test_move_to_game_mapping(self):
        """Test move to game index mapping."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]  # 2 moves, 1 move
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        move_indices = np.array([0, 1, 2])
        game_indices = result.move_to_game(move_indices)

        np.testing.assert_array_equal(game_indices, [0, 0, 1])

    def test_position_to_game_accepts_various_dtypes(self):
        """Test position_to_game accepts various integer dtypes."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Test various integer dtypes (int64 is optimal, others are converted)
        for dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            pos_indices = np.array([0, 1, 2, 3, 4], dtype=dtype)
            game_indices = result.position_to_game(pos_indices)
            np.testing.assert_array_equal(game_indices, [0, 0, 0, 1, 1])

    def test_move_to_game_accepts_various_dtypes(self):
        """Test move_to_game accepts various integer dtypes."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Test various integer dtypes (int64 is optimal, others are converted)
        for dtype in [np.int32, np.int64, np.uint32, np.uint64]:
            move_indices = np.array([0, 1, 2], dtype=dtype)
            game_indices = result.move_to_game(move_indices)
            np.testing.assert_array_equal(game_indices, [0, 0, 1])

    def test_clocks_and_evals(self):
        """Test clock and eval parsing."""
        pgn = """1. e4 { [%eval 0.17] [%clk 0:00:30] } 1... e5 { [%eval 0.19] [%clk 0:00:29] } 1-0"""
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        self.assertAlmostEqual(result.evals[0], 0.17, places=2)
        self.assertAlmostEqual(result.evals[1], 0.19, places=2)
        self.assertAlmostEqual(result.clocks[0], 30.0, places=1)
        self.assertAlmostEqual(result.clocks[1], 29.0, places=1)

    def test_missing_clocks_evals_are_nan(self):
        """Test missing clocks/evals are NaN."""
        pgns = ["1. e4 e5 1-0"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        self.assertTrue(np.isnan(result.clocks[0]))
        self.assertTrue(np.isnan(result.evals[0]))

    def test_headers_preserved(self):
        """Test headers are preserved as dicts."""
        pgn = """[White "Player1"]
[Black "Player2"]
[WhiteElo "1500"]

1. e4 1-0"""
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

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
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        self.assertEqual(len(result), 3)
        self.assertTrue(result.valid[0])
        self.assertFalse(result.valid[1])
        self.assertTrue(result.valid[2])

    def test_checkmate_detection(self):
        """Test checkmate is detected."""
        # Scholar's mate
        pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        game = result[0]
        self.assertTrue(game.is_checkmate)
        self.assertFalse(game.is_stalemate)
        self.assertEqual(game.legal_move_count, 0)

    def test_promotion(self):
        """Test promotion encoding."""
        # Simplified position reaching promotion
        pgn = """[FEN "8/P7/8/8/8/8/8/4K2k w - - 0 1"]

1. a8=Q 1-0"""
        chunked = pa.chunked_array([pa.array([pgn])])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # Promotion to queen = 5
        self.assertEqual(result.promotions[0], 5)

        game = result[0]
        self.assertEqual(game.move_uci(0), "a7a8q")

    def test_num_properties(self):
        """Test num_games, num_moves, num_positions properties."""
        pgns = ["1. e4 e5 1-0", "1. d4 0-1"]  # 2 + 1 = 3 moves, 3 + 2 = 5 positions
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        self.assertEqual(result.num_games, 2)
        self.assertEqual(result.num_moves, 3)
        self.assertEqual(result.num_positions, 5)

    def test_negative_indexing(self):
        """Test negative index access."""
        pgns = ["1. e4 1-0", "1. d4 0-1", "1. c4 1/2-1/2"]
        chunked = pa.chunked_array([pa.array(pgns)])
        result = rust_pgn_reader_python_binding.parse_games_flat(chunked)

        # -1 should be the last game
        last_game = result[-1]
        self.assertEqual(len(last_game), 1)

        # Check that from_squares for c4 is c2
        # c2 = file 2 (c) + rank 1 (2nd rank) * 8 = 2 + 8 = 10
        self.assertEqual(last_game.from_squares[0], 10)  # c2


if __name__ == "__main__":
    unittest.main()
