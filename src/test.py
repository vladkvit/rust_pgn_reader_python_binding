import unittest
import rust_pgn_reader_python_binding


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
        comments_reference = ["asdf", "hello"]
        valid_reference = True
        evals_reference = []
        clock_times_reference = []

        self.assertTrue(extractor.moves == moves_reference)
        self.assertTrue(extractor.comments == comments_reference)
        self.assertTrue(extractor.valid_moves == valid_reference)
        self.assertTrue(extractor.evals == evals_reference)
        self.assertTrue(extractor.clock_times == clock_times_reference)
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
        comments_reference = []
        valid_reference = True
        evals_reference = []
        clock_times_reference = []
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

        self.assertTrue(extractor.moves == moves_reference)
        self.assertTrue(extractor.comments == comments_reference)
        self.assertTrue(extractor.valid_moves == valid_reference)
        self.assertTrue(extractor.evals == evals_reference)
        self.assertTrue(extractor.clock_times == clock_times_reference)
        self.assertTrue(extractor.headers == headers_reference)
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
        self.assertTrue(extractor.moves == moves_reference)
        self.assertTrue(extractor.comments == comments_reference)
        self.assertTrue(extractor.valid_moves == valid_reference)
        self.assertTrue(extractor.evals == evals_reference)
        self.assertTrue(extractor.clock_times == clock_times_reference)
        self.assertFalse(extractor.position_status.is_checkmate)
        self.assertFalse(extractor.position_status.is_stalemate)
        self.assertFalse(extractor.position_status.is_game_over)

    def test_multithreaded(self):
        pgns = [
            "1. Nf3 g6 2. b3 Bg7 3. Nc3 e5 4. Bb2 e4 5. Ng1 d6 6. Rb1 a5 7. Nxe4 Bxb2 8. Rxb2 Nf6 9. Nxf6+ Qxf6 10. Rb1 Ra6 11. e3 Rb6 12. d4 a4 13. bxa4 Nc6 14. Rxb6 cxb6 15. Bb5 Bd7 16. Bxc6 Bxc6 17. Nf3 Bxa4 18. O-O O-O 19. Re1 Rc8 20. Re2 d5 21. Ne5 Qf5 22. Qd3 Bxc2 23. Qxf5 Bxf5 24. h3 b5 25. Rb2 f6 26. Ng4 Rc6 27. Nh6+ Kg7 28. Nxf5+ gxf5 29. Rxb5 Rc7 30. Rxd5 Kg6 31. f4 Kh5 32. Rxf5+ Kh4 33. Rxf6 Kg3 34. d5 Rc1# 0-1",
            "1. e4 {asdf} e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O {hello} Bc5 5. d3 d6 6. h3 h6 7. c3 O-O",
        ]
        extractor = rust_pgn_reader_python_binding.parse_games(pgns)

        comments_reference = [[], ["asdf", "hello"]]

        self.assertTrue(extractor[0].comments == comments_reference[0])
        self.assertTrue(extractor[1].comments == comments_reference[1])

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

        self.assertTrue(extractor[0].moves == moves_reference[0])
        self.assertTrue(extractor[1].moves == moves_reference[1])
        self.assertTrue(extractor[0].position_status.is_checkmate)
        self.assertFalse(extractor[0].position_status.is_stalemate)
        self.assertTrue(extractor[0].position_status.is_game_over)
        self.assertTrue(extractor[0].position_status.legal_move_count == 0)
        self.assertTrue(extractor[0].position_status.turn == 1)
        self.assertTrue(extractor[0].position_status.insufficient_material == (0, 0))

        self.assertTrue(extractor[1].position_status is None)
        extractor[1].update_position_status()
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


if __name__ == "__main__":
    unittest.main()
