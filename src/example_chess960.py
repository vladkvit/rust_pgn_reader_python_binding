import rust_pgn_reader_python_binding

pgn_moves = """
[Event "lc0MG"]
[Site "internet"]
[Date "????.??.??"]
[Round "-"]
[White "lc0.net.714559"]
[Black "lc0.net.714558"]
[Result "1/2-1/2"]
[Variant "chess960"]
[FEN "brkrqnnb/pppppppp/8/8/8/8/PPPPPPPP/BRKRQNNB w KQkq - 0 1"]

1.g3 d5 2.d4 g6 3.b3 Nf6 4.Ne3 b6 5.Nh3 Ne6 6.f4 Ng7 7.g4 h6 8.Nf2 Ne6 9.f5 Nf4 10.Nf1 gxf5 11.Qd2 Ne6 12.gxf5 Ng5 13.Ng3 Qd7 14.Rg1 Rg8 15.O-O-O O-O-O 16.Kb1 Kb8 17.Bb2 h5 18.Qf4 Bb7 19.h4 Nge4 20.Nfxe4 dxe4 21.Nxe4 Nxe4 22.Bxe4 Bxe4 23.Qxe4 e6 24.Rxg8 Rxg8 25.fxe6 fxe6 26.Rf1 Qe7 27.e3 Bf6 28.Ba3 Qd8 29.Qxe6 Bxh4 30.Rf5 Bg5 31.d5 Qc8 32.e4 h4 33.Qf7 Be3 34.d6 Rg1+ 35.Rf1 Rxf1+ 36.Qxf1 h3 37.dxc7+ Kxc7 38.Qc4+ Kb7 39.Qf7+ Qc7 40.Qd5+ Qc6 41.Qf7+ Qc7 42.Qd5+ Qc6 43.Qf7+ Qc7 1/2-1/2 {OL: 0}
"""

result = rust_pgn_reader_python_binding.parse_game(pgn_moves)
game = result[0]

print("Moves (UCI):", game.moves_uci())
print("Valid:", game.is_valid)
print("Outcome:", game.outcome)
print("Num positions:", game.num_positions)
print("Clocks (first 5):", game.clocks[:5].tolist())
