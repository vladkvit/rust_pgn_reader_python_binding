import rust_pgn_reader_python_binding
import numpy as np

pgn_moves = """
[Event "Casual Correspondence game"]
[Site "https:///gdEj47Dv"]
[Date ""]
[White "lichess AI level 8"]
[Black "TherealARB"]
[Result "0-1"]
[UTCDate ""]
[UTCTime ":15"]
[WhiteElo "?"]
[BlackElo "1500"]
[Variant "Standard"]
[TimeControl "-"]
[ECO "C00"]
[Opening "Rat Defense: Small Center Defense"]
[Termination "Normal"]
[Annotator ""]

1. e4 { [%eval ] } 1... e6 { [%eval ] } 
2. d4 { [%eval ] } 2... d6?! { (0 → ) Inaccuracy. d5 was best. } { [%eval ] } { C00 Rat Defense: Small Center Defense } (2... d5 3. Nc3 Nf6 4. e5 Nfd7 5. f4 c5 6. Nf3 Nc6 7. Be3) 
3. c4?! { ( → ) Inaccuracy. Bd3 was best. } { [%eval ] } (3. Bd3 c5 4. dxc5 dxc5 5. Nf3 Ne7 6. Qe2 Bd7 7. Nc3 Nec6) 3... h6? { ( → ) Mistake. d5 was best. } { [%eval ] } (3... d5 4. cxd5 exd5 5. exd5 Nf6 6. Nc3 Be7 7. Nf3 O-O 8. Bc4) 
4. Nf3 { [%eval ] } 4... a6 { [%eval ] } 
5. Nc3 { [%eval 1] } 5... g6 { [%eval ] } 6. Be3 { [%eval ] } 6... b6 { [%eval ] } 7. Bd3 { [%eval ] } 7... Bg7 { [%eval ] } 8. Qd2 { [%eval ] } 8... Bb7 { [%eval 7] } 9. O-O { [%eval ] } 9... Ne7 { [%eval ] } 10. d5 { [%eval ] } 10... e5 { [%eval ] } 11. g3 { [%eval ] } 11... Nd7 { [%eval 4] } 12. a3 { [%eval ] } 12... g5 { [%eval ] } 13. h4 { [%eval 3] } 13... f6 { [%eval ] } 14. Rfc1 { [%eval ] } 14... Ng6 { [%eval ] } 15. h5 { [%eval ] } 15... Nf4 { [%eval ] } 16. gxf4 { [%eval ] } 16... gxf4 { [%eval ] } 17. Bxf4?! { ( → ) Inaccuracy. Kh1 was best. } { [%eval ] } (17. Kh1 fxe3 18. fxe3 f5 19. Rg1 Bf6 20. exf5 Bg5 21. Ne4 Nf6 22. Qg2 c6 23. Nxf6+ Qxf6) 17... exf4 { [%eval 1] } 18. Qxf4 { [%eval ] } 18... Ne5 { [%eval ] } 19. Be2?! { (0 → ) Inaccuracy. Rd1 was best. } { [%eval ] } (19. Rd1 Nxf3+ 20. Qxf3 O-O 21. Kf1 Bc8 22. Re1 Qe7 23. Re3 Qf7 24. Rae1 Bd7 25. Ne2 f5) 19... Nxf3+ { [%eval 1] } 20. Bxf3 { [%eval 5] } 20... Bc8 { [%eval ] } 21. Re1 { [%eval ] } 21... O-O { [%eval ] } 22. Kh2 { [%eval ] } 22... Rf7 { [%eval ] } 23. Rg1 { [%eval ] } 23... f5 { [%eval ] } 24. Rg6 { [%eval -3] } 24... Kh8 { [%eval ] } 25. Rxh6+ { [%eval ] } 25... Bxh6 { [%eval ] } 26. Qxh6+ { [%eval ] } 26... Rh7 { [%eval ] } 27. Qf4 { [%eval ] } 27... Qf6 { [%eval ] } 28. Re1 { [%eval -7] } 28... Bd7 { [%eval -] } 29. e5 { [%eval 8] } 29... Qg7 { [%eval -5] } 30. exd6 { [%eval ] } 30... Rg8 { [%eval ] } 31. Qg3 { [%eval -4] } 31... Qf6 { [%eval -] } 32. Qf4 { [%eval ] } 32... cxd6 { [%eval ] } 33. Ne2 { [%eval ] } 33... Qe5 { [%eval -5] } 34. b4?! { (-5 → ) Inaccuracy. Rg1 was best. } { [%eval ] } (34. Rg1 Qxf4+ 35. Nxf4 Rxg1 36. Kxg1 Kg7 37. Kf1 b5 38. cxb5 Bxb5+ 39. Ke1 Kf6 40. Kd2 a5) 34... Be8 { [%eval ] } 35. Qxe5+ { [%eval ] } 35... dxe5 { [%eval ] } 36. Ng3 { [%eval ] } 36... f4 { [%eval ] } 37. Rxe5 { [%eval ] } 37... fxg3+ { [%eval ] } 38. fxg3 { [%eval -] } 38... Rhg7 { [%eval ] } 39. g4 { [%eval ] } 39... Bd7 { [%eval ] } 40. h6 { [%eval ] } 40... Rg6 { [%eval ] } 41. g5?! ...
66. Kc2 { [%eval #-25] } 66... b4 { [%eval #-6] } 
67. Kd2 { [%eval #-9] } 67... b3 { [%eval #-8] } 68. Bd3 { [%eval #-7] } 68... b2 { [%eval #-7] } 69. Ke3 { [%eval #-6] } 69... Ba2 { [%eval #-11] } 70. Kf2 { [%eval #-8] } 70... Bb1 { [%eval #-6] } 71. Bc4 { [%eval #-6] } 71... Bg6 { [%eval #-4] } 72. Bd3 { [%eval #-4] } 72... Bxd3 { [%eval #-3] } 73. Kf3 { [%eval #-3] } 73... b1=Q { [%eval #-2] } 74. Ke3 { [%eval #-2] } 74... Qf1 { [%eval #-1] } 75. Kd2 { [%eval #-1] } 75... Qae1# { Black wins by checkmate. } 0-1
"""

result = rust_pgn_reader_python_binding.parse_game(pgn_moves, store_comments=True)
game = result[0]

print("Moves (UCI):", game.moves_uci())
print("Comments (first 5):", game.comments[:5])
print("Valid:", game.is_valid)
print("Evals (first 10):", game.evals[:10].tolist())
print("Clocks (first 10):", game.clocks[:10].tolist())
print("Outcome:", game.outcome)
print("Is checkmate:", game.is_checkmate)
print("Is game over:", game.is_game_over)
