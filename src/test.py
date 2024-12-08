import my_own_parser

pgn_moves = "1. e4 {asdf} e5 2. Nf3 Nc6 3. Bb5 Nf6 4. O-O {hello} Bc5 5. d3 d6 6. h3 h6 7. c3 O-O 8. Be3 a6 9. Ba4 Bd7 10. Bxc5 dxc5 11. Bxc6 Bxc6 12. Nxe5 Bb5 13. Re1 Re8 14. c4 Rxe5 15. cxb5 axb5 16. Nc3 b4 17. Nd5 Nxd5 18. exd5 Rxe1+ 19. Qxe1 Qxd5 20. Qe3 b6 21. b3 c6 22. Qe2 b5 23. Rd1 Qd4 24. Qe7 Rxa2 25. Qe8+ Kh7 26. Qxf7 c4 27. Qf5+ Kh8 28. Qf8+ Kh7 29. Qf5+ Kh8 30. Qf8+ Kh7 31. Qf5+ Kh8 32. Qf8+ Kh7 1/2-1/2"
extractor = my_own_parser.parse_moves(pgn_moves)

print(extractor.moves)
print(extractor.comments)
print(extractor.valid_moves)


pgn_full = """
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
extractor = my_own_parser.parse_moves(pgn_full)
print(extractor.moves)
print(extractor.comments)
print(extractor.valid_moves)
