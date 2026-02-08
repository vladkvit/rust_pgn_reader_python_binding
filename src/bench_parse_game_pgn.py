import rust_pgn_reader_python_binding
from datetime import datetime


def split_pgn(file_path):
    """Generator to split a PGN file into individual game strings."""
    with open(file_path, "r", encoding="utf-8") as file:
        game_lines = []
        for line in file:
            if line.strip() == "" and game_lines:
                yield "".join(game_lines)
                game_lines = []
            else:
                game_lines.append(line)
        if game_lines:
            yield "".join(game_lines)


file_path = "lichess_db_standard_rated_2013-07.pgn"

a = datetime.now()
for game_pgn in split_pgn(file_path):
    result = rust_pgn_reader_python_binding.parse_game(game_pgn)
    game = result[0]
    moves = game.moves_uci()

b = datetime.now()
print(b - a)
