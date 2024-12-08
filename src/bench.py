import my_own_parser

from datetime import datetime


def split_pgn(file_path):
    """Generator to split a PGN file into individual game strings."""
    with open(file_path, "r", encoding="utf-8") as file:
        game_lines = []
        for line in file:
            if line.strip() == "" and game_lines:  # End of a game
                yield "".join(game_lines)
                game_lines = []  # Reset for the next game
            else:
                game_lines.append(line)
        if game_lines:  # Yield the last game if the file doesn't end with a blank line
            yield "".join(game_lines)


file_path = "lichess_db_standard_rated_2013-07.pgn"

a = datetime.now()
for game_pgn in split_pgn(file_path):
    # print("Game:")
    # print(game_pgn)
    extractor = my_own_parser.parse_moves(game_pgn)

    # print(extractor.moves)
    # print(extractor.comments)
    # print(extractor.valid_moves)

b = datetime.now()
print(b - a)
