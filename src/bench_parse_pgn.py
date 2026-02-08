import rust_pgn_reader_python_binding
from datetime import datetime


def split_pgn(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    games = []
    current_game = []
    in_movetext = False

    for line in content.splitlines(keepends=True):
        stripped = line.strip()

        if stripped == "":
            if in_movetext:
                games.append("".join(current_game))
                current_game = []
                in_movetext = False
            else:
                current_game.append(line)
        elif stripped.startswith("["):
            current_game.append(line)
        else:
            in_movetext = True
            current_game.append(line)

    if current_game:
        games.append("".join(current_game))

    return games


file_path = "lichess_db_standard_rated_2013-07.pgn"

start = datetime.now()

a = datetime.now()
games = split_pgn(file_path)
b = datetime.now()
print(f"File read & split: {b - a} ({len(games)} games)")

a = datetime.now()
result = rust_pgn_reader_python_binding.parse_games_from_strings(games)
b = datetime.now()
print(f"Parse:             {b - a}")

end = datetime.now()
print(f"Total:             {end - start}")
