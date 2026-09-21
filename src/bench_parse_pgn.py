import rust_pgn_reader_python_binding
from time import perf_counter


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

start = perf_counter()

a = perf_counter()
games = split_pgn(file_path)
b = perf_counter()
print(f"File read & split: {b - a:.4f} ({len(games)} games)")

a = perf_counter()
result = rust_pgn_reader_python_binding.parse_games_from_strings(games)
b = perf_counter()
print(f"Parse:             {b - a:.4f}")

end = perf_counter()
print(f"Total:             {end - start:.4f}")
