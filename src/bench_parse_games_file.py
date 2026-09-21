import rust_pgn_reader_python_binding
from time import perf_counter

file_path = "2013-07-train-00000-of-00001.parquet"

a = perf_counter()
result = rust_pgn_reader_python_binding.parse_games_from_parquet(file_path)

b = perf_counter()
print(f"read+parse (rust reads file): {b - a:.4f}")
