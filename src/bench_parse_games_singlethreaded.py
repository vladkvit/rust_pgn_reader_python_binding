import rust_pgn_reader_python_binding
import pyarrow.parquet as pq
from time import perf_counter

file_path = "2013-07-train-00000-of-00001.parquet"

a = perf_counter()
pf = pq.ParquetFile(file_path)
movetext_arrow_array = pf.read(columns=["movetext"]).column("movetext")

b = perf_counter()
result = rust_pgn_reader_python_binding.parse_games(movetext_arrow_array, num_threads=1)

c = perf_counter()
print(f"read: {b - a:.4f}")
print(f"parse: {c - b:.4f}")
