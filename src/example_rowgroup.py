import rust_pgn_reader_python_binding
import pyarrow.parquet as pq

from datetime import datetime


file_path = "2013-07-train-00000-of-00001.parquet"


a = datetime.now()

pf = pq.ParquetFile(file_path)

for i in range(pf.num_row_groups):
    table = pf.read_row_group(i, columns=["movetext"])
    result = rust_pgn_reader_python_binding.parse_games(
        table.column("movetext"), num_threads=4
    )
    print(f"Row group {i}: {result.num_games} games, {result.num_moves} moves")

b = datetime.now()
print(b - a)
