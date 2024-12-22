import rust_pgn_reader_python_binding
import pyarrow.parquet as pq

from datetime import datetime


file_path = "2013-07-train-00000-of-00001.parquet"


pf = pq.ParquetFile(file_path)
pylist = pf.read(columns=["movetext"]).column("movetext").to_pylist()

a = datetime.now()

for row in pylist:
    extractor = rust_pgn_reader_python_binding.parse_game(row)
    moves = extractor.moves
    comments = extractor.comments

b = datetime.now()
print(b - a)
