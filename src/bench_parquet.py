import rust_pgn_reader_python_binding
from fastparquet import ParquetFile

from datetime import datetime


file_path = "2013-07-train-00000-of-00001.parquet"


a = datetime.now()

pf = ParquetFile(file_path)

for df in pf.iter_row_groups():
    for row in df.movetext:
        extractor = rust_pgn_reader_python_binding.parse_moves(row)

b = datetime.now()
print(b - a)
