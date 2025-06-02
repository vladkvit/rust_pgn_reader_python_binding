import rust_pgn_reader_python_binding
import pyarrow.parquet as pq

from datetime import datetime

file_path = "2013-07-train-00000-of-00001.parquet"


pf = pq.ParquetFile(file_path)

movetext_arrow_array = pf.read(columns=["movetext"]).column("movetext")

a = datetime.now()

extractors = rust_pgn_reader_python_binding.parse_game_moves_arrow_chunked_array(
    movetext_arrow_array
)

b = datetime.now()
print(b - a)
