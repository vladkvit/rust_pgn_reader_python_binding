import queue
import threading
from time import perf_counter

import pyarrow as pa
import pyarrow.parquet as pq
import rust_pgn_reader_python_binding

file_path = "2013-07-train-00000-of-00001.parquet"
batch_size = 16384

q = queue.Queue(maxsize=2)


def produce():
    for batch in pq.ParquetFile(file_path).iter_batches(batch_size, columns=["movetext"]):
        q.put(batch)
    q.put(None)


a = perf_counter()
threading.Thread(target=produce, daemon=True).start()
while (batch := q.get()) is not None:
    rust_pgn_reader_python_binding.parse_games(pa.chunked_array([batch.column("movetext")]))
b = perf_counter()
print(f"interleaved read+parse: {b - a:.4f}")
