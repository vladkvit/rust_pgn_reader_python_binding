"""
Benchmark for parse_games() — parsing speed and data access patterns.

Usage:
    python bench_data_access.py 2013-07-train-00000-of-00001.parquet
"""

import sys
import time

import numpy as np
import pyarrow.parquet as pq

import rust_pgn_reader_python_binding as pgn


def main():
    if len(sys.argv) < 2:
        print("Usage: python bench_data_access.py <parquet_file>")
        return 1

    file_path = sys.argv[1]
    print(f"Loading from: {file_path}")

    pf = pq.ParquetFile(file_path)
    chunked_array = pf.read(columns=["movetext"]).column("movetext")
    print(f"Loaded {len(chunked_array):,} games")

    # Warmup
    _ = pgn.parse_games(chunked_array)

    # Timed parse
    start = time.perf_counter()
    result = pgn.parse_games(chunked_array)
    elapsed = time.perf_counter() - start

    print(f"\nParsing: {elapsed:.3f}s")
    print(
        f"  {result.num_games:,} games, {result.num_moves:,} moves, {result.num_positions:,} positions"
    )
    print(f"  {result.num_games / elapsed:,.0f} games/sec")
    print(f"  {result.num_chunks} chunks")

    # Data access: chunk-level array access
    start = time.perf_counter()
    for chunk in result.chunks:
        _ = chunk.boards.sum()
        _ = chunk.from_squares.sum()
        _ = chunk.to_squares.sum()
    elapsed = time.perf_counter() - start
    print(f"\nChunk array access: {elapsed:.3f}s")

    # Data access: per-game views
    n_access = min(1000, result.num_games)
    start = time.perf_counter()
    for i in range(n_access):
        _ = result[i].boards
    elapsed = time.perf_counter() - start
    print(
        f"Per-game access ({n_access} games): {elapsed:.3f}s ({elapsed / n_access * 1000:.3f}ms/game)"
    )

    # Position-to-game mapping
    indices = np.random.randint(0, result.num_positions, size=1000, dtype=np.int64)
    start = time.perf_counter()
    _ = result.position_to_game(indices)
    elapsed = time.perf_counter() - start
    print(f"Position-to-game (1000 lookups): {elapsed * 1000:.3f}ms")

    # Memory usage
    total_bytes = 0
    for chunk in result.chunks:
        total_bytes += (
            chunk.boards.nbytes
            + chunk.castling.nbytes
            + chunk.en_passant.nbytes
            + chunk.halfmove_clock.nbytes
            + chunk.turn.nbytes
            + chunk.from_squares.nbytes
            + chunk.to_squares.nbytes
            + chunk.promotions.nbytes
            + chunk.clocks.nbytes
            + chunk.evals.nbytes
            + chunk.move_offsets.nbytes
            + chunk.position_offsets.nbytes
        )
    print(
        f"\nMemory: {total_bytes / 1024 / 1024:.1f} MB ({total_bytes / result.num_positions:.0f} bytes/position)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
