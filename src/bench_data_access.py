"""
Benchmark for parse_games() — parsing speed and data access patterns.
"""

import time

import numpy as np
import pyarrow.parquet as pq

import rust_pgn_reader_python_binding as pgn


def main():
    file_path = "2013-07-train-00000-of-00001.parquet"
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

    # Data access: global flat arrays
    start = time.perf_counter()
    _ = result.boards.sum()
    _ = result.from_squares.sum()
    _ = result.to_squares.sum()
    elapsed = time.perf_counter() - start
    print(f"\nFlat array access: {elapsed:.3f}s")

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
    total_bytes = (
        result.boards.nbytes
        + result.castling.nbytes
        + result.en_passant.nbytes
        + result.halfmove_clock.nbytes
        + result.turn.nbytes
        + result.from_squares.nbytes
        + result.to_squares.nbytes
        + result.promotions.nbytes
        + result.clocks.nbytes
        + result.evals.nbytes
        + result.move_offsets.nbytes
        + result.position_offsets.nbytes
    )
    print(
        f"\nMemory: {total_bytes / 1024 / 1024:.1f} MB ({total_bytes / result.num_positions:.0f} bytes/position)"
    )

    return 0


if __name__ == "__main__":
    main()
