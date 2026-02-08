"""
Benchmark for parse_games() PGN parsing.

This benchmark measures:
1. Parsing speed (games/second)
2. Memory efficiency (bytes per position)
3. Data access patterns for ML workloads

Usage:
    python bench_parse_games.py [parquet_file]

If no parquet file is provided, synthetic PGN data will be generated.
"""

import sys
import time
import argparse
from typing import Optional

import numpy as np
import pyarrow as pa

import rust_pgn_reader_python_binding as pgn


def generate_synthetic_pgns(num_games: int, moves_per_game: int = 40) -> list[str]:
    """Generate synthetic PGN games for benchmarking."""
    move_pairs = [
        ("e4", "e5"),
        ("Nf3", "Nc6"),
        ("Bb5", "a6"),
        ("Ba4", "Nf6"),
        ("O-O", "Be7"),
        ("Re1", "b5"),
        ("Bb3", "d6"),
        ("c3", "O-O"),
        ("h3", "Nb8"),
        ("d4", "Nbd7"),
        ("Nbd2", "Bb7"),
        ("Bc2", "Re8"),
        ("Nf1", "Bf8"),
        ("Ng3", "g6"),
        ("Bg5", "h6"),
        ("Bd2", "Bg7"),
        ("a4", "c5"),
        ("d5", "c4"),
        ("b4", "Nc5"),
        ("Be3", "Qc7"),
    ]

    pgns = []
    for i in range(num_games):
        moves = []
        num_pairs = min(moves_per_game // 2, len(move_pairs))
        for j in range(num_pairs):
            white_move, black_move = move_pairs[j]
            moves.append(f"{j + 1}. {white_move} {black_move}")

        movetext = " ".join(moves)
        result = ["1-0", "0-1", "1/2-1/2"][i % 3]

        pgn_str = f"""[Event "Synthetic Game {i}"]
[White "Player{i * 2}"]
[Black "Player{i * 2 + 1}"]
[Result "{result}"]

{movetext} {result}"""
        pgns.append(pgn_str)

    return pgns


def load_parquet_pgns(file_path: str, limit: Optional[int] = None) -> pa.ChunkedArray:
    """Load PGN strings from a parquet file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(file_path)

    table = pf.read()
    for col_name in ["movetext", "pgn", "moves", "game"]:
        if col_name in table.column_names:
            arr = table.column(col_name)
            if limit:
                arr = arr.slice(0, limit)
            return arr

    raise ValueError(
        f"Could not find PGN column. Available columns: {table.column_names}"
    )


def benchmark_parse_games(
    chunked_array: pa.ChunkedArray, num_threads: Optional[int] = None, warmup: int = 1
) -> dict:
    """Benchmark parse_games()."""
    # Warmup
    for _ in range(warmup):
        _ = pgn.parse_games(chunked_array, num_threads=num_threads)

    # Timed run
    start = time.perf_counter()
    result = pgn.parse_games(chunked_array, num_threads=num_threads)
    elapsed = time.perf_counter() - start

    return {
        "method": "parse_games",
        "elapsed_seconds": elapsed,
        "num_games": result.num_games,
        "num_moves": result.num_moves,
        "num_positions": result.num_positions,
        "games_per_second": result.num_games / elapsed,
        "moves_per_second": result.num_moves / elapsed,
        "positions_per_second": result.num_positions / elapsed,
        "valid_games": int(sum(chunk.valid.sum() for chunk in result.chunks)),
        "result": result,
    }


def benchmark_data_access(result) -> dict:
    """Benchmark data access patterns for parse_games result."""
    start = time.perf_counter()

    # Simulate ML data loading: access all boards via chunks
    for chunk in result.chunks:
        _ = chunk.boards.sum()

    # Access moves via chunks
    for chunk in result.chunks:
        _ = chunk.from_squares.sum()
        _ = chunk.to_squares.sum()

    elapsed = time.perf_counter() - start
    return {"access_time": elapsed}


def benchmark_per_game_access(result) -> dict:
    """Benchmark per-game access pattern."""
    start = time.perf_counter()

    for i in range(min(1000, result.num_games)):
        game = result[i]
        _ = game.boards

    elapsed = time.perf_counter() - start
    return {
        "access_time": elapsed,
        "games_accessed": min(1000, result.num_games),
    }


def benchmark_position_to_game_mapping(result) -> dict:
    """Benchmark position-to-game mapping."""
    start = time.perf_counter()

    indices = np.random.randint(0, result.num_positions, size=1000, dtype=np.int64)
    _ = result.position_to_game(indices)

    elapsed = time.perf_counter() - start
    return {
        "access_time": elapsed,
        "positions_accessed": 1000,
    }


def format_number(n: float) -> str:
    """Format large numbers with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return f"{n:.2f}"


def print_results(results: dict, label: str):
    """Print benchmark results."""
    print(f"\n{label}")
    print("-" * 50)
    print(f"  Time:              {results['elapsed_seconds']:.3f}s")
    print(f"  Games:             {results['num_games']:,}")
    print(f"  Moves:             {results['num_moves']:,}")
    print(f"  Positions:         {results['num_positions']:,}")
    print(f"  Valid games:       {results['valid_games']:,}")
    print(f"  Games/sec:         {format_number(results['games_per_second'])}")
    print(f"  Moves/sec:         {format_number(results['moves_per_second'])}")
    print(f"  Positions/sec:     {format_number(results['positions_per_second'])}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark parse_games()")
    parser.add_argument(
        "parquet_file",
        nargs="?",
        help="Path to parquet file with PGN data (optional)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=10000,
        help="Number of synthetic games to generate if no parquet file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of games from parquet file",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads (default: all cores)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--skip-access",
        action="store_true",
        help="Skip data access benchmarks",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("parse_games() Benchmark")
    print("=" * 60)

    # Load or generate data
    if args.parquet_file:
        print(f"\nLoading from: {args.parquet_file}")
        try:
            chunked_array = load_parquet_pgns(args.parquet_file, limit=args.limit)
            print(f"Loaded {len(chunked_array):,} games")
        except Exception as e:
            print(f"Error loading parquet: {e}")
            return 1
    else:
        print(f"\nGenerating {args.num_games:,} synthetic games...")
        pgns = generate_synthetic_pgns(args.num_games)
        chunked_array = pa.chunked_array([pa.array(pgns)])
        print(f"Generated {len(chunked_array):,} games")

    print(f"Threads: {args.threads or 'all cores'}")
    print(f"Warmup iterations: {args.warmup}")

    # Benchmark
    print("\n" + "=" * 60)
    print("PARSING BENCHMARK")
    print("=" * 60)

    results = benchmark_parse_games(
        chunked_array, num_threads=args.threads, warmup=args.warmup
    )
    print_results(results, "parse_games()")

    # Data access benchmarks
    if not args.skip_access:
        print("\n" + "=" * 60)
        print("DATA ACCESS BENCHMARKS")
        print("=" * 60)

        data_access = benchmark_data_access(results["result"])
        print(f"\nArray access time:           {data_access['access_time']:.3f}s")

        per_game_access = benchmark_per_game_access(results["result"])
        print(f"\nPer-game access time:        {per_game_access['access_time']:.3f}s")
        print(f"Games accessed:              {per_game_access['games_accessed']:,}")
        print(
            f"Time per game:               {per_game_access['access_time'] / per_game_access['games_accessed'] * 1000:.3f}ms"
        )

        position_mapping = benchmark_position_to_game_mapping(results["result"])
        print(f"\nPosition-to-game mapping:    {position_mapping['access_time']:.3f}s")
        print(
            f"Positions mapped:            {position_mapping['positions_accessed']:,}"
        )
        print(
            f"Time per lookup:             {position_mapping['access_time'] / position_mapping['positions_accessed'] * 1000000:.3f}us"
        )

    # Memory usage (approximate)
    print("\n" + "=" * 60)
    print("MEMORY USAGE (approximate)")
    print("=" * 60)

    result = results["result"]
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

    print(f"\nArrays total:         {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Bytes per position:   {total_bytes / result.num_positions:.1f}")
    print(f"Bytes per move:       {total_bytes / result.num_moves:.1f}")
    print(f"Number of chunks:     {result.num_chunks}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
