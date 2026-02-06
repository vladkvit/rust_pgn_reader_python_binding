"""
Benchmark comparing parse_games_flat() vs parse_game_moves_arrow_chunked_array().

This benchmark measures:
1. Parsing speed (games/second)
2. Memory efficiency (bytes per position)
3. Data access patterns for ML workloads

Usage:
    python bench_parse_games_flat.py [parquet_file]

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
    # A realistic game template
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
        # Build movetext
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

    # Try common column names for PGN/movetext
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


def benchmark_parse_games_flat(
    chunked_array: pa.ChunkedArray, num_threads: Optional[int] = None, warmup: int = 1
) -> dict:
    """Benchmark parse_games_flat()."""
    # Warmup
    for _ in range(warmup):
        _ = pgn.parse_games_flat(chunked_array, num_threads=num_threads)

    # Timed run
    start = time.perf_counter()
    result = pgn.parse_games_flat(chunked_array, num_threads=num_threads)
    elapsed = time.perf_counter() - start

    return {
        "method": "parse_games_flat",
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


def benchmark_parse_arrow_chunked(
    chunked_array: pa.ChunkedArray, num_threads: Optional[int] = None, warmup: int = 1
) -> dict:
    """Benchmark parse_game_moves_arrow_chunked_array()."""
    # Warmup
    for _ in range(warmup):
        _ = pgn.parse_game_moves_arrow_chunked_array(
            chunked_array, num_threads=num_threads
        )

    # Timed run
    start = time.perf_counter()
    extractors = pgn.parse_game_moves_arrow_chunked_array(
        chunked_array, num_threads=num_threads
    )
    elapsed = time.perf_counter() - start

    num_games = len(extractors)
    num_moves = sum(len(e.moves) for e in extractors)
    num_positions = num_moves + num_games  # Approximate
    valid_games = sum(1 for e in extractors if e.valid_moves)

    return {
        "method": "parse_arrow_chunked",
        "elapsed_seconds": elapsed,
        "num_games": num_games,
        "num_moves": num_moves,
        "num_positions": num_positions,
        "games_per_second": num_games / elapsed,
        "moves_per_second": num_moves / elapsed,
        "positions_per_second": num_positions / elapsed,
        "valid_games": valid_games,
        "result": extractors,
    }


def benchmark_data_access_flat(result) -> dict:
    """Benchmark data access patterns for parse_games_flat result."""
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


def benchmark_data_access_extractors(extractors: list) -> dict:
    """Benchmark data access patterns for list of MoveExtractors."""
    start = time.perf_counter()

    # Simulate accessing all moves (requires iteration)
    total = 0
    for e in extractors:
        for m in e.moves:
            total += m.from_square + m.to_square

    elapsed = time.perf_counter() - start
    return {"access_time": elapsed}


def benchmark_per_game_access_flat(result) -> dict:
    """Benchmark per-game access pattern for parse_games_flat result."""
    start = time.perf_counter()

    # Per-game access pattern (iterate and access boards)
    for i in range(min(1000, result.num_games)):
        game = result[i]
        _ = game.boards

    elapsed = time.perf_counter() - start
    return {
        "access_time": elapsed,
        "games_accessed": min(1000, result.num_games),
    }


def benchmark_position_to_game_mapping(result) -> dict:
    """Benchmark position-to-game mapping for parse_games_flat result."""
    start = time.perf_counter()

    # Position-to-game mapping (still works globally)
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
    parser = argparse.ArgumentParser(
        description="Benchmark parse_games_flat() vs parse_game_moves_arrow_chunked_array()"
    )
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
    print("parse_games_flat() Benchmark")
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

    # Benchmark parse_games_flat
    print("\n" + "=" * 60)
    print("PARSING BENCHMARKS")
    print("=" * 60)

    flat_results = benchmark_parse_games_flat(
        chunked_array, num_threads=args.threads, warmup=args.warmup
    )
    print_results(flat_results, "parse_games_flat()")

    # Benchmark parse_game_moves_arrow_chunked_array
    arrow_results = benchmark_parse_arrow_chunked(
        chunked_array, num_threads=args.threads, warmup=args.warmup
    )
    print_results(arrow_results, "parse_game_moves_arrow_chunked_array()")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    speedup = arrow_results["elapsed_seconds"] / flat_results["elapsed_seconds"]
    print(
        f"\nparse_games_flat() is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
    )

    # Data access benchmarks
    if not args.skip_access:
        print("\n" + "=" * 60)
        print("DATA ACCESS BENCHMARKS")
        print("=" * 60)

        flat_access = benchmark_data_access_flat(flat_results["result"])
        print(f"\nFlat arrays access time:     {flat_access['access_time']:.3f}s")

        extractor_access = benchmark_data_access_extractors(arrow_results["result"])
        print(f"Extractor list access time:  {extractor_access['access_time']:.3f}s")

        access_speedup = extractor_access["access_time"] / flat_access["access_time"]
        print(f"\nFlat arrays are {access_speedup:.2f}x faster for data access")

        # Per-game access benchmarks
        print("\n" + "=" * 60)
        print("PER-GAME ACCESS BENCHMARKS")
        print("=" * 60)

        per_game_access = benchmark_per_game_access_flat(flat_results["result"])
        print(f"\nPer-game access time:        {per_game_access['access_time']:.3f}s")
        print(f"Games accessed:              {per_game_access['games_accessed']:,}")
        print(
            f"Time per game:               {per_game_access['access_time'] / per_game_access['games_accessed'] * 1000:.3f}ms"
        )

        position_mapping = benchmark_position_to_game_mapping(flat_results["result"])
        print(f"\nPosition-to-game mapping:    {position_mapping['access_time']:.3f}s")
        print(
            f"Positions mapped:            {position_mapping['positions_accessed']:,}"
        )
        print(
            f"Time per lookup:             {position_mapping['access_time'] / position_mapping['positions_accessed'] * 1000000:.3f}μs"
        )

    # Memory usage (approximate)
    print("\n" + "=" * 60)
    print("MEMORY USAGE (approximate)")
    print("=" * 60)

    flat_result = flat_results["result"]
    flat_bytes = 0
    for chunk in flat_result.chunks:
        flat_bytes += (
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

    print(f"\nFlat arrays total:    {flat_bytes / 1024 / 1024:.2f} MB")
    print(f"Bytes per position:   {flat_bytes / flat_result.num_positions:.1f}")
    print(f"Bytes per move:       {flat_bytes / flat_result.num_moves:.1f}")
    print(f"Number of chunks:     {flat_result.num_chunks}")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
