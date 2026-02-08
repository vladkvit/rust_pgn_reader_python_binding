"""
Example demonstrating the parse_games() API for ML-optimized PGN parsing.

This API returns flat NumPy arrays suitable for efficient batch processing
in machine learning pipelines.
"""

import numpy as np
import pyarrow as pa

import rust_pgn_reader_python_binding as pgn


# Sample PGN games with annotations
sample_pgns = [
    # Game 1: Sicilian Defense with eval/clock annotations
    """[Event "Online Blitz"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1850"]
[BlackElo "1790"]

1. e4 { [%eval 0.17] [%clk 0:03:00] } 1... c5 { [%eval 0.19] [%clk 0:03:00] }
2. Nf3 { [%eval 0.25] [%clk 0:02:55] } 2... d6 { [%eval 0.30] [%clk 0:02:58] }
3. d4 { [%eval 0.35] [%clk 0:02:50] } 3... cxd4 { [%eval 0.32] [%clk 0:02:55] }
4. Nxd4 { [%eval 0.28] [%clk 0:02:48] } 4... Nf6 { [%eval 0.25] [%clk 0:02:52] }
5. Nc3 { [%eval 0.30] [%clk 0:02:45] } 1-0""",
    # Game 2: Italian Game
    """[Event "Club Championship"]
[White "Magnus"]
[Black "Hikaru"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 0-1""",
    # Game 3: Scholar's Mate (checkmate)
    """[Event "Beginner Game"]
[White "NewPlayer"]
[Black "Victim"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0""",
    # Game 4: Queen's Gambit
    """[Event "Tournament"]
[White "Carlsen"]
[Black "Caruana"]
[Result "1/2-1/2"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 1/2-1/2""",
]


def main():
    print("=" * 60)
    print("parse_games() API Example")
    print("=" * 60)

    # Create PyArrow chunked array from PGN strings
    chunked_array = pa.chunked_array([pa.array(sample_pgns)])

    # Parse all games at once - returns flat NumPy arrays
    result = pgn.parse_games(chunked_array)

    # === Basic Statistics ===
    print(f"\n--- Basic Statistics ---")
    print(f"Number of games:     {result.num_games}")
    print(f"Total moves:         {result.num_moves}")
    print(f"Total positions:     {result.num_positions}")
    print(f"Number of chunks:    {result.num_chunks}")

    # === Chunk Details (escape hatch for raw array access) ===
    print(f"\n--- Chunk Details ---")
    for i, chunk in enumerate(result.chunks):
        print(
            f"  Chunk {i}: {chunk.num_games} games, "
            f"{chunk.num_moves} moves, "
            f"{chunk.num_positions} positions"
        )
        print(f"    boards:       {chunk.boards.shape} ({chunk.boards.dtype})")
        print(
            f"    from_squares: {chunk.from_squares.shape} ({chunk.from_squares.dtype})"
        )

    # === Iterate Over Games ===
    print(f"\n--- Game Details ---")
    for i, game in enumerate(result):
        print(
            f"\nGame {i + 1}: {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')}"
        )
        print(f"  Event:      {game.headers.get('Event', 'N/A')}")
        print(f"  Moves:      {len(game)}")
        print(f"  Positions:  {game.num_positions}")
        print(f"  Valid:      {game.is_valid}")
        print(f"  Checkmate:  {game.is_checkmate}")
        print(f"  Outcome:    {game.outcome}")
        print(
            f"  UCI moves:  {' '.join(game.moves_uci()[:10])}{'...' if len(game) > 10 else ''}"
        )

        # Show eval annotations if available
        valid_evals = game.evals[~np.isnan(game.evals)]
        if len(valid_evals) > 0:
            print(f"  Evals:      {valid_evals[:5].tolist()}...")

    # === Direct Array Access for ML ===
    print(f"\n--- ML-Ready Data Access ---")

    # Access boards via chunks (no single merged array)
    # To concatenate all boards: np.concatenate([c.boards for c in result.chunks])
    chunk0_boards = result.chunks[0].boards
    print(f"Chunk 0 boards: {chunk0_boards.shape}")

    # Get initial position of first game
    game0 = result[0]
    initial_board = game0.initial_board
    print(f"\nInitial board (Game 1):")
    # Print board with piece symbols
    piece_chars = " PNBRQKpnbrqk"
    for rank in range(7, -1, -1):  # Print from rank 8 to rank 1
        row = ""
        for file in range(8):
            sq_idx = rank * 8 + file
            piece = initial_board.flat[sq_idx]
            row += piece_chars[piece] + " "
        print(f"  {rank + 1} | {row}")
    print(f"    +----------------")
    print(f"      a b c d e f g h")

    # === Position-to-Game Mapping ===
    print(f"\n--- Position-to-Game Mapping ---")
    # Useful for shuffling positions while keeping track of game metadata
    sample_positions = np.array([0, 5, 10, 15, 20], dtype=np.int64)
    game_indices = result.position_to_game(sample_positions)
    print(f"Position indices: {sample_positions}")
    print(f"Game indices:     {game_indices}")

    # === Slicing ===
    print(f"\n--- Slicing ---")
    # Get games 1-2 (0-indexed)
    subset = result[1:3]
    print(f"Slice [1:3]: {len(subset)} games")
    for game in subset:
        print(
            f"  - {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')}"
        )

    # === Negative Indexing ===
    print(f"\n--- Negative Indexing ---")
    last_game = result[-1]
    print(
        f"Last game: {last_game.headers.get('White', '?')} vs {last_game.headers.get('Black', '?')}"
    )
    print(f"  Moves: {last_game.moves_uci()}")

    # === Checkmate Detection ===
    print(f"\n--- Checkmate Detection ---")
    for i, game in enumerate(result):
        if game.is_checkmate:
            print(f"Game {i + 1} ended in checkmate!")
            print(f"  Final position legal moves: {game.legal_move_count}")
            print(f"  Is game over: {game.is_game_over}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
