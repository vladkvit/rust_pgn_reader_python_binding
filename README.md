# rust_pgn_reader_python_binding
## Fast PGN parsing bindings for Python
This project adds Python bindings to [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader). In addition, it also parses and extracts [%clk ..] and [%eval ..] tags from comments.

## Installing
`pip install rust_pgn_reader_python_binding`

## API

Three entry points are available:

- `parse_game(pgn)` - Parse a single PGN string
- `parse_games(chunked_array)` - Parse games from a PyArrow ChunkedArray (multithreaded)
- `parse_games_from_strings(pgns)` - Parse a list of PGN strings (multithreaded)

All return a `ParsedGames` container with flat NumPy arrays, supporting:
- Indexing (`result[i]`), slicing (`result[1:3]`), and iteration (`for game in result`)
- Per-game views (`PyGameView`) with zero-copy array slices
- Position-to-game and move-to-game mapping for ML workflows
- Optional comment storage (`store_comments=True`)
- Optional legal move storage (`store_legal_moves=True`)

## Benchmarks
Below are some benchmarks on Lichess's 2013-07 chess games (293,459 games) on a 7800X3D.

| Parser                                                                     | File format | Time   |
|----------------------------------------------------------------------------|-------------|--------|
| [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader/tree/master)  | PGN         | 1s     |
| rust_pgn_reader_python_binding, parse_games (multithreaded)                | parquet     | 0.35s  |
| rust_pgn_reader_python_binding, parse_games_from_strings (multithreaded)   | parquet     | 0.5s   |
| rust_pgn_reader_python_binding, parse_game (single-threaded)               | parquet     | 3.3s   |
| rust_pgn_reader_python_binding, parse_game (single-threaded)               | PGN         | 4.7s   |
| [chess-library](https://github.com/Disservin/chess-library)                | PGN         | 2s     |
| [python-chess](https://github.com/niklasf/python-chess)                    | PGN         | 3+ min |

To replicate, download `2013-07-train-00000-of-00001.parquet` and then run:

`python src/bench_parse_games.py` (recommended — multithreaded parse_games via Arrow)

`python src/bench_parse_games_from_strings.py` (multithreaded parse_games_from_strings)

`python src/bench_parse_game.py` (single-threaded parse_game from parquet)

`python src/bench_parse_game_pgn.py` (single-threaded parse_game from .pgn file)

`python src/bench_data_access.py 2013-07-train-00000-of-00001.parquet` (parsing + data access + memory)

## Building
`maturin develop`

`maturin develop --release`

For a more thorough tutorial, follow https://lukesalamone.github.io/posts/how-to-create-rust-python-bindings/

## Profiling
`py-spy record -s -F -f speedscope --output profile.speedscope -- python ./src/bench_parse_games.py`

Linux/WSL-only:
`py-spy record -s -F -n -f speedscope --output profile.speedscope -- python ./src/bench_parse_games.py`

## Testing
`cargo test`

`python -m unittest src/test.py`
