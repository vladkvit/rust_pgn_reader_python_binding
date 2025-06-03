# rust_pgn_reader_python_binding
## Fast PGN parsing bindings for Python
This project adds Python bindings to [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader). In addition, it also parses and extracts [%clk ..] and [%eval ..] tags from comments.

## Installing
`pip install rust_pgn_reader_python_binding`

## Benchmarks
Below are some benchmarks on Lichess's 2013-07 chess games (293,459	games) on an 7800X3D.

| Parser                                                                     | File format | Time   |
|----------------------------------------------------------------------------|-------------|--------|
| [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader/tree/master)  | PGN         | 1s     |
| rust_pgn_reader_python_binding                                             | PGN         | 4.7s   |
| rust_pgn_reader_python_binding, parse_game (single_threaded)               | parquet     | 3.3s   |
| rust_pgn_reader_python_binding, parse_games (multithreaded)                | parquet     | 0.5s   |
| rust_pgn_reader_python_binding, parse_game_moves_arrow_chunked_array (multithreaded) | parquet     | 0.35s   |
| [chess-library](https://github.com/Disservin/chess-library)                | PGN         | 2s     |
| [python-chess](https://github.com/niklasf/python-chess)                    | PGN         | 3+ min |

To replicate, download `2013-07-train-00000-of-00001.parquet` and then run:

`python bench_parquet.py` (single-threaded parse_game)

`python bench_parquet_parallel.py` (multithreaded parse_games)

`python bench_parquet_arrow.py` (multithreaded parse_games)

## Building
`maturin develop`

`maturin develop --release`

For a more thorough tutorial, follow https://lukesalamone.github.io/posts/how-to-create-rust-python-bindings/

## Profiling
`py-spy record -s -F -f speedscope --output profile.speedscope -- python ./src/bench_parquet.py`

Linux/WSL-only:
`py-spy record -s -F -n -f speedscope --output profile.speedscope -- python ./src/bench_parquet.py`

## Testing
`cargo test`

`python -m unittest src/test.py`