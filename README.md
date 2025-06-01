# rust_pgn_reader_python_binding
## Fast PGN parsing bindings for Python
This project adds Python bindings to [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader). In addition, it also parses and extracts [%clk ..] and [%eval ..] tags from comments.

## Benchmarks
Below are some benchmarks on Lichess's 2013-07 chess games (293,459	games).

| Parser                                                                     | File format | Time   |
|----------------------------------------------------------------------------|-------------|--------|
| [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader/tree/master)  | PGN         | 1s     |
| rust_pgn_reader_python_binding                                             | PGN         | 4.7s   |
| rust_pgn_reader_python_binding, parse_game (single_threaded)               | parquet     | 3.3s   |
| rust_pgn_reader_python_binding, parse_games (multithreaded)                | parquet     | 0.5s   |
| [chess-library](https://github.com/Disservin/chess-library)                | PGN         | 2s     |
| [python-chess](https://github.com/niklasf/python-chess)                    | PGN         | 3+ min |

The main reason for rust_pgn_reader_python_binding being slower (at least in single threaded mode) than rust-pgn-reader is because of vector / string allocations (that show up as malloc's and free's in profiling).

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