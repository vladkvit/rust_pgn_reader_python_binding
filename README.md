# rust_pgn_reader_python_binding
## Fast PGN parsing bindings for Python
This project adds Python bindings to [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader)

## Benchmarks
Below are some benchmarks on Lichess's 2013-07 chess games (293,459	games).

| Parser                                                                     | File format | Time   |
|----------------------------------------------------------------------------|-------------|--------|
| [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader/tree/master)  | PGN         | 1s     |
| rust_pgn_reader_python_binding                                             | PGN         | 4.7s   |
| rust_pgn_reader_python_binding                                             | parquet     | 4.3s   |
| [chess-library](https://github.com/Disservin/chess-library)                | PGN         | 2s     |
| [python-chess](https://github.com/niklasf/python-chess)                    | PGN         | 3+ min |

## Building
`maturin develop`
`maturin develop --release`

For a more thorough tutorial, follow https://lukesalamone.github.io/posts/how-to-create-rust-python-bindings/