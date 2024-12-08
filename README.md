# 
## Fast PGN parsing bindings for Python
This project adds Python bindings to [rust-pgn-reader](https://github.com/niklasf/rust-pgn-reader)

For Lichess's 2013-07 PGN file, it takes about 4.7 seconds to parse.
For Lichess's 2013-07 parquet file from Huggingface, it takes about 4.3 seconds to parse
The native rust-pgn-reader runs through the file in about 1 second.

python-chess takes about 3 minutes
The C++ [chess-library](https://github.com/Disservin/chess-library) takes about 2 seconds

## Building
`maturin develop`
`maturin develop --release`