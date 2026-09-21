# PGO

- Requires maturin >= 1.13 and `rustup component add llvm-tools` (for llvm-profdata).
- `maturin develop` ignores PGO — build the wheel explicitly:

  `maturin build --release --pgo`

- This runs a 3-phase build. 1) instrumented wheel 2) trains by running `pgo-command` from pyproject.toml (`python src/bench_parse_games.py`) in a temp venv 3) optimized rebuild.
- Training needs `2013-07-train-00000-of-00001.parquet` in the working dir.
- Optimized wheel lands in `target/wheels/`; install with `pip install --force-reinstall --no-deps target\wheels\rust_pgn_reader_python_binding-*-win_amd64.whl`.
- Measured vs main (median of 5 interleaved runs): MT parse 0.176s → 0.146s (~17% faster), ST parse 1.34s → 1.23s (~8% faster).
