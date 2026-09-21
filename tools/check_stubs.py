#!/usr/bin/env python3
"""Drift check between the auto-generated PyO3 stub and the hand-maintained one.

Builds with the ``stubgen`` Cargo feature (pyo3 ``experimental-inspect``), lets
``maturin generate-stubs`` emit the stub PyO3 would ship automatically, and
compares its *symbol surface* against the committed, hand-maintained
``rust_pgn_reader_python_binding.pyi``.

Only names are compared (module functions, classes, per-class
methods/properties) — not annotations. The hand-written stub is the source of
truth for typing precision (numpy dtypes, ``__getitem__`` overloads); the
generated stub is the source of truth for *what actually exists at runtime*.
The check fails when the two surfaces diverge, e.g.:

  * generated-only symbol: a Rust method/getter was added (or an existing one
    is missing from the hand-written stub) -> update the hand-written stub.
  * handwritten-only symbol: the stub promises something the extension no
    longer provides -> fix or remove it.

Usage:
    python tools/check_stubs.py                      # generate + compare
    python tools/check_stubs.py --generated PATH     # compare a prebuilt stub

Requires maturin >= 1.15 on PATH (or pass --maturin) and a Rust toolchain
when generating. Exits 0 when in sync, 1 on drift.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDWRITTEN = ROOT / "rust_pgn_reader_python_binding.pyi"


def collect_surface(path: Path) -> dict[str, set[str]]:
    """Symbol surface of a .pyi file: {'<module>': {...}, 'Class': {...}}."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    surface: dict[str, set[str]] = {"<module>": set()}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            surface["<module>"].add(node.name)
        elif isinstance(node, ast.ClassDef):
            surface[node.name] = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return surface


def diff_surfaces(generated: dict, handwritten: dict) -> list[str]:
    problems: list[str] = []

    for scope in sorted(handwritten.keys() - generated.keys()):
        problems.append(f"handwritten-only scope: {scope}")
    for scope in sorted(generated.keys() - handwritten.keys()):
        problems.append(f"generated-only scope: {scope}")

    for scope in sorted(generated.keys() & handwritten.keys()):
        gen, hand = generated[scope], handwritten[scope]
        for name in sorted(gen - hand):
            problems.append(f"{scope}.{name}: missing from hand-written stub")
        for name in sorted(hand - gen):
            problems.append(f"{scope}.{name}: not in generated stub (stale?)")
    return problems


def generate_stub(maturin: str, outdir: Path) -> Path:
    cmd = [
        maturin,
        "generate-stubs",
        "--out",
        str(outdir),
        "--features",
        "stubgen",
        "--find-interpreter",
    ]
    print("+ " + " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True)
    # maturin >= 1.15 writes stubs in package layout: <out>/<module>/__init__.pyi
    return outdir / "rust_pgn_reader_python_binding" / "__init__.pyi"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--generated",
        type=Path,
        help="Use a pre-built generated stub instead of running maturin.",
    )
    parser.add_argument(
        "--handwritten",
        type=Path,
        default=HANDWRITTEN,
        help=f"Path to the hand-maintained stub (default: {HANDWRITTEN})",
    )
    parser.add_argument("--maturin", default="maturin", help="maturin executable.")
    args = parser.parse_args()

    if args.generated is not None:
        generated_path = args.generated
        problems = _compare(generated_path, args.handwritten)
    else:
        with tempfile.TemporaryDirectory(prefix="stubgen-") as tmp:
            generated_path = generate_stub(args.maturin, Path(tmp))
            problems = _compare(generated_path, args.handwritten)

    if problems:
        print("stub drift detected:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("stub surfaces are in sync.")
    return 0


def _compare(generated_path: Path, handwritten_path: Path) -> list[str]:
    generated = collect_surface(generated_path)
    handwritten = collect_surface(handwritten_path)
    return diff_surfaces(generated, handwritten)


if __name__ == "__main__":
    sys.exit(main())
