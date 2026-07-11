"""Shared logic for dumping/comparing reference outputs of the PGN parser.

Used by dump_reference.py (run against the OLD build to create a reference
file) and compare_reference.py (run against the NEW build to verify parity).

Works with both the chunked API (<= 4.x: result.chunks / PyChunkView) and the
flat API (>= 5.0: global arrays directly on ParsedGames), so the exact same
digest computation runs against both builds.

Comparison semantics for invalid games (parse_errors is not None):
- 5.0 allocates per-game array space from pass-1 token counts and zero-fills
  the unwritten tail, so whole-array digests would differ by design.
- Therefore digests cover valid games only, and invalid games are recorded
  individually with digests of their actually-parsed prefix slices.
"""

import hashlib
import json

import numpy as np


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_array(arr) -> str:
    a = np.ascontiguousarray(arr)
    return _sha(a.tobytes())


def digest_json(obj) -> str:
    return _sha(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


class Adapted:
    """Uniform view over old (chunked) and new (flat) ParsedGames results."""

    def __init__(self, result):
        self.legal_moves_stored = False
        if hasattr(result, "chunks"):
            self._from_chunked(result)
        else:
            self._from_flat(result)
        self.num_games = len(self.valid)
        # Positions per game are always parsed moves + 1 (initial position is
        # recorded even for games that error out immediately).
        self.parsed_position_counts = self.parsed_move_counts + 1

    def _from_chunked(self, result):
        chunks = result.chunks

        def cat(name, axis=0):
            return np.concatenate([np.asarray(getattr(c, name)) for c in chunks], axis=axis)

        def cat_list(name):
            out = []
            for c in chunks:
                out.extend(getattr(c, name))
            return out

        for name in (
            "boards",
            "castling",
            "en_passant",
            "halfmove_clock",
            "turn",
            "from_squares",
            "to_squares",
            "promotions",
            "clocks",
            "evals",
            "is_checkmate",
            "is_stalemate",
            "is_insufficient",
            "legal_move_count",
            "valid",
        ):
            setattr(self, name, cat(name))

        self.headers = cat_list("headers")
        self.outcome = cat_list("outcome")
        self.parse_errors = cat_list("parse_errors")
        self.comments = cat_list("comments")

        self.move_counts = np.concatenate(
            [np.diff(np.asarray(c.move_offsets, dtype=np.int64)) for c in chunks]
        )
        self.position_counts = np.concatenate(
            [np.diff(np.asarray(c.position_offsets, dtype=np.int64)) for c in chunks]
        )
        # Old builds only record actually-parsed moves: allocated == parsed.
        self.parsed_move_counts = self.move_counts

        if any(len(np.asarray(c.legal_move_offsets)) > 0 for c in chunks):
            self.legal_moves_stored = True
            self.legal_move_from_squares = cat("legal_move_from_squares")
            self.legal_move_to_squares = cat("legal_move_to_squares")
            self.legal_move_promotions = cat("legal_move_promotions")
            self.legal_move_counts = np.concatenate(
                [np.diff(np.asarray(c.legal_move_offsets, dtype=np.int64)) for c in chunks]
            )

    def _from_flat(self, result):
        for name in (
            "boards",
            "castling",
            "en_passant",
            "halfmove_clock",
            "turn",
            "from_squares",
            "to_squares",
            "promotions",
            "clocks",
            "evals",
            "is_checkmate",
            "is_stalemate",
            "is_insufficient",
            "legal_move_count",
            "valid",
        ):
            setattr(self, name, np.asarray(getattr(result, name)))

        self.headers = result.headers
        self.outcome = result.outcome
        self.parse_errors = result.parse_errors
        self.comments = result.comments

        self.move_counts = np.diff(np.asarray(result.move_offsets, dtype=np.int64))
        self.position_counts = np.diff(np.asarray(result.position_offsets, dtype=np.int64))
        self.parsed_move_counts = np.asarray(result.parsed_move_counts, dtype=np.int64)

        legal_offsets = np.asarray(result.legal_move_offsets)
        if len(legal_offsets) > 0:
            self.legal_moves_stored = True
            self.legal_move_from_squares = np.asarray(result.legal_move_from_squares)
            self.legal_move_to_squares = np.asarray(result.legal_move_to_squares)
            self.legal_move_promotions = np.asarray(result.legal_move_promotions)
            self.legal_move_counts = np.diff(legal_offsets.astype(np.int64))


def _masked_digest(arr, mask):
    if mask.all():
        return digest_array(arr)
    return digest_array(arr[mask])


def summarize(result, include_optional=False) -> dict:
    """Compute a comparable summary dict for a ParsedGames result."""
    a = Adapted(result)
    n = a.num_games
    valid = np.asarray(a.valid, dtype=bool)

    pos_off = np.concatenate([[0], np.cumsum(a.position_counts)]).astype(np.int64)
    move_off = np.concatenate([[0], np.cumsum(a.move_counts)]).astype(np.int64)

    game_of_pos = np.repeat(np.arange(n), a.position_counts)
    game_of_move = np.repeat(np.arange(n), a.move_counts)
    pos_valid = valid[game_of_pos]
    move_valid = valid[game_of_move]

    digests = {
        # Per-position arrays (valid games only)
        "boards": _masked_digest(a.boards, pos_valid),
        "castling": _masked_digest(a.castling, pos_valid),
        "en_passant": _masked_digest(a.en_passant, pos_valid),
        "halfmove_clock": _masked_digest(a.halfmove_clock, pos_valid),
        "turn": _masked_digest(a.turn, pos_valid),
        # Per-move arrays (valid games only)
        "from_squares": _masked_digest(a.from_squares, move_valid),
        "to_squares": _masked_digest(a.to_squares, move_valid),
        "promotions": _masked_digest(a.promotions, move_valid),
        "clocks": _masked_digest(a.clocks, move_valid),
        "evals": _masked_digest(a.evals, move_valid),
        # Per-game arrays (all games; identical for old/new by design)
        "is_checkmate": digest_array(a.is_checkmate),
        "is_stalemate": digest_array(a.is_stalemate),
        "is_insufficient": digest_array(a.is_insufficient),
        "legal_move_count": digest_array(a.legal_move_count),
        "valid": digest_array(valid),
        "parsed_move_counts_valid": digest_array(a.parsed_move_counts[valid]),
        # Per-game metadata (all games)
        "headers": digest_json(a.headers),
        "outcome": digest_json(a.outcome),
        "parse_errors": digest_json(a.parse_errors),
    }

    if include_optional:
        move_valid_list = move_valid.tolist()
        digests["comments"] = digest_json(
            [c for c, ok in zip(a.comments, move_valid_list) if ok]
        )
        if a.legal_moves_stored:
            entry_valid = np.repeat(pos_valid, a.legal_move_counts)
            digests["legal_move_counts"] = _masked_digest(a.legal_move_counts, pos_valid)
            digests["legal_move_from_squares"] = _masked_digest(
                a.legal_move_from_squares, entry_valid
            )
            digests["legal_move_to_squares"] = _masked_digest(
                a.legal_move_to_squares, entry_valid
            )
            digests["legal_move_promotions"] = _masked_digest(
                a.legal_move_promotions, entry_valid
            )

    invalid_games = []
    for idx in np.flatnonzero(~valid):
        idx = int(idx)
        pm = int(a.parsed_move_counts[idx])
        pp = pm + 1
        ps, ms = int(pos_off[idx]), int(move_off[idx])
        invalid_games.append(
            {
                "index": idx,
                "parse_error": a.parse_errors[idx],
                "outcome": a.outcome[idx],
                "parsed_moves": pm,
                "digests": {
                    "boards": digest_array(a.boards[ps : ps + pp]),
                    "castling": digest_array(a.castling[ps : ps + pp]),
                    "en_passant": digest_array(a.en_passant[ps : ps + pp]),
                    "halfmove_clock": digest_array(a.halfmove_clock[ps : ps + pp]),
                    "turn": digest_array(a.turn[ps : ps + pp]),
                    "from_squares": digest_array(a.from_squares[ms : ms + pm]),
                    "to_squares": digest_array(a.to_squares[ms : ms + pm]),
                    "promotions": digest_array(a.promotions[ms : ms + pm]),
                    "clocks": digest_array(a.clocks[ms : ms + pm]),
                    "evals": digest_array(a.evals[ms : ms + pm]),
                },
            }
        )

    return {
        "num_games": int(n),
        "num_invalid": int((~valid).sum()),
        "valid_moves_total": int(a.parsed_move_counts[valid].sum()),
        "valid_positions_total": int(a.parsed_position_counts[valid].sum()),
        "digests": digests,
        "invalid_games": invalid_games,
    }


def build_reference(parquet_path: str, subset_n: int = 2000) -> dict:
    import pyarrow.parquet as pq
    import rust_pgn_reader_python_binding as rp

    col = pq.ParquetFile(parquet_path).read(columns=["movetext"]).column("movetext")

    result = rp.parse_games(col)
    main = summarize(result)
    del result

    subset_col = col.slice(0, subset_n)
    subset_result = rp.parse_games(
        subset_col, store_comments=True, store_legal_moves=True
    )
    subset = summarize(subset_result, include_optional=True)
    del subset_result

    return {
        "parquet": parquet_path,
        "subset_n": subset_n,
        "main": main,
        "subset": subset,
    }


def diff_dicts(ref: dict, new: dict, path: str = "") -> list:
    """Recursively diff two summary dicts, returning mismatch descriptions."""
    mismatches = []
    if isinstance(ref, dict) and isinstance(new, dict):
        for key in sorted(set(ref) | set(new)):
            if key not in ref:
                mismatches.append(f"{path}.{key}: missing in reference")
            elif key not in new:
                mismatches.append(f"{path}.{key}: missing in new output")
            else:
                mismatches.extend(diff_dicts(ref[key], new[key], f"{path}.{key}"))
    elif isinstance(ref, list) and isinstance(new, list):
        if len(ref) != len(new):
            mismatches.append(f"{path}: length {len(ref)} != {len(new)}")
        else:
            for i, (r, n) in enumerate(zip(ref, new)):
                mismatches.extend(diff_dicts(r, n, f"{path}[{i}]"))
    elif ref != new:
        mismatches.append(f"{path}: {ref!r} != {new!r}")
    return mismatches
