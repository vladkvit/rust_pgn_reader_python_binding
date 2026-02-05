"""
Ergonomic Python wrappers for rust_pgn_reader_python_binding.ParsedGames.

Usage:
    import rust_pgn_reader_python_binding as pgn
    from python.wrapper import add_ergonomic_methods, GameView

    result = pgn.parse_games_flat(chunked_array)
    add_ergonomic_methods(type(result))

    for game in result:
        print(game.headers)
"""

from __future__ import annotations

import numpy as np
from typing import Iterator, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from rust_pgn_reader_python_binding import ParsedGames


class GameView:
    """
    Zero-copy view into a single game's data within a ParsedGames result.

    Board indexing note: Boards use square indexing (a1=0, h8=63).
    To convert to rank/file array indexing used by some Python code:
        rank = square // 8
        file = square % 8
        # For [7-rank, file] layout: board_2d[7 - rank, file]
    """

    __slots__ = (
        "_data",
        "_idx",
        "_move_start",
        "_move_end",
        "_pos_start",
        "_pos_end",
    )

    def __init__(self, data: "ParsedGames", idx: int):
        self._data = data
        self._idx = idx
        self._move_start = int(data.move_offsets[idx])
        self._move_end = int(data.move_offsets[idx + 1])
        self._pos_start = int(data.position_offsets[idx])
        self._pos_end = int(data.position_offsets[idx + 1])

    def __len__(self) -> int:
        """Number of moves in this game."""
        return self._move_end - self._move_start

    @property
    def num_positions(self) -> int:
        """Number of positions recorded for this game."""
        return self._pos_end - self._pos_start

    # === Board state views ===

    @property
    def boards(self) -> np.ndarray:
        """Board positions, shape (num_positions, 8, 8)."""
        return self._data.boards[self._pos_start : self._pos_end]

    @property
    def initial_board(self) -> np.ndarray:
        """Initial board, shape (8, 8)."""
        return self._data.boards[self._pos_start]

    @property
    def final_board(self) -> np.ndarray:
        """Final board, shape (8, 8)."""
        return self._data.boards[self._pos_end - 1]

    @property
    def castling(self) -> np.ndarray:
        """Castling rights [K,Q,k,q], shape (num_positions, 4)."""
        return self._data.castling[self._pos_start : self._pos_end]

    @property
    def en_passant(self) -> np.ndarray:
        """En passant file (-1 if none), shape (num_positions,)."""
        return self._data.en_passant[self._pos_start : self._pos_end]

    @property
    def halfmove_clock(self) -> np.ndarray:
        """Halfmove clock, shape (num_positions,)."""
        return self._data.halfmove_clock[self._pos_start : self._pos_end]

    @property
    def turn(self) -> np.ndarray:
        """Side to move (True=white), shape (num_positions,)."""
        return self._data.turn[self._pos_start : self._pos_end]

    # === Move views ===

    @property
    def from_squares(self) -> np.ndarray:
        """From squares, shape (num_moves,)."""
        return self._data.from_squares[self._move_start : self._move_end]

    @property
    def to_squares(self) -> np.ndarray:
        """To squares, shape (num_moves,)."""
        return self._data.to_squares[self._move_start : self._move_end]

    @property
    def promotions(self) -> np.ndarray:
        """Promotions (-1=none), shape (num_moves,)."""
        return self._data.promotions[self._move_start : self._move_end]

    @property
    def clocks(self) -> np.ndarray:
        """Clock times in seconds (NaN if missing), shape (num_moves,)."""
        return self._data.clocks[self._move_start : self._move_end]

    @property
    def evals(self) -> np.ndarray:
        """Engine evals (NaN if missing), shape (num_moves,)."""
        return self._data.evals[self._move_start : self._move_end]

    # === Per-game metadata ===

    @property
    def headers(self) -> Dict[str, str]:
        """Raw PGN headers as dict."""
        return self._data.headers[self._idx]

    @property
    def is_checkmate(self) -> bool:
        """Final position is checkmate."""
        return bool(self._data.is_checkmate[self._idx])

    @property
    def is_stalemate(self) -> bool:
        """Final position is stalemate."""
        return bool(self._data.is_stalemate[self._idx])

    @property
    def is_insufficient(self) -> tuple:
        """Insufficient material (white, black)."""
        return (
            bool(self._data.is_insufficient[self._idx, 0]),
            bool(self._data.is_insufficient[self._idx, 1]),
        )

    @property
    def legal_move_count(self) -> int:
        """Legal moves in final position."""
        return int(self._data.legal_move_count[self._idx])

    @property
    def is_valid(self) -> bool:
        """Whether game parsed successfully."""
        return bool(self._data.valid[self._idx])

    # === Convenience methods ===

    def move_uci(self, move_idx: int) -> str:
        """Get UCI string for move at index."""
        files = "abcdefgh"
        ranks = "12345678"
        from_sq = int(self.from_squares[move_idx])
        to_sq = int(self.to_squares[move_idx])
        promo = int(self.promotions[move_idx])

        uci = f"{files[from_sq % 8]}{ranks[from_sq // 8]}{files[to_sq % 8]}{ranks[to_sq // 8]}"
        if promo >= 0:
            promo_chars = {2: "n", 3: "b", 4: "r", 5: "q"}
            uci += promo_chars.get(promo, "")
        return uci

    def moves_uci(self) -> list:
        """Get all moves as UCI strings."""
        return [self.move_uci(i) for i in range(len(self))]

    def __repr__(self) -> str:
        white = self.headers.get("White", "?")
        black = self.headers.get("Black", "?")
        return (
            f"<GameView {white} vs {black}, {len(self)} moves, valid={self.is_valid}>"
        )


class BatchSlice:
    """Lazy iterator over a slice of games."""

    __slots__ = ("_data", "_indices")

    def __init__(self, data: "ParsedGames", indices: range):
        self._data = data
        self._indices = indices

    def __iter__(self) -> Iterator[GameView]:
        for i in self._indices:
            yield GameView(self._data, i)

    def __len__(self) -> int:
        return len(self._indices)

    def __repr__(self) -> str:
        return f"<BatchSlice [{self._indices.start}:{self._indices.stop}:{self._indices.step}], {len(self)} games>"


# === Functions to add ergonomic methods to ParsedGames ===


def _parsed_games_len(self) -> int:
    """Number of games in result."""
    return len(self.move_offsets) - 1


def _parsed_games_getitem(self, idx):
    """Access game(s) by index or slice."""
    n_games = len(self.move_offsets) - 1
    if isinstance(idx, int):
        if idx < 0:
            idx += n_games
        if not 0 <= idx < n_games:
            raise IndexError(f"Game index {idx} out of range [0, {n_games})")
        return GameView(self, idx)
    elif isinstance(idx, slice):
        start, stop, step = idx.indices(n_games)
        return BatchSlice(self, range(start, stop, step))
    raise TypeError(f"Invalid index type: {type(idx)}")


def _parsed_games_iter(self) -> Iterator[GameView]:
    """Iterate over all games."""
    for i in range(len(self.move_offsets) - 1):
        yield GameView(self, i)


def _position_to_game(self, position_indices: np.ndarray) -> np.ndarray:
    """
    Map position indices to game indices.

    Useful after shuffling/sampling positions to look up game metadata.

    Args:
        position_indices: Array of indices into boards array

    Returns:
        Array of game indices (same shape as input)
    """
    return (
        np.searchsorted(self.position_offsets[:-1], position_indices, side="right") - 1
    )


def _move_to_game(self, move_indices: np.ndarray) -> np.ndarray:
    """
    Map move indices to game indices.

    Args:
        move_indices: Array of indices into from_squares, to_squares, etc.

    Returns:
        Array of game indices (same shape as input)
    """
    return np.searchsorted(self.move_offsets[:-1], move_indices, side="right") - 1


@property
def _num_games(self) -> int:
    """Number of games."""
    return len(self.move_offsets) - 1


@property
def _num_moves(self) -> int:
    """Total moves across all games."""
    return int(self.move_offsets[-1])


@property
def _num_positions(self) -> int:
    """Total positions recorded."""
    return int(self.position_offsets[-1])


def add_ergonomic_methods(parsed_games_class):
    """
    Add ergonomic methods to the ParsedGames class.

    Call once after importing the module:
        import rust_pgn_reader_python_binding as pgn
        from python.wrapper import add_ergonomic_methods
        add_ergonomic_methods(pgn.ParsedGames)
    """
    parsed_games_class.__len__ = _parsed_games_len
    parsed_games_class.__getitem__ = _parsed_games_getitem
    parsed_games_class.__iter__ = _parsed_games_iter
    parsed_games_class.position_to_game = _position_to_game
    parsed_games_class.move_to_game = _move_to_game
    parsed_games_class.num_games = _num_games
    parsed_games_class.num_moves = _num_moves
    parsed_games_class.num_positions = _num_positions
