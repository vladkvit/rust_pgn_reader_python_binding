from typing import List, Optional, Tuple, Dict, Iterator, Union, overload
import pyarrow
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

class PyGameView:
    """Zero-copy view into a single game's data within a ParsedGames result.

    Board indexing note: Boards use square indexing (a1=0, h8=63).
    To convert to rank/file:
        rank = square // 8
        file = square % 8
    """

    def __len__(self) -> int:
        """Number of moves in this game."""
        ...

    @property
    def num_positions(self) -> int:
        """Number of positions recorded for this game."""
        ...

    # === Board state views ===

    @property
    def boards(self) -> NDArray[np.uint8]:
        """Board positions, shape (num_positions, 8, 8)."""
        ...

    @property
    def initial_board(self) -> NDArray[np.uint8]:
        """Initial board position, shape (8, 8)."""
        ...

    @property
    def final_board(self) -> NDArray[np.uint8]:
        """Final board position, shape (8, 8)."""
        ...

    @property
    def castling(self) -> NDArray[np.bool_]:
        """Castling rights [K,Q,k,q], shape (num_positions, 4)."""
        ...

    @property
    def en_passant(self) -> NDArray[np.int8]:
        """En passant file (-1 if none), shape (num_positions,)."""
        ...

    @property
    def halfmove_clock(self) -> NDArray[np.uint8]:
        """Halfmove clock, shape (num_positions,)."""
        ...

    @property
    def turn(self) -> NDArray[np.bool_]:
        """Side to move (True=white), shape (num_positions,)."""
        ...

    # === Move views ===

    @property
    def from_squares(self) -> NDArray[np.uint8]:
        """From squares, shape (num_moves,)."""
        ...

    @property
    def to_squares(self) -> NDArray[np.uint8]:
        """To squares, shape (num_moves,)."""
        ...

    @property
    def promotions(self) -> NDArray[np.int8]:
        """Promotions (-1=none, 2=N, 3=B, 4=R, 5=Q), shape (num_moves,)."""
        ...

    @property
    def clocks(self) -> NDArray[np.float32]:
        """Clock times in seconds (NaN if missing), shape (num_moves,)."""
        ...

    @property
    def evals(self) -> NDArray[np.float32]:
        """Engine evals (NaN if missing), shape (num_moves,)."""
        ...

    # === Per-game metadata ===

    @property
    def headers(self) -> Dict[str, str]:
        """Raw PGN headers as dict."""
        ...

    @property
    def outcome(self) -> Optional[str]:
        """Game outcome from movetext: 'White', 'Black', 'Draw', 'Unknown', or None."""
        ...

    @property
    def is_checkmate(self) -> bool:
        """Final position is checkmate."""
        ...

    @property
    def is_stalemate(self) -> bool:
        """Final position is stalemate."""
        ...

    @property
    def is_insufficient(self) -> Tuple[bool, bool]:
        """Insufficient material (white, black)."""
        ...

    @property
    def legal_move_count(self) -> int:
        """Legal move count in final position."""
        ...

    @property
    def is_valid(self) -> bool:
        """Whether game parsed successfully."""
        ...

    @property
    def is_game_over(self) -> bool:
        """Whether the game is over (checkmate, stalemate, or both sides insufficient)."""
        ...

    @property
    def comments(self) -> List[Optional[str]]:
        """Raw text comments per move (only populated when store_comments=True)."""
        ...

    @property
    def legal_moves(self) -> List[List[Tuple[int, int, int]]]:
        """Legal moves at each position (only populated when store_legal_moves=True).
        Each entry is a list of (from_square, to_square, promotion) tuples."""
        ...

    # === Convenience methods ===

    def move_uci(self, move_idx: int) -> str:
        """Get UCI string for move at index."""
        ...

    def moves_uci(self) -> List[str]:
        """Get all moves as UCI strings."""
        ...

    def __repr__(self) -> str: ...

class ParsedGamesIter:
    """Iterator over games in a ParsedGames result."""

    def __iter__(self) -> "ParsedGamesIter": ...
    def __next__(self) -> PyGameView: ...

class PyChunkView:
    """View into a single chunk's raw numpy arrays.

    Access via ``parsed_games.chunks[i]``. Each chunk corresponds to one
    parsing thread's output. Use this for advanced access patterns like
    manual concatenation or custom batching.
    """

    @property
    def num_games(self) -> int: ...
    @property
    def num_moves(self) -> int: ...
    @property
    def num_positions(self) -> int: ...
    @property
    def boards(self) -> NDArray[np.uint8]:
        """Board positions, shape (N_positions, 8, 8), dtype uint8."""
        ...
    @property
    def castling(self) -> NDArray[np.bool_]:
        """Castling rights [K,Q,k,q], shape (N_positions, 4), dtype bool."""
        ...
    @property
    def en_passant(self) -> NDArray[np.int8]: ...
    @property
    def halfmove_clock(self) -> NDArray[np.uint8]: ...
    @property
    def turn(self) -> NDArray[np.bool_]: ...
    @property
    def from_squares(self) -> NDArray[np.uint8]: ...
    @property
    def to_squares(self) -> NDArray[np.uint8]: ...
    @property
    def promotions(self) -> NDArray[np.int8]: ...
    @property
    def clocks(self) -> NDArray[np.float32]: ...
    @property
    def evals(self) -> NDArray[np.float32]: ...
    @property
    def move_offsets(self) -> NDArray[np.uint32]: ...
    @property
    def position_offsets(self) -> NDArray[np.uint32]: ...
    @property
    def is_checkmate(self) -> NDArray[np.bool_]: ...
    @property
    def is_stalemate(self) -> NDArray[np.bool_]: ...
    @property
    def is_insufficient(self) -> NDArray[np.bool_]: ...
    @property
    def legal_move_count(self) -> NDArray[np.uint16]: ...
    @property
    def valid(self) -> NDArray[np.bool_]: ...
    @property
    def headers(self) -> List[Dict[str, str]]: ...
    @property
    def outcome(self) -> List[Optional[str]]: ...
    @property
    def comments(self) -> List[Optional[str]]: ...
    @property
    def legal_move_from_squares(self) -> NDArray[np.uint8]: ...
    @property
    def legal_move_to_squares(self) -> NDArray[np.uint8]: ...
    @property
    def legal_move_promotions(self) -> NDArray[np.int8]: ...
    @property
    def legal_move_offsets(self) -> NDArray[np.uint32]: ...
    def __repr__(self) -> str: ...

class ParsedGames:
    """Chunked container for parsed chess games, optimized for ML training.

    Internally stores data in multiple chunks (one per parsing thread) to
    avoid the cost of merging. Per-game access is O(log(num_chunks)) via
    binary search on precomputed boundaries.

    Board layout:
        Boards use square indexing: a1=0, b1=1, ..., h8=63
        Piece encoding: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
    """

    # === Computed properties ===

    @property
    def num_games(self) -> int:
        """Number of games in the result."""
        ...

    @property
    def num_moves(self) -> int:
        """Total number of moves across all games."""
        ...

    @property
    def num_positions(self) -> int:
        """Total number of board positions recorded."""
        ...

    @property
    def num_chunks(self) -> int:
        """Number of internal chunks."""
        ...

    # === Escape hatch: raw chunk access ===

    @property
    def chunks(self) -> List[PyChunkView]:
        """Access raw per-chunk data.

        Each chunk corresponds to one parsing thread's output. Use this
        for advanced access patterns like manual concatenation.
        """
        ...

    # === Sequence protocol ===

    def __len__(self) -> int:
        """Number of games in the result."""
        ...

    @overload
    def __getitem__(self, idx: int) -> PyGameView: ...
    @overload
    def __getitem__(self, idx: slice) -> List[PyGameView]: ...
    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[PyGameView, List[PyGameView]]:
        """Access game(s) by index or slice."""
        ...

    def __iter__(self) -> ParsedGamesIter:
        """Iterate over all games."""
        ...

    # === Mapping utilities ===

    def position_to_game(self, position_indices: npt.ArrayLike) -> NDArray[np.int64]:
        """Map global position indices to game indices.

        Useful after shuffling/sampling positions to look up game metadata.

        Args:
            position_indices: Array of indices into the global position space.
                Accepts any integer dtype; int64 is optimal (avoids conversion).

        Returns:
            Array of game indices (same shape as input)
        """
        ...

    def move_to_game(self, move_indices: npt.ArrayLike) -> NDArray[np.int64]:
        """Map global move indices to game indices.

        Args:
            move_indices: Array of indices into the global move space.
                Accepts any integer dtype; int64 is optimal (avoids conversion).

        Returns:
            Array of game indices (same shape as input)
        """
        ...

def parse_game(
    pgn: str,
    store_comments: bool = False,
    store_legal_moves: bool = False,
) -> ParsedGames:
    """Parse a single PGN game string.

    Convenience wrapper for parsing a single game. Returns a ParsedGames
    container with one game.

    Args:
        pgn: PGN game string
        store_comments: Whether to store raw text comments (default: False)
        store_legal_moves: Whether to store legal moves at each position (default: False)

    Returns:
        ParsedGames object containing the parsed game
    """
    ...

def parse_games(
    pgn_chunked_array: pyarrow.ChunkedArray,
    num_threads: Optional[int] = None,
    chunk_multiplier: Optional[int] = None,
    store_comments: bool = False,
    store_legal_moves: bool = False,
) -> ParsedGames:
    """Parse chess games from a PyArrow ChunkedArray into flat NumPy arrays.

    This API is optimized for ML training pipelines, returning flat NumPy arrays
    that can be efficiently batched and processed.

    Args:
        pgn_chunked_array: PyArrow ChunkedArray containing PGN strings
        num_threads: Number of threads for parallel parsing (default: all CPUs)
        chunk_multiplier: Multiplier for number of chunks (default: 1)
        store_comments: Whether to store raw text comments (default: False)
        store_legal_moves: Whether to store legal moves at each position (default: False)

    Returns:
        ParsedGames object containing flat arrays and iteration support
    """
    ...

def parse_games_from_strings(
    pgns: List[str],
    num_threads: Optional[int] = None,
    store_comments: bool = False,
    store_legal_moves: bool = False,
) -> ParsedGames:
    """Parse multiple PGN game strings in parallel.

    Convenience wrapper for when you have a list of strings rather than an Arrow array.

    Args:
        pgns: List of PGN game strings
        num_threads: Number of threads for parallel parsing (default: all CPUs)
        store_comments: Whether to store raw text comments (default: False)
        store_legal_moves: Whether to store legal moves at each position (default: False)

    Returns:
        ParsedGames object containing flat arrays and iteration support
    """
    ...
