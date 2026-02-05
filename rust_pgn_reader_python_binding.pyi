from typing import List, Optional, Tuple, Dict, Iterator, Union, overload
import pyarrow
import numpy as np
from numpy.typing import NDArray

class PyUciMove:
    from_square: int
    to_square: int
    promotion: Optional[int]

    def __init__(
        self, from_square: int, to_square: int, promotion: Optional[int]
    ) -> None: ...
    @property
    def get_from_square_name(self) -> str: ...
    @property
    def get_to_square_name(self) -> str: ...
    @property
    def get_promotion_name(self) -> Optional[str]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class PositionStatus:
    is_checkmate: bool
    is_stalemate: bool
    legal_move_count: int
    is_game_over: bool
    insufficient_material: Tuple[bool, bool]
    turn: bool

class MoveExtractor:
    moves: List[PyUciMove]
    valid_moves: bool
    comments: List[Optional[str]]
    evals: List[Optional[float]]
    clock_times: List[Optional[Tuple[int, int, float]]]
    outcome: Optional[str]
    headers: List[Tuple[str, str]]
    castling_rights: List[Optional[Tuple[bool, bool, bool, bool]]]
    position_status: Optional[PositionStatus]

    def __init__(self, store_legal_moves: bool = False) -> None: ...
    def turn(self) -> bool: ...
    def update_position_status(self) -> None: ...
    @property
    def legal_moves(self) -> List[List[PyUciMove]]: ...

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

class ParsedGames:
    """Flat array container for parsed chess games, optimized for ML training.

    Indexing:
        - N_games: Number of games
        - N_moves: Total moves across all games
        - N_positions: Total board positions recorded

    Board layout:
        Boards use square indexing: a1=0, b1=1, ..., h8=63
        Piece encoding: 0=empty, 1-6=white PNBRQK, 7-12=black pnbrqk
    """

    # === Board state arrays (N_positions) ===

    @property
    def boards(self) -> NDArray[np.uint8]:
        """Board positions, shape (N_positions, 8, 8), dtype uint8."""
        ...

    @property
    def castling(self) -> NDArray[np.bool_]:
        """Castling rights [K,Q,k,q], shape (N_positions, 4), dtype bool."""
        ...

    @property
    def en_passant(self) -> NDArray[np.int8]:
        """En passant file (-1 if none), shape (N_positions,), dtype int8."""
        ...

    @property
    def halfmove_clock(self) -> NDArray[np.uint8]:
        """Halfmove clock, shape (N_positions,), dtype uint8."""
        ...

    @property
    def turn(self) -> NDArray[np.bool_]:
        """Side to move (True=white), shape (N_positions,), dtype bool."""
        ...

    # === Move arrays (N_moves) ===

    @property
    def from_squares(self) -> NDArray[np.uint8]:
        """From squares, shape (N_moves,), dtype uint8."""
        ...

    @property
    def to_squares(self) -> NDArray[np.uint8]:
        """To squares, shape (N_moves,), dtype uint8."""
        ...

    @property
    def promotions(self) -> NDArray[np.int8]:
        """Promotions (-1=none, 2=N, 3=B, 4=R, 5=Q), shape (N_moves,), dtype int8."""
        ...

    @property
    def clocks(self) -> NDArray[np.float32]:
        """Clock times in seconds (NaN if missing), shape (N_moves,), dtype float32."""
        ...

    @property
    def evals(self) -> NDArray[np.float32]:
        """Engine evals (NaN if missing), shape (N_moves,), dtype float32."""
        ...

    # === Offsets ===

    @property
    def move_offsets(self) -> NDArray[np.uint32]:
        """Move offsets for CSR-style indexing, shape (N_games + 1,), dtype uint32.

        Game i's moves: move_offsets[i]..move_offsets[i+1]
        """
        ...

    @property
    def position_offsets(self) -> NDArray[np.uint32]:
        """Position offsets for CSR-style indexing, shape (N_games + 1,), dtype uint32.

        Game i's positions: position_offsets[i]..position_offsets[i+1]
        """
        ...

    # === Final position status (N_games) ===

    @property
    def is_checkmate(self) -> NDArray[np.bool_]:
        """Final position is checkmate, shape (N_games,), dtype bool."""
        ...

    @property
    def is_stalemate(self) -> NDArray[np.bool_]:
        """Final position is stalemate, shape (N_games,), dtype bool."""
        ...

    @property
    def is_insufficient(self) -> NDArray[np.bool_]:
        """Insufficient material (white, black), shape (N_games, 2), dtype bool."""
        ...

    @property
    def legal_move_count(self) -> NDArray[np.uint16]:
        """Legal move count in final position, shape (N_games,), dtype uint16."""
        ...

    # === Parse status (N_games) ===

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Whether game parsed successfully, shape (N_games,), dtype bool."""
        ...

    # === Raw headers (N_games) ===

    @property
    def headers(self) -> List[Dict[str, str]]:
        """Raw PGN headers as list of dicts."""
        ...

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

    def position_to_game(
        self, position_indices: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Map position indices to game indices.

        Useful after shuffling/sampling positions to look up game metadata.

        Args:
            position_indices: Array of indices into boards array

        Returns:
            Array of game indices (same shape as input)
        """
        ...

    def move_to_game(self, move_indices: NDArray[np.int64]) -> NDArray[np.int64]:
        """Map move indices to game indices.

        Args:
            move_indices: Array of indices into from_squares, to_squares, etc.

        Returns:
            Array of game indices (same shape as input)
        """
        ...

def parse_game(pgn: str, store_legal_moves: bool = False) -> MoveExtractor: ...
def parse_games(
    pgns: List[str], num_threads: Optional[int] = None, store_legal_moves: bool = False
) -> List[MoveExtractor]: ...
def parse_game_moves_arrow_chunked_array(
    pgn_chunked_array: pyarrow.ChunkedArray,
    num_threads: Optional[int] = None,
    store_legal_moves: bool = False,
) -> List[MoveExtractor]: ...
def parse_games_flat(
    pgn_chunked_array: pyarrow.ChunkedArray,
    num_threads: Optional[int] = None,
) -> ParsedGames:
    """Parse chess games from a PyArrow ChunkedArray into flat NumPy arrays.

    This API is optimized for ML training pipelines, returning flat NumPy arrays
    that can be efficiently batched and processed.

    Args:
        pgn_chunked_array: PyArrow ChunkedArray containing PGN strings
        num_threads: Number of threads for parallel parsing (default: all CPUs)

    Returns:
        ParsedGames object containing flat arrays and iteration support
    """
    ...
