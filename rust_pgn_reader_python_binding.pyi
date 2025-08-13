from typing import List, Optional, Tuple
import pyarrow

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

def parse_game(pgn: str, store_legal_moves: bool = False) -> MoveExtractor: ...
def parse_games(
    pgns: List[str], num_threads: Optional[int] = None, store_legal_moves: bool = False
) -> List[MoveExtractor]: ...
def parse_game_moves_arrow_chunked_array(
    pgn_chunked_array: pyarrow.ChunkedArray,
    num_threads: Optional[int] = None,
    store_legal_moves: bool = False,
) -> List[MoveExtractor]: ...
