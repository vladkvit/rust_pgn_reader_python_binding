use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PySlice};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: searchsorted-based index mapping (used by position_to_game / move_to_game)
// ---------------------------------------------------------------------------

fn index_to_game<'py>(
    py: Python<'py>,
    offsets: &Py<PyAny>,
    indices: &Bound<'py, PyAny>,
) -> PyResult<Py<PyArray1<i64>>> {
    let offsets = offsets.bind(py);
    let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;

    let numpy = py.import("numpy")?;

    let int64_dtype = numpy.getattr("int64")?;
    let indices = numpy.call_method1("asarray", (indices,))?.call_method(
        "astype",
        (int64_dtype,),
        Some(&[("copy", false)].into_py_dict(py)?),
    )?;

    let len = offsets.len()?;
    let slice_obj = PySlice::new(py, 0, (len - 1) as isize, 1);
    let offsets_slice = offsets.call_method1("__getitem__", (slice_obj,))?;

    let result = numpy.call_method1(
        "searchsorted",
        (
            offsets_slice,
            indices,
            pyo3::types::PyString::new(py, "right"),
        ),
    )?;

    let one = 1i64.into_pyobject(py)?;
    let result = result.call_method1("__sub__", (one,))?;

    Ok(result.extract()?)
}

// ---------------------------------------------------------------------------
// Helper: UCI string formatting
// ---------------------------------------------------------------------------

const FILES: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
const RANKS: [char; 8] = ['1', '2', '3', '4', '5', '6', '7', '8'];
const PROMO_CHARS: [char; 6] = ['_', '_', 'n', 'b', 'r', 'q']; // index 2=N, 3=B, 4=R, 5=Q

fn format_uci(from_sq: u8, to_sq: u8, promo: i8) -> String {
    let mut uci = format!(
        "{}{}{}{}",
        FILES[(from_sq % 8) as usize],
        RANKS[(from_sq / 8) as usize],
        FILES[(to_sq % 8) as usize],
        RANKS[(to_sq / 8) as usize]
    );
    if promo >= 0 && (promo as usize) < PROMO_CHARS.len() {
        uci.push(PROMO_CHARS[promo as usize]);
    }
    uci
}

// ---------------------------------------------------------------------------
// ParsedGames
// ---------------------------------------------------------------------------

/// Flat container for parsed chess games, optimized for ML training.
///
/// All data lives in single flat NumPy arrays indexed by the CSR offset
/// arrays ``move_offsets`` / ``position_offsets`` (length num_games + 1).
/// Supports integer indexing, slicing, and iteration to access individual
/// games as ``PyGameView`` objects.
///
/// Invalid games (``parse_errors[i] is not None``): their array ranges are
/// sized from a pre-parse token count; only the first ``parsed_move_counts[i]``
/// moves (and that many + 1 positions) contain data, the remainder of the
/// range is sentinel-filled (promotions/en_passant = -1, clocks/evals = NaN,
/// zeros elsewhere).
///
/// Properties:
///     num_games: Total number of games.
///     num_moves: Total move slots across all games (== move_offsets[-1]).
///     num_positions: Total position slots recorded.
///     boards, castling, ...: Flat global arrays (see PyGameView docs for
///         encodings).
///
/// Methods:
///     position_to_game(indices): Map position indices to game indices.
///     move_to_game(indices): Map move indices to game indices.
#[pyclass]
pub struct ParsedGames {
    // Per-position NumPy arrays
    pub boards: Py<PyAny>,         // (N_positions, 8, 8) u8
    pub castling: Py<PyAny>,       // (N_positions, 4) bool
    pub en_passant: Py<PyAny>,     // (N_positions,) i8
    pub halfmove_clock: Py<PyAny>, // (N_positions,) u8
    pub turn: Py<PyAny>,           // (N_positions,) bool

    // Per-move NumPy arrays
    pub from_squares: Py<PyAny>, // (N_moves,) u8
    pub to_squares: Py<PyAny>,   // (N_moves,) u8
    pub promotions: Py<PyAny>,   // (N_moves,) i8
    pub clocks: Py<PyAny>,       // (N_moves,) f32
    pub evals: Py<PyAny>,        // (N_moves,) f32

    // Global CSR offsets and per-game NumPy arrays
    pub move_offsets: Py<PyAny>,       // (N_games + 1,) u32
    pub position_offsets: Py<PyAny>,   // (N_games + 1,) u32
    pub parsed_move_counts: Py<PyAny>, // (N_games,) u32
    pub is_checkmate: Py<PyAny>,       // (N_games,) bool
    pub is_stalemate: Py<PyAny>,       // (N_games,) bool
    pub is_insufficient: Py<PyAny>,    // (N_games, 2) bool
    pub legal_move_count: Py<PyAny>,   // (N_games,) u16
    pub valid: Py<PyAny>,              // (N_games,) bool

    // Per-game Rust-side data
    pub headers: Vec<HashMap<String, String>>,
    pub outcome: Vec<Option<String>>, // "White", "Black", "Draw", "Unknown", or None
    pub parse_errors: Vec<Option<String>>, // None if valid, Some(msg) if not

    // Optional: raw text comments (per-move), only populated when store_comments=true
    pub comments: Vec<Option<String>>,

    // Optional: legal moves at each position (CSR arrays + offsets)
    pub legal_move_from_squares: Py<PyAny>, // (N_legal_moves,) u8
    pub legal_move_to_squares: Py<PyAny>,   // (N_legal_moves,) u8
    pub legal_move_promotions: Py<PyAny>,   // (N_legal_moves,) i8
    pub legal_move_offsets: Py<PyAny>,      // (N_positions + 1,) u32

    // Rust-side offset copies for fast game-view construction
    pub move_offsets_rs: Vec<u32>,
    pub position_offsets_rs: Vec<u32>,
    pub parsed_move_counts_rs: Vec<u32>,

    pub num_games: usize,
    pub num_moves: usize,
    pub num_positions: usize,
    pub num_legal_moves: usize,
}

/// Generate a `#[pymethods]` block with NumPy-array getters for ParsedGames.
macro_rules! parsed_games_array_getters {
    ($($name:ident),+ $(,)?) => {
        #[pymethods]
        impl ParsedGames {
            $(
                #[getter]
                fn $name(&self, py: Python<'_>) -> Py<PyAny> {
                    self.$name.clone_ref(py)
                }
            )+
        }
    };
}

/// Generate a `#[pymethods]` block with Vec-cloning getters for ParsedGames.
macro_rules! parsed_games_vec_getters {
    ($($name:ident -> $ret:ty),+ $(,)?) => {
        #[pymethods]
        impl ParsedGames {
            $(
                #[getter]
                fn $name(&self) -> $ret {
                    self.$name.clone()
                }
            )+
        }
    };
}

parsed_games_array_getters!(
    boards,
    castling,
    en_passant,
    halfmove_clock,
    turn,
    from_squares,
    to_squares,
    promotions,
    clocks,
    evals,
    move_offsets,
    position_offsets,
    parsed_move_counts,
    is_checkmate,
    is_stalemate,
    is_insufficient,
    legal_move_count,
    valid,
    legal_move_from_squares,
    legal_move_to_squares,
    legal_move_promotions,
    legal_move_offsets,
);

parsed_games_vec_getters!(
    headers -> Vec<HashMap<String, String>>,
    outcome -> Vec<Option<String>>,
    parse_errors -> Vec<Option<String>>,
    comments -> Vec<Option<String>>,
);

#[pymethods]
impl ParsedGames {
    /// Number of games in the result.
    #[getter]
    fn num_games(&self) -> usize {
        self.num_games
    }

    /// Total number of move slots across all games.
    #[getter]
    fn num_moves(&self) -> usize {
        self.num_moves
    }

    /// Total number of board position slots recorded.
    #[getter]
    fn num_positions(&self) -> usize {
        self.num_positions
    }

    fn __len__(&self) -> usize {
        self.num_games
    }

    fn __repr__(&self) -> String {
        format!(
            "<ParsedGames: {} games, {} moves, {} positions>",
            self.num_games, self.num_moves, self.num_positions
        )
    }

    fn __getitem__(slf: Py<Self>, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let n_games = slf.borrow(py).num_games;

        // Handle integer index
        if let Ok(mut i) = idx.extract::<isize>() {
            if i < 0 {
                i += n_games as isize;
            }
            if i < 0 || i >= n_games as isize {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Game index {} out of range [0, {})",
                    i, n_games
                )));
            }
            let game_view = PyGameView::new(py, slf.clone_ref(py), i as usize)?;
            return Ok(Py::new(py, game_view)?.into_any());
        }

        // Handle slice — returns list of PyGameView
        if let Ok(slice) = idx.cast::<PySlice>() {
            let indices = slice.indices(n_games as isize)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step as usize;

            let mut views: Vec<Py<PyGameView>> = Vec::new();
            let mut i = start;
            while i < stop {
                let game_view = PyGameView::new(py, slf.clone_ref(py), i)?;
                views.push(Py::new(py, game_view)?);
                i += step;
            }
            return Ok(PyList::new(py, views)?.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Invalid index type: expected int or slice, got {}",
            idx.get_type().name()?
        )))
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<ParsedGamesIter> {
        let total = slf.borrow(py).num_games;
        Ok(ParsedGamesIter {
            data: slf,
            index: 0,
            length: total,
        })
    }

    /// Map position indices to game indices.
    ///
    /// Given an array of indices into the global position space, returns
    /// an array of the corresponding game indices. Useful after shuffling
    /// or sampling positions to look up game metadata.
    ///
    /// Args:
    ///     position_indices: Array of position indices.
    ///         Accepts any integer dtype; int64 is optimal (avoids conversion).
    ///
    /// Returns:
    ///     numpy.ndarray[int64]: Game indices (same shape as input).
    fn position_to_game<'py>(
        &self,
        py: Python<'py>,
        position_indices: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        index_to_game(py, &self.position_offsets, position_indices)
    }

    /// Map move indices to game indices.
    ///
    /// Given an array of indices into the global move space, returns
    /// an array of the corresponding game indices.
    ///
    /// Args:
    ///     move_indices: Array of move indices.
    ///         Accepts any integer dtype; int64 is optimal (avoids conversion).
    ///
    /// Returns:
    ///     numpy.ndarray[int64]: Game indices (same shape as input).
    fn move_to_game<'py>(
        &self,
        py: Python<'py>,
        move_indices: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        index_to_game(py, &self.move_offsets, move_indices)
    }
}

// ---------------------------------------------------------------------------
// ParsedGamesIter
// ---------------------------------------------------------------------------

/// Iterator over games in a ParsedGames result.
#[pyclass]
pub struct ParsedGamesIter {
    data: Py<ParsedGames>,
    index: usize,
    length: usize,
}

#[pymethods]
impl ParsedGamesIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> PyResult<Option<PyGameView>> {
        if slf.index >= slf.length {
            return Ok(None);
        }
        let game_view = PyGameView::new(py, slf.data.clone_ref(py), slf.index)?;
        slf.index += 1;
        Ok(Some(game_view))
    }
}

// ---------------------------------------------------------------------------
// PyGameView — macros + impl
// ---------------------------------------------------------------------------

/// Zero-copy view into a single game within a ParsedGames result.
///
/// Provides access to board positions, moves, metadata, and annotations
/// for one game. All array properties return numpy array slices (views,
/// not copies) into the parent data. For invalid games the view covers
/// only the actually-parsed prefix of the game's range.
///
/// Board encoding:
///     Piece values: 0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K (white),
///     7=p, 8=n, 9=b, 10=r, 11=q, 12=k (black).
///
/// Square indexing:
///     a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63.
///     rank = square // 8, file = square % 8.
///
/// Move encoding:
///     from_squares / to_squares: source and destination square indices.
///     promotions: -1=none, 2=N, 3=B, 4=R, 5=Q.
///     clocks: remaining time in seconds (NaN if missing).
///     evals: engine evaluation in pawns (NaN if missing).
#[pyclass]
pub struct PyGameView {
    data: Py<ParsedGames>,
    /// Global game index.
    game_idx: usize,
    /// Move range within the global move arrays (parsed moves only).
    move_start: usize,
    move_end: usize,
    /// Position range within the global position arrays (parsed only).
    pos_start: usize,
    pos_end: usize,
}

impl PyGameView {
    /// Create a new game view for game index `game_idx`.
    pub fn new(py: Python<'_>, data: Py<ParsedGames>, game_idx: usize) -> PyResult<Self> {
        let borrowed = data.borrow(py);

        if game_idx >= borrowed.num_games {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "Invalid game index",
            ));
        }

        let move_start = borrowed.move_offsets_rs[game_idx] as usize;
        let pos_start = borrowed.position_offsets_rs[game_idx] as usize;
        let parsed = borrowed.parsed_move_counts_rs[game_idx] as usize;

        drop(borrowed);

        Ok(Self {
            data,
            game_idx,
            move_start,
            move_end: move_start + parsed,
            pos_start,
            pos_end: pos_start + parsed + 1,
        })
    }
}

/// Generate a `#[pymethods]` block with position-sliced getters for PyGameView.
macro_rules! game_view_pos_getters {
    ($($name:ident),+ $(,)?) => {
        #[pymethods]
        impl PyGameView {
            $(
                #[getter]
                fn $name<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                    let borrowed = self.data.borrow(py);
                    let arr = borrowed.$name.bind(py);
                    let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
                    let slice = arr.call_method1("__getitem__", (slice_obj,))?;
                    Ok(slice.unbind())
                }
            )+
        }
    };
}

/// Generate a `#[pymethods]` block with move-sliced getters for PyGameView.
macro_rules! game_view_move_getters {
    ($($name:ident),+ $(,)?) => {
        #[pymethods]
        impl PyGameView {
            $(
                #[getter]
                fn $name<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                    let borrowed = self.data.borrow(py);
                    let arr = borrowed.$name.bind(py);
                    let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
                    let slice = arr.call_method1("__getitem__", (slice_obj,))?;
                    Ok(slice.unbind())
                }
            )+
        }
    };
}

game_view_pos_getters!(boards, castling, en_passant, halfmove_clock, turn);
game_view_move_getters!(from_squares, to_squares, promotions, clocks, evals);

#[pymethods]
impl PyGameView {
    /// Number of moves in this game.
    fn __len__(&self) -> usize {
        self.move_end - self.move_start
    }

    /// Number of positions recorded for this game.
    #[getter]
    fn num_positions(&self) -> usize {
        self.pos_end - self.pos_start
    }

    /// Initial board position, shape (8, 8).
    #[getter]
    fn initial_board<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let boards = borrowed.boards.bind(py);
        let slice = boards.call_method1("__getitem__", (self.pos_start,))?;
        Ok(slice.unbind())
    }

    /// Final board position, shape (8, 8).
    #[getter]
    fn final_board<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let boards = borrowed.boards.bind(py);
        let slice = boards.call_method1("__getitem__", (self.pos_end - 1,))?;
        Ok(slice.unbind())
    }

    // === Per-game metadata ===

    /// Raw PGN headers as dict.
    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<HashMap<String, String>> {
        let borrowed = self.data.borrow(py);
        Ok(borrowed.headers[self.game_idx].clone())
    }

    /// Final position is checkmate.
    #[getter]
    fn is_checkmate(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.is_checkmate.bind(py);
        let arr: &Bound<'_, PyArray1<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.game_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Final position is stalemate.
    #[getter]
    fn is_stalemate(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.is_stalemate.bind(py);
        let arr: &Bound<'_, PyArray1<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.game_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Insufficient material (white, black).
    #[getter]
    fn is_insufficient(&self, py: Python<'_>) -> PyResult<(bool, bool)> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.is_insufficient.bind(py);
        let arr: &Bound<'_, PyArray2<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let base = self.game_idx * 2;
        let white = slice
            .get(base)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let black = slice
            .get(base + 1)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        Ok((white, black))
    }

    /// Legal move count in final position.
    #[getter]
    fn legal_move_count(&self, py: Python<'_>) -> PyResult<u16> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.legal_move_count.bind(py);
        let arr: &Bound<'_, PyArray1<u16>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.game_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Whether game parsed successfully.
    #[getter]
    fn is_valid(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.valid.bind(py);
        let arr: &Bound<'_, PyArray1<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.game_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Whether the game is over (checkmate, stalemate, or both sides have insufficient material).
    #[getter]
    fn is_game_over(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);

        let cm_arr = borrowed.is_checkmate.bind(py);
        let cm_arr: &Bound<'_, PyArray1<bool>> = cm_arr.cast()?;
        let checkmate = cm_arr
            .readonly()
            .as_slice()?
            .get(self.game_idx)
            .copied()
            .unwrap_or(false);

        let sm_arr = borrowed.is_stalemate.bind(py);
        let sm_arr: &Bound<'_, PyArray1<bool>> = sm_arr.cast()?;
        let stalemate = sm_arr
            .readonly()
            .as_slice()?
            .get(self.game_idx)
            .copied()
            .unwrap_or(false);

        let ins_arr = borrowed.is_insufficient.bind(py);
        let ins_arr: &Bound<'_, PyArray2<bool>> = ins_arr.cast()?;
        let ins_slice = ins_arr.readonly();
        let ins_slice = ins_slice.as_slice()?;
        let base = self.game_idx * 2;
        let insuf_white = ins_slice.get(base).copied().unwrap_or(false);
        let insuf_black = ins_slice.get(base + 1).copied().unwrap_or(false);

        Ok(checkmate || stalemate || (insuf_white && insuf_black))
    }

    /// Game outcome from movetext: "White", "Black", "Draw", "Unknown", or None.
    #[getter]
    fn outcome(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let borrowed = self.data.borrow(py);
        Ok(borrowed.outcome[self.game_idx].clone())
    }

    /// Parse error message if the game failed to parse, or None if valid.
    #[getter]
    fn parse_error(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let borrowed = self.data.borrow(py);
        Ok(borrowed.parse_errors[self.game_idx].clone())
    }

    /// Raw text comments per move (only populated when store_comments=true).
    /// Returns list[str | None] of length num_moves.
    #[getter]
    fn comments(&self, py: Python<'_>) -> PyResult<Vec<Option<String>>> {
        let borrowed = self.data.borrow(py);
        if borrowed.comments.is_empty() {
            return Ok(Vec::new());
        }
        Ok(borrowed.comments[self.move_start..self.move_end].to_vec())
    }

    /// Legal moves at each position in this game.
    /// Returns list of lists: [(from, to, promotion), ...] per position.
    /// Only populated when store_legal_moves=true.
    #[getter]
    fn legal_moves(&self, py: Python<'_>) -> PyResult<Vec<Vec<(u8, u8, i8)>>> {
        let borrowed = self.data.borrow(py);

        if borrowed.num_legal_moves == 0 {
            return Ok(Vec::new());
        }

        let offsets_arr = borrowed.legal_move_offsets.bind(py);
        let offsets_arr: &Bound<'_, PyArray1<u32>> = offsets_arr.cast()?;
        let offsets_ro = offsets_arr.readonly();
        let offsets_slice = offsets_ro.as_slice()?;

        let from_arr = borrowed.legal_move_from_squares.bind(py);
        let from_arr: &Bound<'_, PyArray1<u8>> = from_arr.cast()?;
        let from_ro = from_arr.readonly();
        let from_slice = from_ro.as_slice()?;

        let to_arr = borrowed.legal_move_to_squares.bind(py);
        let to_arr: &Bound<'_, PyArray1<u8>> = to_arr.cast()?;
        let to_ro = to_arr.readonly();
        let to_slice = to_ro.as_slice()?;

        let promo_arr = borrowed.legal_move_promotions.bind(py);
        let promo_arr: &Bound<'_, PyArray1<i8>> = promo_arr.cast()?;
        let promo_ro = promo_arr.readonly();
        let promo_slice = promo_ro.as_slice()?;

        let mut result = Vec::with_capacity(self.pos_end - self.pos_start);
        for pos_idx in self.pos_start..self.pos_end {
            let start = offsets_slice[pos_idx] as usize;
            let end = offsets_slice[pos_idx + 1] as usize;
            let mut moves = Vec::with_capacity(end - start);
            for i in start..end {
                moves.push((from_slice[i], to_slice[i], promo_slice[i]));
            }
            result.push(moves);
        }
        Ok(result)
    }

    // === Convenience methods ===

    /// Get UCI string for move at index (e.g. "e2e4", "a7a8q").
    fn move_uci(&self, py: Python<'_>, move_idx: usize) -> PyResult<String> {
        if move_idx >= self.move_end - self.move_start {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Move index {} out of range [0, {})",
                move_idx,
                self.move_end - self.move_start
            )));
        }

        let borrowed = self.data.borrow(py);
        let from_arr = borrowed.from_squares.bind(py);
        let from_arr: &Bound<'_, PyArray1<u8>> = from_arr.cast()?;
        let to_arr = borrowed.to_squares.bind(py);
        let to_arr: &Bound<'_, PyArray1<u8>> = to_arr.cast()?;
        let promo_arr = borrowed.promotions.bind(py);
        let promo_arr: &Bound<'_, PyArray1<i8>> = promo_arr.cast()?;

        let abs_idx = self.move_start + move_idx;
        let from_sq = from_arr.readonly().as_slice()?[abs_idx];
        let to_sq = to_arr.readonly().as_slice()?[abs_idx];
        let promo = promo_arr.readonly().as_slice()?[abs_idx];

        Ok(format_uci(from_sq, to_sq, promo))
    }

    /// Get all moves as UCI strings (e.g. ["e2e4", "e7e5", "g1f3"]).
    fn moves_uci(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let borrowed = self.data.borrow(py);

        let from_arr = borrowed.from_squares.bind(py);
        let from_arr: &Bound<'_, PyArray1<u8>> = from_arr.cast()?;
        let to_arr = borrowed.to_squares.bind(py);
        let to_arr: &Bound<'_, PyArray1<u8>> = to_arr.cast()?;
        let promo_arr = borrowed.promotions.bind(py);
        let promo_arr: &Bound<'_, PyArray1<i8>> = promo_arr.cast()?;

        let from_slice = from_arr.readonly();
        let from_slice = from_slice.as_slice()?;
        let to_slice = to_arr.readonly();
        let to_slice = to_slice.as_slice()?;
        let promo_slice = promo_arr.readonly();
        let promo_slice = promo_slice.as_slice()?;

        let n_moves = self.move_end - self.move_start;
        let mut result = Vec::with_capacity(n_moves);
        for i in self.move_start..self.move_end {
            result.push(format_uci(from_slice[i], to_slice[i], promo_slice[i]));
        }
        Ok(result)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let headers = self.headers(py)?;
        let white = headers.get("White").map(|s| s.as_str()).unwrap_or("?");
        let black = headers.get("Black").map(|s| s.as_str()).unwrap_or("?");
        let n_moves = self.move_end - self.move_start;
        let is_valid = self.is_valid(py)?;
        Ok(format!(
            "<PyGameView {} vs {}, {} moves, valid={}>",
            white, black, n_moves, is_valid
        ))
    }
}
