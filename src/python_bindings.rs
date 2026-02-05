use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PySlice};
use shakmaty::{Role, Square};
use std::collections::HashMap;

// Definition of PyUciMove
#[pyclass(get_all, set_all, module = "rust_pgn_reader_python_binding")]
#[derive(Clone, Debug)]
pub struct PyUciMove {
    pub from_square: u8,
    pub to_square: u8,
    pub promotion: Option<u8>,
}

#[pymethods]
impl PyUciMove {
    #[new]
    pub fn new(from_square: u8, to_square: u8, promotion: Option<u8>) -> Self {
        PyUciMove {
            from_square,
            to_square,
            promotion,
        }
    }

    #[getter]
    fn get_from_square_name(&self) -> String {
        Square::new(self.from_square as u32).to_string()
    }

    #[getter]
    fn get_to_square_name(&self) -> String {
        Square::new(self.to_square as u32).to_string()
    }

    #[getter]
    fn get_promotion_name(&self) -> Option<String> {
        self.promotion.and_then(|p_u8| {
            Role::try_from(p_u8)
                .map(|role| format!("{:?}", role)) // Get the debug representation (e.g., "Queen")
                .ok()
        })
    }

    // __str__ method for Python representation
    fn __str__(&self) -> String {
        let promo_str = self.promotion.map_or("".to_string(), |p_u8| {
            Role::try_from(p_u8)
                .map(|role| role.char().to_string())
                .unwrap_or_else(|_| "".to_string()) // Handle potential error if u8 is not a valid Role
        });
        format!(
            "{}{}{}",
            Square::new(self.from_square as u32),
            Square::new(self.to_square as u32),
            promo_str
        )
    }

    // __repr__ for a more developer-friendly representation
    fn __repr__(&self) -> String {
        let promo_repr = self.promotion.map_or("None".to_string(), |p_u8| {
            Role::try_from(p_u8)
                .map(|role| format!("Some('{}')", role.char()))
                .unwrap_or_else(|_| format!("Some(InvalidRole({}))", p_u8))
        });
        format!(
            "PyUciMove(from_square={}, to_square={}, promotion={})",
            Square::new(self.from_square as u32),
            Square::new(self.to_square as u32),
            promo_repr
        )
    }
}

#[pyclass]
/// Holds the status of a chess position.
#[derive(Clone)]
pub struct PositionStatus {
    #[pyo3(get)]
    pub is_checkmate: bool,

    #[pyo3(get)]
    pub is_stalemate: bool,

    #[pyo3(get)]
    pub legal_move_count: usize,

    #[pyo3(get)]
    pub is_game_over: bool,

    #[pyo3(get)]
    pub insufficient_material: (bool, bool),

    #[pyo3(get)]
    pub turn: bool,
}

/// Flat array container for parsed chess games, optimized for ML training.
#[pyclass]
pub struct ParsedGames {
    // === Board state arrays (N_positions) ===
    /// Board positions, shape (N_positions, 8, 8), dtype uint8
    #[pyo3(get)]
    pub boards: Py<PyAny>,

    /// Castling rights [K,Q,k,q], shape (N_positions, 4), dtype bool
    #[pyo3(get)]
    pub castling: Py<PyAny>,

    /// En passant file (-1 if none), shape (N_positions,), dtype int8
    #[pyo3(get)]
    pub en_passant: Py<PyAny>,

    /// Halfmove clock, shape (N_positions,), dtype uint8
    #[pyo3(get)]
    pub halfmove_clock: Py<PyAny>,

    /// Side to move (true=white), shape (N_positions,), dtype bool
    #[pyo3(get)]
    pub turn: Py<PyAny>,

    // === Move arrays (N_moves) ===
    /// From squares, shape (N_moves,), dtype uint8
    #[pyo3(get)]
    pub from_squares: Py<PyAny>,

    /// To squares, shape (N_moves,), dtype uint8
    #[pyo3(get)]
    pub to_squares: Py<PyAny>,

    /// Promotions (-1=none, 2=N, 3=B, 4=R, 5=Q), shape (N_moves,), dtype int8
    #[pyo3(get)]
    pub promotions: Py<PyAny>,

    /// Clock times in seconds (NaN if missing), shape (N_moves,), dtype float32
    #[pyo3(get)]
    pub clocks: Py<PyAny>,

    /// Engine evals (NaN if missing), shape (N_moves,), dtype float32
    #[pyo3(get)]
    pub evals: Py<PyAny>,

    // === Offsets ===
    /// Move offsets for CSR-style indexing, shape (N_games + 1,), dtype uint32
    /// Game i's moves: move_offsets[i]..move_offsets[i+1]
    #[pyo3(get)]
    pub move_offsets: Py<PyAny>,

    /// Position offsets for CSR-style indexing, shape (N_games + 1,), dtype uint32
    /// Game i's positions: position_offsets[i]..position_offsets[i+1]
    #[pyo3(get)]
    pub position_offsets: Py<PyAny>,

    // === Final position status (N_games) ===
    /// Final position is checkmate, shape (N_games,), dtype bool
    #[pyo3(get)]
    pub is_checkmate: Py<PyAny>,

    /// Final position is stalemate, shape (N_games,), dtype bool
    #[pyo3(get)]
    pub is_stalemate: Py<PyAny>,

    /// Insufficient material (white, black), shape (N_games, 2), dtype bool
    #[pyo3(get)]
    pub is_insufficient: Py<PyAny>,

    /// Legal move count in final position, shape (N_games,), dtype uint16
    #[pyo3(get)]
    pub legal_move_count: Py<PyAny>,

    // === Parse status (N_games) ===
    /// Whether game parsed successfully, shape (N_games,), dtype bool
    #[pyo3(get)]
    pub valid: Py<PyAny>,

    // === Raw headers (N_games) ===
    /// Raw PGN headers as list of dicts
    #[pyo3(get)]
    pub headers: Vec<HashMap<String, String>>,
}

#[pymethods]
impl ParsedGames {
    /// Number of games in the result.
    #[getter]
    fn num_games(&self) -> usize {
        self.headers.len()
    }

    /// Total number of moves across all games.
    #[getter]
    fn num_moves(&self, py: Python<'_>) -> PyResult<usize> {
        let offsets = self.move_offsets.bind(py);
        let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;
        let readonly = offsets.readonly();
        let slice = readonly.as_slice()?;
        Ok(slice.last().copied().unwrap_or(0) as usize)
    }

    /// Total number of board positions recorded.
    #[getter]
    fn num_positions(&self, py: Python<'_>) -> PyResult<usize> {
        let offsets = self.position_offsets.bind(py);
        let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;
        let readonly = offsets.readonly();
        let slice = readonly.as_slice()?;
        Ok(slice.last().copied().unwrap_or(0) as usize)
    }

    fn __len__(&self) -> usize {
        self.headers.len()
    }

    fn __getitem__(slf: Py<Self>, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let n_games = slf.borrow(py).headers.len();

        // Handle integer index
        if let Ok(mut i) = idx.extract::<isize>() {
            // Handle negative indexing
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

        // Handle slice
        if let Ok(slice) = idx.cast::<PySlice>() {
            let indices = slice.indices(n_games as isize)?;
            let start = indices.start as usize;
            let stop = indices.stop as usize;
            let step = indices.step as usize;

            // For simplicity, we return a list of PyGameView objects
            let mut views: Vec<Py<PyGameView>> = Vec::new();
            let mut i = start;
            while i < stop {
                let game_view = PyGameView::new(py, slf.clone_ref(py), i)?;
                views.push(Py::new(py, game_view)?);
                i += step;
            }
            return Ok(pyo3::types::PyList::new(py, views)?.into_any().unbind());
        }

        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Invalid index type: expected int or slice, got {}",
            idx.get_type().name()?
        )))
    }

    fn __iter__(slf: Py<Self>, py: Python<'_>) -> PyResult<ParsedGamesIter> {
        let n_games = slf.borrow(py).headers.len();
        Ok(ParsedGamesIter {
            data: slf,
            index: 0,
            length: n_games,
        })
    }

    /// Map position indices to game indices.
    ///
    /// Useful after shuffling/sampling positions to look up game metadata.
    ///
    /// Args:
    ///     position_indices: Array of indices into boards array.
    ///         Accepts any integer dtype; int64 is optimal (avoids conversion).
    ///
    /// Returns:
    ///     Array of game indices (same shape as input)
    fn position_to_game<'py>(
        &self,
        py: Python<'py>,
        position_indices: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        let offsets = self.position_offsets.bind(py);
        let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;

        // Get numpy module for searchsorted
        let numpy = py.import("numpy")?;

        // Convert input to int64 array (no-op if already int64)
        let int64_dtype = numpy.getattr("int64")?;
        let position_indices = numpy
            .call_method1("asarray", (position_indices,))?
            .call_method(
                "astype",
                (int64_dtype,),
                Some(&[("copy", false)].into_py_dict(py)?),
            )?;

        // offsets[:-1] - all but last element
        let len = offsets.len()?;
        let slice_obj = PySlice::new(py, 0, (len - 1) as isize, 1);
        let offsets_slice = offsets.call_method1("__getitem__", (slice_obj,))?;

        // searchsorted(offsets[:-1], position_indices, side='right') - 1
        let result = numpy.call_method1(
            "searchsorted",
            (
                offsets_slice,
                position_indices,
                pyo3::types::PyString::new(py, "right"),
            ),
        )?;

        // Subtract 1
        let one = 1i64.into_pyobject(py)?;
        let result = result.call_method1("__sub__", (one,))?;

        Ok(result.extract()?)
    }

    /// Map move indices to game indices.
    ///
    /// Args:
    ///     move_indices: Array of indices into from_squares, to_squares, etc.
    ///         Accepts any integer dtype; int64 is optimal (avoids conversion).
    ///
    /// Returns:
    ///     Array of game indices (same shape as input)
    fn move_to_game<'py>(
        &self,
        py: Python<'py>,
        move_indices: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        let offsets = self.move_offsets.bind(py);
        let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;

        let numpy = py.import("numpy")?;

        // Convert input to int64 array (no-op if already int64)
        let int64_dtype = numpy.getattr("int64")?;
        let move_indices = numpy
            .call_method1("asarray", (move_indices,))?
            .call_method(
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
                move_indices,
                pyo3::types::PyString::new(py, "right"),
            ),
        )?;

        let one = 1i64.into_pyobject(py)?;
        let result = result.call_method1("__sub__", (one,))?;

        Ok(result.extract()?)
    }
}

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

/// Zero-copy view into a single game's data within a ParsedGames result.
///
/// Board indexing note: Boards use square indexing (a1=0, h8=63).
/// To convert to rank/file:
///     rank = square // 8
///     file = square % 8
#[pyclass]
pub struct PyGameView {
    data: Py<ParsedGames>,
    idx: usize,
    move_start: usize,
    move_end: usize,
    pos_start: usize,
    pos_end: usize,
}

impl PyGameView {
    pub fn new(py: Python<'_>, data: Py<ParsedGames>, idx: usize) -> PyResult<Self> {
        let borrowed = data.borrow(py);

        let move_offsets = borrowed.move_offsets.bind(py);
        let move_offsets: &Bound<'_, PyArray1<u32>> = move_offsets.cast()?;
        let pos_offsets = borrowed.position_offsets.bind(py);
        let pos_offsets: &Bound<'_, PyArray1<u32>> = pos_offsets.cast()?;

        let move_offsets_ro = move_offsets.readonly();
        let move_offsets_slice = move_offsets_ro.as_slice()?;
        let pos_offsets_ro = pos_offsets.readonly();
        let pos_offsets_slice = pos_offsets_ro.as_slice()?;

        let move_start = move_offsets_slice
            .get(idx)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let move_end = move_offsets_slice
            .get(idx + 1)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let pos_start = pos_offsets_slice
            .get(idx)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let pos_end = pos_offsets_slice
            .get(idx + 1)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;

        drop(borrowed);

        Ok(Self {
            data,
            idx,
            move_start: *move_start as usize,
            move_end: *move_end as usize,
            pos_start: *pos_start as usize,
            pos_end: *pos_end as usize,
        })
    }
}

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

    // === Board state views ===

    /// Board positions, shape (num_positions, 8, 8).
    #[getter]
    fn boards<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let boards = borrowed.boards.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = boards.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
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

    /// Castling rights [K,Q,k,q], shape (num_positions, 4).
    #[getter]
    fn castling<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.castling.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// En passant file (-1 if none), shape (num_positions,).
    #[getter]
    fn en_passant<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.en_passant.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Halfmove clock, shape (num_positions,).
    #[getter]
    fn halfmove_clock<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.halfmove_clock.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Side to move (True=white), shape (num_positions,).
    #[getter]
    fn turn<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.turn.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    // === Move views ===

    /// From squares, shape (num_moves,).
    #[getter]
    fn from_squares<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.from_squares.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// To squares, shape (num_moves,).
    #[getter]
    fn to_squares<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.to_squares.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Promotions (-1=none, 2=N, 3=B, 4=R, 5=Q), shape (num_moves,).
    #[getter]
    fn promotions<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.promotions.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Clock times in seconds (NaN if missing), shape (num_moves,).
    #[getter]
    fn clocks<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.clocks.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Engine evals (NaN if missing), shape (num_moves,).
    #[getter]
    fn evals<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.evals.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    // === Per-game metadata ===

    /// Raw PGN headers as dict.
    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<HashMap<String, String>> {
        let borrowed = self.data.borrow(py);
        Ok(borrowed.headers[self.idx].clone())
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
            .get(self.idx)
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
            .get(self.idx)
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
        // Array is shape (n_games, 2), so index is idx * 2 for white, idx * 2 + 1 for black
        let base = self.idx * 2;
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
            .get(self.idx)
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
            .get(self.idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    // === Convenience methods ===

    /// Get UCI string for move at index.
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

        let from_ro = from_arr.readonly();
        let to_ro = to_arr.readonly();
        let promo_ro = promo_arr.readonly();

        let from_slice = from_ro.as_slice()?;
        let to_slice = to_ro.as_slice()?;
        let promo_slice = promo_ro.as_slice()?;

        let abs_idx = self.move_start + move_idx;
        let from_sq = from_slice
            .get(abs_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid move index"))?;
        let to_sq = to_slice
            .get(abs_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid move index"))?;
        let promo = promo_slice
            .get(abs_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid move index"))?;

        let files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        let ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];

        let mut uci = format!(
            "{}{}{}{}",
            files[(from_sq % 8) as usize],
            ranks[(from_sq / 8) as usize],
            files[(to_sq % 8) as usize],
            ranks[(to_sq / 8) as usize]
        );

        if promo >= 0 {
            let promo_chars = ['_', '_', 'n', 'b', 'r', 'q']; // 2=N, 3=B, 4=R, 5=Q
            if (promo as usize) < promo_chars.len() {
                uci.push(promo_chars[promo as usize]);
            }
        }

        Ok(uci)
    }

    /// Get all moves as UCI strings.
    fn moves_uci(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let n_moves = self.move_end - self.move_start;
        let mut result = Vec::with_capacity(n_moves);
        for i in 0..n_moves {
            result.push(self.move_uci(py, i)?);
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

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{Role, Square};

    #[test]
    fn test_py_uci_move_no_promotion() {
        let uci_move = PyUciMove::new(Square::E2 as u8, Square::E4 as u8, None);
        assert_eq!(uci_move.from_square, Square::E2 as u8);
        assert_eq!(uci_move.to_square, Square::E4 as u8);
        assert_eq!(uci_move.promotion, None);
        assert_eq!(uci_move.get_from_square_name(), "e2");
        assert_eq!(uci_move.get_to_square_name(), "e4");
        assert_eq!(uci_move.get_promotion_name(), None);
        assert_eq!(uci_move.__str__(), "e2e4");
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=e2, to_square=e4, promotion=None)"
        );
    }

    #[test]
    fn test_py_uci_move_with_queen_promotion() {
        let uci_move = PyUciMove::new(Square::E7 as u8, Square::E8 as u8, Some(Role::Queen as u8));
        assert_eq!(uci_move.from_square, Square::E7 as u8);
        assert_eq!(uci_move.to_square, Square::E8 as u8);
        assert_eq!(uci_move.promotion, Some(Role::Queen as u8));
        assert_eq!(uci_move.get_from_square_name(), "e7");
        assert_eq!(uci_move.get_to_square_name(), "e8");
        assert_eq!(uci_move.get_promotion_name(), Some("Queen".to_string()));
        assert_eq!(uci_move.__str__(), "e7e8q");
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=e7, to_square=e8, promotion=Some('q'))"
        );
    }

    #[test]
    fn test_py_uci_move_with_rook_promotion() {
        let uci_move = PyUciMove::new(Square::A7 as u8, Square::A8 as u8, Some(Role::Rook as u8));
        assert_eq!(uci_move.from_square, Square::A7 as u8);
        assert_eq!(uci_move.to_square, Square::A8 as u8);
        assert_eq!(uci_move.promotion, Some(Role::Rook as u8));
        assert_eq!(uci_move.get_from_square_name(), "a7");
        assert_eq!(uci_move.get_to_square_name(), "a8");
        assert_eq!(uci_move.get_promotion_name(), Some("Rook".to_string()));
        assert_eq!(uci_move.__str__(), "a7a8r");
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=a7, to_square=a8, promotion=Some('r'))"
        );
    }

    #[test]
    fn test_py_uci_move_invalid_promotion_val() {
        // Test with a u8 value that doesn't correspond to a valid Role
        let uci_move = PyUciMove::new(Square::B7 as u8, Square::B8 as u8, Some(99)); // 99 is not a valid Role
        assert_eq!(uci_move.from_square, Square::B7 as u8);
        assert_eq!(uci_move.to_square, Square::B8 as u8);
        assert_eq!(uci_move.promotion, Some(99));
        assert_eq!(uci_move.get_from_square_name(), "b7");
        assert_eq!(uci_move.get_to_square_name(), "b8");
        assert_eq!(uci_move.get_promotion_name(), None); // Should be None as 99 is invalid
        assert_eq!(uci_move.__str__(), "b7b8"); // Should produce no promotion char
        assert_eq!(
            uci_move.__repr__(),
            "PyUciMove(from_square=b7, to_square=b8, promotion=Some(InvalidRole(99)))"
        );
    }
}
