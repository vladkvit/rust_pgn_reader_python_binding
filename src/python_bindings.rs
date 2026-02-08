use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList, PySlice};
use std::collections::HashMap;

/// Internal per-chunk data. Not exposed to Python directly.
/// Each chunk corresponds to one thread's output during parallel parsing.
pub struct ChunkData {
    // Per-position numpy arrays
    pub boards: Py<PyAny>,         // (N_positions, 8, 8) u8
    pub castling: Py<PyAny>,       // (N_positions, 4) bool
    pub en_passant: Py<PyAny>,     // (N_positions,) i8
    pub halfmove_clock: Py<PyAny>, // (N_positions,) u8
    pub turn: Py<PyAny>,           // (N_positions,) bool

    // Per-move numpy arrays
    pub from_squares: Py<PyAny>, // (N_moves,) u8
    pub to_squares: Py<PyAny>,   // (N_moves,) u8
    pub promotions: Py<PyAny>,   // (N_moves,) i8
    pub clocks: Py<PyAny>,       // (N_moves,) f32
    pub evals: Py<PyAny>,        // (N_moves,) f32

    // Per-game arrays (CSR offsets are local to this chunk)
    pub move_offsets: Py<PyAny>,     // (N_games + 1,) u32
    pub position_offsets: Py<PyAny>, // (N_games + 1,) u32
    pub is_checkmate: Py<PyAny>,     // (N_games,) bool
    pub is_stalemate: Py<PyAny>,     // (N_games,) bool
    pub is_insufficient: Py<PyAny>,  // (N_games, 2) bool
    pub legal_move_count: Py<PyAny>, // (N_games,) u16
    pub valid: Py<PyAny>,            // (N_games,) bool
    pub headers: Vec<HashMap<String, String>>,
    pub outcome: Vec<Option<String>>, // Per-game: "White", "Black", "Draw", "Unknown", or None

    // Optional: raw text comments (per-move), only populated when store_comments=true
    pub comments: Vec<Option<String>>,

    // Optional: legal moves at each position (CSR arrays + offsets)
    pub legal_move_from_squares: Py<PyAny>, // (N_legal_moves,) u8
    pub legal_move_to_squares: Py<PyAny>,   // (N_legal_moves,) u8
    pub legal_move_promotions: Py<PyAny>,   // (N_legal_moves,) i8
    pub legal_move_offsets: Py<PyAny>,      // (N_positions + 1,) u32

    // Metadata
    pub num_games: usize,
    pub num_moves: usize,
    pub num_positions: usize,
    pub num_legal_moves: usize,
}

/// Chunked container for parsed chess games, optimized for ML training.
///
/// Internally stores data in multiple chunks (one per parsing thread) to
/// avoid the cost of merging. Per-game access is O(log(num_chunks)) via
/// binary search on precomputed boundaries.
#[pyclass]
pub struct ParsedGames {
    pub chunks: Vec<ChunkData>,

    // Global prefix sums for O(1) chunk lookup.
    // game_boundaries[i] = total games in chunks 0..i
    // So game_boundaries = [0, chunk0.num_games, chunk0+chunk1, ..., total_games]
    pub game_boundaries: Vec<usize>,
    pub move_boundaries: Vec<usize>,
    pub position_boundaries: Vec<usize>,

    pub total_games: usize,
    pub total_moves: usize,
    pub total_positions: usize,

    // Global offsets for position_to_game / move_to_game (precomputed)
    pub global_move_offsets: Py<PyAny>,
    pub global_position_offsets: Py<PyAny>,
}

impl ParsedGames {
    /// Locate which chunk a global game index belongs to.
    /// Returns (chunk_index, local_game_index).
    fn locate_game(&self, global_idx: usize) -> (usize, usize) {
        // Binary search: find the last boundary <= global_idx
        // game_boundaries = [0, n0, n0+n1, ...], length = num_chunks + 1
        let chunk_idx = match self.game_boundaries.binary_search(&global_idx) {
            Ok(i) => {
                // Exact match. If it's the last boundary, back up one.
                if i >= self.chunks.len() {
                    self.chunks.len() - 1
                } else {
                    i
                }
            }
            Err(i) => i - 1, // insertion point - 1
        };
        let local_idx = global_idx - self.game_boundaries[chunk_idx];
        (chunk_idx, local_idx)
    }
}

#[pymethods]
impl ParsedGames {
    /// Number of games in the result.
    #[getter]
    fn num_games(&self) -> usize {
        self.total_games
    }

    /// Total number of moves across all games.
    #[getter]
    fn num_moves(&self) -> usize {
        self.total_moves
    }

    /// Total number of board positions recorded.
    #[getter]
    fn num_positions(&self) -> usize {
        self.total_positions
    }

    /// Number of internal chunks.
    #[getter]
    fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    fn __len__(&self) -> usize {
        self.total_games
    }

    fn __getitem__(slf: Py<Self>, py: Python<'_>, idx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let n_games = slf.borrow(py).total_games;

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
        let total = slf.borrow(py).total_games;
        Ok(ParsedGamesIter {
            data: slf,
            index: 0,
            length: total,
        })
    }

    /// Escape hatch: access raw per-chunk data.
    ///
    /// Returns a list of chunk view objects, each exposing numpy arrays
    /// for that chunk's data. Use this for advanced/custom access patterns.
    #[getter]
    fn chunks(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let n_chunks = slf.borrow(py).chunks.len();
        let mut views: Vec<Py<PyChunkView>> = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            views.push(Py::new(
                py,
                PyChunkView {
                    parent: slf.clone_ref(py),
                    chunk_idx: i,
                },
            )?);
        }
        Ok(PyList::new(py, views)?.into_any().unbind())
    }

    /// Map position indices to game indices.
    ///
    /// Useful after shuffling/sampling positions to look up game metadata.
    ///
    /// Args:
    ///     position_indices: Array of indices into the global position space.
    ///         Accepts any integer dtype; int64 is optimal (avoids conversion).
    ///
    /// Returns:
    ///     Array of game indices (same shape as input)
    fn position_to_game<'py>(
        &self,
        py: Python<'py>,
        position_indices: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        let offsets = self.global_position_offsets.bind(py);
        let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;

        let numpy = py.import("numpy")?;

        let int64_dtype = numpy.getattr("int64")?;
        let position_indices = numpy
            .call_method1("asarray", (position_indices,))?
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
                position_indices,
                pyo3::types::PyString::new(py, "right"),
            ),
        )?;

        let one = 1i64.into_pyobject(py)?;
        let result = result.call_method1("__sub__", (one,))?;

        Ok(result.extract()?)
    }

    /// Map move indices to game indices.
    ///
    /// Args:
    ///     move_indices: Array of indices into the global move space.
    ///         Accepts any integer dtype; int64 is optimal (avoids conversion).
    ///
    /// Returns:
    ///     Array of game indices (same shape as input)
    fn move_to_game<'py>(
        &self,
        py: Python<'py>,
        move_indices: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyArray1<i64>>> {
        let offsets = self.global_move_offsets.bind(py);
        let offsets: &Bound<'_, PyArray1<u32>> = offsets.cast()?;

        let numpy = py.import("numpy")?;

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

/// Escape hatch: view into a single chunk's raw numpy arrays.
///
/// Access via `parsed_games.chunks[i]`. Each chunk corresponds to one
/// parsing thread's output. Use this for advanced access patterns like
/// manual concatenation or custom batching.
#[pyclass]
pub struct PyChunkView {
    parent: Py<ParsedGames>,
    chunk_idx: usize,
}

#[pymethods]
impl PyChunkView {
    #[getter]
    fn num_games(&self, py: Python<'_>) -> usize {
        self.parent.borrow(py).chunks[self.chunk_idx].num_games
    }

    #[getter]
    fn num_moves(&self, py: Python<'_>) -> usize {
        self.parent.borrow(py).chunks[self.chunk_idx].num_moves
    }

    #[getter]
    fn num_positions(&self, py: Python<'_>) -> usize {
        self.parent.borrow(py).chunks[self.chunk_idx].num_positions
    }

    #[getter]
    fn boards(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .boards
            .clone_ref(py)
    }

    #[getter]
    fn castling(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .castling
            .clone_ref(py)
    }

    #[getter]
    fn en_passant(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .en_passant
            .clone_ref(py)
    }

    #[getter]
    fn halfmove_clock(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .halfmove_clock
            .clone_ref(py)
    }

    #[getter]
    fn turn(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .turn
            .clone_ref(py)
    }

    #[getter]
    fn from_squares(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .from_squares
            .clone_ref(py)
    }

    #[getter]
    fn to_squares(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .to_squares
            .clone_ref(py)
    }

    #[getter]
    fn promotions(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .promotions
            .clone_ref(py)
    }

    #[getter]
    fn clocks(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .clocks
            .clone_ref(py)
    }

    #[getter]
    fn evals(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .evals
            .clone_ref(py)
    }

    #[getter]
    fn move_offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .move_offsets
            .clone_ref(py)
    }

    #[getter]
    fn position_offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .position_offsets
            .clone_ref(py)
    }

    #[getter]
    fn is_checkmate(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .is_checkmate
            .clone_ref(py)
    }

    #[getter]
    fn is_stalemate(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .is_stalemate
            .clone_ref(py)
    }

    #[getter]
    fn is_insufficient(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .is_insufficient
            .clone_ref(py)
    }

    #[getter]
    fn legal_move_count(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .legal_move_count
            .clone_ref(py)
    }

    #[getter]
    fn valid(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .valid
            .clone_ref(py)
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> Vec<HashMap<String, String>> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .headers
            .clone()
    }

    #[getter]
    fn outcome(&self, py: Python<'_>) -> Vec<Option<String>> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .outcome
            .clone()
    }

    /// Raw text comments per move (only populated when store_comments=true).
    #[getter]
    fn comments(&self, py: Python<'_>) -> Vec<Option<String>> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .comments
            .clone()
    }

    /// Legal move from-squares for all positions in this chunk.
    #[getter]
    fn legal_move_from_squares(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .legal_move_from_squares
            .clone_ref(py)
    }

    /// Legal move to-squares for all positions in this chunk.
    #[getter]
    fn legal_move_to_squares(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .legal_move_to_squares
            .clone_ref(py)
    }

    /// Legal move promotions for all positions in this chunk.
    #[getter]
    fn legal_move_promotions(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .legal_move_promotions
            .clone_ref(py)
    }

    /// CSR offsets for legal moves (per-position). Length = num_positions + 1.
    #[getter]
    fn legal_move_offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.parent.borrow(py).chunks[self.chunk_idx]
            .legal_move_offsets
            .clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let borrowed = self.parent.borrow(py);
        let chunk = &borrowed.chunks[self.chunk_idx];
        format!(
            "<PyChunkView chunk={}, {} games, {} moves, {} positions>",
            self.chunk_idx, chunk.num_games, chunk.num_moves, chunk.num_positions
        )
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
    /// Index of the chunk this game lives in.
    chunk_idx: usize,
    /// Local game index within the chunk.
    local_idx: usize,
    /// Move range within the chunk's move arrays.
    move_start: usize,
    move_end: usize,
    /// Position range within the chunk's position arrays.
    pos_start: usize,
    pos_end: usize,
}

impl PyGameView {
    /// Create a new game view for global game index `global_idx`.
    pub fn new(py: Python<'_>, data: Py<ParsedGames>, global_idx: usize) -> PyResult<Self> {
        let borrowed = data.borrow(py);

        let (chunk_idx, local_idx) = borrowed.locate_game(global_idx);
        let chunk = &borrowed.chunks[chunk_idx];

        // Read this chunk's local CSR offsets
        let move_offsets = chunk.move_offsets.bind(py);
        let move_offsets: &Bound<'_, PyArray1<u32>> = move_offsets.cast()?;
        let pos_offsets = chunk.position_offsets.bind(py);
        let pos_offsets: &Bound<'_, PyArray1<u32>> = pos_offsets.cast()?;

        let move_offsets_ro = move_offsets.readonly();
        let move_offsets_slice = move_offsets_ro.as_slice()?;
        let pos_offsets_ro = pos_offsets.readonly();
        let pos_offsets_slice = pos_offsets_ro.as_slice()?;

        let move_start = move_offsets_slice
            .get(local_idx)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let move_end = move_offsets_slice
            .get(local_idx + 1)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let pos_start = pos_offsets_slice
            .get(local_idx)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;
        let pos_end = pos_offsets_slice
            .get(local_idx + 1)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))?;

        drop(borrowed);

        Ok(Self {
            data,
            chunk_idx,
            local_idx,
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
        let boards = borrowed.chunks[self.chunk_idx].boards.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = boards.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Initial board position, shape (8, 8).
    #[getter]
    fn initial_board<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let boards = borrowed.chunks[self.chunk_idx].boards.bind(py);
        let slice = boards.call_method1("__getitem__", (self.pos_start,))?;
        Ok(slice.unbind())
    }

    /// Final board position, shape (8, 8).
    #[getter]
    fn final_board<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let boards = borrowed.chunks[self.chunk_idx].boards.bind(py);
        let slice = boards.call_method1("__getitem__", (self.pos_end - 1,))?;
        Ok(slice.unbind())
    }

    /// Castling rights [K,Q,k,q], shape (num_positions, 4).
    #[getter]
    fn castling<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].castling.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// En passant file (-1 if none), shape (num_positions,).
    #[getter]
    fn en_passant<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].en_passant.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Halfmove clock, shape (num_positions,).
    #[getter]
    fn halfmove_clock<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].halfmove_clock.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Side to move (True=white), shape (num_positions,).
    #[getter]
    fn turn<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].turn.bind(py);
        let slice_obj = PySlice::new(py, self.pos_start as isize, self.pos_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    // === Move views ===

    /// From squares, shape (num_moves,).
    #[getter]
    fn from_squares<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].from_squares.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// To squares, shape (num_moves,).
    #[getter]
    fn to_squares<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].to_squares.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Promotions (-1=none, 2=N, 3=B, 4=R, 5=Q), shape (num_moves,).
    #[getter]
    fn promotions<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].promotions.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Clock times in seconds (NaN if missing), shape (num_moves,).
    #[getter]
    fn clocks<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].clocks.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    /// Engine evals (NaN if missing), shape (num_moves,).
    #[getter]
    fn evals<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].evals.bind(py);
        let slice_obj = PySlice::new(py, self.move_start as isize, self.move_end as isize, 1);
        let slice = arr.call_method1("__getitem__", (slice_obj,))?;
        Ok(slice.unbind())
    }

    // === Per-game metadata ===

    /// Raw PGN headers as dict.
    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<HashMap<String, String>> {
        let borrowed = self.data.borrow(py);
        Ok(borrowed.chunks[self.chunk_idx].headers[self.local_idx].clone())
    }

    /// Final position is checkmate.
    #[getter]
    fn is_checkmate(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].is_checkmate.bind(py);
        let arr: &Bound<'_, PyArray1<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.local_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Final position is stalemate.
    #[getter]
    fn is_stalemate(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].is_stalemate.bind(py);
        let arr: &Bound<'_, PyArray1<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.local_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Insufficient material (white, black).
    #[getter]
    fn is_insufficient(&self, py: Python<'_>) -> PyResult<(bool, bool)> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].is_insufficient.bind(py);
        let arr: &Bound<'_, PyArray2<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        let base = self.local_idx * 2;
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
        let arr = borrowed.chunks[self.chunk_idx].legal_move_count.bind(py);
        let arr: &Bound<'_, PyArray1<u16>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.local_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Whether game parsed successfully.
    #[getter]
    fn is_valid(&self, py: Python<'_>) -> PyResult<bool> {
        let borrowed = self.data.borrow(py);
        let arr = borrowed.chunks[self.chunk_idx].valid.bind(py);
        let arr: &Bound<'_, PyArray1<bool>> = arr.cast()?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice()?;
        slice
            .get(self.local_idx)
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("Invalid game index"))
    }

    /// Whether the game is over (checkmate, stalemate, or both sides have insufficient material).
    #[getter]
    fn is_game_over(&self, py: Python<'_>) -> PyResult<bool> {
        let checkmate = self.is_checkmate(py)?;
        let stalemate = self.is_stalemate(py)?;
        let (insuf_white, insuf_black) = self.is_insufficient(py)?;
        Ok(checkmate || stalemate || (insuf_white && insuf_black))
    }

    /// Game outcome from movetext: "White", "Black", "Draw", "Unknown", or None.
    #[getter]
    fn outcome(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let borrowed = self.data.borrow(py);
        Ok(borrowed.chunks[self.chunk_idx].outcome[self.local_idx].clone())
    }

    /// Raw text comments per move (only populated when store_comments=true).
    /// Returns list[str | None] of length num_moves.
    #[getter]
    fn comments(&self, py: Python<'_>) -> PyResult<Vec<Option<String>>> {
        let borrowed = self.data.borrow(py);
        let chunk_comments = &borrowed.chunks[self.chunk_idx].comments;
        if chunk_comments.is_empty() {
            return Ok(Vec::new());
        }
        Ok(chunk_comments[self.move_start..self.move_end].to_vec())
    }

    /// Legal moves at each position in this game.
    /// Returns list of lists: [[from, to, promotion], ...] per position.
    /// Only populated when store_legal_moves=true.
    #[getter]
    fn legal_moves(&self, py: Python<'_>) -> PyResult<Vec<Vec<(u8, u8, i8)>>> {
        let borrowed = self.data.borrow(py);
        let chunk = &borrowed.chunks[self.chunk_idx];

        // Check if legal moves were stored
        if chunk.num_legal_moves == 0 {
            return Ok(Vec::new());
        }

        let offsets_arr = chunk.legal_move_offsets.bind(py);
        let offsets_arr: &Bound<'_, PyArray1<u32>> = offsets_arr.cast()?;
        let offsets_ro = offsets_arr.readonly();
        let offsets_slice = offsets_ro.as_slice()?;

        let from_arr = chunk.legal_move_from_squares.bind(py);
        let from_arr: &Bound<'_, PyArray1<u8>> = from_arr.cast()?;
        let from_ro = from_arr.readonly();
        let from_slice = from_ro.as_slice()?;

        let to_arr = chunk.legal_move_to_squares.bind(py);
        let to_arr: &Bound<'_, PyArray1<u8>> = to_arr.cast()?;
        let to_ro = to_arr.readonly();
        let to_slice = to_ro.as_slice()?;

        let promo_arr = chunk.legal_move_promotions.bind(py);
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
        let chunk = &borrowed.chunks[self.chunk_idx];
        let from_arr = chunk.from_squares.bind(py);
        let from_arr: &Bound<'_, PyArray1<u8>> = from_arr.cast()?;
        let to_arr = chunk.to_squares.bind(py);
        let to_arr: &Bound<'_, PyArray1<u8>> = to_arr.cast()?;
        let promo_arr = chunk.promotions.bind(py);
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
