"""NumPy array allocation helpers for GOTM column physics.

GOTM arrays use a ``0:nlev`` index convention (Fortran), giving nlev+1 elements.

Single-column kernels use shape ``(nlev + 1,)``.
Batch kernels (called by Dask tasks) use shape ``(batch_size, nlev + 1)``,
where batch_size columns are processed in parallel with numba.prange.
All arrays are float64, C-contiguous.
"""

import numpy as np

__all__ = ["ColumnWorkspace", "make_column_array"]


def make_column_array(
    nlev: int,
    *,
    n_cols: int | None = None,
    fill: float = 0.0,
) -> np.ndarray:
    """Return a C-contiguous float64 column array initialised to *fill*."""
    shape: tuple[int, ...] = (nlev + 1,) if n_cols is None else (n_cols, nlev + 1)
    arr = np.empty(shape, dtype=np.float64)
    arr[:] = fill
    return arr


class ColumnWorkspace:
    """Base for physics module workspace classes.

    Subclasses declare np.ndarray attributes in __init__ by calling
    ``make_column_array``.  Only the ndarray members are ever passed into
    @numba.njit functions — never the workspace object itself.
    """

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        self._nlev = nlev
        self._n_cols = n_cols

    @property
    def nlev(self) -> int:
        return self._nlev

    @property
    def n_cols(self) -> int | None:
        return self._n_cols
