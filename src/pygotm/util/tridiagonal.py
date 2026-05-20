"""
Tridiagonal (Thomas algorithm) solver — translation of ``mtridiagonal.F90``.

Provides a Numba-JIT Thomas algorithm solver for tridiagonal linear systems,
used by the diffusion, turbulence-closure, and momentum routines throughout
pyGOTM.

The main diagonal is stored in ``bu``, the upper diagonal in ``au``, the lower
diagonal in ``cu``, and the right-hand side in ``du``.  Work arrays ``ru`` and
``qu`` hold intermediate values during forward substitution and back
substitution respectively.

Public interface: :func:`init_tridiagonal`, :func:`tridiagonal`,
:func:`clean_tridiagonal`, :class:`TridiagonalWorkspace`,
:class:`TridiagonalBatchWorkspace`.

Original FORTRAN authors: Hans Burchard, Karsten Bolding.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "TridiagonalBatchWorkspace",
    "TridiagonalWorkspace",
    "clean_tridiagonal",
    "init_tridiagonal",
    "tridiagonal",
]


class TridiagonalWorkspace(ColumnWorkspace):
    """Single-column tridiagonal workspace — arrays shape (nlev+1,)."""

    def __init__(self, nlev: int) -> None:
        super().__init__(nlev)
        shape = (nlev + 1,)
        self.au = np.zeros(shape, dtype=np.float64)
        self.bu = np.zeros(shape, dtype=np.float64)
        self.cu = np.zeros(shape, dtype=np.float64)
        self.du = np.zeros(shape, dtype=np.float64)
        self.ru = np.zeros(shape, dtype=np.float64)
        self.qu = np.zeros(shape, dtype=np.float64)


class TridiagonalBatchWorkspace(ColumnWorkspace):
    """Batch tridiagonal workspace — arrays shape (batch_size, nlev+1)."""

    def __init__(self, nlev: int, batch_size: int) -> None:
        super().__init__(nlev, n_cols=batch_size)
        shape = (batch_size, nlev + 1)
        self.au = np.zeros(shape, dtype=np.float64)
        self.bu = np.zeros(shape, dtype=np.float64)
        self.cu = np.zeros(shape, dtype=np.float64)
        self.du = np.zeros(shape, dtype=np.float64)
        self.ru = np.zeros(shape, dtype=np.float64)
        self.qu = np.zeros(shape, dtype=np.float64)


def init_tridiagonal(nlev: int) -> TridiagonalWorkspace:
    """Allocate the tridiagonal coefficients and Thomas work arrays."""
    return TridiagonalWorkspace(nlev=nlev)


def clean_tridiagonal(workspace: TridiagonalWorkspace) -> None:
    """No-op — NumPy workspaces are garbage-collected normally."""


# Suppress the unused import warning: make_column_array is re-exported for
# callers that previously used it via this module.
_ = make_column_array


@numba.njit(cache=True)
def tridiagonal(
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
    value: np.ndarray,
    fi: int,
    lt: int,
) -> None:
    """Solve a tridiagonal system with the Thomas algorithm (simplified Gaussian elimination).

    A linear equation with tridiagonal matrix structure is solved here.  The main
    diagonal is stored in ``bu``, the upper diagonal in ``au``, the lower diagonal
    in ``cu``, and the right-hand side in ``du``.  Indices run from ``fi`` to ``lt``
    inclusive.
    """

    if fi == lt:
        value[fi] = du[fi] / bu[fi]
    else:
        ru[lt] = au[lt] / bu[lt]
        qu[lt] = du[lt] / bu[lt]

        for offset in range(lt - fi - 1):
            i = lt - 1 - offset
            denominator = bu[i] - cu[i] * ru[i + 1]
            ru[i] = au[i] / denominator
            qu[i] = (du[i] - cu[i] * qu[i + 1]) / denominator

        denominator = bu[fi] - cu[fi] * ru[fi + 1]
        qu[fi] = (du[fi] - cu[fi] * qu[fi + 1]) / denominator

        value[fi] = qu[fi]
        for i in range(fi + 1, lt + 1):
            value[i] = qu[i] - ru[i] * value[i - 1]
