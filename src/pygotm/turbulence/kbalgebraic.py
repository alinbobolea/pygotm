# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The algebraic kb-equation\label{sec:kbalgebraic}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "KBAlgebraicWorkspace",
    "step_kbalgebraic",
]


class KBAlgebraicWorkspace(ColumnWorkspace):
    """Workspace arrays for the algebraic buoyancy-variance closure."""

    tke: np.ndarray
    eps: np.ndarray
    kb: np.ndarray
    Pb: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.kb = make_column_array(nlev, n_cols=n_cols)
        self.Pb = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_kbalgebraic(
    nlev: int,
    ctt: float,
    kb_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    Pb: np.ndarray,
) -> None:
    r"""Advance the algebraic buoyancy-variance closure (single column)."""
    for i in range(nlev + 1):
        kb[i] = ctt * tke[i] / eps[i] * Pb[i]

        if kb[i] < kb_min:
            kb[i] = kb_min


@numba.njit(parallel=True, cache=True)
def step_kbalgebraic(
    batch_size: int,
    nlev: int,
    ctt: float,
    kb_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    Pb: np.ndarray,
) -> None:
    r"""Advance the algebraic buoyancy-variance closure (batch)."""
    for b in numba.prange(batch_size):
        _step_kbalgebraic(nlev, ctt, kb_min, tke[b], eps[b], kb[b], Pb[b])
