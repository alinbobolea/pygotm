# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update time scale ratio
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = ["RRatioWorkspace", "step_r_ratio"]


class RRatioWorkspace(ColumnWorkspace):
    """Workspace arrays for dissipation-time-scale ratio updates."""

    tke: np.ndarray
    eps: np.ndarray
    kb: np.ndarray
    epsb: np.ndarray
    r: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.kb = make_column_array(nlev, n_cols=n_cols)
        self.epsb = make_column_array(nlev, n_cols=n_cols)
        self.r = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_r_ratio(
    nlev: int,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
    r: np.ndarray,
) -> None:
    r"""Update the dissipation-time-scale ratio r (single column)."""
    for i in range(nlev + 1):
        r[i] = kb[i] * eps[i] / (epsb[i] * tke[i])


@numba.njit(parallel=True, cache=True)
def step_r_ratio(
    batch_size: int,
    nlev: int,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
    r: np.ndarray,
) -> None:
    r"""Update the dissipation-time-scale ratio r (batch)."""
    for b in numba.prange(batch_size):
        _step_r_ratio(nlev, tke[b], eps[b], kb[b], epsb[b], r[b])
