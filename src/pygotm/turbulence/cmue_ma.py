# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The Munk and Anderson (1948) stability function\label{sec:cmueMA}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "CmueMAWorkspace",
    "step_cmue_ma",
]

_RI_EPSILON: float = 1.0e-8
_RI_THRESHOLD: float = 1.0e-10


class CmueMAWorkspace(ColumnWorkspace):
    """Workspace arrays for Munk-Anderson stability functions."""

    as_: np.ndarray
    an: np.ndarray
    cmue1: np.ndarray
    cmue2: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.as_ = make_column_array(nlev, n_cols=n_cols)
        self.an = make_column_array(nlev, n_cols=n_cols)
        self.cmue1 = make_column_array(nlev, n_cols=n_cols)
        self.cmue2 = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_cmue_ma(
    nlev: int,
    cm0_fix: float,
    prandtl0_fix: float,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
) -> None:
    r"""Update Munk-Anderson stability functions (single column)."""
    for i in range(1, nlev):
        ri = an[i] / (as_[i] + _RI_EPSILON)
        prandtl = prandtl0_fix
        if ri >= _RI_THRESHOLD:
            prandtl = (
                prandtl0_fix
                * (1.0 + 3.33 * ri) ** 1.5
                / math.sqrt(1.0 + 10.0 * ri)
            )
        cmue1[i] = cm0_fix
        cmue2[i] = cm0_fix / prandtl


@numba.njit(parallel=True, cache=True)
def step_cmue_ma(
    batch_size: int,
    nlev: int,
    cm0_fix: float,
    prandtl0_fix: float,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
) -> None:
    r"""Update Munk-Anderson stability functions (batch)."""
    for b in numba.prange(batch_size):
        _step_cmue_ma(
            nlev, cm0_fix, prandtl0_fix,
            as_[b], an[b], cmue1[b], cmue2[b],
        )
