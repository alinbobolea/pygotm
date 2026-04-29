# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The algebraic k-equation\label{sec:tkealgebraic}
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
    "TKEAlgebraicWorkspace",
    "step_tkealgebraic",
]

_C_FILT: float = 1.0


class TKEAlgebraicWorkspace(ColumnWorkspace):
    """Workspace arrays for the algebraic TKE closure."""

    tke: np.ndarray
    tkeo: np.ndarray
    L: np.ndarray
    NN: np.ndarray
    SS: np.ndarray
    cmue1: np.ndarray
    cmue2: np.ndarray
    u_taus: np.ndarray
    u_taub: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.tkeo = make_column_array(nlev, n_cols=n_cols)
        self.L = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)
        self.cmue1 = make_column_array(nlev, n_cols=n_cols)
        self.cmue2 = make_column_array(nlev, n_cols=n_cols)
        self.u_taus = make_column_array(nlev, n_cols=n_cols)
        self.u_taub = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_tkealgebraic(
    nlev: int,
    k_min: float,
    cm0: float,
    cde: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    L: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    u_taus: float,
    u_taub: float,
) -> None:
    r"""Advance the algebraic TKE closure (single column)."""
    for i in range(nlev + 1):
        tkeo[i] = tke[i]

    for i in range(1, nlev):
        tke[i] = (
            _C_FILT
            * (
                L[i]
                * L[i]
                / cde
                * (cmue1[i] * SS[i] - cmue2[i] * NN[i])
            )
            + (1.0 - _C_FILT) * tkeo[i]
        )

    boundary_scale = math.sqrt(cm0 * cde)
    tke[0] = u_taub * u_taub / boundary_scale
    tke[nlev] = u_taus * u_taus / boundary_scale

    for i in range(nlev + 1):
        if tke[i] < k_min:
            tke[i] = k_min


@numba.njit(parallel=True, cache=True)
def step_tkealgebraic(
    batch_size: int,
    nlev: int,
    k_min: float,
    cm0: float,
    cde: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    L: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    u_taus: np.ndarray,
    u_taub: np.ndarray,
) -> None:
    r"""Advance the algebraic TKE closure (batch)."""
    for b in numba.prange(batch_size):
        _step_tkealgebraic(
            nlev, k_min, cm0, cde,
            tke[b], tkeo[b], L[b], NN[b], SS[b], cmue1[b], cmue2[b],
            u_taus[b, 0], u_taub[b, 0],
        )
