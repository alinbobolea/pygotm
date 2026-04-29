# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update internal wave mixing\label{sec:internalWaves}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = ["InternalWaveWorkspace", "step_internal_wave"]

_SHEAR_EPSILON = 1.0e-10


class InternalWaveWorkspace(ColumnWorkspace):
    """Workspace arrays for internal-wave mixing."""

    tke: np.ndarray
    num: np.ndarray
    nuh: np.ndarray
    NN: np.ndarray
    SS: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.nuh = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_internal_wave(
    nlev: int,
    iw_model: int,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    tke: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> None:
    r"""Apply the Kantha-Clayson internal-wave mixing fallback (single column)."""
    if iw_model == 2:
        rich2 = rich_cr * rich_cr

        for i in range(1, nlev):
            if tke[i] <= klimiw:
                rich = NN[i] / (SS[i] + _SHEAR_EPSILON)
                if rich < rich_cr:
                    if rich > 0.0:
                        pot = 1.0 - rich * rich / rich2
                        x = numshear * pot * pot * pot
                        num[i] = numiw + x
                        nuh[i] = nuhiw + x
                    else:
                        num[i] = numiw + numshear
                        nuh[i] = nuhiw + numshear
                else:
                    num[i] = numiw
                    nuh[i] = nuhiw


@numba.njit(parallel=True, cache=True)
def step_internal_wave(
    batch_size: int,
    nlev: int,
    iw_model: int,
    klimiw: float,
    rich_cr: float,
    numiw: float,
    nuhiw: float,
    numshear: float,
    tke: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> None:
    r"""Apply the Kantha-Clayson internal-wave mixing fallback (batch)."""
    for b in numba.prange(batch_size):
        _step_internal_wave(
            nlev, iw_model, klimiw, rich_cr, numiw, nuhiw, numshear,
            tke[b], num[b], nuh[b], NN[b], SS[b],
        )
