# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update turbulence production\label{sec:production}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "ProductionWorkspace",
    "step_production",
]


class ProductionWorkspace(ColumnWorkspace):
    """Workspace arrays for production kernels."""

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)
        self.xP = make_column_array(nlev, n_cols=n_cols)
        self.SSCSTK = make_column_array(nlev, n_cols=n_cols)
        self.SSSTK = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.nuh = make_column_array(nlev, n_cols=n_cols)
        self.nucl = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Pb = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.PSTK = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_production(
    nlev: int,
    iw_model: int,
    alpha: float,
    has_xP: int,
    has_sscstk: int,
    has_ssstk: int,
    NN: np.ndarray,
    SS: np.ndarray,
    xP: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nucl: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
) -> None:
    r"""Update turbulence production terms (single column)."""
    alpha_eff = 0.0
    if iw_model == 1:
        alpha_eff = alpha

    for i in range(nlev + 1):
        P[i] = num[i] * (SS[i] + alpha_eff * NN[i])
        B[i] = -nuh[i] * NN[i]
        Pb[i] = -B[i] * NN[i]

        if has_xP != 0:
            Px[i] = xP[i]

        if has_sscstk != 0:
            P[i] = P[i] + nucl[i] * SSCSTK[i]
            PSTK[i] = num[i] * SSCSTK[i]

        if has_ssstk != 0:
            if has_sscstk == 0:
                PSTK[i] = 0.0
            PSTK[i] = PSTK[i] + nucl[i] * SSSTK[i]


@numba.njit(parallel=True, cache=True)
def step_production(
    batch_size: int,
    nlev: int,
    iw_model: int,
    alpha: float,
    has_xP: int,
    has_sscstk: int,
    has_ssstk: int,
    NN: np.ndarray,
    SS: np.ndarray,
    xP: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    num: np.ndarray,
    nuh: np.ndarray,
    nucl: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Pb: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
) -> None:
    r"""Update turbulence production terms (batch)."""
    for b in numba.prange(batch_size):
        _step_production(
            nlev, iw_model, alpha, has_xP, has_sscstk, has_ssstk,
            NN[b], SS[b], xP[b], SSCSTK[b], SSSTK[b],
            num[b], nuh[b], nucl[b], P[b], B[b], Pb[b], Px[b], PSTK[b],
        )
