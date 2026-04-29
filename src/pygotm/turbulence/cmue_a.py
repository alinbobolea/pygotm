# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The non-local, exact weak-equilibrium stability function \label{sec:cmueA}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "CmueAWorkspace",
    "step_cmue_a",
]


class CmueAWorkspace(ColumnWorkspace):
    """Workspace arrays for exact weak-equilibrium stability functions."""

    eps: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    Pb: np.ndarray
    epsb: np.ndarray
    as_: np.ndarray
    an: np.ndarray
    at: np.ndarray
    r: np.ndarray
    cmue1: np.ndarray
    cmue2: np.ndarray
    gam: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.Pb = make_column_array(nlev, n_cols=n_cols)
        self.epsb = make_column_array(nlev, n_cols=n_cols)
        self.as_ = make_column_array(nlev, n_cols=n_cols)
        self.an = make_column_array(nlev, n_cols=n_cols)
        self.at = make_column_array(nlev, n_cols=n_cols)
        self.r = make_column_array(nlev, n_cols=n_cols)
        self.cmue1 = make_column_array(nlev, n_cols=n_cols)
        self.cmue2 = make_column_array(nlev, n_cols=n_cols)
        self.gam = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_cmue_a(
    nlev: int,
    cm0: float,
    cc1: float,
    ct1: float,
    a1: float,
    a2: float,
    a3: float,
    a5: float,
    at1: float,
    at2: float,
    at3: float,
    at4: float,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    Pb: np.ndarray,
    epsb: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    r: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    gam: np.ndarray,
) -> None:
    r"""Update the exact weak-equilibrium stability functions (single column)."""

    cm3_inv = 1.0 / (cm0 * cm0 * cm0)

    xd0 = 36.0
    xd1 = 84.0 * a5 * at3
    xd2 = 9.0 * (at2 * at2 - at1 * at1)
    xd3 = -12.0 * (a2 * a2 - 3.0 * a3 * a3)
    xd4 = 12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2)
    xd5 = 12.0 * a5 * at3 * (a3 * a3 - a2 * a2)
    xd6 = 48.0 * a5 * a5 * at3 * at3
    xd7 = 3.0 * (a2 * a2 - 3.0 * a3 * a3) * (at1 * at1 - at2 * at2)

    xn0 = 36.0 * a1
    xn1 = -12.0 * a5 * at3 * (at1 + at2)
    xn2 = 8.0 * a5 * at3 * (6.0 * a1 - a2 - 3.0 * a3)
    xn3 = 9.0 * a1 * (at2 * at2 - at1 * at1)
    xn4 = 36.0 * a5 * at4 * (at1 + at2)
    xn5 = 24.0 * a5 * at4 * (a2 + 3.0 * a3)

    xt0 = 12.0 * at3
    xt1 = 12.0 * a5 * at3 * at3
    xt2 = 9.0 * a1 * at3 * (at1 - at2)
    xt3 = (6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2 * a2 - 3.0 * a3 * a3)) * at3

    xg0 = 36.0 * at4
    xg1 = 36.0 * a5 * at3 * at4
    xg2 = -12.0 * at4 * (a2 * a2 - 3.0 * a3 * a3)

    for i in range(1, nlev):
        pe = (P[i] + Px[i] + B[i]) / eps[i]
        pbeb = Pb[i] / epsb[i]
        r_i = 1.0 / r[i]

        n_val = pe + 0.5 * cc1 - 1.0
        nt_val = 0.5 * (pe - 1.0) + ct1 + 0.5 * r_i * (pbeb - 1.0)
        nt_val = (pe - 1.0) + ct1

        n_sq = n_val * n_val
        n_cube = n_sq * n_val
        nt_sq = nt_val * nt_val

        d0 = xd0 * n_cube * nt_sq
        d1 = xd1 * n_sq * nt_val
        d2 = xd2 * n_cube + xd3 * n_val * nt_sq
        d3 = xd4 * n_val + xd5 * nt_val
        d4 = xd6 * n_val
        d5 = xd7 * n_val

        n0 = xn0 * n_sq * nt_sq
        n1 = xn1 * n_sq + xn2 * n_val * nt_val
        n2 = xn3 * n_sq
        n3 = xn4 * n_sq + xn5 * n_val * nt_val

        nt0 = xt0 * n_cube * nt_val
        nt1 = xt1 * n_sq
        nt2 = xt2 * n_sq + xt3 * n_val * nt_val

        gam0 = xg0 * n_cube * nt_val
        gam1 = xg1 * n_sq
        gam2 = xg2 * n_val * nt_val

        d_cm = (
            d0
            + d1 * an[i]
            + d2 * as_[i]
            + d3 * an[i] * as_[i]
            + d4 * an[i] * an[i]
            + d5 * as_[i] * as_[i]
        )
        n_cm = n0 + n1 * an[i] + n2 * as_[i] + n3 * at[i]
        n_cmp = nt0 + nt1 * an[i] + nt2 * as_[i]
        n_gam = (gam0 + gam1 * an[i] + gam2 * as_[i]) * at[i]

        cmue1[i] = cm3_inv * n_cm / d_cm
        cmue2[i] = cm3_inv * n_cmp / d_cm
        gam[i] = n_gam / d_cm


@numba.njit(parallel=True, cache=True)
def step_cmue_a(
    batch_size: int,
    nlev: int,
    cm0: float,
    cc1: float,
    ct1: float,
    a1: float,
    a2: float,
    a3: float,
    a5: float,
    at1: float,
    at2: float,
    at3: float,
    at4: float,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    Pb: np.ndarray,
    epsb: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    r: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    gam: np.ndarray,
) -> None:
    r"""Update the exact weak-equilibrium stability functions (batch)."""
    for b in numba.prange(batch_size):
        _step_cmue_a(
            nlev, cm0, cc1, ct1, a1, a2, a3, a5, at1, at2, at3, at4,
            eps[b], P[b], B[b], Px[b], Pb[b], epsb[b],
            as_[b], an[b], at[b], r[b], cmue1[b], cmue2[b], gam[b],
        )
