# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The quasi-equilibrium stability functions \label{sec:cmueD}
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
    "CmueDWorkspace",
    "step_cmue_d",
]

_AN_LIMIT_FACT: float = 0.5
_SMALL: float = 1.0e-10


class CmueDWorkspace(ColumnWorkspace):
    """Workspace arrays for quasi-equilibrium stability functions."""

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
def _step_cmue_d(
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
    at5: float,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
) -> None:
    r"""Update the quasi-equilibrium stability functions (single column)."""

    n_val = 0.5 * cc1
    nt_val = ct1

    n_sq = n_val * n_val
    n_cube = n_sq * n_val
    nt_sq = nt_val * nt_val

    d0 = 36.0 * n_cube * nt_sq
    d1 = 84.0 * a5 * at3 * n_sq * nt_val + 36.0 * at5 * n_cube * nt_val
    d2 = 9.0 * (at2 * at2 - at1 * at1) * n_cube - 12.0 * (
        a2 * a2 - 3.0 * a3 * a3
    ) * n_val * nt_sq
    d3 = (
        12.0 * a5 * at3 * (a2 * at1 - 3.0 * a3 * at2) * n_val
        + 12.0 * a5 * at3 * (a3 * a3 - a2 * a2) * nt_val
        + 12.0 * at5 * (3.0 * a3 * a3 - a2 * a2) * n_val * nt_val
    )
    d4 = 48.0 * a5 * a5 * at3 * at3 * n_val + 36.0 * a5 * at3 * at5 * n_sq
    d5 = 3.0 * (a2 * a2 - 3.0 * a3 * a3) * (at1 * at1 - at2 * at2) * n_val

    n0 = 36.0 * a1 * n_sq * nt_sq
    n1 = (
        -12.0 * a5 * at3 * (at1 + at2) * n_sq
        + 8.0 * a5 * at3 * (6.0 * a1 - a2 - 3.0 * a3) * n_val * nt_val
        + 36.0 * a1 * at5 * n_sq * nt_val
    )
    n2 = 9.0 * a1 * (at2 * at2 - at1 * at1) * n_sq

    nt0 = 12.0 * at3 * n_cube * nt_val
    nt1 = 12.0 * a5 * at3 * at3 * n_sq
    nt2 = 9.0 * a1 * at3 * (at1 - at2) * n_sq + (
        6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2 * a2 - 3.0 * a3 * a3)
    ) * at3 * n_val * nt_val

    cm3_inv = 1.0 / (cm0 * cm0 * cm0)

    an_min_num = -(d1 + nt0) + math.sqrt((d1 + nt0) * (d1 + nt0) - 4.0 * d0 * (d4 + nt1))
    an_min_den = 2.0 * (d4 + nt1)
    an_min = an_min_num / an_min_den

    for i in range(1, nlev):
        if an[i] < _AN_LIMIT_FACT * an_min:
            an[i] = _AN_LIMIT_FACT * an_min

        tmp0 = -d0 - (d1 + nt0) * an[i] - (d4 + nt1) * an[i] * an[i]
        tmp1 = -d2 + n0 + (n1 - d3 - nt2) * an[i]

        if abs(n2 - d5) < _SMALL:
            as_[i] = -tmp0 / tmp1
        else:
            tmp2 = n2 - d5
            as_[i] = (-tmp1 + math.sqrt(tmp1 * tmp1 - 4.0 * tmp0 * tmp2)) / (
                2.0 * tmp2
            )

        d_cm = (
            d0
            + d1 * an[i]
            + d2 * as_[i]
            + d3 * an[i] * as_[i]
            + d4 * an[i] * an[i]
            + d5 * as_[i] * as_[i]
        )
        n_cm = n0 + n1 * an[i] + n2 * as_[i]
        n_cmp = nt0 + nt1 * an[i] + nt2 * as_[i]

        cmue1[i] = cm3_inv * n_cm / d_cm
        cmue2[i] = cm3_inv * n_cmp / d_cm


@numba.njit(parallel=True, cache=True)
def step_cmue_d(
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
    at5: float,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
) -> None:
    r"""Update the quasi-equilibrium stability functions (batch)."""
    for b in numba.prange(batch_size):
        _step_cmue_d(
            nlev, cm0, cc1, ct1, a1, a2, a3, a5, at1, at2, at3, at5,
            as_[b], an[b], cmue1[b], cmue2[b],
        )
