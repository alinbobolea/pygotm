# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!
! !ROUTINE: The Langmuir turbulence quasi-equilibrium stability functions after Harcourt (2015)\label{sec:cmueDH15}
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
    "CmueDH15Workspace",
    "step_cmue_d_h15",
]

_SMALL: float = 1.0e-8
_SQRT2: float = 1.4142135623730951
_H15_VON_KARMAN: float = 0.41

_MY_A1: float = 0.92
_MY_A2: float = 0.74
_MY_B1: float = 16.6
_MY_B2: float = 10.1
_MY_C1: float = 0.08
_MY_C2: float = 0.7
_MY_C3: float = 0.2
_H15_GHMIN: float = -0.28
_H15_GHOFF: float = 0.003
_H15_GVOFF: float = 0.006
_H15_SXMAX: float = 2.12

_H15_SHN0: float = _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
_H15_SHNH: float = -9.0 * _MY_A1 * _MY_A2 * (_MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1))
_H15_SHNS: float = (
    9.0 * _MY_A1 * _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1) * (2.0 * _MY_A1 + _MY_A2)
)
_H15_SHNV: float = (
    9.0
    * _MY_A1
    * _MY_A2
    * (
        _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1 - 3.0 * _MY_C1)
        - 2.0 * _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1 + 3.0 * _MY_C1)
    )
)
_H15_SHDAH: float = -9.0 * _MY_A1 * _MY_A2
_H15_SHDAV: float = -36.0 * _MY_A1 * _MY_A1
_H15_SHDBH: float = -3.0 * _MY_A2 * (6.0 * _MY_A1 + _MY_B2 * (1.0 - _MY_C3))
_H15_SHDV: float = -9.0 * _MY_A2 * _MY_A2 * (1.0 - _MY_C2)
_H15_SHDVH: float = (
    -162.0 * _MY_A1 * _MY_A1 * _MY_A2 * (2.0 * _MY_A1 + (2.0 - _MY_C2) * _MY_A2)
)
_H15_SHDVV: float = 324.0 * _MY_A1 * _MY_A1 * _MY_A2 * _MY_A2 * (1.0 - _MY_C2)
_H15_SSN0: float = _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
_H15_SSDH: float = -9.0 * _MY_A1 * _MY_A2
_H15_SSDV: float = -9.0 * _MY_A1 * _MY_A1
_H15_SMN0: float = _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1 - 3.0 * _MY_C1)
_H15_SMNHSH: float = 9.0 * _MY_A1 * (2.0 * _MY_A1 + _MY_A2 * (1.0 - _MY_C2))
_H15_SMNSSS: float = 27.0 * _MY_A1 * _MY_A1
_H15_SMDH: float = -9.0 * _MY_A1 * _MY_A2
_H15_SMDV: float = -36.0 * _MY_A1 * _MY_A1
_H15_SCALE: float = 4.0 / (_MY_B1 * _MY_B1)


class CmueDH15Workspace(ColumnWorkspace):
    """Workspace arrays for Harcourt (2015) Langmuir stability functions."""

    as_: np.ndarray
    an: np.ndarray
    av: np.ndarray
    aw: np.ndarray
    SPF: np.ndarray
    cmue1: np.ndarray
    cmue2: np.ndarray
    cmue3: np.ndarray
    sq_var: np.ndarray
    sl_var: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.as_ = make_column_array(nlev, n_cols=n_cols)
        self.an = make_column_array(nlev, n_cols=n_cols)
        self.av = make_column_array(nlev, n_cols=n_cols)
        self.aw = make_column_array(nlev, n_cols=n_cols)
        self.SPF = make_column_array(nlev, n_cols=n_cols)
        self.cmue1 = make_column_array(nlev, n_cols=n_cols)
        self.cmue2 = make_column_array(nlev, n_cols=n_cols)
        self.cmue3 = make_column_array(nlev, n_cols=n_cols)
        self.sq_var = make_column_array(nlev, n_cols=n_cols)
        self.sl_var = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_cmue_d_h15(
    nlev: int,
    length_lim: int,
    sq: float,
    sl: float,
    as_: np.ndarray,
    an: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    SPF: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
    sq_var: np.ndarray,
    sl_var: np.ndarray,
) -> None:
    r"""Update Harcourt (2015) quasi-equilibrium Langmuir stability functions (single column)."""

    for i in range(1, nlev):
        gh = -_H15_SCALE * an[i]
        gm = _H15_SCALE * as_[i]
        gv = _H15_SCALE * av[i]
        gs = _H15_SCALE * aw[i]
        sh = _SMALL
        ss = _SMALL
        sm = _SMALL

        if length_lim == 0:
            tmp1 = 1.0
            tmp2 = _H15_GHMIN / min(_H15_GHMIN, gh)
            tmp1 = min(tmp1, tmp2)

            if tmp1 < 1.0:
                gh = gh * tmp1
                gv = gv * tmp1
                gs = gs * tmp1

        tmp0 = 2.0

        if gv > 0.0:
            tmp1 = (_H15_SHDAH + _H15_SHDBH) * gh + (_H15_SHDAV + _H15_SHDV) * gv
            tmp1 = tmp1 + (
                (_H15_SHDAH * _H15_GHOFF + _H15_SHDAV * _H15_GVOFF) * (_H15_SHDBH * gh)
                + (_H15_SHDVH * _H15_GHOFF + _H15_SHDVV * _H15_GVOFF) * gv
            )
            tmp1 = tmp1 + (
                (_H15_SHDAH * gh + _H15_SHDAV * gv) * (_H15_SHDBH * _H15_GHOFF)
                + (_H15_SHDVH * gh + _H15_SHDVV * gv) * _H15_GVOFF
            )

            tmp2 = (_H15_SHDAH * gh + _H15_SHDAV * gv) * (_H15_SHDBH * gh) + (
                _H15_SHDVH * gh + _H15_SHDVV * gv
            ) * gv

            tmp4 = (
                1.0
                + (_H15_SHDAH + _H15_SHDBH) * _H15_GHOFF
                + (_H15_SHDAV + _H15_SHDV) * _H15_GVOFF
                + (_H15_SHDAH * _H15_GHOFF + _H15_SHDAV * _H15_GVOFF)
                * (_H15_SHDBH * _H15_GHOFF)
                + (_H15_SHDVH * _H15_GHOFF + _H15_SHDVV * _H15_GVOFF) * _H15_GVOFF
            )

            tmp3 = tmp1 * tmp1 - 4.0 * tmp2 * tmp4

            if tmp3 >= 0.0 and tmp2 < 0.0:
                tmp3 = (-tmp1 + math.sqrt(tmp3)) / (2.0 * tmp2)
            elif tmp3 >= 0.0 and tmp3 > 0.0:
                tmp3 = (-tmp1 - math.sqrt(tmp3)) / (2.0 * tmp2)
            else:
                tmp3 = 2.0

            if tmp3 > 0.0 and tmp3 < 1.0:
                tmp0 = min(tmp0, tmp3)

        gv = gv * SPF[i]
        gs = gs * SPF[i] * SPF[i]

        if gh > 0.0:
            tmp1 = 2.0 * (_H15_SHDAH + _H15_SHDBH) * gh + (_H15_SHDAV + _H15_SHDV) * gv
            tmp2 = (2.0 * _H15_SHDAH * gh + _H15_SHDAV * gv) * (
                2.0 * _H15_SHDBH * gh
            ) + (2.0 * _H15_SHDVH * gh + _H15_SHDVV * gv) * gv
            tmp4 = 1.0
            tmp3 = tmp1 * tmp1 - 4.0 * tmp2 * tmp4

            if tmp3 >= 0.0 and tmp2 < 0.0:
                tmp3 = (-tmp1 + math.sqrt(tmp3)) / (2.0 * tmp2)
            elif tmp3 >= 0.0 and tmp3 > 0.0:
                tmp3 = (-tmp1 - math.sqrt(tmp3)) / (2.0 * tmp2)
            else:
                tmp3 = 2.0

            if tmp3 > 0.0 and tmp3 < 1.0:
                tmp0 = min(tmp0, tmp3)

        if tmp0 > 0.0 and tmp0 < 1.0:
            gh = tmp0 * gh
            gm = tmp0 * gm
            gv = tmp0 * gv
            gs = tmp0 * gs

        tmp1 = _H15_SHN0 + _H15_SHNH * gh + _H15_SHNS * gs + _H15_SHNV * gv
        if tmp1 < 0.0:
            sh = _SMALL
        else:
            tmp2 = (1.0 + _H15_SHDAH * gh + _H15_SHDAV * gv) * (
                1.0 + _H15_SHDBH * gh
            ) + (_H15_SHDV + _H15_SHDVH * gh + _H15_SHDVV * gv) * gv
            if tmp2 <= 0.0:
                sh = _H15_SXMAX
            else:
                sh = min(max(_SMALL, tmp1 / tmp2), _H15_SXMAX)

        tmp2 = 1.0 + _H15_SSDH * gh + _H15_SSDV * gv
        if tmp2 < 0.0:
            ss = _H15_SXMAX
        else:
            ss = min(max(_SMALL, _H15_SSN0 / tmp2), _H15_SXMAX)

        tmp1 = _H15_SMN0 + _H15_SMNHSH * gh * sh + _H15_SMNSSS * gs * ss
        if tmp1 < _SMALL and tmp1 >= 0.0:
            gh = gh + _SMALL
            gv = gv + _SMALL
            tmp1 = _H15_SMN0 + _H15_SMNHSH * gh * sh + _H15_SMNSSS * gs * ss
        elif tmp1 > -_SMALL and tmp1 < 0.0:
            gh = gh - _SMALL
            gv = gv - _SMALL
            tmp1 = _H15_SMN0 + _H15_SMNHSH * gh * sh + _H15_SMNSSS * gs * ss

        if tmp1 < 0.0:
            sm = _SMALL
        else:
            tmp2 = 1.0 + _H15_SMDH * gh + _H15_SMDV * gv
            if tmp2 <= 0.0:
                sm = _H15_SXMAX
            else:
                sm = min(max(_SMALL, tmp1 / tmp2), _H15_SXMAX)

        ss = ss * SPF[i]

        cmue1[i] = _SQRT2 * sm
        cmue2[i] = _SQRT2 * sh
        cmue3[i] = _SQRT2 * ss

        sq_var[i] = math.sqrt(sq * sq + (_H15_VON_KARMAN * sh) ** 2)
        sl_var[i] = math.sqrt(sl * sl + (_H15_VON_KARMAN * sh) ** 2)


@numba.njit(parallel=True, cache=True)
def step_cmue_d_h15(
    batch_size: int,
    nlev: int,
    length_lim: int,
    sq: float,
    sl: float,
    as_: np.ndarray,
    an: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    SPF: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
    sq_var: np.ndarray,
    sl_var: np.ndarray,
) -> None:
    r"""Update Harcourt (2015) quasi-equilibrium Langmuir stability functions (batch)."""
    for b in numba.prange(batch_size):
        _step_cmue_d_h15(
            nlev,
            length_lim,
            sq,
            sl,
            as_[b],
            an[b],
            av[b],
            aw[b],
            SPF[b],
            cmue1[b],
            cmue2[b],
            cmue3[b],
            sq_var[b],
            sl_var[b],
        )
