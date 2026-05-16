# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Some algebraic length-scale relations \label{sec:algebraiclength}
!
! !INTERFACE:
!   subroutine algebraiclength(method,nlev,z0b,z0s,depth,h,NN)
!
! !DESCRIPTION:
! This subroutine computes the vertical profile of the turbulent
! scale $l$ from different types of analytical expressions. These
! range from simple geometrical forms to more complicated expressions
! taking into account the effects of stratification and shear. The
! users can select their method in the input file {\tt gotm.yaml}.
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array
from pygotm.turbulence.turbulence import Blackadar as _BLACKADAR
from pygotm.turbulence.turbulence import Parabolic as _PARABOLIC
from pygotm.turbulence.turbulence import Robert_Ouellet as _ROBERT_OUELLET
from pygotm.turbulence.turbulence import Triangular as _TRIANGULAR
from pygotm.turbulence.turbulence import Xing_Davies as _XING_DAVIES

__all__ = [
    "AlgebraicLengthWorkspace",
    "step_algebraiclength",
]

_BETA_XING: float = 2.0
_GAMMA_BLACKADAR: float = 0.2


class AlgebraicLengthWorkspace(ColumnWorkspace):
    """Workspace arrays for algebraic mixing-length profiles."""

    tke: np.ndarray
    eps: np.ndarray
    L: np.ndarray
    h: np.ndarray
    NN: np.ndarray
    depth: np.ndarray
    z0b: np.ndarray
    z0s: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.L = make_column_array(nlev, n_cols=n_cols)
        self.h = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.depth = make_column_array(nlev, n_cols=n_cols)
        self.z0b = make_column_array(nlev, n_cols=n_cols)
        self.z0s = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_algebraiclength(
    method: int,
    nlev: int,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: int,
    eps_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    L: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    depth: float,
    z0b: float,
    z0s: float,
) -> None:
    r"""Update algebraic mixing-length profiles and dissipation (single column)."""

    db = 0.0
    ds = 0.0

    if method == _PARABOLIC:
        for i in range(1, nlev):
            db = db + h[i]
            ds = depth - db
            L[i] = kappa * (ds + z0s) * (db + z0b) / (ds + db + z0b + z0s)
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s

    elif method == _TRIANGULAR:
        for i in range(1, nlev):
            db = db + h[i]
            ds = depth - db
            L[i] = kappa * min(ds + z0s, db + z0b)
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s

    elif method == _XING_DAVIES:
        for i in range(1, nlev):
            db = db + h[i]
            ds = depth - db
            db_xing = db * math.exp(-_BETA_XING * db / depth)
            L[i] = kappa * (ds + z0s) * (db_xing + z0b) / (ds + db_xing + z0s + z0b)
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s

    elif method == _ROBERT_OUELLET:
        for i in range(1, nlev):
            db = db + h[i]
            ds = depth - db
            L[i] = kappa * (db + z0b) * math.sqrt(1.0 - (db - z0s) / depth)
        L[0] = kappa * z0b
        L[nlev] = kappa * (depth + z0b) * math.sqrt(z0s / depth)

    elif method == _BLACKADAR:
        int_qz = 0.0
        int_q = 0.0

        for i in range(1, nlev):
            db = db + h[i]
            root_tke = math.sqrt(tke[i])
            int_qz = int_qz + root_tke * (db + z0b) * h[i]
            int_q = int_q + root_tke * h[i]

        la = _GAMMA_BLACKADAR * int_qz / int_q

        db = 0.0
        for i in range(1, nlev):
            db = db + h[i]
            ds = depth - db
            L[i] = 1.0 / (
                1.0 / (kappa * (ds + z0s)) + 1.0 / (kappa * (db + z0b)) + 1.0 / la
            )

        L[0] = kappa * z0b
        L[nlev] = kappa * z0s

    for i in range(nlev + 1):
        if NN[i] > 0.0 and length_lim != 0:
            lcrit = math.sqrt(2.0 * galp * galp * tke[i] / NN[i])
            if L[i] > lcrit:
                L[i] = lcrit

        tke32 = math.sqrt(tke[i] * tke[i] * tke[i])
        eps[i] = cde * tke32 / L[i]

        if eps[i] < eps_min:
            eps[i] = eps_min
            L[i] = cde * tke32 / eps_min


@numba.njit(parallel=True, cache=True)
def step_algebraiclength(
    batch_size: int,
    method: int,
    nlev: int,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: int,
    eps_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    L: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    depth: np.ndarray,
    z0b: np.ndarray,
    z0s: np.ndarray,
) -> None:
    r"""Update algebraic mixing-length profiles and dissipation (batch)."""
    for b in numba.prange(batch_size):
        _step_algebraiclength(
            method,
            nlev,
            kappa,
            cde,
            galp,
            length_lim,
            eps_min,
            tke[b],
            eps[b],
            L[b],
            h[b],
            NN[b],
            depth[b, 0],
            z0b[b, 0],
            z0s[b, 0],
        )
