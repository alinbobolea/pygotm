# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic kb-equation \label{sec:kbeq}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array
from pygotm.util.diff_face import diff_face

__all__ = [
    "KBEquationWorkspace",
    "step_kbeq",
]

_CNPAR: float = 1.0
_ZERO: float = 0.0


class KBEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the dynamic buoyancy-variance equation."""

    kb: np.ndarray
    h: np.ndarray
    Pb: np.ndarray
    epsb: np.ndarray
    nuh: np.ndarray
    avh: np.ndarray
    l_sour: np.ndarray
    q_sour: np.ndarray
    au: np.ndarray
    bu: np.ndarray
    cu: np.ndarray
    du: np.ndarray
    ru: np.ndarray
    qu: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.kb = make_column_array(nlev, n_cols=n_cols)
        self.h = make_column_array(nlev, n_cols=n_cols)
        self.Pb = make_column_array(nlev, n_cols=n_cols)
        self.epsb = make_column_array(nlev, n_cols=n_cols)
        self.nuh = make_column_array(nlev, n_cols=n_cols)
        self.avh = make_column_array(nlev, n_cols=n_cols)
        self.l_sour = make_column_array(nlev, n_cols=n_cols)
        self.q_sour = make_column_array(nlev, n_cols=n_cols)
        self.au = make_column_array(nlev, n_cols=n_cols)
        self.bu = make_column_array(nlev, n_cols=n_cols)
        self.cu = make_column_array(nlev, n_cols=n_cols)
        self.du = make_column_array(nlev, n_cols=n_cols)
        self.ru = make_column_array(nlev, n_cols=n_cols)
        self.qu = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_kbeq(
    nlev: int,
    dt: float,
    kb_min: float,
    k_ubc: int,
    k_lbc: int,
    kb: np.ndarray,
    h: np.ndarray,
    Pb: np.ndarray,
    epsb: np.ndarray,
    nuh: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic buoyancy-variance equation (single column)."""
    for i in range(nlev + 1):
        avh[i] = nuh[i]
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        prod = Pb[i]
        diss = epsb[i]

        prod_pos = 0.5 * (prod + abs(prod))
        prod_neg = prod - prod_pos

        q_sour[i] = prod_pos
        l_sour[i] = (prod_neg - diss) / kb[i]

    diff_face(
        nlev,
        dt,
        _CNPAR,
        h,
        k_ubc,
        k_lbc,
        _ZERO,
        _ZERO,
        avh,
        l_sour,
        q_sour,
        kb,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )

    kb[nlev] = _ZERO
    kb[0] = _ZERO

    for i in range(nlev + 1):
        if kb[i] < kb_min:
            kb[i] = kb_min


@numba.njit(parallel=True, cache=True)
def step_kbeq(
    batch_size: int,
    nlev: int,
    dt: float,
    kb_min: float,
    k_ubc: int,
    k_lbc: int,
    kb: np.ndarray,
    h: np.ndarray,
    Pb: np.ndarray,
    epsb: np.ndarray,
    nuh: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic buoyancy-variance equation (batch)."""
    for b in numba.prange(batch_size):
        _step_kbeq(
            nlev,
            dt,
            kb_min,
            k_ubc,
            k_lbc,
            kb[b],
            h[b],
            Pb[b],
            epsb[b],
            nuh[b],
            avh[b],
            l_sour[b],
            q_sour[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
