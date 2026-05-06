# ruff: noqa: E501
"""
Algebraic closure for buoyancy variance :math:`k_b`.

Implements GOTM Section 4.7.30 (kbalgebraic.F90) — computes the buoyancy
variance :math:`k_b = \\langle b'^2 \\rangle / 2` under the algebraic
equilibrium assumption.

The equilibrium condition :math:`P_b = \\varepsilon_b` (Eq. 171) is assumed,
where :math:`P_b` is the buoyancy-variance production.  Using the definition
of the time-scale ratio :math:`r = c_b` (Eq. 66), this gives (Eq. 172):

.. math::

   k_b = \\frac{k_{b\\varepsilon}}{k\\varepsilon} P_b
       = r \\frac{k}{\\varepsilon} P_b
       = c_b \\frac{k}{\\varepsilon} P_b \\point

In the code, the coefficient :math:`c_b` is passed as ``ctt`` (the scalar
turbulence time-scale ratio).

The result is clipped to ``kb_min`` to prevent negative values.

Author (original Fortran): Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "KBAlgebraicWorkspace",
    "step_kbalgebraic",
    "step_kbalgebraic_single",
]


class KBAlgebraicWorkspace(ColumnWorkspace):
    """Workspace arrays for the algebraic buoyancy-variance closure."""

    tke: np.ndarray
    eps: np.ndarray
    kb: np.ndarray
    Pb: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.kb = make_column_array(nlev, n_cols=n_cols)
        self.Pb = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_kbalgebraic(
    nlev: int,
    ctt: float,
    kb_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    Pb: np.ndarray,
) -> None:
    r"""Advance the algebraic buoyancy-variance closure (single column)."""
    for i in range(nlev + 1):
        kb[i] = ctt * tke[i] / eps[i] * Pb[i]

        if kb[i] < kb_min:
            kb[i] = kb_min


@numba.njit(parallel=True, cache=True)
def step_kbalgebraic(
    batch_size: int,
    nlev: int,
    ctt: float,
    kb_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    Pb: np.ndarray,
) -> None:
    r"""Advance the algebraic buoyancy-variance closure (batch)."""
    for b in numba.prange(batch_size):
        _step_kbalgebraic(nlev, ctt, kb_min, tke[b], eps[b], kb[b], Pb[b])


step_kbalgebraic_single = _step_kbalgebraic
