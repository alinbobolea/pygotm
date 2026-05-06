# ruff: noqa: E501
"""
Algebraic closure for buoyancy-variance dissipation :math:`\\varepsilon_b`.

Implements GOTM Section 4.7.32 (epsbalgebraic.F90) — computes the molecular
rate of destruction of buoyancy variance :math:`\\varepsilon_b` under the
algebraic equilibrium assumption.

Assuming a constant time-scale ratio :math:`r = c_b` (Eq. 66), the
production–dissipation balance for :math:`k_b` gives (Eq. 179):

.. math::

   \\varepsilon_b = \\frac{1}{c_b} \\frac{\\varepsilon}{k} k_b \\point

In the code, :math:`c_b` is passed as ``ctt`` (scalar turbulence time-scale
ratio), so:

.. math::

   \\varepsilon_b = \\frac{1}{c_{tt}} \\frac{\\varepsilon}{k} k_b \\point

The result is clipped to ``epsb_min``.

Author (original Fortran): Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "EpsBAlgebraicWorkspace",
    "step_epsbalgebraic",
    "step_epsbalgebraic_single",
]


class EpsBAlgebraicWorkspace(ColumnWorkspace):
    """Workspace arrays for the algebraic buoyancy-destruction closure."""

    tke: np.ndarray
    eps: np.ndarray
    kb: np.ndarray
    epsb: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.kb = make_column_array(nlev, n_cols=n_cols)
        self.epsb = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_epsbalgebraic(
    nlev: int,
    ctt: float,
    epsb_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
) -> None:
    r"""Advance the algebraic buoyancy-destruction closure (single column)."""
    one_over_ctt = 1.0 / ctt

    for i in range(nlev + 1):
        epsb[i] = one_over_ctt * eps[i] / tke[i] * kb[i]

        if epsb[i] < epsb_min:
            epsb[i] = epsb_min


@numba.njit(parallel=True, cache=True)
def step_epsbalgebraic(
    batch_size: int,
    nlev: int,
    ctt: float,
    epsb_min: float,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    epsb: np.ndarray,
) -> None:
    r"""Advance the algebraic buoyancy-destruction closure (batch)."""
    for b in numba.prange(batch_size):
        _step_epsbalgebraic(nlev, ctt, epsb_min, tke[b], eps[b], kb[b], epsb[b])


step_epsbalgebraic_single = _step_epsbalgebraic
