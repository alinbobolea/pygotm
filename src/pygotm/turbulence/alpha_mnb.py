# ruff: noqa: E501
"""
Dimensionless stability parameters :math:`\\alpha_M`, :math:`\\alpha_N`, :math:`\\alpha_b`.

Implements GOTM Section 4.7.21 — updates the non-dimensional shear and
buoyancy parameters used by the algebraic stress models (Eq. 149):

.. math::

   \\alpha_M = \\frac{k^2}{\\varepsilon^2} M^2, \\quad
   \\alpha_N = \\frac{k^2}{\\varepsilon^2} N^2, \\quad
   \\alpha_b = \\frac{k}{\\varepsilon^2} \\langle b'^2 \\rangle \\comma

where :math:`k/\\varepsilon` is the turbulence time scale :math:`\\tau`.
In the code, ``at`` stores :math:`\\alpha_b = (k/\\varepsilon)(k_b/\\varepsilon)`.

When Stokes drift is active, two additional parameters are computed:

.. math::

   \\alpha_v = \\frac{k^2}{\\varepsilon^2} \\cdot \\mathrm{SSCSTK}, \\quad
   \\alpha_w = \\frac{k^2}{\\varepsilon^2} \\cdot \\mathrm{SSSTK} \\comma

stored in ``av`` and ``aw`` respectively.

All parameters are floored at :math:`10^{-10}` to prevent numerical
singularities in the stability-function evaluations.

Author (original Fortran): Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "AlphaMNBWorkspace",
    "step_alpha_mnb",
    "step_alpha_mnb_single",
]

_MIN_NONNEGATIVE_ALPHA: float = float(np.float32(1.0e-10))


class AlphaMNBWorkspace(ColumnWorkspace):
    """Workspace arrays for alpha_M, alpha_N, alpha_B, and Stokes terms."""

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.kb = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)
        self.SSCSTK = make_column_array(nlev, n_cols=n_cols)
        self.SSSTK = make_column_array(nlev, n_cols=n_cols)
        self.as_ = make_column_array(nlev, n_cols=n_cols)
        self.an = make_column_array(nlev, n_cols=n_cols)
        self.at = make_column_array(nlev, n_cols=n_cols)
        self.av = make_column_array(nlev, n_cols=n_cols)
        self.aw = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_alpha_mnb(
    nlev: int,
    has_sscstk: int,
    has_ssstk: int,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
) -> None:
    r"""Update alpha_M, alpha_N, alpha_b, and optional Stokes terms (single column)."""
    for i in range(nlev + 1):
        tau2 = tke[i] * tke[i] / (eps[i] * eps[i])
        as_[i] = tau2 * SS[i]
        an[i] = tau2 * NN[i]
        at[i] = tke[i] / eps[i] * kb[i] / eps[i]

        if as_[i] < _MIN_NONNEGATIVE_ALPHA:
            as_[i] = _MIN_NONNEGATIVE_ALPHA
        if at[i] < _MIN_NONNEGATIVE_ALPHA:
            at[i] = _MIN_NONNEGATIVE_ALPHA

    if has_sscstk != 0 and has_ssstk != 0:
        for i in range(nlev + 1):
            tau2 = tke[i] * tke[i] / (eps[i] * eps[i])
            av[i] = tau2 * SSCSTK[i]
            aw[i] = tau2 * SSSTK[i]
            if aw[i] < _MIN_NONNEGATIVE_ALPHA:
                aw[i] = _MIN_NONNEGATIVE_ALPHA


@numba.njit(parallel=True, cache=True)
def step_alpha_mnb(
    batch_size: int,
    nlev: int,
    has_sscstk: int,
    has_ssstk: int,
    tke: np.ndarray,
    eps: np.ndarray,
    kb: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
) -> None:
    r"""Update alpha_M, alpha_N, alpha_b, and optional Stokes terms (batch)."""
    for b in numba.prange(batch_size):
        _step_alpha_mnb(
            nlev,
            has_sscstk,
            has_ssstk,
            tke[b],
            eps[b],
            kb[b],
            NN[b],
            SS[b],
            SSCSTK[b],
            SSSTK[b],
            as_[b],
            an[b],
            at[b],
            av[b],
            aw[b],
        )


step_alpha_mnb_single = _step_alpha_mnb
