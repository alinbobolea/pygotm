# ruff: noqa: E501
"""
Local weak-equilibrium stability functions :math:`c_\\mu` and :math:`c_\\mu'`.

Implements GOTM Section 4.7.38 (cmue_c.F90) — computes the stability functions
under the **local weak-equilibrium** assumption (Canuto et al. 2001; Cheng et al.
2002).  Two simplifications are applied relative to the non-local variant:

1. The buoyancy variance :math:`k_b = \\varepsilon_b` (local production–dissipation
   balance) eliminates the :math:`\\bar{T}` dependence from Eq. 71.
2. The :math:`\\Gamma`-term in Eq. 74 drops out, so the solution is characterised
   by :math:`c_\\mu` and :math:`c_\\mu'` only.

The result gives the eddy viscosity and diffusivity:

.. math::

   \\nu_t = c_\\mu \\frac{k^2}{\\varepsilon}, \\quad
   \\kappa_t = c_\\mu' \\frac{k^2}{\\varepsilon} \\point

The denominator and numerators are polynomials (Eqs. 191–194):

.. math::

   D &= d_0 + d_1 \\bar{N}^2 + d_2 \\bar{S}^2
          + d_3 \\bar{N}^2 \\bar{S}^2 + d_4 \\bar{N}^4 + d_5 \\bar{S}^4 \\comma \\\\
   N_n &= n_0 + n_1 \\bar{N}^2 + n_2 \\bar{S}^2 \\comma \\\\
   N_b &= n_{b0} + n_{b1} \\bar{N}^2 + n_{b2} \\bar{S}^2 \\point

Here :math:`\\bar{S}^2 = \\alpha_M` and :math:`\\bar{N}^2 = \\alpha_N`.  The
coefficients :math:`d_i`, :math:`n_i`, :math:`n_{bi}` are derived from the
model constants :math:`a_1, a_2, a_3, a_5, a_{t1}, a_{t2}, a_{t3}, a_{t5}` and
:math:`\\mathcal{N} = c_1/2`, :math:`\\mathcal{N}_b = c_{t1}` (see code for
explicit expressions).

A clipping on :math:`\\alpha_N` is applied (``anLimitFact = 0.5``), and a
corresponding limiter on :math:`\\alpha_M` prevents negative stability
functions in strongly sheared flows.

Author (original Fortran): Lars Umlauf.
"""

import math

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "CmueCWorkspace",
    "step_cmue_c",
    "step_cmue_c_single",
]

_AS_LIMIT_FACT: float = 1.0
_AN_LIMIT_FACT: float = 0.5


class CmueCWorkspace(ColumnWorkspace):
    """Workspace arrays for local weak-equilibrium stability functions."""

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
def _step_cmue_c(
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
    r"""Update the local weak-equilibrium stability functions (single column)."""

    n_val = 0.5 * cc1
    nt_val = ct1

    n_sq = n_val * n_val
    n_cube = n_sq * n_val
    nt_sq = nt_val * nt_val

    d0 = 36.0 * n_cube * nt_sq
    d1 = 84.0 * a5 * at3 * n_sq * nt_val + 36.0 * at5 * n_cube * nt_val
    d2 = (
        9.0 * (at2 * at2 - at1 * at1) * n_cube
        - 12.0 * (a2 * a2 - 3.0 * a3 * a3) * n_val * nt_sq
    )
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
    nt2 = (
        9.0 * a1 * at3 * (at1 - at2) * n_sq
        + (6.0 * a1 * (a2 - 3.0 * a3) - 4.0 * (a2 * a2 - 3.0 * a3 * a3))
        * at3
        * n_val
        * nt_val
    )

    cm3_inv = 1.0 / (cm0 * cm0 * cm0)

    an_min_num = -(d1 + nt0) + math.sqrt(
        (d1 + nt0) * (d1 + nt0) - 4.0 * d0 * (d4 + nt1)
    )
    an_min_den = 2.0 * (d4 + nt1)
    an_min = an_min_num / an_min_den

    for i in range(1, nlev):
        if an[i] < _AN_LIMIT_FACT * an_min:
            an[i] = _AN_LIMIT_FACT * an_min

        as_max_num = (
            d0 * n0
            + (d0 * n1 + d1 * n0) * an[i]
            + (d1 * n1 + d4 * n0) * an[i] * an[i]
            + d4 * n1 * an[i] * an[i] * an[i]
        )
        as_max_den = d2 * n0 + (d2 * n1 + d3 * n0) * an[i] + d3 * n1 * an[i] * an[i]
        as_max = as_max_num / as_max_den
        if as_[i] > _AS_LIMIT_FACT * as_max:
            as_[i] = _AS_LIMIT_FACT * as_max

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
def step_cmue_c(
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
    r"""Update the local weak-equilibrium stability functions (batch)."""
    for b in numba.prange(batch_size):
        _step_cmue_c(
            nlev,
            cm0,
            cc1,
            ct1,
            a1,
            a2,
            a3,
            a5,
            at1,
            at2,
            at3,
            at5,
            as_[b],
            an[b],
            cmue1[b],
            cmue2[b],
        )


step_cmue_c_single = _step_cmue_c
