# ruff: noqa: E501
"""
Algebraic velocity-variance components :math:`\\langle u'^2\\rangle`,
:math:`\\langle v'^2\\rangle`, :math:`\\langle w'^2\\rangle`.

Implements GOTM Section 4.7.33 (variances.F90) — derives the three diagonal
Reynolds-stress components from algebraic expressions (Eq. 180) using the
solution of the second-moment closure Eqs. 70 and 74:

.. math::

   \\frac{\\langle u'^2 \\rangle}{k} &=
     \\frac{2}{3} + \\frac{1}{\\mathcal{N}\\varepsilon}
     \\left[
       \\left(\\frac{a_2}{3} + a_3\\right) \\nu_t
       \\left(\\frac{\\partial U}{\\partial z}\\right)^2
       - \\frac{2 a_2}{3} \\nu_t
       \\left(\\frac{\\partial V}{\\partial z}\\right)^2
       - \\frac{4}{3} a_5 G
     \\right] \\comma \\\\
   \\frac{\\langle v'^2 \\rangle}{k} &=
     \\frac{2}{3} + \\frac{1}{\\mathcal{N}\\varepsilon}
     \\left[
       \\left(\\frac{a_2}{3} + a_3\\right) \\nu_t
       \\left(\\frac{\\partial V}{\\partial z}\\right)^2
       - \\frac{2 a_2}{3} \\nu_t
       \\left(\\frac{\\partial U}{\\partial z}\\right)^2
       - \\frac{4}{3} a_5 G
     \\right] \\comma \\\\
   \\frac{\\langle w'^2 \\rangle}{k} &=
     \\frac{2}{3} + \\frac{1}{\\mathcal{N}\\varepsilon}
     \\left[
       \\left(\\frac{a_2}{3} - a_3\\right)(P + P_x)
       + \\frac{8}{3} a_5 G
     \\right] \\comma

where :math:`\\mathcal{N} = c_1/2` is the rapid-return-to-isotropy parameter.
In the code: ``n_value = 0.5 * cc1``.  The directional shear components
``SSU`` and ``SSV`` enter the :math:`u'` and :math:`v'` variance expressions.

The variances fall back to the isotropic value :math:`2k/3` when
:math:`\\varepsilon \\le 0` or :math:`\\mathcal{N} \\le 0`.

Author (original Fortran): Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = ["VariancesWorkspace", "step_variances", "step_variances_single"]

_TWO_THIRDS = 2.0 / 3.0
_FOUR_THIRDS = 4.0 / 3.0
_EIGHT_THIRDS = 8.0 / 3.0


class VariancesWorkspace(ColumnWorkspace):
    """Workspace arrays for algebraic velocity variances."""

    tke: np.ndarray
    eps: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    num: np.ndarray
    SSU: np.ndarray
    SSV: np.ndarray
    uu: np.ndarray
    vv: np.ndarray
    ww: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.SSU = make_column_array(nlev, n_cols=n_cols)
        self.SSV = make_column_array(nlev, n_cols=n_cols)
        self.uu = make_column_array(nlev, n_cols=n_cols)
        self.vv = make_column_array(nlev, n_cols=n_cols)
        self.ww = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _step_variances(
    nlev: int,
    cc1: float,
    ct1: float,
    a2: float,
    a3: float,
    a5: float,
    tke: np.ndarray,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    num: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
) -> None:
    r"""Update the algebraic velocity variances (single column)."""
    n_value = 0.5 * cc1
    fac3 = a2 / 3.0 + a3
    fac4 = a2 / 3.0 - a3
    fac5 = _TWO_THIRDS * a2

    for i in range(nlev + 1):
        if eps[i] <= 0.0 or n_value <= 0.0:
            uu[i] = _TWO_THIRDS * tke[i]
            vv[i] = _TWO_THIRDS * tke[i]
            ww[i] = _TWO_THIRDS * tke[i]
            continue

        fac2 = 1.0 / (n_value * eps[i])

        uu[i] = tke[i] * (
            _TWO_THIRDS
            + fac2
            * (
                fac3 * num[i] * SSU[i]
                - fac5 * num[i] * SSV[i]
                - _FOUR_THIRDS * a5 * B[i]
            )
        )

        vv[i] = tke[i] * (
            _TWO_THIRDS
            + fac2
            * (
                fac3 * num[i] * SSV[i]
                - fac5 * num[i] * SSU[i]
                - _FOUR_THIRDS * a5 * B[i]
            )
        )

        ww[i] = tke[i] * (
            _TWO_THIRDS
            + fac2
            * (
                fac4 * (P[i] + Px[i])
                + _EIGHT_THIRDS * a5 * B[i]
            )
        )


@numba.njit(parallel=True, cache=True)
def step_variances(
    batch_size: int,
    nlev: int,
    cc1: float,
    ct1: float,
    a2: float,
    a3: float,
    a5: float,
    tke: np.ndarray,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    num: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    uu: np.ndarray,
    vv: np.ndarray,
    ww: np.ndarray,
) -> None:
    r"""Update the algebraic velocity variances (batch)."""
    for b in numba.prange(batch_size):
        _step_variances(
            nlev, cc1, ct1, a2, a3, a5,
            tke[b], eps[b], P[b], B[b], Px[b], num[b], SSU[b], SSV[b],
            uu[b], vv[b], ww[b],
        )


step_variances_single = _step_variances
