# ruff: noqa: E501
"""
Vertical shear-frequency squared.

Implements GOTM Section 3.2.13.  The squared shear frequency is defined as
(Eq. 36):

.. math::

   M^2 = \\left( \\frac{\\partial U}{\\partial z} \\right)^2
       + \\left( \\frac{\\partial V}{\\partial z} \\right)^2 \\point

Energy-conserving discretisation
---------------------------------

The :math:`U`- and :math:`V`-contributions are discretised with the
energy-conserving scheme of Burchard (2002) (Eq. 37).  Defining the
averaged velocities :math:`\\tilde{U}_j = \\tfrac{1}{2}(\\hat{U}_j + U_j^n)`
(where :math:`\\hat{U}` is the updated and :math:`U^n` is the old velocity),
the discrete shear is

.. math::

   M^2_{U,j} = \\frac{1}{2} \\left[
     \\frac{(\\hat{U}_{j+1}-\\hat{U}_j)(\\hat{U}_{j+1}-U_j^n)}{h_{j+1/2}\\,h_j}
   + \\frac{(\\hat{U}_{j+1}-\\hat{U}_j)(U^n_{j+1}-\\hat{U}_j)}{h_{j+1/2}\\,h_{j+1}}
   \\right] \\comma

and analogously for :math:`M^2_V`.  The scheme guarantees that no spurious
mean kinetic energy is generated in the conversion to turbulent kinetic energy.

Stokes drift contributions
--------------------------

When Stokes drift shear profiles :math:`(\\partial u_s/\\partial z, \\partial v_s/\\partial z)` are supplied, two additional arrays are computed:

* ``SSCSTK``: cross-production term
  :math:`(\\partial u_s/\\partial z)(\\partial U/\\partial z) + (\\partial v_s/\\partial z)(\\partial V/\\partial z)`,
  used in the Stokes-drift shear production in the turbulence equations.

* ``SSSTK``: squared Stokes shear
  :math:`(\\partial u_s/\\partial z)^2 + (\\partial v_s/\\partial z)^2`,
  entering the extra Stokes production term.

The resulting :math:`M^2` drives shear production in all two-equation
turbulence closures.

Author (original Fortran): Lars Umlauf.
"""

import numba
import numpy as np

from pygotm.meanflow.meanflow import MeanflowState

__all__ = ["shear", "step_shear_single"]


@numba.njit(cache=True)
def step_shear_single(
    nlev: int,
    cnpar: float,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    uo: np.ndarray,
    vo: np.ndarray,
    dusdz: np.ndarray,
    dvsdz: np.ndarray,
    SS: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    SSCSTK: np.ndarray,
    SSSTK: np.ndarray,
) -> None:
    """Compute shear-frequency squared for one 0:nlev column."""

    for i in range(1, nlev):
        h_mid = 0.5 * (h[i + 1] + h[i])

        num_a_u = cnpar * (u[i + 1] - u[i]) * (u[i + 1] - uo[i]) + (1.0 - cnpar) * (
            uo[i + 1] - uo[i]
        ) * (uo[i + 1] - u[i])
        num_b_u = cnpar * (u[i + 1] - u[i]) * (uo[i + 1] - u[i]) + (1.0 - cnpar) * (
            uo[i + 1] - uo[i]
        ) * (u[i + 1] - uo[i])
        SSU[i] = 0.5 * (num_a_u / h_mid / h[i] + num_b_u / h_mid / h[i + 1])

        num_a_v = cnpar * (v[i + 1] - v[i]) * (v[i + 1] - vo[i]) + (1.0 - cnpar) * (
            vo[i + 1] - vo[i]
        ) * (vo[i + 1] - v[i])
        num_b_v = cnpar * (v[i + 1] - v[i]) * (vo[i + 1] - v[i]) + (1.0 - cnpar) * (
            vo[i + 1] - vo[i]
        ) * (v[i + 1] - vo[i])
        SSV[i] = 0.5 * (num_a_v / h_mid / h[i] + num_b_v / h_mid / h[i + 1])

        SS[i] = SSU[i] + SSV[i]
        SSCSTK[i] = (
            dusdz[i] * (u[i + 1] - u[i]) / h_mid + dvsdz[i] * (v[i + 1] - v[i]) / h_mid
        )
        SSSTK[i] = dusdz[i] * dusdz[i] + dvsdz[i] * dvsdz[i]

    SSU[0] = SSU[1]
    SSU[nlev] = SSU[nlev - 1]
    SSV[0] = SSV[1]
    SSV[nlev] = SSV[nlev - 1]
    SS[0] = SS[1]
    SS[nlev] = SS[nlev - 1]
    SSCSTK[0] = SSCSTK[1]
    SSCSTK[nlev] = SSCSTK[nlev - 1]
    SSSTK[0] = SSSTK[1]
    SSSTK[nlev] = SSSTK[nlev - 1]


def shear(
    state: MeanflowState,
    nlev: int,
    cnpar: float,
    dusdz: np.ndarray | None = None,
    dvsdz: np.ndarray | None = None,
) -> None:
    """Compute the shear-frequency squared (M²) at layer interfaces.

    Updates ``state.SS``, ``state.SSU``, ``state.SSV``, ``state.SSCSTK``,
    and ``state.SSSTK`` in-place using the energy-conserving Burchard (2002)
    discretisation.

    Parameters
    ----------
    state:
        MeanflowState with h, u, v, uo, vo, SS, SSU, SSV, SSCSTK, SSSTK.
        All arrays have shape (nlev+1,); index 0 = seabed, nlev = surface.
    nlev:
        Number of model layers.
    cnpar:
        Numerical implicitness parameter (0 = explicit, 1 = fully implicit;
        0.5 for Crank-Nicolson). Controls weighting between old and new
        time-level velocity differences in the shear production formula.
    dusdz:
        Stokes drift shear in x, shape (nlev+1,) [s⁻¹]. Defaults to zeros.
    dvsdz:
        Stokes drift shear in y, shape (nlev+1,) [s⁻¹]. Defaults to zeros.
    """
    assert state.h is not None
    assert state.u is not None
    assert state.v is not None
    assert state.uo is not None
    assert state.vo is not None
    assert state.SS is not None
    assert state.SSU is not None
    assert state.SSV is not None
    assert state.SSCSTK is not None
    assert state.SSSTK is not None

    h = state.h
    u = state.u
    v = state.v
    uo = state.uo
    vo = state.vo

    n = nlev + 1
    _dusdz = dusdz if dusdz is not None else np.zeros(n)
    _dvsdz = dvsdz if dvsdz is not None else np.zeros(n)

    step_shear_single(
        nlev,
        cnpar,
        h,
        u,
        v,
        uo,
        vo,
        _dusdz,
        _dvsdz,
        state.SS,
        state.SSU,
        state.SSV,
        state.SSCSTK,
        state.SSSTK,
    )
