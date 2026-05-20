# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculate steady-state Richardson number from c3\label{sec:Rist}
!
! !INTERFACE:
!   REALTYPE function  compute_rist(c1,c2,c3)
!
! !DESCRIPTION:
! Numerically computes the steady-state Richardson-number $Ri_{st}$
! for two-equations models from the given
! $c_{\psi 3}$ and the parameters
! $c_{\psi 1}$ and $c_{\psi 2}$ according to \eq{Ri_st}.
! A (very tricky) double Newton-iteration is used to solve the resulting
! implicit non-linear equation.
!
! !USES:
!   use turbulence, only:           as,an,cmue1,cmue2
!   use turbulence, only:           cm0
!   use turbulence, only:           turb_method,stab_method
!   use turbulence, only:           cm0_fix,Prandtl0_fix
!   use turbulence, only:           Constant
!   use turbulence, only:           Munk_Anderson
!   use turbulence, only:           Schumann_Gerz
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   REALTYPE, intent(in)           :: c1,c2,c3
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Hans Burchard, Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!     integer                      :: i,j
!     integer,parameter            :: imax=100
!     REALTYPE                     :: cc3,fc,fp,step,ann
!     REALTYPE                     :: ffc,ffp,Ri,Rii
!     REALTYPE                     :: NN(0:2),SS(0:2)
!     logical                      :: converged
!     REALTYPE,parameter           :: e=1.e-9,ee=1.e-4
!
!-----------------------------------------------------------------------
!BOC
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from pygotm.turbulence.compute_cpsi3 import (
    _IMAX,
    _MAX_STEP,
    _NEWTON_TOL,
    _require_finite_scalar,
    _require_positive_cm0,
    _solve_ann_for_ri,
    _StabilityConvergenceError,
)

if TYPE_CHECKING:
    from pygotm.turbulence.turbulence import TurbulenceState

__all__ = ["compute_rist"]

_RI_EPSILON_OUTER = 1.0e-4
_RI_INITIAL_GUESS = 0.18
_ANN_INITIAL_GUESS = 0.1
_FORTRAN_FAILURE_SENTINEL = -999.0


def _compute_c3_for_ri(
    state: TurbulenceState,
    *,
    c1: float,
    c2: float,
    ri: float,
) -> float:
    _, _, cmue1_value, cmue2_value = _solve_ann_for_ri(
        state,
        ri=ri,
        initial_ann=_ANN_INITIAL_GUESS,
    )
    return c2 + (c1 - c2) / ri * cmue1_value / cmue2_value


def compute_rist(state: TurbulenceState, c1: float, c2: float, c3: float) -> float:
    r"""Compute the steady-state Richardson number implied by ``c_{\psi 3}``."""

    _require_positive_cm0(state)
    _require_finite_scalar("c1", c1)
    _require_finite_scalar("c2", c2)
    _require_finite_scalar("c3", c3)

    ri = _RI_INITIAL_GUESS
    converged = True

    for _ in range(_IMAX + 1):
        try:
            ffc = _compute_c3_for_ri(state, c1=c1, c2=c2, ri=ri) - c3
            ffp = (
                _compute_c3_for_ri(
                    state,
                    c1=c1,
                    c2=c2,
                    ri=ri + _RI_EPSILON_OUTER,
                )
                - c3
            )
        except _StabilityConvergenceError:
            converged = False
            break

        derivative = (ffp - ffc) / _RI_EPSILON_OUTER
        if derivative == 0.0 or not isfinite(derivative):
            converged = False
            break

        step = -ffc / derivative
        ri = ri + 0.25 * step

        if abs(step) > _MAX_STEP:
            converged = False
            break
        if abs(step) < _NEWTON_TOL:
            break

    if not converged:
        return _FORTRAN_FAILURE_SENTINEL
    return ri
