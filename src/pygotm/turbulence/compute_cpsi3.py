# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculate c3 from steady-state Richardson number\label{sec:c3}
!
! !INTERFACE:
!   REALTYPE function compute_cpsi3(c1,c2,Ri)
!
! !DESCRIPTION:
! Numerically computes $c_{\psi 3}$ for two-equation models from  given
! steady-state Richardson-number $Ri_{st}$ and parameters
! $c_{\psi 1}$ and $c_{\psi 2}$ according to \eq{Ri_st}.
! A Newton-iteration is used to solve the resulting
! implicit non-linear equation.
!
! !USES:
!   use turbulence, only:           an,as,cmue1,cmue2
!   use turbulence, only:           cm0,cm0_fix,Prandtl0_fix
!   use turbulence, only:           turb_method,stab_method
!   use turbulence, only:           Constant
!   use turbulence, only:           Munk_Anderson
!   use turbulence, only:           Schumann_Gerz
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   REALTYPE, intent(in)            :: c1,c2,Ri
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard, Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!     integer                       :: i
!     integer,parameter             :: imax=100
!     REALTYPE                      :: fc,fp,step,ann
!     REALTYPE,parameter            :: e=1.e-8
!
!-----------------------------------------------------------------------
!BOC
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from math import exp, isfinite, sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pygotm.turbulence.turbulence import TurbulenceState

__all__ = ["compute_cpsi3"]

_IMAX = 100
_EPSILON = 1.0e-8
_NEWTON_TOL = 1.0e-10
_MAX_STEP = 100.0
_AN_LIMIT_FACT = 0.5
_SMALL = 1.0e-10
_RI_EPSILON = 1.0e-8
_RI_THRESHOLD = 1.0e-10
_SG_LIMIT = 3.0
_RI_INFINITY = 0.25
_PROBE_INDEX = 1


class _StabilityConvergenceError(RuntimeError):
    """Raised when the translated Newton iteration leaves the valid regime."""


def _require_finite_scalar(name: str, value: float) -> None:
    if not isfinite(value):
        msg = f"{name} must be finite"
        raise ValueError(msg)


def _require_positive_scalar(name: str, value: float) -> None:
    _require_finite_scalar(name, value)
    if value <= 0.0:
        msg = f"{name} must be positive"
        raise ValueError(msg)


def _require_positive_cm0(state: TurbulenceState) -> None:
    _require_positive_scalar("state.cm0", state.cm0)


def _store_probe_values(
    state: TurbulenceState,
    *,
    an_value: float,
    as_value: float,
    cmue1_value: float,
    cmue2_value: float,
) -> None:
    """Mirror the Fortran probe-point side effects when arrays are allocated."""

    if state.an is not None and state.an.size > _PROBE_INDEX:
        state.an[_PROBE_INDEX] = an_value
    if state.as_ is not None and state.as_.size > _PROBE_INDEX:
        state.as_[_PROBE_INDEX] = as_value
    if state.cmue1 is not None and state.cmue1.size > _PROBE_INDEX:
        state.cmue1[_PROBE_INDEX] = cmue1_value
    if state.cmue2 is not None and state.cmue2.size > _PROBE_INDEX:
        state.cmue2[_PROBE_INDEX] = cmue2_value


def _cmue_d_scalar(
    state: TurbulenceState,
    *,
    an_value: float,
) -> tuple[float, float, float, float]:
    """Evaluate the quasi-equilibrium stability functions at one probe point."""

    n_val = 0.5 * state.cc1
    nt_val = state.ct1
    n_sq = n_val * n_val
    n_cube = n_sq * n_val
    nt_sq = nt_val * nt_val

    d0 = 36.0 * n_cube * nt_sq
    d1 = (
        84.0 * state.a5 * state.at3 * n_sq * nt_val + 36.0 * state.at5 * n_cube * nt_val
    )
    d2 = (
        9.0 * (state.at2 * state.at2 - state.at1 * state.at1) * n_cube
        - 12.0 * (state.a2 * state.a2 - 3.0 * state.a3 * state.a3) * n_val * nt_sq
    )
    d3 = (
        12.0
        * state.a5
        * state.at3
        * (state.a2 * state.at1 - 3.0 * state.a3 * state.at2)
        * n_val
        + 12.0
        * state.a5
        * state.at3
        * (state.a3 * state.a3 - state.a2 * state.a2)
        * nt_val
        + 12.0
        * state.at5
        * (3.0 * state.a3 * state.a3 - state.a2 * state.a2)
        * n_val
        * nt_val
    )
    d4 = (
        48.0 * state.a5 * state.a5 * state.at3 * state.at3 * n_val
        + 36.0 * state.a5 * state.at3 * state.at5 * n_sq
    )
    d5 = (
        3.0
        * (state.a2 * state.a2 - 3.0 * state.a3 * state.a3)
        * (state.at1 * state.at1 - state.at2 * state.at2)
        * n_val
    )

    n0 = 36.0 * state.a1 * n_sq * nt_sq
    n1 = (
        -12.0 * state.a5 * state.at3 * (state.at1 + state.at2) * n_sq
        + 8.0
        * state.a5
        * state.at3
        * (6.0 * state.a1 - state.a2 - 3.0 * state.a3)
        * n_val
        * nt_val
        + 36.0 * state.a1 * state.at5 * n_sq * nt_val
    )
    n2 = 9.0 * state.a1 * (state.at2 * state.at2 - state.at1 * state.at1) * n_sq

    nt0 = 12.0 * state.at3 * n_cube * nt_val
    nt1 = 12.0 * state.a5 * state.at3 * state.at3 * n_sq
    nt2 = (
        9.0 * state.a1 * state.at3 * (state.at1 - state.at2) * n_sq
        + (
            6.0 * state.a1 * (state.a2 - 3.0 * state.a3)
            - 4.0 * (state.a2 * state.a2 - 3.0 * state.a3 * state.a3)
        )
        * state.at3
        * n_val
        * nt_val
    )

    an_discriminant = (d1 + nt0) * (d1 + nt0) - 4.0 * d0 * (d4 + nt1)
    if an_discriminant < 0.0:
        msg = "negative discriminant while evaluating the quasi-equilibrium stability function"
        raise RuntimeError(msg)
    an_min = (-(d1 + nt0) + sqrt(an_discriminant)) / (2.0 * (d4 + nt1))
    an_value = max(an_value, _AN_LIMIT_FACT * an_min)

    tmp0 = -d0 - (d1 + nt0) * an_value - (d4 + nt1) * an_value * an_value
    tmp1 = -d2 + n0 + (n1 - d3 - nt2) * an_value

    if abs(n2 - d5) < _SMALL:
        as_value = -tmp0 / tmp1
    else:
        tmp2 = n2 - d5
        as_discriminant = tmp1 * tmp1 - 4.0 * tmp0 * tmp2
        if as_discriminant < 0.0:
            msg = "negative discriminant while solving for the shear number"
            raise RuntimeError(msg)
        as_value = (-tmp1 + sqrt(as_discriminant)) / (2.0 * tmp2)

    d_cm = (
        d0
        + d1 * an_value
        + d2 * as_value
        + d3 * an_value * as_value
        + d4 * an_value * an_value
        + d5 * as_value * as_value
    )
    n_cm = n0 + n1 * an_value + n2 * as_value
    n_cmp = nt0 + nt1 * an_value + nt2 * as_value
    cm3_inv = 1.0 / (state.cm0 * state.cm0 * state.cm0)
    return an_value, as_value, cm3_inv * n_cm / d_cm, cm3_inv * n_cmp / d_cm


def _evaluate_stability(
    state: TurbulenceState,
    *,
    ann: float,
    ri: float,
) -> tuple[float, float, float, float]:
    """Return ``(an, as, cmue1, cmue2)`` for one scalar Richardson probe."""

    from pygotm.turbulence.turbulence import (
        Constant,
        Munk_Anderson,
        Schumann_Gerz,
        first_order,
    )

    an_value = ann
    as_value = an_value / ri

    # These solves run during model analysis, not in the time-stepping hot path.
    # Keep them scalar and reuse the translated stability-function algebra directly.
    if state.turb_method == first_order:
        _require_positive_scalar("state.cm0_fix", state.cm0_fix)
        _require_positive_scalar("state.Prandtl0_fix", state.Prandtl0_fix)

        if state.stab_method == Constant:
            cmue1_value = state.cm0_fix
            cmue2_value = state.cm0_fix / state.Prandtl0_fix
        elif state.stab_method == Munk_Anderson:
            ri_value = an_value / (as_value + _RI_EPSILON)
            prandtl = state.Prandtl0_fix
            if ri_value >= _RI_THRESHOLD:
                prandtl = (
                    state.Prandtl0_fix
                    * (1.0 + 3.33 * ri_value) ** 1.5
                    / sqrt(1.0 + 10.0 * ri_value)
                )
            cmue1_value = state.cm0_fix
            cmue2_value = state.cm0_fix / prandtl
        elif state.stab_method == Schumann_Gerz:
            ri_value = an_value / (as_value + _RI_EPSILON)
            prandtl = state.Prandtl0_fix
            if ri_value >= _RI_THRESHOLD:
                prandtl = (
                    state.Prandtl0_fix
                    * exp(-ri_value / (state.Prandtl0_fix * _RI_INFINITY))
                    + ri_value / _RI_INFINITY
                )
            cmue1_value = state.cm0_fix
            cmue2_value = state.cm0_fix / min(_SG_LIMIT, prandtl)
        else:
            msg = f"unsupported first-order stability function {state.stab_method}"
            raise NotImplementedError(msg)
    else:
        an_value, as_value, cmue1_value, cmue2_value = _cmue_d_scalar(
            state,
            an_value=an_value,
        )

    _store_probe_values(
        state,
        an_value=an_value,
        as_value=as_value,
        cmue1_value=cmue1_value,
        cmue2_value=cmue2_value,
    )
    return an_value, as_value, cmue1_value, cmue2_value


def _solve_ann_for_ri(
    state: TurbulenceState,
    *,
    ri: float,
    initial_ann: float,
) -> tuple[float, float, float, float]:
    """Solve the translated inner Newton iteration for one Richardson number."""

    if not isfinite(ri) or ri <= 0.0:
        msg = "steady-state Richardson number left the positive domain"
        raise _StabilityConvergenceError(msg)

    ann = initial_ann
    cm0_inv3 = state.cm0 ** (-3)

    for _ in range(_IMAX + 1):
        an_value, as_value, cmue1_value, cmue2_value = _evaluate_stability(
            state,
            ann=ann,
            ri=ri,
        )
        fc = cmue1_value * an_value / ri - cmue2_value * an_value - cm0_inv3

        an_step, _, cmue1_step, cmue2_step = _evaluate_stability(
            state,
            ann=ann + _EPSILON,
            ri=ri,
        )
        fp = cmue1_step * an_step / ri - cmue2_step * an_step - cm0_inv3

        derivative = (fp - fc) / _EPSILON
        if derivative == 0.0 or not isfinite(derivative):
            msg = "Method for calculating the steady-state Richardson relation does not converge."
            raise _StabilityConvergenceError(msg)

        step = -fc / derivative
        ann = ann + 0.5 * step

        if abs(step) > _MAX_STEP:
            msg = "Method for calculating the steady-state Richardson relation does not converge."
            raise _StabilityConvergenceError(msg)
        if abs(step) < _NEWTON_TOL:
            break

    return _evaluate_stability(state, ann=ann, ri=ri)


def compute_cpsi3(state: TurbulenceState, c1: float, c2: float, Ri: float) -> float:
    r"""Compute ``c_{\psi 3}`` from a target steady-state Richardson number."""

    _require_positive_cm0(state)
    _require_finite_scalar("c1", c1)
    _require_finite_scalar("c2", c2)
    _require_positive_scalar("Ri", Ri)

    try:
        _, _, cmue1_value, cmue2_value = _solve_ann_for_ri(
            state,
            ri=Ri,
            initial_ann=5.0,
        )
    except _StabilityConvergenceError as exc:
        msg = (
            "Method for calculating c3 does not converge. "
            "Probably, the prescribed steady-state Richardson number is outside "
            "the range of the chosen stability function."
        )
        raise RuntimeError(msg) from exc

    return c2 + (c1 - c2) / Ri * cmue1_value / cmue2_value
