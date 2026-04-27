"""Reference helpers for scalar turbulence-model analysis tests."""

from __future__ import annotations

from math import exp, sqrt

import numpy as np

from pygotm.turbulence.turbulence import (
    Constant,
    Munk_Anderson,
    Schumann_Gerz,
    TurbulenceState,
    first_order,
)

_IMAX = 100
_EPSILON = 1.0e-8
_NEWTON_TOL = 1.0e-10
_AN_LIMIT_FACT = 0.5
_SMALL = 1.0e-10
_RI_EPSILON = 1.0e-8
_RI_THRESHOLD = 1.0e-10
_SG_LIMIT = 3.0
_RI_INFINITY = 0.25


def configure_second_order_state(state: TurbulenceState) -> None:
    """Populate the second-order closure coefficients used in the tests."""

    state.cc1 = 5.0
    state.cc2 = 0.8
    state.cc3 = 1.968
    state.cc4 = 1.136
    state.cc5 = 0.0
    state.cc6 = 0.4
    state.ct1 = 5.95
    state.ct2 = 0.6
    state.ct3 = 1.0
    state.ct4 = 0.0
    state.ct5 = 0.3333
    state.ctt = 0.72

    state.a1 = 2.0 / 3.0 - state.cc2 / 2.0
    state.a2 = 1.0 - state.cc3 / 2.0
    state.a3 = 1.0 - state.cc4 / 2.0
    state.a4 = state.cc5 / 2.0
    state.a5 = 0.5 - state.cc6 / 2.0
    state.at1 = 1.0 - state.ct2
    state.at2 = 1.0 - state.ct3
    state.at3 = 2.0 * (1.0 - state.ct4)
    state.at4 = 2.0 * (1.0 - state.ct5)
    state.at5 = 2.0 * state.ctt * (1.0 - state.ct5)

    n_val = state.cc1 / 2.0
    state.cm0 = (
        (
            state.a2 * state.a2
            - 3.0 * state.a3 * state.a3
            + 3.0 * state.a1 * n_val
        )
        / (3.0 * n_val * n_val)
    ) ** 0.25


def _cmue_d_reference(
    state: TurbulenceState,
    *,
    an_value: float,
) -> tuple[float, float, float, float]:
    n_val = 0.5 * state.cc1
    nt_val = state.ct1

    d0 = 36.0 * n_val**3 * nt_val**2
    d1 = (
        84.0 * state.a5 * state.at3 * n_val**2 * nt_val
        + 36.0 * state.at5 * n_val**3 * nt_val
    )
    d2 = 9.0 * (state.at2**2 - state.at1**2) * n_val**3 - 12.0 * (
        state.a2**2 - 3.0 * state.a3**2
    ) * n_val * nt_val**2
    d3 = (
        12.0
        * state.a5
        * state.at3
        * (state.a2 * state.at1 - 3.0 * state.a3 * state.at2)
        * n_val
        + 12.0
        * state.a5
        * state.at3
        * (state.a3**2 - state.a2**2)
        * nt_val
        + 12.0
        * state.at5
        * (3.0 * state.a3**2 - state.a2**2)
        * n_val
        * nt_val
    )
    d4 = (
        48.0 * state.a5**2 * state.at3**2 * n_val
        + 36.0 * state.a5 * state.at3 * state.at5 * n_val**2
    )
    d5 = 3.0 * (state.a2**2 - 3.0 * state.a3**2) * (
        state.at1**2 - state.at2**2
    ) * n_val

    n0 = 36.0 * state.a1 * n_val**2 * nt_val**2
    n1 = (
        -12.0 * state.a5 * state.at3 * (state.at1 + state.at2) * n_val**2
        + 8.0
        * state.a5
        * state.at3
        * (6.0 * state.a1 - state.a2 - 3.0 * state.a3)
        * n_val
        * nt_val
        + 36.0 * state.a1 * state.at5 * n_val**2 * nt_val
    )
    n2 = 9.0 * state.a1 * (state.at2**2 - state.at1**2) * n_val**2
    nt0 = 12.0 * state.at3 * n_val**3 * nt_val
    nt1 = 12.0 * state.a5 * state.at3**2 * n_val**2
    nt2 = 9.0 * state.a1 * state.at3 * (state.at1 - state.at2) * n_val**2 + (
        6.0 * state.a1 * (state.a2 - 3.0 * state.a3)
        - 4.0 * (state.a2**2 - 3.0 * state.a3**2)
    ) * state.at3 * n_val * nt_val

    an_min = (
        -(d1 + nt0) + np.sqrt((d1 + nt0) ** 2 - 4.0 * d0 * (d4 + nt1))
    ) / (2.0 * (d4 + nt1))
    an_value = max(an_value, _AN_LIMIT_FACT * an_min)

    tmp0 = -d0 - (d1 + nt0) * an_value - (d4 + nt1) * an_value**2
    tmp1 = -d2 + n0 + (n1 - d3 - nt2) * an_value
    if abs(n2 - d5) < _SMALL:
        as_value = -tmp0 / tmp1
    else:
        tmp2 = n2 - d5
        as_value = (-tmp1 + np.sqrt(tmp1 * tmp1 - 4.0 * tmp0 * tmp2)) / (
            2.0 * tmp2
        )

    d_cm = (
        d0
        + d1 * an_value
        + d2 * as_value
        + d3 * an_value * as_value
        + d4 * an_value**2
        + d5 * as_value**2
    )
    n_cm = n0 + n1 * an_value + n2 * as_value
    n_cmp = nt0 + nt1 * an_value + nt2 * as_value

    cm3_inv = 1.0 / state.cm0**3
    return an_value, as_value, cm3_inv * n_cm / d_cm, cm3_inv * n_cmp / d_cm


def _evaluate_cmue_reference(
    state: TurbulenceState,
    *,
    ann: float,
    ri: float,
) -> tuple[float, float, float, float]:
    if state.turb_method == first_order:
        an_value = ann
        as_value = ann / ri

        if state.stab_method == Constant:
            return (
                an_value,
                as_value,
                state.cm0_fix,
                state.cm0_fix / state.Prandtl0_fix,
            )
        if state.stab_method == Munk_Anderson:
            ri_value = an_value / (as_value + _RI_EPSILON)
            prandtl = state.Prandtl0_fix
            if ri_value >= _RI_THRESHOLD:
                prandtl = (
                    state.Prandtl0_fix
                    * (1.0 + 3.33 * ri_value) ** 1.5
                    / sqrt(1.0 + 10.0 * ri_value)
                )
            return an_value, as_value, state.cm0_fix, state.cm0_fix / prandtl
        if state.stab_method == Schumann_Gerz:
            ri_value = an_value / (as_value + _RI_EPSILON)
            prandtl = state.Prandtl0_fix
            if ri_value >= _RI_THRESHOLD:
                prandtl = (
                    state.Prandtl0_fix
                    * exp(-ri_value / (state.Prandtl0_fix * _RI_INFINITY))
                    + ri_value / _RI_INFINITY
                )
            return (
                an_value,
                as_value,
                state.cm0_fix,
                state.cm0_fix / min(_SG_LIMIT, prandtl),
            )
        msg = f"unsupported stability function {state.stab_method}"
        raise NotImplementedError(msg)

    return _cmue_d_reference(state, an_value=ann)


def reference_compute_cpsi3(
    state: TurbulenceState,
    *,
    c1: float,
    c2: float,
    ri: float,
    initial_ann: float = 5.0,
) -> float:
    """Compute a reference ``cpsi3`` value without calling production code."""

    ann = initial_ann
    for _ in range(_IMAX + 1):
        an_value, _, cmue1_value, cmue2_value = _evaluate_cmue_reference(
            state,
            ann=ann,
            ri=ri,
        )
        fc = cmue1_value * an_value / ri - cmue2_value * an_value - state.cm0**(-3)

        an_step, _, cmue1_step, cmue2_step = _evaluate_cmue_reference(
            state,
            ann=ann + _EPSILON,
            ri=ri,
        )
        fp = cmue1_step * an_step / ri - cmue2_step * an_step - state.cm0**(-3)

        step = -fc / ((fp - fc) / _EPSILON)
        ann = ann + 0.5 * step
        if abs(step) < _NEWTON_TOL:
            break

    _, _, cmue1_value, cmue2_value = _evaluate_cmue_reference(
        state,
        ann=ann,
        ri=ri,
    )
    return c2 + (c1 - c2) / ri * cmue1_value / cmue2_value
