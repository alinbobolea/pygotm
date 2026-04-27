"""Tests for pygotm.turbulence.compute_rist."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from turbulence_model_analysis_reference import (
    configure_second_order_state,
    reference_compute_cpsi3,
)

from pygotm.turbulence.compute_rist import compute_rist
from pygotm.turbulence.turbulence import (
    Constant,
    Munk_Anderson,
    Schumann_Gerz,
    TurbulenceState,
    first_order,
    init_turbulence,
    post_init_turbulence,
    second_order,
)

_NLEV = 12
_C1 = 1.44
_C2 = 1.92


def _make_first_order_state(stab_method: int) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, turb_method=first_order, stab_method=stab_method)
    post_init_turbulence(state, _NLEV)
    state.cm0 = state.cm0_fix
    return state


def _make_second_order_state() -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, turb_method=second_order)
    post_init_turbulence(state, _NLEV)
    configure_second_order_state(state)
    return state


@pytest.mark.parametrize(
    ("state_factory", "ri_target"),
    [
        (lambda: _make_first_order_state(Constant), 0.25),
        (lambda: _make_first_order_state(Munk_Anderson), 0.22),
        (lambda: _make_first_order_state(Schumann_Gerz), 0.18),
        (_make_second_order_state, 0.20),
    ],
)
def test_recovers_target_ri_for_supported_closures(
    state_factory: Callable[[], TurbulenceState],
    ri_target: float,
) -> None:
    state = state_factory()
    c3_value = reference_compute_cpsi3(state, c1=_C1, c2=_C2, ri=ri_target)

    ri_value = compute_rist(state, _C1, _C2, c3_value)

    assert ri_value == pytest.approx(ri_target, rel=1.0e-9, abs=1.0e-12)
    assert math.isfinite(ri_value)


def test_returns_fortran_sentinel_for_out_of_range_c3() -> None:
    state = _make_first_order_state(Constant)

    value = compute_rist(state, _C1, _C2, 2.5)

    assert value == -999.0


def test_rejects_nonfinite_c3() -> None:
    state = _make_first_order_state(Constant)

    with pytest.raises(ValueError, match="c3 must be finite"):
        compute_rist(state, _C1, _C2, float("nan"))
