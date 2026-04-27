"""Tests for pygotm.turbulence.compute_cpsi3."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from turbulence_model_analysis_reference import (
    configure_second_order_state,
    reference_compute_cpsi3,
)

from pygotm.turbulence.compute_cpsi3 import compute_cpsi3
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
    ("state_factory", "ri"),
    [
        (lambda: _make_first_order_state(Constant), 0.25),
        (lambda: _make_first_order_state(Munk_Anderson), 0.22),
        (lambda: _make_first_order_state(Schumann_Gerz), 0.18),
        (_make_second_order_state, 0.20),
    ],
)
def test_matches_reference_for_supported_closures(
    state_factory: Callable[[], TurbulenceState],
    ri: float,
) -> None:
    state = state_factory()
    expected = reference_compute_cpsi3(state, c1=_C1, c2=_C2, ri=ri)

    value = compute_cpsi3(state, _C1, _C2, ri)

    assert value == pytest.approx(expected, rel=1.0e-10, abs=1.0e-12)
    assert math.isfinite(value)


def test_updates_probe_arrays_when_allocated() -> None:
    state = _make_first_order_state(Constant)
    value = compute_cpsi3(state, _C1, _C2, 0.25)

    assert state.an is not None
    assert state.as_ is not None
    assert state.cmue1 is not None
    assert state.cmue2 is not None

    assert value == pytest.approx(0.4992, rel=1.0e-12, abs=1.0e-12)
    assert state.an[1] > 0.0
    assert state.as_[1] == pytest.approx(state.an[1] / 0.25, rel=1.0e-12)
    assert state.cmue1[1] == pytest.approx(state.cm0_fix)
    assert state.cmue2[1] == pytest.approx(state.cm0_fix / state.Prandtl0_fix)


def test_rejects_nonpositive_ri() -> None:
    state = _make_first_order_state(Constant)

    with pytest.raises(ValueError, match="Ri must be positive"):
        compute_cpsi3(state, _C1, _C2, 0.0)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    state = _make_second_order_state()

    values = np.array(
        [
            compute_cpsi3(state, _C1, _C2, ri)
            for ri in np.linspace(0.14, 0.24, 5)
        ]
    )

    assert np.isfinite(values).all()
