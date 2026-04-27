"""Tests for pygotm.airsea.airsea_fluxes."""

from __future__ import annotations

import pytest

from pygotm.airsea.airsea_fluxes import FAIRALL, KONDO, airsea_fluxes
from pygotm.airsea.airsea_variables import AirSeaState
from pygotm.airsea.fairall import fairall
from pygotm.airsea.humidity import humidity
from pygotm.airsea.kondo import kondo


def _make_state() -> AirSeaState:
    state = AirSeaState()
    humidity(state, 1, 75.0, 101325.0, 18.0, 15.0)
    return state


def test_dispatch_to_kondo_matches_direct_call() -> None:
    state = _make_state()
    expected = kondo(state, 18.0, 15.0, 5.0, 2.0, 0.0)
    actual = airsea_fluxes(KONDO, state, 18.0, 15.0, 5.0, 2.0, 0.0)
    assert actual == pytest.approx(expected, rel=1.0e-12)


def test_dispatch_to_fairall_matches_direct_call() -> None:
    state = _make_state()
    expected = fairall(state, 18.0, 15.0, 5.0, 2.0, 0.0)
    actual = airsea_fluxes(FAIRALL, state, 18.0, 15.0, 5.0, 2.0, 0.0)
    assert actual == pytest.approx(expected, rel=1.0e-12)


def test_method_zero_returns_all_zeroes() -> None:
    state = _make_state()
    assert airsea_fluxes(0, state, 18.0, 15.0, 5.0, 2.0, 0.0) == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.0)
    )


def test_invalid_method_raises_value_error() -> None:
    state = _make_state()
    with pytest.raises(ValueError, match="invalid airsea flux method"):
        airsea_fluxes(99, state, 18.0, 15.0, 5.0, 2.0, 0.0)
