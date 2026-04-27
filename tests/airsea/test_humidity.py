"""Tests for pygotm.airsea.humidity."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.airsea.airsea_variables import AirSeaState, const06, kelvin, rgas
from pygotm.airsea.humidity import humidity

_A1 = 6.107799961
_A2 = 4.436518521e-1
_A3 = 1.428945805e-2
_A4 = 2.650648471e-4
_A5 = 3.031240396e-6
_A6 = 2.034080948e-8
_A7 = 6.136820929e-11

_AIRP = 101325.0
_TW = 18.0
_TA = 15.0


def _saturation_vapor_pressure(temp_c: float) -> float:
    pressure_mb = _A1 + temp_c * (
        _A2
        + temp_c
        * (_A3 + temp_c * (_A4 + temp_c * (_A5 + temp_c * (_A6 + temp_c * _A7))))
    )
    return pressure_mb * 100.0


def _specific_humidity(airp: float, vapour_pressure: float) -> float:
    return const06 * vapour_pressure / (airp - 0.377 * vapour_pressure)


def _expected_surface_terms(tw: float, airp: float) -> tuple[float, float]:
    es = 0.98 * _saturation_vapor_pressure(tw)
    qs = _specific_humidity(airp, es)
    return es, qs


def test_import_and_smoke_relative_humidity() -> None:
    state = AirSeaState()
    humidity(state, 1, 75.0, _AIRP, _TW, _TA)
    assert state.ea > 0.0
    assert state.qa > 0.0
    assert state.rhoa > 0.0


def test_relative_humidity_matches_fortran_formula() -> None:
    state = AirSeaState()
    humidity(state, 1, 75.0, _AIRP, _TW, _TA)

    expected_es, expected_qs = _expected_surface_terms(_TW, _AIRP)
    expected_ea = 0.75 * _saturation_vapor_pressure(_TA)
    expected_qa = _specific_humidity(_AIRP, expected_ea)
    expected_rhoa = _AIRP / (rgas * (_TA + kelvin) * (1.0 + const06 * expected_qa))

    assert state.ta == pytest.approx(_TA)
    assert state.es == pytest.approx(expected_es)
    assert state.qs == pytest.approx(expected_qs)
    assert state.ea == pytest.approx(expected_ea)
    assert state.qa == pytest.approx(expected_qa)
    assert state.rhoa == pytest.approx(expected_rhoa)
    assert state.qa < state.qs


def test_wet_bulb_celsius_matches_fortran_formula() -> None:
    state = AirSeaState()
    twet = 12.0
    humidity(state, 2, twet, _AIRP, _TW, _TA)

    expected_es, expected_qs = _expected_surface_terms(_TW, _AIRP)
    expected_ea = _saturation_vapor_pressure(twet)
    expected_ea = expected_ea - 6.6e-4 * (1.0 + 1.15e-3 * twet) * _AIRP * (_TA - twet)
    expected_qa = _specific_humidity(_AIRP, expected_ea)
    expected_rhoa = _AIRP / (rgas * (_TA + kelvin) * (1.0 + const06 * expected_qa))

    assert state.es == pytest.approx(expected_es)
    assert state.qs == pytest.approx(expected_qs)
    assert state.ea == pytest.approx(expected_ea)
    assert state.qa == pytest.approx(expected_qa)
    assert state.rhoa == pytest.approx(expected_rhoa)
    assert state.qa < state.qs


def test_wet_bulb_kelvin_input_matches_celsius_path() -> None:
    state_c = AirSeaState()
    state_k = AirSeaState()
    humidity(state_c, 2, 12.0, _AIRP, _TW, _TA)
    humidity(state_k, 2, 12.0 + kelvin, _AIRP, _TW, _TA)

    assert state_k.es == pytest.approx(state_c.es)
    assert state_k.qs == pytest.approx(state_c.qs)
    assert state_k.ea == pytest.approx(state_c.ea)
    assert state_k.qa == pytest.approx(state_c.qa)
    assert state_k.rhoa == pytest.approx(state_c.rhoa)


def test_dew_point_celsius_matches_fortran_formula() -> None:
    state = AirSeaState()
    dew = 10.0
    humidity(state, 3, dew, _AIRP, _TW, _TA)

    expected_es, expected_qs = _expected_surface_terms(_TW, _AIRP)
    expected_ea = _saturation_vapor_pressure(dew)
    expected_qa = _specific_humidity(_AIRP, expected_ea)
    expected_rhoa = _AIRP / (rgas * (_TA + kelvin) * (1.0 + const06 * expected_qa))

    assert state.es == pytest.approx(expected_es)
    assert state.qs == pytest.approx(expected_qs)
    assert state.ea == pytest.approx(expected_ea)
    assert state.qa == pytest.approx(expected_qa)
    assert state.rhoa == pytest.approx(expected_rhoa)
    assert state.qa < state.qs


def test_dew_point_kelvin_input_matches_celsius_path() -> None:
    state_c = AirSeaState()
    state_k = AirSeaState()
    humidity(state_c, 3, 10.0, _AIRP, _TW, _TA)
    humidity(state_k, 3, 10.0 + kelvin, _AIRP, _TW, _TA)

    assert state_k.es == pytest.approx(state_c.es)
    assert state_k.qs == pytest.approx(state_c.qs)
    assert state_k.ea == pytest.approx(state_c.ea)
    assert state_k.qa == pytest.approx(state_c.qa)
    assert state_k.rhoa == pytest.approx(state_c.rhoa)


def test_specific_humidity_input_preserves_given_qa() -> None:
    state = AirSeaState()
    qa = 0.0095
    humidity(state, 4, qa, _AIRP, _TW, _TA)

    expected_es, expected_qs = _expected_surface_terms(_TW, _AIRP)
    expected_ea = qa * _AIRP / (const06 + 0.378 * qa)
    expected_rhoa = _AIRP / (rgas * (_TA + kelvin) * (1.0 + const06 * qa))

    assert state.es == pytest.approx(expected_es)
    assert state.qs == pytest.approx(expected_qs)
    assert state.ea == pytest.approx(expected_ea)
    assert state.qa == pytest.approx(qa)
    assert state.rhoa == pytest.approx(expected_rhoa)


@pytest.mark.parametrize(
    ("hum_method", "hum_value"),
    [
        (1, 80.0),
        (2, 12.0),
        (3, 10.0),
        (4, 0.0095),
    ],
)
def test_humidity_outputs_are_finite_for_all_supported_methods(
    hum_method: int,
    hum_value: float,
) -> None:
    state = AirSeaState()
    humidity(state, hum_method, hum_value, _AIRP, _TW, _TA)

    for value in (state.es, state.ea, state.qs, state.qa, state.rhoa, state.ta):
        assert np.isfinite(value)
        assert value >= 0.0


def test_invalid_humidity_method_raises_value_error() -> None:
    state = AirSeaState()
    with pytest.raises(ValueError, match="not a valid hum_method"):
        humidity(state, 99, 0.0, _AIRP, _TW, _TA)
