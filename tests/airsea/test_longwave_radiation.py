"""Tests for pygotm.airsea.longwave_radiation."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.airsea_variables import (
    BERLIAND_BERLIAND,
    BIGNAMI,
    CLARK,
    HASTENRATH_LAMB,
    JOSEY1,
    JOSEY2,
    AirSeaState,
    bolz,
    emiss,
)
from pygotm.airsea.humidity import humidity
from pygotm.airsea.longwave_radiation import longwave_radiation

_AIRP = 101325.0
_TW_C = 18.0
_TA_C = 15.0
_TW_K = 291.15
_TA_K = 288.15
_CLOUD = 0.5
_DLAT = 45.0


def _make_state() -> AirSeaState:
    state = AirSeaState()
    humidity(state, 1, 75.0, _AIRP, _TW_C, _TA_C)
    return state


def _expected_longwave(
    method: int,
    state: AirSeaState,
    dlat: float,
    tw: float,
    ta: float,
    cloud: float,
) -> tuple[float, float]:
    cloud_correction_factor = (
        0.497202,
        0.501885,
        0.506568,
        0.511250,
        0.515933,
        0.520616,
        0.525299,
        0.529982,
        0.534665,
        0.539348,
        0.544031,
        0.548714,
        0.553397,
        0.558080,
        0.562763,
        0.567446,
        0.572129,
        0.576812,
        0.581495,
        0.586178,
        0.590861,
        0.595544,
        0.600227,
        0.604910,
        0.609593,
        0.614276,
        0.618959,
        0.623641,
        0.628324,
        0.633007,
        0.637690,
        0.642373,
        0.647056,
        0.651739,
        0.656422,
        0.661105,
        0.665788,
        0.670471,
        0.675154,
        0.679837,
        0.684520,
        0.689203,
        0.693886,
        0.698569,
        0.703252,
        0.707935,
        0.712618,
        0.717301,
        0.721984,
        0.726667,
        0.731350,
        0.736032,
        0.740715,
        0.745398,
        0.750081,
        0.754764,
        0.759447,
        0.764130,
        0.768813,
        0.773496,
        0.778179,
        0.782862,
        0.787545,
        0.792228,
        0.796911,
        0.801594,
        0.806277,
        0.810960,
        0.815643,
        0.820326,
        0.825009,
        0.829692,
        0.834375,
        0.839058,
        0.843741,
        0.848423,
        0.853106,
        0.857789,
        0.862472,
        0.867155,
        0.871838,
        0.876521,
        0.881204,
        0.885887,
        0.890570,
        0.895253,
        0.899936,
        0.904619,
        0.909302,
        0.913985,
        0.918668,
    )
    ccf = cloud_correction_factor[int(math.floor(abs(dlat) + 0.5))]
    ea = state.ea

    if method == CLARK:
        x1 = (1.0 - ccf * cloud * cloud) * tw**4
        x2 = 0.39 - 0.05 * math.sqrt(ea * 0.01)
        x3 = 4.0 * tw**3 * (tw - ta)
        return -emiss * bolz * (x1 * x2 + x3), ea
    if method == HASTENRATH_LAMB:
        x1 = (1.0 - ccf * cloud * cloud) * tw**4
        x2 = 0.39 - 0.056 * math.sqrt(1000.0 * state.qa)
        x3 = 4.0 * tw**3 * (tw - ta)
        return -emiss * bolz * (x1 * x2 + x3), ea
    if method == BIGNAMI:
        x1 = (1.0 + 0.1762 * cloud * cloud) * ta**4
        x2 = 0.653 + 0.00535 * (ea * 0.01)
        x3 = emiss * tw**4
        return -bolz * (-x1 * x2 + x3), ea
    if method == BERLIAND_BERLIAND:
        x1 = (1.0 - 0.6823 * cloud * cloud) * ta**4
        x2 = 0.39 - 0.05 * math.sqrt(0.01 * ea)
        x3 = 4.0 * ta**3 * (tw - ta)
        return -emiss * bolz * (x1 * x2 + x3), ea
    if method == JOSEY1:
        x1 = emiss * tw**4
        x2 = (10.77 * cloud + 2.34) * cloud - 18.44
        x3 = 0.955 * (ta + x2) ** 4
        return -bolz * (x1 - x3), ea

    ea = max(ea, 10.0)
    x1 = emiss * tw**4
    x2 = 34.07 + 4157.0 / math.log(2.1718e10 / ea)
    x2 = (10.77 * cloud + 2.34) * cloud - 18.44 + 0.84 * (x2 - ta + 4.01)
    x3 = 0.955 * (ta + x2) ** 4
    return -bolz * (x1 - x3), ea


def test_import_and_smoke_all_supported_methods() -> None:
    state = _make_state()
    for method in (CLARK, HASTENRATH_LAMB, BIGNAMI, BERLIAND_BERLIAND, JOSEY1, JOSEY2):
        flux = longwave_radiation(state, method, _DLAT, _TW_K, _TA_K, _CLOUD)
        assert math.isfinite(flux)


@pytest.mark.parametrize(
    "method",
    [CLARK, HASTENRATH_LAMB, BIGNAMI, BERLIAND_BERLIAND, JOSEY1, JOSEY2],
)
def test_methods_match_fortran_formula(method: int) -> None:
    state = _make_state()
    expected, expected_ea = _expected_longwave(
        method,
        state,
        _DLAT,
        _TW_K,
        _TA_K,
        _CLOUD,
    )
    flux = longwave_radiation(state, method, _DLAT, _TW_K, _TA_K, _CLOUD)
    assert flux == pytest.approx(expected, rel=1.0e-12)
    assert state.ea == pytest.approx(expected_ea, rel=1.0e-12)


def test_josey2_clamps_vapour_pressure_to_ten_pascal() -> None:
    state = _make_state()
    state.ea = 5.0
    expected, expected_ea = _expected_longwave(
        JOSEY2,
        state,
        _DLAT,
        _TW_K,
        _TA_K,
        _CLOUD,
    )
    state.ea = 5.0
    flux = longwave_radiation(state, JOSEY2, _DLAT, _TW_K, _TA_K, _CLOUD)
    assert flux == pytest.approx(expected, rel=1.0e-12)
    assert state.ea == pytest.approx(expected_ea)


def test_invalid_method_raises_value_error() -> None:
    state = _make_state()
    with pytest.raises(ValueError, match="illegal longwave_radiation_method"):
        longwave_radiation(state, 99, _DLAT, _TW_K, _TA_K, _CLOUD)
