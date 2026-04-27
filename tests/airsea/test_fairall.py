"""Tests for pygotm.airsea.fairall."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.airsea_variables import AirSeaState
from pygotm.airsea.fairall import fairall, psi
from pygotm.airsea.humidity import humidity

_AIRP = 101325.0
_SST = 18.0
_AIRT = 15.0


def _make_state() -> AirSeaState:
    state = AirSeaState()
    humidity(state, 1, 75.0, _AIRP, _SST, _AIRT)
    return state


def test_import_and_smoke() -> None:
    state = _make_state()
    outputs = fairall(state, _SST, _AIRT, 5.0, 2.0, 0.0)
    assert all(math.isfinite(value) for value in outputs)


@pytest.mark.parametrize(
    ("iflag", "zol", "expected"),
    [
        (1, 0.2, -0.94),
        (2, 0.2, -0.94),
        (1, 0.0, 0.0),
    ],
)
def test_psi_stable_and_neutral_cases_match_closed_form(
    iflag: int,
    zol: float,
    expected: float,
) -> None:
    assert psi(iflag, zol) == pytest.approx(expected, rel=1.0e-12)


def test_psi_unstable_wind_branch_matches_formula() -> None:
    zol = -0.5
    chik = (1.0 - 16.0 * zol) ** 0.25
    psik = (
        2.0 * math.log(0.5 * (1.0 + chik))
        + math.log(0.5 * (1.0 + chik * chik))
        - 2.0 * math.atan(chik)
        + 0.5 * math.pi
    )
    chic = (1.0 - 12.87 * zol) ** (1.0 / 3.0)
    psic = (
        1.5 * math.log((1.0 / 3.0) * (1.0 + chic + chic * chic))
        - 1.7320508 * math.atan((1.0 + 2.0 * chic) / 1.7320508)
        + math.pi / 1.7320508
    )
    fw = 1.0 / (1.0 + zol * zol)
    expected = fw * psik + (1.0 - fw) * psic
    assert psi(1, zol) == pytest.approx(expected, rel=1.0e-12)


def test_invalid_psi_flag_raises() -> None:
    with pytest.raises(ValueError, match="invalid iflag"):
        psi(99, -0.5)


def test_reference_case_matches_regression_values() -> None:
    state = _make_state()
    assert fairall(state, _SST, _AIRT, 5.0, 2.0, 0.0) == pytest.approx(
        (
            0.0,
            0.044229368585926354,
            0.01769174743437054,
            -125.30373966958629,
            -31.35559939321449,
        ),
        rel=1.0e-12,
    )


def test_celsius_and_kelvin_inputs_are_equivalent() -> None:
    state_c = _make_state()
    state_k = _make_state()
    celsius = fairall(state_c, _SST, _AIRT, 5.0, 2.0, 0.0)
    kelvin = fairall(state_k, _SST + 273.15, _AIRT + 273.15, 5.0, 2.0, 0.0)
    assert kelvin == pytest.approx(celsius, rel=1.0e-12)


def test_zero_wind_returns_zero_fluxes_and_stress() -> None:
    state = _make_state()
    assert fairall(state, _SST, _AIRT, 0.0, 0.0, 0.0) == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.0)
    )


def test_large_stable_richardson_number_suppresses_fluxes() -> None:
    state = _make_state()
    assert fairall(state, 5.0, 25.0, 0.1, 0.0, 0.0) == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.0)
    )


def test_rain_impact_and_evaporation_branch_changes_outputs() -> None:
    dry_state = _make_state()
    wet_state = _make_state()
    wet_state.rain_impact = True
    wet_state.calc_evaporation = True

    dry = fairall(dry_state, _SST, _AIRT, 5.0, 2.0, 1.0e-6)
    wet = fairall(wet_state, _SST, _AIRT, 5.0, 2.0, 1.0e-6)

    assert wet[0] != pytest.approx(0.0)
    assert wet[1] > dry[1]
    assert wet[2] > dry[2]
    assert wet[4] < dry[4]
