"""Tests for pygotm.airsea.kondo."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.airsea_variables import (
    AirSeaState,
    const06,
    cpa,
    cpw,
    kelvin,
    rgas,
    rho_0,
)
from pygotm.airsea.humidity import humidity
from pygotm.airsea.kondo import kondo

_AIRP = 101325.0
_SST = 18.0
_AIRT = 15.0


def _make_state() -> AirSeaState:
    state = AirSeaState()
    humidity(state, 1, 75.0, _AIRP, _SST, _AIRT)
    return state


def _expected_kondo(
    state: AirSeaState,
    sst: float,
    airt: float,
    u10: float,
    v10: float,
    precip: float,
) -> tuple[float, float, float, float, float]:
    ae_d = (0.0, 0.771, 0.867, 1.2, 0.0)
    ae_h = (0.0, 0.927, 1.15, 1.17, 1.652)
    ae_e = (0.0, 0.969, 1.18, 1.196, 1.68)
    be_d = (1.08, 0.0858, 0.0667, 0.025, 0.073)
    be_h = (1.185, 0.0546, 0.01, 0.0075, -0.017)
    be_e = (1.23, 0.0521, 0.01, 0.008, -0.016)
    ce_h = (0.0, 0.0, 0.0, -0.00045, 0.0)
    ce_e = (0.0, 0.0, 0.0, -0.0004, 0.0)
    pe_d = (-0.15, 1.0, 1.0, 1.0, 1.0)
    pe_h = (-0.157, 1.0, 1.0, 1.0, 1.0)
    pe_e = (-0.16, 1.0, 1.0, 1.0, 1.0)
    eps = 1.0e-12

    evap = 0.0
    w = math.hypot(u10, v10)
    latent_heat = (2.5 - 0.00234 * sst) * 1.0e6

    if sst < 100.0:
        tw = sst
    else:
        tw = sst - kelvin

    if airt < 100.0:
        ta = airt
        ta_k = airt + kelvin
    else:
        ta = airt - kelvin
        ta_k = airt

    s0 = 0.25 * (sst - airt) / (w + 1.0e-10) ** 2
    s = s0 * abs(s0) / (abs(s0) + 0.01)

    if w < 2.2:
        x = math.log(w + eps)
        cdd = (be_d[0] * math.exp(pe_d[0] * x)) * 1.0e-3
        chd = (be_h[0] * math.exp(pe_h[0] * x)) * 1.0e-3
        ced = (be_e[0] * math.exp(pe_e[0] * x)) * 1.0e-3
    elif w < 5.0:
        x = math.exp(math.log(w + eps))
        cdd = (ae_d[1] + be_d[1] * x) * 1.0e-3
        chd = (ae_h[1] + be_h[1] * x) * 1.0e-3
        ced = (ae_e[1] + be_e[1] * x) * 1.0e-3
    elif w < 8.0:
        x = math.exp(math.log(w + eps))
        cdd = (ae_d[2] + be_d[2] * x) * 1.0e-3
        chd = (ae_h[2] + be_h[2] * x) * 1.0e-3
        ced = (ae_e[2] + be_e[2] * x) * 1.0e-3
    elif w < 25.0:
        x = math.exp(math.log(w + eps))
        cdd = (ae_d[3] + be_d[3] * x) * 1.0e-3
        chd = (ae_h[3] + be_h[3] * x + ce_h[3] * (w - 8.0) ** 2) * 1.0e-3
        ced = (ae_e[3] + be_e[3] * x + ce_e[3] * (w - 8.0) ** 2) * 1.0e-3
    else:
        x = math.exp(math.log(w + eps))
        cdd = (ae_d[4] + be_d[4] * x) * 1.0e-3
        chd = (ae_h[4] + be_h[4] * x) * 1.0e-3
        ced = (ae_e[4] + be_e[4] * x) * 1.0e-3

    if s < 0.0:
        if s > -3.3:
            x = 0.1 + 0.03 * s + 0.9 * math.exp(4.8 * s)
        else:
            x = 0.0
        cdd = x * cdd
        chd = x * chd
        ced = x * ced
    else:
        cdd = cdd * (1.0 + 0.47 * math.sqrt(s))
        chd = chd * (1.0 + 0.63 * math.sqrt(s))
        ced = ced * (1.0 + 0.63 * math.sqrt(s))

    qh = -chd * cpa * state.rhoa * w * (sst - airt)
    qe = -ced * latent_heat * state.rhoa * w * (state.qs - state.qa)

    if state.rain_impact:
        rainfall = precip * 1000.0
        x1 = 2.11e-5 * (ta_k / kelvin) ** 1.94
        x2 = 0.02411 * (1.0 + ta * (3.309e-3 - 1.44e-6 * ta)) / (state.rhoa * cpa)
        x3 = state.qa * latent_heat / (rgas * ta_k * ta_k)
        cd_rain = 1.0 / (1.0 + const06 * (x3 * latent_heat * x1) / (cpa * x2))
        cd_rain = (
            cd_rain * cpw * ((tw - ta) + (state.qs - state.qa) * latent_heat / cpa)
        )
        qh = qh - rainfall * cd_rain
        if state.calc_evaporation:
            evap = state.rhoa / rho_0 * ced * w * (state.qa - state.qs)

    tmp = cdd * state.rhoa * w
    taux = tmp * u10
    tauy = tmp * v10

    if state.rain_impact:
        rainfall = precip * 1000.0
        tmp = 0.85 * rainfall
        taux = taux + tmp * u10
        tauy = tauy + tmp * v10

    return evap, taux, tauy, qe, qh


def test_import_and_smoke() -> None:
    state = _make_state()
    outputs = kondo(state, _SST, _AIRT, 5.0, 2.0, 0.0)
    assert all(math.isfinite(value) for value in outputs)


@pytest.mark.parametrize("wind_speed", [1.0, 3.0, 6.0, 10.0, 30.0])
def test_piecewise_wind_branches_match_fortran_formula(wind_speed: float) -> None:
    state = _make_state()
    expected = _expected_kondo(state, _SST, _AIRT, wind_speed, 0.0, 0.0)
    actual = kondo(state, _SST, _AIRT, wind_speed, 0.0, 0.0)
    assert actual == pytest.approx(expected, rel=1.0e-12)


def test_zero_wind_returns_zero_fluxes_and_stresses() -> None:
    state = _make_state()
    assert kondo(state, _SST, _AIRT, 0.0, 0.0, 0.0) == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.0)
    )


def test_rain_impact_and_evaporation_branch_matches_formula() -> None:
    state = _make_state()
    state.rain_impact = True
    state.calc_evaporation = True
    expected = _expected_kondo(state, _SST, _AIRT, 5.0, 2.0, 1.0e-6)
    actual = kondo(state, _SST, _AIRT, 5.0, 2.0, 1.0e-6)
    assert actual == pytest.approx(expected, rel=1.0e-12)


def test_stable_and_unstable_cases_remain_finite() -> None:
    state = _make_state()
    unstable = kondo(state, 20.0, 15.0, 5.0, 0.0, 0.0)
    stable = kondo(state, 10.0, 15.0, 5.0, 0.0, 0.0)
    assert all(math.isfinite(value) for value in unstable)
    assert all(math.isfinite(value) for value in stable)
