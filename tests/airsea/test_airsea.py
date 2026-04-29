"""Tests for pygotm.airsea.airsea."""

from __future__ import annotations

import math

import pytest

from pygotm.airsea.airsea import (
    AirSeaDriverState,
    do_airsea,
    init_airsea,
    integrated_fluxes,
    post_init_airsea,
    set_sst,
    set_ssuv,
    surface_fluxes,
)
from pygotm.airsea.airsea_fluxes import KONDO, airsea_fluxes
from pygotm.airsea.airsea_variables import CLARK, CONST, PAYNE
from pygotm.airsea.albedo_water import albedo_water
from pygotm.airsea.humidity import humidity
from pygotm.airsea.longwave_radiation import longwave_radiation
from pygotm.airsea.shortwave_radiation import shortwave_radiation
from pygotm.airsea.solar_zenith_angle import solar_zenith_angle
from pygotm.constants import KELVIN_OFFSET_C

_AIRP = 101325.0
_AIRT = 15.0
_SST = 18.0
_HUM = 75.0
_CLOUD = 0.3
_U10 = 5.0
_V10 = 2.0
_LAT = 45.0
_LON = -20.0
_YEARDAY = 100
_SECS = 12 * 3600


def _make_driver() -> AirSeaDriverState:
    state = AirSeaDriverState()
    init_airsea(state)
    post_init_airsea(state, _LAT, _LON)
    return state


def test_import_and_instantiate() -> None:
    state = AirSeaDriverState()
    assert state is not None


def test_init_airsea_overrides_and_rejects_unknown_fields() -> None:
    state = AirSeaDriverState()
    init_airsea(state, fluxes_method=KONDO, hum_method=4, const_albedo=0.2)
    assert state.fluxes_method == KONDO
    assert state.hum_method == 4
    assert state.const_albedo == pytest.approx(0.2)

    with pytest.raises(AttributeError, match="unknown airsea configuration field"):
        init_airsea(state, not_a_field=1)


def test_post_init_airsea_resets_runtime_state_and_stores_location() -> None:
    state = AirSeaDriverState()
    state.w = 5.0
    state.tx = 1.0
    state.int_total = 12.0
    post_init_airsea(state, _LAT, _LON)
    assert state.w == pytest.approx(0.0)
    assert state.tx == pytest.approx(0.0)
    assert state.int_total == pytest.approx(0.0)
    assert state.dlat == pytest.approx(_LAT)
    assert state.dlon == pytest.approx(_LON)


def test_set_sst_and_set_ssuv_follow_ssuv_method() -> None:
    state = _make_driver()
    set_sst(state, _SST)
    assert state.sst == pytest.approx(_SST)

    set_ssuv(state, 0.4, -0.1)
    assert state.ssu == pytest.approx(0.4)
    assert state.ssv == pytest.approx(-0.1)

    state.ssuv_method = 0
    set_ssuv(state, 1.0, 1.0)
    assert state.ssu == pytest.approx(0.4)
    assert state.ssv == pytest.approx(-0.1)


def test_do_airsea_bulk_fluxes_matches_component_routines() -> None:
    state = _make_driver()
    init_airsea(
        state,
        fluxes_method=KONDO,
        shortwave_method=3,
        albedo_method=PAYNE,
        longwave_method=CLARK,
    )
    set_sst(state, _SST)
    do_airsea(
        state,
        yearday=_YEARDAY,
        secs=_SECS,
        airp=_AIRP,
        airt=_AIRT,
        hum=_HUM,
        cloud=_CLOUD,
        u10=_U10,
        v10=_V10,
    )

    reference = AirSeaDriverState()
    post_init_airsea(reference, _LAT, _LON)
    set_sst(reference, _SST)
    humidity(reference, state.hum_method, _HUM, _AIRP, _SST, _AIRT)
    ql = longwave_radiation(
        reference,
        state.longwave_method,
        _LAT,
        _SST + KELVIN_OFFSET_C,
        _AIRT + KELVIN_OFFSET_C,
        _CLOUD,
    )
    evap, tx, ty, qe, qh = airsea_fluxes(
        state.fluxes_method,
        reference,
        _SST,
        _AIRT,
        _U10,
        _V10,
        0.0,
    )
    zenith = solar_zenith_angle(_YEARDAY, _SECS / 3600.0, _LON, _LAT)
    shortwave = shortwave_radiation(zenith, _YEARDAY, _LON, _LAT, _CLOUD)
    albedo = albedo_water(PAYNE, zenith, _YEARDAY)

    assert state.evap == pytest.approx(evap, rel=1.0e-12)
    assert state.tx == pytest.approx(tx, rel=1.0e-12)
    assert state.ty == pytest.approx(ty, rel=1.0e-12)
    assert state.qe == pytest.approx(qe, rel=1.0e-12)
    assert state.qh == pytest.approx(qh, rel=1.0e-12)
    assert state.ql == pytest.approx(ql, rel=1.0e-12)
    assert state.heat == pytest.approx(ql + qe + qh, rel=1.0e-12)
    assert state.shortwave == pytest.approx(shortwave, rel=1.0e-12)
    assert state.albedo == pytest.approx(albedo, rel=1.0e-12)
    assert state.w == pytest.approx(math.hypot(_U10, _V10), rel=1.0e-12)


def test_do_airsea_applies_calculated_flux_scale_factors() -> None:
    state = _make_driver()
    init_airsea(
        state,
        fluxes_method=KONDO,
        shortwave_method=3,
        albedo_method=PAYNE,
        longwave_method=CLARK,
        heat_scale_factor=1.2,
        shortwave_scale_factor=0.97,
    )
    set_sst(state, _SST)
    do_airsea(
        state,
        yearday=_YEARDAY,
        secs=_SECS,
        airp=_AIRP,
        airt=_AIRT,
        hum=_HUM,
        cloud=_CLOUD,
        u10=_U10,
        v10=_V10,
    )

    unscaled_heat = state.ql + state.qe + state.qh
    zenith = solar_zenith_angle(_YEARDAY, _SECS / 3600.0, _LON, _LAT)
    unscaled_shortwave = shortwave_radiation(zenith, _YEARDAY, _LON, _LAT, _CLOUD)

    assert state.heat == pytest.approx(1.2 * unscaled_heat, rel=1.0e-12)
    assert state.shortwave == pytest.approx(0.97 * unscaled_shortwave, rel=1.0e-12)


def test_ssuv_absolute_ignores_surface_current() -> None:
    """ssuv_method=0 (absolute): set_ssuv is a no-op; effective wind equals the 10-m wind."""
    state = _make_driver()
    init_airsea(state, fluxes_method=KONDO, ssuv_method=0)
    set_sst(state, _SST)
    set_ssuv(state, 1.0, 0.0)  # 1 m/s surface current — must be ignored
    do_airsea(
        state,
        yearday=_YEARDAY,
        secs=_SECS,
        airp=_AIRP,
        airt=_AIRT,
        hum=_HUM,
        cloud=_CLOUD,
        u10=5.0,
        v10=0.0,
    )
    assert state.w == pytest.approx(5.0)  # hypot(5-0, 0-0): ssu/ssv unchanged by set_ssuv


def test_ssuv_relative_subtracts_surface_current_from_wind() -> None:
    """ssuv_method=1 (relative): surface current is subtracted from wind; effective speed is reduced."""
    state = _make_driver()
    init_airsea(state, fluxes_method=KONDO, ssuv_method=1)
    set_sst(state, _SST)
    set_ssuv(state, 1.0, 0.0)  # 1 m/s eastward surface current
    do_airsea(
        state,
        yearday=_YEARDAY,
        secs=_SECS,
        airp=_AIRP,
        airt=_AIRT,
        hum=_HUM,
        cloud=_CLOUD,
        u10=5.0,
        v10=0.0,
    )
    assert state.w == pytest.approx(4.0)  # hypot(5-1, 0-0) = 4.0


def test_do_airsea_requires_meteorology_when_bulk_fluxes_enabled() -> None:
    state = _make_driver()
    init_airsea(state, fluxes_method=KONDO)
    with pytest.raises(ValueError, match="airp, airt, and hum are required"):
        do_airsea(state, yearday=_YEARDAY, secs=_SECS)


def test_do_airsea_prescribed_fluxes_keep_w_zero_and_apply_drag_scaling() -> None:
    state = _make_driver()
    init_airsea(
        state,
        fluxes_method=0,
        shortwave_method=2,
        shortwave_type=2,
        albedo_method=CONST,
        const_albedo=0.12,
    )
    state.bio_drag_scale = 1.5
    do_airsea(
        state,
        yearday=_YEARDAY,
        secs=_SECS,
        shortwave=250.0,
        heat=-80.0,
        tx=0.2,
        ty=-0.1,
    )
    assert state.shortwave == pytest.approx(250.0)
    assert state.heat == pytest.approx(-80.0)
    assert state.tx == pytest.approx(0.3)
    assert state.ty == pytest.approx(-0.15)
    assert state.albedo == pytest.approx(0.12)
    assert state.w == pytest.approx(0.0)


def test_surface_fluxes_returns_state_components() -> None:
    state = _make_driver()
    init_airsea(state, fluxes_method=KONDO, longwave_method=CLARK)
    sensible, latent, longwave = surface_fluxes(
        state,
        _SST,
        airp=_AIRP,
        airt=_AIRT,
        hum=_HUM,
        cloud=_CLOUD,
        u10=_U10,
        v10=_V10,
    )
    assert sensible == pytest.approx(state.qh, rel=1.0e-12)
    assert latent == pytest.approx(state.qe, rel=1.0e-12)
    assert longwave == pytest.approx(state.ql, rel=1.0e-12)


def test_integrated_fluxes_accumulate_running_totals() -> None:
    state = _make_driver()
    state.precip = 2.0e-6
    state.evap = -1.0e-6
    state.shortwave = 200.0
    state.heat = -50.0
    integrated_fluxes(state, 10.0)
    assert state.int_precip == pytest.approx(2.0e-5)
    assert state.int_evap == pytest.approx(-1.0e-5)
    assert state.int_fwf == pytest.approx(1.0e-5)
    assert state.int_swr == pytest.approx(2000.0)
    assert state.int_heat == pytest.approx(-500.0)
    assert state.int_total == pytest.approx(1500.0)


def test_integrated_fluxes_can_use_net_shortwave_after_albedo() -> None:
    state = _make_driver()
    state.shortwave = 200.0
    state.heat = -50.0

    integrated_fluxes(state, 10.0, shortwave=160.0)

    assert state.int_swr == pytest.approx(1600.0)
    assert state.int_heat == pytest.approx(-500.0)
    assert state.int_total == pytest.approx(1100.0)


def test_sst_obs_overrides_output_sst() -> None:
    state = _make_driver()
    init_airsea(state, fluxes_method=0)
    set_sst(state, _SST)
    do_airsea(
        state,
        yearday=_YEARDAY,
        secs=_SECS,
        shortwave=100.0,
        heat=-20.0,
        sst_obs=12.5,
    )
    assert state.sst == pytest.approx(12.5)
