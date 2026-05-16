# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: airsea --- atmospheric fluxes \label{sec:airsea}
!
! !DESCRIPTION:
!  This module calculates the heat, momentum
!  and freshwater fluxes between the ocean and the atmosphere as well as
!  the incoming solar radiation. Fluxes and solar radiation may be
!  prescribed. Alternatively, they may be calculated by means
!  of bulk formulae from observed or modelled meteorological
!  parameters and the solar radiation may be calculated
!  from longitude, latitude, time and cloudiness.
!  Albedo correction is applied according to a configuration variable.
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_airsea, post_init_airsea
!   public do_airsea
!   public clean_airsea
!   public set_sst
!   public set_ssuv
!   public surface_fluxes
!   public integrated_fluxes
!EOP
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from pygotm.airsea.airsea_fluxes import airsea_fluxes
from pygotm.airsea.airsea_variables import CLARK, CONST, PAYNE, AirSeaState, bolz, emiss
from pygotm.airsea.albedo_water import albedo_water
from pygotm.airsea.humidity import humidity
from pygotm.airsea.longwave_radiation import longwave_radiation
from pygotm.airsea.shortwave_radiation import shortwave_radiation
from pygotm.airsea.solar_zenith_angle import solar_zenith_angle
from pygotm.constants import KELVIN_OFFSET_C

__all__ = [
    "AirSeaDriverState",
    "clean_airsea",
    "do_airsea",
    "init_airsea",
    "integrated_fluxes",
    "post_init_airsea",
    "set_sst",
    "set_ssuv",
    "surface_fluxes",
]

ScalarOverride = bool | int | float


class AirSeaDriverState(AirSeaState):
    """State and configuration for the translated ``airsea.F90`` driver."""

    def __init__(self) -> None:
        super().__init__()

        self.shortwave_method: int = 1
        self.shortwave_type: int = 1
        self.shortwave_scale_factor: float = 1.0
        self.longwave_method: int = CLARK
        self.longwave_type: int = 1
        self.hum_method: int = 1
        self.albedo_method: int = PAYNE
        self.const_albedo: float = 0.0
        self.fluxes_method: int = 0
        self.heat_scale_factor: float = 1.0
        self.ssuv_method: int = 1

        self.dlon: float = 0.0
        self.dlat: float = 0.0

        self.w: float = 0.0
        self.shortwave: float = 0.0
        self.ql: float = 0.0
        self.albedo: float = 0.0
        self.heat: float = 0.0
        self.qe: float = 0.0
        self.qh: float = 0.0
        self.tx: float = 0.0
        self.ty: float = 0.0
        self.precip: float = 0.0
        self.evap: float = 0.0
        self.sst: float = 0.0
        self.sss: float = 0.0
        self.ssu: float = 0.0
        self.ssv: float = 0.0

        self.int_precip: float = 0.0
        self.int_evap: float = 0.0
        self.int_fwf: float = 0.0
        self.int_swr: float = 0.0
        self.int_heat: float = 0.0
        self.int_total: float = 0.0

        self.bio_drag_scale: float = 1.0
        self.bio_albedo: float = 0.0


def init_airsea(
    state: AirSeaDriverState,
    *,
    overrides: Mapping[str, ScalarOverride] | None = None,
    **keyword_overrides: ScalarOverride,
) -> None:
    """Apply scalar configuration overrides to ``state``."""

    merged_overrides: dict[str, ScalarOverride] = {}
    if overrides is not None:
        merged_overrides.update(overrides)
    merged_overrides.update(keyword_overrides)

    for name, value in merged_overrides.items():
        if not hasattr(state, name):
            msg = f"unknown airsea configuration field {name!r}"
            raise AttributeError(msg)
        setattr(state, name, value)


def post_init_airsea(state: AirSeaDriverState, lat: float, lon: float) -> None:
    """Reset runtime air-sea state and store latitude/longitude."""

    state.w = 0.0
    state.shortwave = 0.0
    state.albedo = 0.0
    state.heat = 0.0
    state.ql = 0.0
    state.qe = 0.0
    state.qh = 0.0
    state.tx = 0.0
    state.ty = 0.0
    state.precip = 0.0
    state.evap = 0.0
    state.sst = 0.0
    state.ssu = 0.0
    state.ssv = 0.0

    state.es = 0.0
    state.ea = 0.0
    state.qs = 0.0
    state.qa = 0.0
    state.L = 0.0
    state.rhoa = 0.0

    state.bio_drag_scale = 1.0
    state.bio_albedo = 0.0

    state.int_precip = 0.0
    state.int_evap = 0.0
    state.int_fwf = 0.0
    state.int_swr = 0.0
    state.int_heat = 0.0
    state.int_total = 0.0

    state.dlon = lon
    state.dlat = lat


def surface_fluxes(
    state: AirSeaDriverState,
    surface_temp: float,
    *,
    airp: float,
    airt: float,
    hum: float,
    cloud: float,
    u10: float,
    v10: float,
    precip: float = 0.0,
    longwave: float | None = None,
) -> tuple[float, float, float]:
    """Update the state from meteorology and return ``(qh, qe, ql)``."""

    set_sst(state, surface_temp)
    _flux_from_meteo(
        state,
        airp=airp,
        airt=airt,
        hum=hum,
        cloud=cloud,
        u10=u10,
        v10=v10,
        precip=precip,
        longwave=longwave,
    )
    return state.qh, state.qe, state.ql


def do_airsea(
    state: AirSeaDriverState,
    *,
    yearday: int,
    secs: int,
    airp: float | None = None,
    airt: float | None = None,
    hum: float | None = None,
    cloud: float = 0.0,
    u10: float = 0.0,
    v10: float = 0.0,
    precip: float = 0.0,
    shortwave: float | None = None,
    heat: float | None = None,
    tx: float | None = None,
    ty: float | None = None,
    longwave: float | None = None,
    sst_obs: float | None = None,
) -> None:
    """Perform one air-sea update using either bulk formulae or prescribed fluxes."""

    have_zenith_angle = False
    zenith_angle = 0.0
    state.precip = precip

    if state.fluxes_method != 0:
        if airp is None or airt is None or hum is None:
            msg = "airp, airt, and hum are required when fluxes_method != 0"
            raise ValueError(msg)
        _flux_from_meteo(
            state,
            airp=airp,
            airt=airt,
            hum=hum,
            cloud=cloud,
            u10=u10,
            v10=v10,
            precip=precip,
            longwave=longwave,
        )

        if state.shortwave_method == 3:
            zenith_angle = solar_zenith_angle(
                yearday, secs / 3600.0, state.dlon, state.dlat
            )
            have_zenith_angle = True
            state.shortwave = (
                shortwave_radiation(
                    zenith_angle,
                    yearday,
                    state.dlon,
                    state.dlat,
                    cloud,
                )
                * state.shortwave_scale_factor
            )
        state.heat = state.heat * state.heat_scale_factor
    else:
        state.qe = 0.0
        state.qh = 0.0
        state.ql = 0.0
        state.heat = 0.0 if heat is None else heat
        state.tx = 0.0 if tx is None else tx
        state.ty = 0.0 if ty is None else ty
        state.w = 0.0

    if state.shortwave_method != 3:
        state.shortwave = 0.0 if shortwave is None else shortwave

    if state.shortwave_method == 3 or state.shortwave_type == 2:
        if not have_zenith_angle:
            zenith_angle = solar_zenith_angle(
                yearday, secs / 3600.0, state.dlon, state.dlat
            )
        if state.albedo_method == CONST:
            state.albedo = state.const_albedo
        else:
            state.albedo = albedo_water(state.albedo_method, zenith_angle, yearday)
    else:
        state.albedo = 0.0

    state.tx = state.tx * state.bio_drag_scale
    state.ty = state.ty * state.bio_drag_scale

    if sst_obs is not None:
        state.sst = sst_obs


def clean_airsea(state: AirSeaDriverState) -> None:
    """Finalize the air-sea module.

    The translated Phase 4 driver has no open-file resources yet, so this is a no-op.
    """

    del state


def integrated_fluxes(
    state: AirSeaDriverState,
    dt: float,
    *,
    shortwave: float | None = None,
) -> None:
    """Integrate freshwater and heat fluxes over ``dt`` seconds."""

    state.int_precip = state.int_precip + dt * state.precip
    state.int_evap = state.int_evap + dt * state.evap
    state.int_fwf = state.int_precip + state.int_evap
    state.int_swr = state.int_swr + dt * (
        state.shortwave if shortwave is None else shortwave
    )
    state.int_heat = state.int_heat + dt * state.heat
    state.int_total = state.int_swr + state.int_heat


def set_sst(state: AirSeaDriverState, temp: float) -> None:
    """Set the model sea-surface temperature used by bulk flux calculations."""

    state.sst = temp


def set_ssuv(state: AirSeaDriverState, uvel: float, vvel: float) -> None:
    """Set the surface current used to form relative wind speed."""

    if state.ssuv_method != 0:
        state.ssu = uvel
        state.ssv = vvel


def _flux_from_meteo(
    state: AirSeaDriverState,
    *,
    airp: float,
    airt: float,
    hum: float,
    cloud: float,
    u10: float,
    v10: float,
    precip: float,
    longwave: float | None,
) -> None:
    if state.sst < 100.0:
        tw = state.sst
        tw_k = state.sst + KELVIN_OFFSET_C
    else:
        tw = state.sst - KELVIN_OFFSET_C
        tw_k = state.sst

    if airt < 100.0:
        ta = airt
        ta_k = airt + KELVIN_OFFSET_C
    else:
        ta = airt - KELVIN_OFFSET_C
        ta_k = airt

    humidity(state, state.hum_method, hum, airp, tw, ta)

    if state.longwave_method == 0:
        state.ql = 0.0 if longwave is None else longwave
    elif state.longwave_method == 2:
        if longwave is None:
            msg = "longwave is required when longwave_method == 2"
            raise ValueError(msg)
        if state.longwave_type == 1:
            state.ql = longwave
        else:
            state.ql = longwave - bolz * emiss * (tw_k**4)
    else:
        state.ql = longwave_radiation(
            state,
            state.longwave_method,
            state.dlat,
            tw_k,
            ta_k,
            cloud,
        )

    state.evap, state.tx, state.ty, state.qe, state.qh = airsea_fluxes(
        state.fluxes_method,
        state,
        tw,
        ta,
        u10 - state.ssu,
        v10 - state.ssv,
        precip,
    )
    state.heat = state.ql + state.qe + state.qh
    state.w = math.hypot(u10 - state.ssu, v10 - state.ssv)
