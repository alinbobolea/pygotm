"""Winton three-layer sea-ice thermodynamics.

Winton (2000) reformulates a three-layer sea-ice model with variable heat
capacity. This module implements the same prognostic structure used by the
paper: snow thickness, ice thickness, upper-layer temperature, lower-layer
temperature, diagnosed surface temperature, optics, and top/bottom melt energy.

The surface flux callback used by ocean components is represented here by an
explicit linearized surface flux pair ``A + B * Ts`` so pyGOTM can pass already
computed air-sea flux components into a Numba kernel.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from pygotm.icethm._util import clamp01, freezing_temperature_winton
from pygotm.icethm.constants import (
    ALB_ICE,
    ALB_OCEAN,
    ALB_SNOW,
    C_ICE,
    K_ICE,
    K_SNOW,
    L_ICE,
    MIN_ICE_THICKNESS,
    OPT_DEP_ICE,
    PEN_ICE,
    RHO_ICE,
    RHO_SNOW,
    T_RANGE_MELT,
)


@numba.njit(cache=True)
def ice_optics(
    hice: float,
    hsnow: float,
    Ts: float,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
) -> None:
    """Update Winton shortwave albedo and penetrating transmissivity."""

    if hice <= 0.0:
        albedo_ice[0] = ALB_OCEAN
        transmissivity[0] = 1.0
        return

    if hsnow > 0.0:
        albedo = ALB_SNOW
    else:
        f_h = clamp01(hice / 0.5)
        albedo = f_h * ALB_ICE + (1.0 - f_h) * ALB_OCEAN
        if Ts > -T_RANGE_MELT:
            albedo -= 0.075 * clamp01((Ts + T_RANGE_MELT) / T_RANGE_MELT)

    albedo_ice[0] = clamp01(albedo)
    transmissivity[0] = PEN_ICE * math.exp(-hice / OPT_DEP_ICE)


def linearize_surface_flux(
    Ts0: float,
    Qsw_absorbed: float,
    Ql: float,
    Qh: float,
    Qe: float,
) -> tuple[float, float]:
    """Return ``A, B`` for a local surface flux approximation ``A + B * Ts``.

    The available pyGOTM flux components are already evaluated for the current
    surface state. In the absence of a callback that can re-evaluate turbulent
    fluxes at perturbed ice surface temperatures, the linearization keeps the
    current net flux exact at ``Ts0`` and uses a conservative negative slope.
    """

    slope = -4.0
    intercept = Qsw_absorbed + Ql + Qh + Qe - slope * Ts0
    return intercept, slope


@numba.njit(cache=True)
def even_up(
    Hice: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    bottom_energy: np.ndarray,
) -> None:
    """Move excess lower-layer heat into melt energy and keep layers ordered."""

    if Hice[0] <= 0.0:
        T1[0] = 0.0
        T2[0] = 0.0
        return

    if T2[0] > 0.0:
        bottom_energy[0] += RHO_ICE * C_ICE * Hice[0] * T2[0] * 0.5
        T2[0] = 0.0
    if T1[0] > 0.0:
        bottom_energy[0] += RHO_ICE * C_ICE * Hice[0] * T1[0] * 0.5
        T1[0] = 0.0
    if T1[0] < T2[0]:
        avg = 0.5 * (T1[0] + T2[0])
        T1[0] = avg
        T2[0] = avg


@numba.njit(cache=True)
def ice3lay_temp(
    Tf: float,
    dt: float,
    A_flux: float,
    B_flux: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    Tice_surface: np.ndarray,
    surface_energy: np.ndarray,
    bottom_energy: np.ndarray,
) -> None:
    """Advance Winton upper/lower ice temperatures with conductive coupling."""

    hice = Hice[0]
    if hice <= 0.0:
        Tice_surface[0] = Tf
        T1[0] = Tf
        T2[0] = Tf
        return

    hsnow = Hsnow[0]
    k12 = 4.0 * K_ICE * K_SNOW / max(K_SNOW * hice + 4.0 * K_ICE * hsnow, 1.0e-12)
    k32 = 2.0 * K_ICE / max(hice, MIN_ICE_THICKNESS)
    denom = B_flux - k12
    if abs(denom) < 1.0e-12:
        Ts = T1[0]
    else:
        Ts = (k12 * T1[0] - A_flux) / denom
    if Ts > 0.0:
        surface_energy[0] += (A_flux + B_flux * 0.0 - k12 * (T1[0] - 0.0)) * dt
        Ts = 0.0

    heatcap = RHO_ICE * C_ICE * max(0.5 * hice, MIN_ICE_THICKNESS)
    d1 = (k12 * (Ts - T1[0]) + k32 * (T2[0] - T1[0])) * dt / heatcap
    d2 = (k32 * (T1[0] - T2[0]) + k32 * (Tf - T2[0])) * dt / heatcap
    T1[0] += d1
    T2[0] += d2
    if T1[0] > 0.0:
        surface_energy[0] += heatcap * T1[0]
        T1[0] = 0.0
    if T2[0] > 0.0:
        bottom_energy[0] += heatcap * T2[0]
        T2[0] = 0.0
    Tice_surface[0] = Ts
    even_up(Hice, T1, T2, bottom_energy)


@numba.njit(cache=True)
def ice3lay_resize(
    Tf: float,
    dt: float,
    precip: float,
    ocean_ice_flux: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    surface_energy: np.ndarray,
    bottom_energy: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
) -> None:
    """Resize snow and ice after surface and basal energy updates."""

    if precip > 0.0 and Hice[0] > 0.0:
        Hsnow[0] += precip * dt * 1000.0 / RHO_SNOW

    latent = RHO_ICE * L_ICE
    if surface_energy[0] > 0.0 and Hice[0] > 0.0:
        surface_melt = surface_energy[0] / latent
        if Hsnow[0] > 0.0:
            snow_equiv = surface_melt * RHO_ICE / RHO_SNOW
            used = min(Hsnow[0], snow_equiv)
            Hsnow[0] -= used
            surface_melt -= used * RHO_SNOW / RHO_ICE
        if surface_melt > 0.0:
            if surface_melt > Hice[0]:
                surface_melt = Hice[0]
            Hice[0] -= surface_melt
        surface_energy[0] = 0.0

    basal_energy = bottom_energy[0] + ocean_ice_flux * dt
    ocean_ice_heat_flux[0] = ocean_ice_flux
    if Hice[0] > 0.0 and basal_energy > 0.0:
        basal_melt = basal_energy / latent
        if basal_melt > Hice[0]:
            basal_melt = Hice[0]
        Hice[0] -= basal_melt
        bottom_energy[0] = 0.0
    elif basal_energy < 0.0:
        growth = -basal_energy / latent
        Hice[0] += growth
        bottom_energy[0] = 0.0

    freeboard = Hice[0] * (RHO_ICE / 1025.0 - 1.0) + Hsnow[0] * RHO_SNOW / 1025.0
    if freeboard > 0.0 and Hsnow[0] > 0.0:
        converted_snow = min(Hsnow[0], freeboard * 1025.0 / RHO_SNOW)
        Hsnow[0] -= converted_snow
        Hice[0] += converted_snow * RHO_SNOW / RHO_ICE

    if Hice[0] <= MIN_ICE_THICKNESS:
        Hice[0] = 0.0
        Hsnow[0] = 0.0
        T1[0] = Tf
        T2[0] = Tf


@numba.njit(cache=True)
def step_winton(
    T_w: float,
    S_w: float,
    h_sfc: float,
    dt: float,
    Qsw: float,
    Ql: float,
    Qh: float,
    Qe: float,
    precip: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    Tice_surface: np.ndarray,
    ice_cover: np.ndarray,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
    Tf: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
    ocean_ice_flux: np.ndarray,
    ocean_ice_salt_flux: np.ndarray,
    surface_ice_energy: np.ndarray,
    bottom_ice_energy: np.ndarray,
) -> None:
    """Advance Winton three-layer ice state by one timestep."""

    tf = freezing_temperature_winton(S_w)
    Tf[0] = tf

    if Hice[0] <= 0.0 and T_w <= tf:
        Hice[0] = max(MIN_ICE_THICKNESS, -C_ICE * h_sfc * (T_w - tf) / L_ICE)
        T1[0] = tf
        T2[0] = tf

    if Hice[0] <= 0.0:
        ice_cover[0] = 0
        albedo_ice[0] = ALB_OCEAN
        transmissivity[0] = 1.0
        Tice_surface[0] = tf
        ocean_ice_heat_flux[0] = 0.0
        ocean_ice_salt_flux[0] = 0.0
        return

    ice_cover[0] = 2
    ice_optics(Hice[0], Hsnow[0], Tice_surface[0], albedo_ice, transmissivity)
    qsw_absorbed = Qsw * (1.0 - albedo_ice[0]) * (1.0 - transmissivity[0])
    A_flux = qsw_absorbed + Ql + Qh + Qe
    B_flux = -4.0
    ice3lay_temp(
        tf,
        dt,
        A_flux,
        B_flux,
        Hice,
        Hsnow,
        T1,
        T2,
        Tice_surface,
        surface_ice_energy,
        bottom_ice_energy,
    )
    ice3lay_resize(
        tf,
        dt,
        precip,
        ocean_ice_flux[0],
        Hice,
        Hsnow,
        T1,
        T2,
        surface_ice_energy,
        bottom_ice_energy,
        ocean_ice_heat_flux,
    )
    ice_optics(Hice[0], Hsnow[0], Tice_surface[0], albedo_ice, transmissivity)
    ocean_ice_salt_flux[0] = 0.0
    if Hice[0] <= 0.0:
        ice_cover[0] = 0
    else:
        ice_cover[0] = 2
