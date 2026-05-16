"""Compact MyLake-style lake-ice thermodynamics.

This implementation follows the MyLake family of slab-ice ideas used by
Saloranta and Andersen (2007): supercooling first creates frazil ice, frazil
consolidates into solid ice, conductive Stefan growth acts through the ice slab,
and positive surface or basal energy melts existing ice. The kernel exposes the
same state variables as the other pyGOTM ice models for shared diagnostics.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from pygotm.icethm._util import freezing_temperature
from pygotm.icethm.constants import (
    ALB_ICE,
    ALB_OCEAN,
    C_WATER_VOL,
    K_ICE,
    L_ICE,
    MAX_FRAZIL,
    MIN_ICE_THICKNESS,
    MYLAKE_ATTN,
    RHO_ICE,
)


@numba.njit(cache=True)
def step_mylake(
    T_w: float,
    S_sfc: float,
    T_air: float,
    h_sfc: float,
    Qsw: float,
    Qh: float,
    Qe: float,
    Ql: float,
    dt: float,
    Hice: np.ndarray,
    Hfrazil: np.ndarray,
    Tice_surface: np.ndarray,
    ice_cover: np.ndarray,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
    Tf: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
    ocean_ice_salt_flux: np.ndarray,
) -> None:
    """Advance a single-column MyLake-style ice slab by one timestep."""

    tf = freezing_temperature(S_sfc)
    Tf[0] = tf
    water_depth = max(h_sfc, MIN_ICE_THICKNESS)
    latent_per_m = RHO_ICE * L_ICE
    supercool = tf - T_w
    if supercool > 0.0:
        Hfrazil[0] += C_WATER_VOL * water_depth * supercool / latent_per_m

    if Hice[0] <= 0.0 and Hfrazil[0] >= MAX_FRAZIL:
        Hice[0] = Hfrazil[0]
        Hfrazil[0] = 0.0

    ocean_heat = 0.0
    if Hice[0] > 0.0:
        if T_air < tf:
            denom = max(Hice[0], 0.05)
            Hice[0] += K_ICE * (tf - T_air) * dt / (latent_per_m * denom)

        net_surface = Qsw + Qh + Qe + Ql
        if net_surface > 0.0:
            melt = net_surface * dt / latent_per_m
            if melt > Hice[0]:
                melt = Hice[0]
            Hice[0] -= melt

        warm = T_w - tf
        if warm > 0.0 and Hice[0] > 0.0:
            basal = C_WATER_VOL * water_depth * warm / latent_per_m
            if basal > Hice[0]:
                basal = Hice[0]
            Hice[0] -= basal
            ocean_heat = basal * latent_per_m / dt

    if Hice[0] <= MIN_ICE_THICKNESS:
        Hice[0] = 0.0
        ice_cover[0] = 0
        albedo_ice[0] = ALB_OCEAN
        transmissivity[0] = 1.0
        Tice_surface[0] = tf
    else:
        ice_cover[0] = 2
        albedo_ice[0] = ALB_ICE
        transmissivity[0] = math.exp(-Hice[0] * MYLAKE_ATTN)
        Tice_surface[0] = min(tf, T_air)

    if Hfrazil[0] < 0.0:
        Hfrazil[0] = 0.0
    ocean_ice_heat_flux[0] = ocean_heat
    ocean_ice_salt_flux[0] = 0.0
