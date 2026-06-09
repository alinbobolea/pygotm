"""Compact MyLake-style lake-ice thermodynamics.

This implementation translates the STIM MyLake slab-ice routine used by
GOTM-Lake. It preserves the same surface and basal growth diagnostics and
mutates the surface-water temperature in the same places as the Fortran
``Tw`` argument.
"""

import math

import numba
import numpy as np

from pygotm.icethm._util import freezing_temperature
from pygotm.icethm.constants import (
    C_WATER_VOL,
    L_ICE,
    MAX_FRAZIL,
    MYLAKE_ALBEDO,
    MYLAKE_K_ICE,
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
    precip: float,
    Hice: np.ndarray,
    Hfrazil: np.ndarray,
    dHis: np.ndarray,
    dHib: np.ndarray,
    Tice_surface: np.ndarray,
    ice_cover: np.ndarray,
    albedo_ice: np.ndarray,
    attenuation_ice: np.ndarray,
    transmissivity: np.ndarray,
    Tf: np.ndarray,
    ocean_ice_flux: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
    ocean_ice_salt_flux: np.ndarray,
    bottom_ice_energy: np.ndarray,
) -> float:
    """Advance a single-column MyLake ice slab and return updated water T."""

    tf = freezing_temperature(S_sfc)
    Tf[0] = tf
    water_depth = h_sfc
    if water_depth <= 0.0:
        water_depth = 1.0e-12
    latent_per_m = RHO_ICE * L_ICE

    bottom_ice_energy[0] = (T_w - tf) * water_depth * C_WATER_VOL
    bottom_ice_energy[0] += ocean_ice_flux[0] * dt
    dHis[0] = 0.0
    dHib[0] = -bottom_ice_energy[0] / latent_per_m

    if ice_cover[0] == 0:
        if dHib[0] > 0.0:
            Hfrazil[0] += dHib[0] + dt * precip
        if Hfrazil[0] >= MAX_FRAZIL:
            ice_cover[0] = 2
            Hice[0] = Hfrazil[0]
            Hfrazil[0] = 0.0
        if Hfrazil[0] < 0.0:
            T_w = -Hfrazil[0] * latent_per_m / (water_depth * C_WATER_VOL) + tf
            Hfrazil[0] = 0.0
    else:
        T_w = tf
        if T_air < tf:
            dHis[0] = Hice[0]
            alpha = 1.0 / (10.0 * Hice[0])
            Tice_surface[0] = (alpha * tf + T_air) / (1.0 + alpha)
            growth_arg = Hice[0] * Hice[0]
            growth_arg += (
                2.0 * MYLAKE_K_ICE / latent_per_m * dt * (tf - Tice_surface[0])
            )
            if growth_arg < 0.0:
                Hice[0] = 0.0
            else:
                Hice[0] = math.sqrt(growth_arg)
            dHis[0] = Hice[0] - dHis[0]
        else:
            Tice_surface[0] = 0.0
            qflux = Qh + Qe + Ql
            dHis[0] = -dt * (Qsw + qflux) / latent_per_m
            if dHis[0] < 0.0:
                Hice[0] += dHis[0]

        Hice[0] += dt * precip
        Hice[0] += dHib[0]

        if Hice[0] <= 0.0:
            T_w = -Hice[0] * latent_per_m / (water_depth * C_WATER_VOL) + tf
            ice_cover[0] = 0
            Hice[0] = 0.0
            attenuation_ice[0] = 0.0
            transmissivity[0] = 1.0
        else:
            albedo_ice[0] = MYLAKE_ALBEDO
            transmissivity[0] = math.exp(-Hice[0] * attenuation_ice[0])

    ocean_ice_heat_flux[0] = 0.0
    ocean_ice_salt_flux[0] = 0.0
    return T_w
