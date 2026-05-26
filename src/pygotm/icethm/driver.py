"""Driver helpers for pyGOTM ice thermodynamics models."""

import numba
import numpy as np

from pygotm.icethm._util import freezing_temperature, require_ice_state
from pygotm.icethm.constants import C_WATER_VOL
from pygotm.icethm.models.basal_melt import step_basal_melt
from pygotm.icethm.models.lebedev import step_lebedev
from pygotm.icethm.models.mylake import step_mylake
from pygotm.icethm.models.simple import step_simple
from pygotm.icethm.models.winton import step_winton
from pygotm.icethm.params import IceModelEnum, IceParams
from pygotm.icethm.state import IceState, make_ice_state


def init_ice(params: IceParams, *, T_air_init: float, S_sfc_init: float) -> IceState:
    """Initialize mutable ice state from immutable YAML-derived parameters."""

    state = make_ice_state()
    state.Hice[0] = params.Hice_init
    state.Hsnow[0] = params.Hsnow_init
    state.ocean_ice_flux[0] = params.ocean_ice_flux_init
    state.Tf[0] = freezing_temperature(float(S_sfc_init))
    if params.model == IceModelEnum.WINTON:
        state.T1[0] = float(T_air_init)
        state.T2[0] = state.Tf[0]
        state.Tice_surface[0] = float(T_air_init)
    elif params.model == IceModelEnum.BASAL_MELT:
        state.Tice_surface[0] = 0.0
    elif params.model == IceModelEnum.SIMPLE:
        # Simple limiter has no prognostic ice surface temperature; match
        # Fortran reference which leaves Tice_surface at its allocated zero.
        state.Tice_surface[0] = 0.0
    else:
        state.Tice_surface[0] = state.Tf[0]
    if state.Hice[0] > 0.0:
        state.ice_cover[0] = 2
    require_ice_state(state)
    return state


@numba.njit(cache=True)
def step_ice(
    model: int,
    T_w: float,
    S_w: float,
    T_air: float,
    h_sfc: float,
    dt: float,
    diff_t_up: float,
    Qsw: float,
    Ql: float,
    Qh: float,
    Qe: float,
    precip: float,
    ustar: float,
    winton_surface_flux_a: float,
    winton_surface_flux_b: float,
    Hice: np.ndarray,
    Hsnow: np.ndarray,
    Hfrazil: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    Tice_surface: np.ndarray,
    fdd: np.ndarray,
    ice_cover: np.ndarray,
    Tf: np.ndarray,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
    ocean_ice_flux: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
    ocean_ice_salt_flux: np.ndarray,
    surface_ice_energy: np.ndarray,
    bottom_ice_energy: np.ndarray,
    melt_rate: np.ndarray,
    T_melt: np.ndarray,
    S_melt: np.ndarray,
) -> float:
    """Dispatch one ice-model step and return the surface temperature flux."""

    if model == 0:
        Tf[0] = freezing_temperature(S_w)
        return diff_t_up
    if model == 1:
        return float(step_simple(T_w, S_w, diff_t_up, Tf, Hice, ice_cover))
    if model == 2:
        basal_melt_cache_version = 1
        if basal_melt_cache_version < 0:
            return diff_t_up
        step_basal_melt(
            T_w,
            S_w,
            h_sfc,
            Hice[0],
            ustar,
            melt_rate,
            T_melt,
            S_melt,
            ocean_ice_heat_flux,
            ocean_ice_salt_flux,
            Tf,
        )
        return float(diff_t_up - ocean_ice_heat_flux[0] / C_WATER_VOL)
    if model == 3:
        step_lebedev(
            T_air,
            T_w,
            S_w,
            dt,
            fdd,
            Hice,
            ice_cover,
            albedo_ice,
            transmissivity,
            Tf,
        )
        return diff_t_up
    if model == 4:
        step_mylake(
            T_w,
            S_w,
            T_air,
            h_sfc,
            Qsw,
            Qh,
            Qe,
            Ql,
            dt,
            Hice,
            Hfrazil,
            Tice_surface,
            ice_cover,
            albedo_ice,
            transmissivity,
            Tf,
            ocean_ice_heat_flux,
            ocean_ice_salt_flux,
        )
        return float(diff_t_up - ocean_ice_heat_flux[0] / C_WATER_VOL)

    winton_cache_version = 1
    if winton_cache_version < 0:
        return diff_t_up
    step_winton(
        T_w,
        S_w,
        h_sfc,
        dt,
        Qsw,
        Ql,
        Qh,
        Qe,
        precip,
        winton_surface_flux_a,
        winton_surface_flux_b,
        Hice,
        Hsnow,
        Hfrazil,
        T1,
        T2,
        Tice_surface,
        ice_cover,
        albedo_ice,
        transmissivity,
        Tf,
        ocean_ice_heat_flux,
        ocean_ice_flux,
        ocean_ice_salt_flux,
        surface_ice_energy,
        bottom_ice_energy,
    )
    return float(diff_t_up - ocean_ice_heat_flux[0] / C_WATER_VOL)


def compute_diff_t_up_from_ice(
    state: IceState,
    *,
    diff_t_up: float,
    rho0: float,
    cp: float,
) -> float:
    """Apply diagnosed ocean-ice heat flux to a pyGOTM temperature flux."""

    return diff_t_up - float(state.ocean_ice_heat_flux[0]) / (rho0 * cp)


def outputs_to_buffers(
    state: IceState,
    reference_scalars: dict[str, np.ndarray],
    slot: int,
) -> None:
    """Write ice scalar diagnostics into runtime reference scalar buffers."""

    mapping = {
        "Hfrazil": state.Hfrazil,
        "Hice": state.Hice,
        "T1": state.T1,
        "T2": state.T2,
        "Tf": state.Tf,
        "Tice_surface": state.Tice_surface,
        "bottom_ice_energy": state.bottom_ice_energy,
        "ocean_ice_flux": state.ocean_ice_flux,
        "ocean_ice_heat_flux": state.ocean_ice_heat_flux,
        "ocean_ice_salt_flux": state.ocean_ice_salt_flux,
        "surface_ice_energy": state.surface_ice_energy,
    }
    for name, array in mapping.items():
        buffer = reference_scalars.get(name)
        if buffer is not None:
            buffer[slot] = float(array[0])
