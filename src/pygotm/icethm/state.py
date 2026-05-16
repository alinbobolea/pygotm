"""Mutable ice thermodynamics state.

Numba kernels mutate length-one ``np.float64`` and ``np.int32`` arrays for scalar
state. This mirrors the rest of pyGOTM's compiled runtime style while keeping
all simulation state explicit and serializable from YAML-derived parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pygotm.icethm.constants import ALB_OCEAN


@dataclass
class IceState:
    """Container for mutable scalar ice state arrays."""

    Hice: np.ndarray
    Hsnow: np.ndarray
    Hfrazil: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    Tice_surface: np.ndarray
    fdd: np.ndarray
    ice_cover: np.ndarray
    Tf: np.ndarray
    albedo_ice: np.ndarray
    transmissivity: np.ndarray
    ocean_ice_flux: np.ndarray
    ocean_ice_heat_flux: np.ndarray
    ocean_ice_salt_flux: np.ndarray
    surface_ice_energy: np.ndarray
    bottom_ice_energy: np.ndarray
    melt_rate: np.ndarray
    T_melt: np.ndarray
    S_melt: np.ndarray


def _scalar(value: float) -> np.ndarray:
    return np.array([float(value)], dtype=np.float64)


def _int_scalar(value: int) -> np.ndarray:
    return np.array([int(value)], dtype=np.int32)


def make_ice_state() -> IceState:
    """Return default no-ice state.

    Defaults represent open water: no ice or snow, ocean albedo, full shortwave
    transmissivity, and zero ice-ocean fluxes.
    """

    return IceState(
        Hice=_scalar(0.0),
        Hsnow=_scalar(0.0),
        Hfrazil=_scalar(0.0),
        T1=_scalar(0.0),
        T2=_scalar(0.0),
        Tice_surface=_scalar(0.0),
        fdd=_scalar(0.0),
        ice_cover=_int_scalar(0),
        Tf=_scalar(0.0),
        albedo_ice=_scalar(ALB_OCEAN),
        transmissivity=_scalar(1.0),
        ocean_ice_flux=_scalar(0.0),
        ocean_ice_heat_flux=_scalar(0.0),
        ocean_ice_salt_flux=_scalar(0.0),
        surface_ice_energy=_scalar(0.0),
        bottom_ice_energy=_scalar(0.0),
        melt_rate=_scalar(0.0),
        T_melt=_scalar(0.0),
        S_melt=_scalar(0.0),
    )
