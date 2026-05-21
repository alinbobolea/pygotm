"""Ice-shelf basal melt closure translated from STIM ``stim_basal_melt.F90``."""

from __future__ import annotations

import math

import numba
import numpy as np

from pygotm.icethm.constants import (
    C_ICE_BASAL,
    C_WATER_BASAL,
    L_ICE,
    LAMBDA1,
    LAMBDA2,
    LAMBDA3,
    RHO_ICE_BASAL,
    RHO_WATER_BASAL,
    T_ICE_CORE,
)

_Z0_ICE: float = 0.01
_NU: float = 1.95e-6
_KAPPA: float = 0.4
_BS: float = 8.5
_PR_TURB: float = 0.7
_PR_TEMP: float = 13.8
_PR_SALT: float = 2432.0


@numba.njit(cache=True)
def basal_freezing_temperature(S_b: float, H_ice: float) -> float:
    """Return the pressure-adjusted interface freezing temperature."""

    draft = H_ice * RHO_ICE_BASAL / RHO_WATER_BASAL
    return LAMBDA1 * S_b + LAMBDA2 + LAMBDA3 * draft


@numba.njit(cache=True)
def step_basal_melt(
    T_w: float,
    S_w: float,
    h_sfc: float,
    H_ice: float,
    ustar: float,
    melt_rate: np.ndarray,
    T_melt: np.ndarray,
    S_melt: np.ndarray,
    ocean_ice_heat_flux: np.ndarray,
    ocean_ice_salt_flux: np.ndarray,
    Tf: np.ndarray,
) -> None:
    """Solve one basal-melt interface step for a single water column."""

    _ = Tf
    if ustar < 1.0e-10:
        melt_rate[0] = 0.0
        T_melt[0] = basal_freezing_temperature(S_w, H_ice)
        S_melt[0] = S_w
        ocean_ice_heat_flux[0] = 0.0
        ocean_ice_salt_flux[0] = 0.0
        return

    beta_t = 0.55 * math.exp(0.5 * _KAPPA * _BS) * math.sqrt(_Z0_ICE * ustar / _NU)
    beta_t = beta_t * (_PR_TEMP ** (2.0 / 3.0) - 0.2)
    beta_t = beta_t - _PR_TURB * _BS + 9.5

    beta_s = 0.55 * math.exp(0.5 * _KAPPA * _BS) * math.sqrt(_Z0_ICE * ustar / _NU)
    beta_s = beta_s * (_PR_SALT ** (2.0 / 3.0) - 0.2)
    beta_s = beta_s - _PR_TURB * _BS + 9.5

    log_t = math.log((0.5 * h_sfc + _Z0_ICE) / _Z0_ICE) + _KAPPA / _PR_TURB * beta_t
    log_s = math.log((0.5 * h_sfc + _Z0_ICE) / _Z0_ICE) + _KAPPA / _PR_TURB * beta_s
    a1t = _KAPPA * ustar / (_PR_TURB * log_t)
    a1s = _KAPPA * ustar / (_PR_TURB * log_s)

    ll = LAMBDA2 + LAMBDA3 * H_ice * RHO_ICE_BASAL / RHO_WATER_BASAL
    s1 = LAMBDA1 * (a1t - a1s * C_ICE_BASAL / C_WATER_BASAL)
    s2 = (
        a1s * S_w * LAMBDA1 * C_ICE_BASAL / C_WATER_BASAL
        - a1s / C_WATER_BASAL * (C_ICE_BASAL * (ll - T_ICE_CORE) + L_ICE)
        - a1t * (T_w - ll)
    )
    s3 = a1s * S_w / C_WATER_BASAL * (C_ICE_BASAL * (ll - T_ICE_CORE) + L_ICE)
    pp = s2 / s1
    qq = s3 / s1
    disc = 0.25 * pp * pp - qq
    if disc < 0.0:
        disc = 0.0

    S_b = -0.5 * pp + math.sqrt(disc)
    T_b = LAMBDA1 * S_b + ll
    rate = (
        a1t
        * (T_w - T_b)
        / (C_ICE_BASAL / C_WATER_BASAL * (T_b - T_ICE_CORE) + L_ICE / C_WATER_BASAL)
    )

    melt_rate[0] = rate
    T_melt[0] = T_b
    S_melt[0] = S_b
    ocean_ice_heat_flux[0] = (
        (a1t * (T_w - T_b) - rate * T_b) * C_WATER_BASAL * RHO_WATER_BASAL
    )
    ocean_ice_salt_flux[0] = a1s * (S_w - S_b) - rate * S_b
