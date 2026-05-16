"""Ice-shelf basal melt three-equation closure.

The kernel solves the Holland-Jenkins ice-ocean interface system:

``T_b = lambda_1 S_b + lambda_2 + lambda_3 H``

``gamma_T (T - T_b) =
M (L/c_w + c_i/c_w (T_b - T_i))``

``gamma_S (S - S_b) = M S_b``

where ``M`` is positive for melting. The pressure/freezing relation is expressed
with the McDougall-Jackett potential-temperature coefficients used by GOTM's
ice-shelf cases. The reported ``ocean_ice_heat_flux`` is positive when melting
extracts heat from the ocean.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from pygotm.icethm.constants import (
    C_ICE_BASAL,
    C_WATER_BASAL,
    DEFAULT_BASAL_USTAR,
    GAMMA_S,
    GAMMA_T,
    L_ICE,
    LAMBDA1,
    LAMBDA2,
    LAMBDA3,
    RHO_ICE_BASAL,
    T_ICE_CORE,
)


@numba.njit(cache=True)
def basal_freezing_temperature(S_b: float, H_ice: float) -> float:
    """Return the pressure-adjusted interface freezing temperature."""

    return LAMBDA1 * S_b + LAMBDA2 + LAMBDA3 * H_ice


@numba.njit(cache=True)
def step_basal_melt(
    T_w: float,
    S_w: float,
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

    ustar_eff = ustar
    if ustar_eff <= 0.0:
        ustar_eff = DEFAULT_BASAL_USTAR
    gamma_t = GAMMA_T * ustar_eff
    gamma_s = GAMMA_S * ustar_eff

    ll = LAMBDA2 + LAMBDA3 * max(0.0, H_ice)
    a0 = L_ICE / C_WATER_BASAL + (C_ICE_BASAL / C_WATER_BASAL) * (ll - T_ICE_CORE)
    a1 = (C_ICE_BASAL / C_WATER_BASAL) * LAMBDA1

    qa = -gamma_t * LAMBDA1 + gamma_s * a1
    qb = gamma_t * (T_w - ll) - gamma_s * S_w * a1 + gamma_s * a0
    qc = -gamma_s * S_w * a0
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        disc = 0.0

    if abs(qa) < 1.0e-30:
        if abs(qb) < 1.0e-30:
            S_b = S_w
        else:
            S_b = -qc / qb
    else:
        sqrt_disc = math.sqrt(disc)
        root_a = (-qb - sqrt_disc) / (2.0 * qa)
        root_b = (-qb + sqrt_disc) / (2.0 * qa)
        S_b = S_w
        if math.isfinite(root_a) and root_a > 0.0:
            S_b = root_a
        if math.isfinite(root_b) and root_b > 0.0:
            if S_b <= 0.0 or abs(root_b - S_w) < abs(S_b - S_w):
                S_b = root_b
        if T_w > basal_freezing_temperature(S_w, max(0.0, H_ice)):
            if math.isfinite(root_a) and 0.0 < root_a < S_w:
                S_b = root_a
            if math.isfinite(root_b) and 0.0 < root_b < S_w:
                if S_b >= S_w or root_b > S_b:
                    S_b = root_b

    if not math.isfinite(S_b) or S_b <= 0.0:
        S_b = max(1.0e-12, S_w)

    T_b = basal_freezing_temperature(S_b, max(0.0, H_ice))
    rate = gamma_s * (S_w - S_b) / S_b
    if not math.isfinite(rate):
        rate = 0.0

    melt_rate[0] = rate
    T_melt[0] = T_b
    S_melt[0] = S_b
    Tf[0] = basal_freezing_temperature(S_w, max(0.0, H_ice))
    ocean_ice_heat_flux[0] = RHO_ICE_BASAL * L_ICE * rate
    ocean_ice_salt_flux[0] = rate * (S_b - S_w)
