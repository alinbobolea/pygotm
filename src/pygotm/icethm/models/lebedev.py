"""Lebedev freezing-degree-day ice growth model.

The model follows the empirical relation used by Lebedev (1938) and later lake
ice applications: accumulated freezing degree days produce ice thickness
``Hice = 0.01 * fac * fdd**exp``. The surface albedo is fixed and shortwave
transmissivity decays exponentially with thickness.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from pygotm.icethm._util import freezing_temperature
from pygotm.icethm.constants import (
    ALB_OCEAN,
    LEBEDEV_ALBEDO,
    LEBEDEV_ATTN,
    LEBEDEV_EXP,
    LEBEDEV_FAC,
)


@numba.njit(cache=True)
def step_lebedev(
    T_air: float,
    T_w: float,
    S_sfc: float,
    dt: float,
    fdd: np.ndarray,
    Hice: np.ndarray,
    ice_cover: np.ndarray,
    albedo_ice: np.ndarray,
    transmissivity: np.ndarray,
    Tf: np.ndarray,
) -> None:
    """Advance empirical freezing-degree-day ice thickness by one timestep."""

    tf = freezing_temperature(S_sfc)
    Tf[0] = tf
    day_fraction = dt / 86400.0
    if T_air < tf:
        fdd[0] += (tf - T_air) * day_fraction
    elif T_air > tf:
        fdd[0] -= (T_air - tf) * day_fraction
        if fdd[0] < 0.0:
            fdd[0] = 0.0

    if fdd[0] <= 1.0 or T_w > tf:
        Hice[0] = 0.0
        ice_cover[0] = 0
        albedo_ice[0] = ALB_OCEAN
        transmissivity[0] = 1.0
        return

    Hice[0] = 0.01 * LEBEDEV_FAC * fdd[0] ** LEBEDEV_EXP
    ice_cover[0] = 2
    albedo_ice[0] = LEBEDEV_ALBEDO
    transmissivity[0] = math.exp(Hice[0] / LEBEDEV_ATTN)
    if transmissivity[0] < 0.0:
        transmissivity[0] = 0.0
    elif transmissivity[0] > 1.0:
        transmissivity[0] = 1.0
