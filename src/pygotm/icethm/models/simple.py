"""Simple GOTM ice limiter.

The simple model is a boundary-condition limiter rather than a prognostic ice
model. It computes the linear freezing point ``Tf = -0.0575 S`` and suppresses
warming temperature flux into water that is already at or below freezing.
"""

from __future__ import annotations

import numba
import numpy as np

from pygotm.icethm._util import freezing_temperature


@numba.njit(cache=True)
def step_simple(
    T_sfc: float,
    S_sfc: float,
    diff_t_up: float,
    Tf_out: np.ndarray,
    Hice_out: np.ndarray,
    ice_cover_out: np.ndarray,
) -> float:
    """Return the modified upward temperature diffusivity boundary flux."""

    tf = freezing_temperature(S_sfc)
    Tf_out[0] = tf
    Hice_out[0] = 0.0
    ice_cover_out[0] = 0
    if T_sfc <= tf and diff_t_up > 0.0:
        return 0.0
    return diff_t_up
