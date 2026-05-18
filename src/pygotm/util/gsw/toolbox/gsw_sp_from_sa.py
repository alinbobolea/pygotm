"""gsw_sp_from_sa -- Practical Salinity from Absolute Salinity.

Direct translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_sp_from_sa.f90
"""

from __future__ import annotations

import numpy as np

from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sso, gsw_ups
from pygotm.util.gsw.toolbox.gsw_sa_from_sp import _is_baltic
from pygotm.util.gsw.toolbox.gsw_saar import gsw_saar

__all__ = ["gsw_sp_from_sa"]


def gsw_sp_from_sa(
    sa: object, p: object, long: float, lat: float
) -> np.ndarray | float:
    """Calculate Practical Salinity from Absolute Salinity."""

    sa_array, p_array = np.broadcast_arrays(
        np.asarray(sa, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
    )
    scalar = sa_array.ndim == 0
    if _is_baltic(float(long), float(lat)):
        result = (35.0 / (gsw_sso - 0.087)) * (sa_array - 0.087)
    else:
        result = (sa_array / gsw_ups) / (1.0 + np.asarray(gsw_saar(p_array, long, lat)))
    return float(result) if scalar else result
