"""gsw_sa_from_sp -- Absolute Salinity from Practical Salinity.

Direct translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_sa_from_sp.f90
"""

from __future__ import annotations

import numpy as np

from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sso, gsw_ups
from pygotm.util.gsw.toolbox.gsw_saar import gsw_saar

__all__ = ["gsw_sa_from_sp"]

_XB_LEFT = np.asarray([12.6, 7.0, 26.0], dtype=np.float64)
_YB_LEFT = np.asarray([50.0, 59.0, 69.0], dtype=np.float64)
_XB_RIGHT = np.asarray([45.0, 26.0], dtype=np.float64)
_YB_RIGHT = np.asarray([50.0, 69.0], dtype=np.float64)


def _util_indx(x: np.ndarray, z: float) -> int:
    n = x.size
    if z > x[0] and z < x[n - 1]:
        kl = 0
        ku = n - 1
        while ku - kl > 1:
            km = (ku + kl) // 2
            if z > x[km]:
                kl = km
            else:
                ku = km
        ki = kl
        if z == x[ki + 1]:
            ki += 1
        return ki
    if z <= x[0]:
        return 0
    return n - 2


def _xinterp1(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    k = _util_indx(x, x0)
    r = (x0 - x[k]) / (x[k + 1] - x[k])
    return float(y[k] + r * (y[k + 1] - y[k]))


def _is_baltic(long: float, lat: float) -> bool:
    if not (_XB_LEFT[1] < long < _XB_RIGHT[0] and _YB_LEFT[0] < lat < _YB_LEFT[2]):
        return False
    xx_left = _xinterp1(_YB_LEFT, _XB_LEFT, lat)
    xx_right = _xinterp1(_YB_RIGHT, _XB_RIGHT, lat)
    return bool(xx_left <= long <= xx_right)


def gsw_sa_from_sp(
    sp: object, p: object, long: float, lat: float
) -> np.ndarray | float:
    """Calculate Absolute Salinity from Practical Salinity."""

    sp_array, p_array = np.broadcast_arrays(
        np.asarray(sp, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
    )
    scalar = sp_array.ndim == 0
    if _is_baltic(float(long), float(lat)):
        result = ((gsw_sso - 0.087) / 35.0) * sp_array + 0.087
    else:
        result = gsw_ups * sp_array * (1.0 + np.asarray(gsw_saar(p_array, long, lat)))
    return float(result) if scalar else result
