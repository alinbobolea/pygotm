"""gsw_sigma0 — potential density anomaly at reference pressure 0 dbar.

Direct Numba translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_sigma0.f90

sigma0 = 1/specvol(SA, CT, 0) - 1000  [kg/m³]
"""

import math

import numba

from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sfac, offset
from pygotm.util.gsw.modules.gsw_mod_specvol_coefficients import (
    v000, v010, v020, v030, v040, v050, v060,
    v100, v110, v120, v130, v140, v150,
    v200, v210, v220, v230, v240,
    v300, v310, v320, v330,
    v400, v410, v420,
    v500, v510,
    v600,
)

__all__ = ["gsw_sigma0"]


@numba.njit(cache=True, fastmath=False)
def gsw_sigma0(sa: float, ct: float) -> float:
    """Potential density anomaly [kg/m³] with reference pressure 0 dbar.

    Translated from gsw_sigma0.f90.  Uses only the p=0 terms of the
    specific-volume polynomial (Roquet et al. 2015).

    Parameters
    ----------
    sa : float
        Absolute Salinity [g/kg]
    ct : float
        Conservative Temperature [°C]

    Returns
    -------
    float
        sigma0 = potential_density - 1000  [kg/m³]
    """
    xs = math.sqrt(gsw_sfac * sa + offset)
    ys = ct * 0.025

    vp0 = (
        v000 + xs * (v010 + xs * (v020 + xs * (v030 + xs * (v040 + xs * (v050
            + v060 * xs)))))
        + ys * (
            v100 + xs * (v110 + xs * (v120 + xs * (v130 + xs * (v140 + v150 * xs))))
            + ys * (
                v200 + xs * (v210 + xs * (v220 + xs * (v230 + v240 * xs)))
                + ys * (
                    v300 + xs * (v310 + xs * (v320 + v330 * xs))
                    + ys * (
                        v400 + xs * (v410 + v420 * xs)
                        + ys * (v500 + v510 * xs + v600 * ys)
                    )
                )
            )
        )
    )

    return 1.0 / vp0 - 1000.0
