"""gsw_specvol — specific volume from SA, CT, p.

Direct Numba translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_specvol.f90

Uses the 75-term Roquet et al. (2015) polynomial.
Inputs follow the Fortran argument order: sa, ct, p.
"""

import math

import numba

from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sfac, offset
from pygotm.util.gsw.modules.gsw_mod_specvol_coefficients import (
    v000, v001, v002, v003, v004, v005, v006,
    v010, v011, v012, v013, v014,
    v020, v021, v022, v023,
    v030, v031, v032,
    v040, v041, v042,
    v050, v051,
    v060,
    v100, v101, v102, v103, v104,
    v110, v111, v112, v113,
    v120, v121, v122,
    v130, v131, v132,
    v140, v141,
    v150,
    v200, v201, v202, v203,
    v210, v211, v212,
    v220, v221, v222,
    v230, v231,
    v240,
    v300, v301, v302,
    v310, v311, v312,
    v320, v321,
    v330,
    v400, v401, v402,
    v410, v411,
    v420,
    v500, v501,
    v510,
    v600,
)

__all__ = ["gsw_specvol"]


@numba.njit(cache=True, fastmath=False)
def gsw_specvol(sa: float, ct: float, p: float) -> float:
    """Specific volume [m³/kg] from Absolute Salinity, Conservative Temperature, pressure.

    Translated from gsw_specvol.f90 (Roquet et al. 2015, 75-term polynomial).

    Parameters
    ----------
    sa : float
        Absolute Salinity [g/kg]
    ct : float
        Conservative Temperature [°C]
    p : float
        Sea pressure [dbar]  (absolute pressure minus 10.1325 dbar)

    Returns
    -------
    float
        Specific volume [m³/kg]
    """
    xs = math.sqrt(gsw_sfac * sa + offset)
    ys = ct * 0.025
    z = p * 1.0e-4

    return (
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
        + z * (
            v001 + xs * (v011 + xs * (v021 + xs * (v031 + xs * (v041 + v051 * xs))))
            + ys * (
                v101 + xs * (v111 + xs * (v121 + xs * (v131 + v141 * xs)))
                + ys * (
                    v201 + xs * (v211 + xs * (v221 + v231 * xs))
                    + ys * (
                        v301 + xs * (v311 + v321 * xs)
                        + ys * (v401 + v411 * xs + v501 * ys)
                    )
                )
            )
            + z * (
                v002 + xs * (v012 + xs * (v022 + xs * (v032 + v042 * xs)))
                + ys * (
                    v102 + xs * (v112 + xs * (v122 + v132 * xs))
                    + ys * (
                        v202 + xs * (v212 + v222 * xs)
                        + ys * (v302 + v312 * xs + v402 * ys)
                    )
                )
                + z * (
                    v003 + xs * (v013 + v023 * xs)
                    + ys * (v103 + v113 * xs + v203 * ys)
                    + z * (v004 + v014 * xs + v104 * ys + z * (v005 + v006 * z))
                )
            )
        )
    )
