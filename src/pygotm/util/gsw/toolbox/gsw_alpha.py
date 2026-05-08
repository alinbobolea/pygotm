"""gsw_alpha — thermal expansion coefficient w.r.t. Conservative Temperature.

Direct Numba translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_alpha.f90

alpha = 0.025 * v_CT_part / specvol  [1/K]
"""

import math

import numba

from pygotm.util.gsw.modules.gsw_mod_specvol_coefficients import (
    a000,
    a001,
    a002,
    a003,
    a004,
    a010,
    a011,
    a012,
    a013,
    a020,
    a021,
    a022,
    a030,
    a031,
    a032,
    a040,
    a041,
    a050,
    a100,
    a101,
    a102,
    a103,
    a110,
    a111,
    a112,
    a120,
    a121,
    a122,
    a130,
    a131,
    a140,
    a200,
    a201,
    a202,
    a210,
    a211,
    a212,
    a220,
    a221,
    a230,
    a300,
    a301,
    a302,
    a310,
    a311,
    a320,
    a400,
    a401,
    a410,
    a500,
)
from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sfac, offset
from pygotm.util.gsw.toolbox.gsw_specvol import gsw_specvol

__all__ = ["gsw_alpha"]


@numba.njit(cache=True, fastmath=False)
def gsw_alpha(sa: float, ct: float, p: float) -> float:
    """Thermal expansion coefficient [1/K] w.r.t. Conservative Temperature.

    Translated from gsw_alpha.f90 (Roquet et al. 2015).

    Parameters
    ----------
    sa : float
        Absolute Salinity [g/kg]
    ct : float
        Conservative Temperature [°C]
    p : float
        Sea pressure [dbar]

    Returns
    -------
    float
        alpha [1/K]
    """
    xs = math.sqrt(gsw_sfac * sa + offset)
    ys = ct * 0.025
    z = p * 1.0e-4

    v_ct_part = (
        a000 + xs * (a100 + xs * (a200 + xs * (a300 + xs * (a400 + a500 * xs))))
        + ys * (
            a010 + xs * (a110 + xs * (a210 + xs * (a310 + a410 * xs)))
            + ys * (
                a020 + xs * (a120 + xs * (a220 + a320 * xs))
                + ys * (
                    a030 + xs * (a130 + a230 * xs)
                    + ys * (a040 + a140 * xs + a050 * ys)
                )
            )
        )
        + z * (
            a001 + xs * (a101 + xs * (a201 + xs * (a301 + a401 * xs)))
            + ys * (
                a011 + xs * (a111 + xs * (a211 + a311 * xs))
                + ys * (
                    a021 + xs * (a121 + a221 * xs)
                    + ys * (a031 + a131 * xs + a041 * ys)
                )
            )
            + z * (
                a002 + xs * (a102 + xs * (a202 + a302 * xs))
                + ys * (
                    a012 + xs * (a112 + a212 * xs)
                    + ys * (a022 + a122 * xs + a032 * ys)
                )
                + z * (a003 + a103 * xs + a013 * ys + a004 * z)
            )
        )
    )

    return float(0.025 * v_ct_part / gsw_specvol(sa, ct, p))
