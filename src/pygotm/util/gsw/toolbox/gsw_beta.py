"""gsw_beta — haline contraction coefficient at constant Conservative Temperature.

Direct Numba translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_beta.f90

beta = -v_SA_part * 0.5 * gsw_sfac / (specvol * xs)  [kg/g]
"""

import math

import numba

from pygotm.util.gsw.modules.gsw_mod_specvol_coefficients import (
    b000,
    b001,
    b002,
    b003,
    b004,
    b010,
    b011,
    b012,
    b013,
    b020,
    b021,
    b022,
    b030,
    b031,
    b032,
    b040,
    b041,
    b050,
    b100,
    b101,
    b102,
    b103,
    b110,
    b111,
    b112,
    b120,
    b121,
    b122,
    b130,
    b131,
    b140,
    b200,
    b201,
    b202,
    b210,
    b211,
    b212,
    b220,
    b221,
    b230,
    b300,
    b301,
    b302,
    b310,
    b311,
    b320,
    b400,
    b401,
    b410,
    b500,
)
from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sfac, offset
from pygotm.util.gsw.toolbox.gsw_specvol import gsw_specvol

__all__ = ["gsw_beta"]


@numba.njit(cache=True, fastmath=False)
def gsw_beta(sa: float, ct: float, p: float) -> float:
    """Haline contraction coefficient [kg/g] at constant Conservative Temperature.

    Translated from gsw_beta.f90 (Roquet et al. 2015).

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
        beta [kg/g]
    """
    xs = math.sqrt(gsw_sfac * sa + offset)
    ys = ct * 0.025
    z = p * 1.0e-4

    v_sa_part = (
        b000
        + xs * (b100 + xs * (b200 + xs * (b300 + xs * (b400 + b500 * xs))))
        + ys
        * (
            b010
            + xs * (b110 + xs * (b210 + xs * (b310 + b410 * xs)))
            + ys
            * (
                b020
                + xs * (b120 + xs * (b220 + b320 * xs))
                + ys
                * (b030 + xs * (b130 + b230 * xs) + ys * (b040 + b140 * xs + b050 * ys))
            )
        )
        + z
        * (
            b001
            + xs * (b101 + xs * (b201 + xs * (b301 + b401 * xs)))
            + ys
            * (
                b011
                + xs * (b111 + xs * (b211 + b311 * xs))
                + ys
                * (b021 + xs * (b121 + b221 * xs) + ys * (b031 + b131 * xs + b041 * ys))
            )
            + z
            * (
                b002
                + xs * (b102 + xs * (b202 + b302 * xs))
                + ys
                * (b012 + xs * (b112 + b212 * xs) + ys * (b022 + b122 * xs + b032 * ys))
                + z * (b003 + b103 * xs + b013 * ys + b004 * z)
            )
        )
    )

    return float(-v_sa_part * 0.5 * gsw_sfac / (gsw_specvol(sa, ct, p) * xs))
