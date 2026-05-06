"""gsw_rho — in-situ density from SA, CT, p.

Direct Numba translation of:
    gotm-model/code/extern/gsw/toolbox/gsw_rho.f90

rho = 1 / gsw_specvol(sa, ct, p)  [kg/m³]
"""

import numba

from pygotm.util.gsw.toolbox.gsw_specvol import gsw_specvol

__all__ = ["gsw_rho"]


@numba.njit(cache=True, fastmath=False)
def gsw_rho(sa: float, ct: float, p: float) -> float:
    """In-situ density [kg/m³] from Absolute Salinity, Conservative Temperature, pressure.

    Translated from gsw_rho.f90.

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
        In-situ density [kg/m³]
    """
    return 1.0 / gsw_specvol(sa, ct, p)
