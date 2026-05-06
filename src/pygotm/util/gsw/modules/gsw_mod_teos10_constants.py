"""TEOS-10 scalar constants.

Direct translation of:
    gotm-model/code/extern/gsw/modules/gsw_mod_teos10_constants.f90

All values are identical to the Fortran parameters so that Numba-compiled
callers produce bit-for-bit matching results.
"""

import numba
import numpy as np

__all__ = [
    "gsw_cp0",
    "gsw_sfac",
    "offset",
]

# Specific heat of seawater at SA=0, CT=0, p=0  [J/(kg K)]
gsw_cp0: float = 3991.86795711963

# sfac = 1 / (40 * gsw_ups)  where gsw_ups = gsw_sso/35  = 35.16504/35
gsw_sfac: float = 0.0248826675584615

# deltaS = 24;  offset = deltaS * gsw_sfac
offset: float = 5.971840214030754e-1
