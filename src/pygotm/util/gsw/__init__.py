"""Numba-compiled TEOS-10 GSW library.

Mirrors the Fortran folder layout:

    extern/gsw/
        modules/
            gsw_mod_teos10_constants.f90   → modules/gsw_mod_teos10_constants.py
            gsw_mod_specvol_coefficients.f90 → modules/gsw_mod_specvol_coefficients.py
        toolbox/
            gsw_saar.f90     → toolbox/gsw_saar.py
            gsw_sa_from_sp.f90 → toolbox/gsw_sa_from_sp.py
            gsw_sp_from_sa.f90 → toolbox/gsw_sp_from_sa.py
            gsw_specvol.f90  → toolbox/gsw_specvol.py
            gsw_sigma0.f90   → toolbox/gsw_sigma0.py
            gsw_alpha.f90    → toolbox/gsw_alpha.py
            gsw_beta.f90     → toolbox/gsw_beta.py
            gsw_rho.f90      → toolbox/gsw_rho.py

The equation-of-state functions used from the compiled time loop are scalar
@numba.njit functions. The salinity conversion helpers are vectorized Python
wrappers around GOTM's bundled GSW SAAR data and are used during setup/output.
"""

from pygotm.util.gsw.toolbox import (
    gsw_alpha,
    gsw_beta,
    gsw_rho,
    gsw_sa_from_sp,
    gsw_saar,
    gsw_sigma0,
    gsw_sp_from_sa,
    gsw_specvol,
)

__all__ = [
    "gsw_alpha",
    "gsw_beta",
    "gsw_rho",
    "gsw_sa_from_sp",
    "gsw_saar",
    "gsw_sigma0",
    "gsw_sp_from_sa",
    "gsw_specvol",
]
