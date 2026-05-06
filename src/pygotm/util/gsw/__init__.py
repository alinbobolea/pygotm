"""Numba-compiled TEOS-10 GSW library.

Mirrors the Fortran folder layout:

    extern/gsw/
        modules/
            gsw_mod_teos10_constants.f90   → modules/gsw_mod_teos10_constants.py
            gsw_mod_specvol_coefficients.f90 → modules/gsw_mod_specvol_coefficients.py
        toolbox/
            gsw_specvol.f90  → toolbox/gsw_specvol.py
            gsw_sigma0.f90   → toolbox/gsw_sigma0.py
            gsw_alpha.f90    → toolbox/gsw_alpha.py
            gsw_beta.f90     → toolbox/gsw_beta.py
            gsw_rho.f90      → toolbox/gsw_rho.py

All functions are scalar @numba.njit so they can be called from inside
compiled time-loop functions without leaving nopython mode.
"""

from pygotm.util.gsw.toolbox import (
    gsw_alpha,
    gsw_beta,
    gsw_rho,
    gsw_sigma0,
    gsw_specvol,
)

__all__ = [
    "gsw_alpha",
    "gsw_beta",
    "gsw_rho",
    "gsw_sigma0",
    "gsw_specvol",
]
