"""
Flux conversion for KPP turbulence model — translation of ``convert_fluxes.F90``.

Converts surface heat, salinity, and shortwave radiative fluxes into the
temperature and buoyancy flux forms required by the KPP turbulence closure.
Three surface contributions are handled:

1. Surface heat flux → temperature flux and thermal buoyancy flux.
2. Surface salinity flux (P − E) → salinity flux and haline buoyancy flux.
3. Shortwave radiative flux profile → radiative temperature and buoyancy flux
   profiles.

Only called when the KPP turbulence model is active.  Callers outside GOTM
must supply fluxes consistent with their own equation of state.

Original author: Lars Umlauf.

Public interface: :func:`convert_fluxes`.
"""

from __future__ import annotations

import numpy as np

from pygotm.util.density import DensityState, get_alpha, get_beta

__all__ = ["convert_fluxes"]


def convert_fluxes(
    state: DensityState,
    nlev: int,
    gravity: float,
    shf: float,
    ssf: float,
    rad: np.ndarray,
    T_srf: float,
    S_srf: float,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Convert surface and radiative fluxes to temperature and buoyancy fluxes.

    The algorithm proceeds in three steps:

    1. Evaluate the thermal expansion coefficient :math:`\\alpha_0` and haline
       contraction coefficient :math:`\\beta_0` at the surface using the
       configured equation of state.
    2. Convert the surface heat flux and salinity flux to temperature and
       buoyancy fluxes:

       .. math::

          t_{\\mathrm{flux}} = -\\frac{q_{\\mathrm{shf}}}{\\rho_0 c_p},\\quad
          bt_{\\mathrm{flux}} = g\\,\\alpha_0\\,t_{\\mathrm{flux}}

       .. math::

          s_{\\mathrm{flux}} = -q_{\\mathrm{ssf}},\\quad
          bs_{\\mathrm{flux}} = -g\\,\\beta_0\\,s_{\\mathrm{flux}}

    3. Convert the shortwave radiative profile to temperature and buoyancy
       flux profiles:

       .. math::

          t_{\\mathrm{rad}} = \\frac{\\mathrm{rad}}{\\rho_0 c_p},\\quad
          b_{\\mathrm{rad}} = g\\,\\alpha\\,t_{\\mathrm{rad}}

    Parameters
    ----------
    state : DensityState
        Initialised density state providing rho0, cp, alpha, beta.
    nlev : int
        Number of vertical layers.
    gravity : float
        Gravitational acceleration [m/s²].
    shf : float
        Surface heat flux [W/m²].  Positive = into ocean.
    ssf : float
        Surface salinity flux [psu·m/s or kg/(m²·s)].  Sign: positive = freshwater
        input → salinity decrease.
    rad : np.ndarray, shape (nlev+1,)
        Shortwave radiative heat flux profile [W/m²].
    T_srf : float
        Surface temperature [°C] — used to evaluate α, β at the surface.
    S_srf : float
        Surface salinity [g/kg] — used to evaluate α, β at the surface.

    Returns
    -------
    t_flux : float
        Temperature flux from surface heat flux [K·m/s].
    s_flux : float
        Salinity flux from P-E [psu·m/s].
    bt_flux : float
        Buoyancy flux from surface heat flux [m²/s³].
    bs_flux : float
        Buoyancy flux from surface salinity flux [m²/s³].
    t_rad : np.ndarray, shape (nlev+1,)
        Temperature flux profile from shortwave radiation [K·m/s].
    b_rad : np.ndarray, shape (nlev+1,)
        Buoyancy flux profile from shortwave radiation [m²/s³].
    """
    assert state.alpha is not None, "call init_density before convert_fluxes"

    alpha0 = get_alpha(state, S_srf, T_srf, 0.0)
    beta0 = get_beta(state, S_srf, T_srf, 0.0)

    # Temperature flux and associated buoyancy flux
    t_flux = -shf / (state.rho0 * state.cp)
    bt_flux = gravity * alpha0 * t_flux

    # Salinity flux and associated buoyancy flux
    s_flux = -ssf
    bs_flux = -gravity * beta0 * s_flux

    # Radiative temperature and buoyancy flux profiles
    t_rad = rad / (state.rho0 * state.cp)
    b_rad = gravity * state.alpha * t_rad

    return t_flux, s_flux, bt_flux, bs_flux, t_rad, b_rad
