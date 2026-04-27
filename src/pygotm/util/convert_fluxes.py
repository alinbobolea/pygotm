r"""!-----------------------------------------------------------------------
!BOP
! !ROUTINE: Convert between buoyancy fluxes and others \label{sec:convertFluxes}
!
! !INTERFACE:
!    subroutine  convert_fluxes(nlev,gravity,swf,shf,ssf,rad,Tsrf,Ssrf,
!                               tFlux,sFlux,btFlux,bsFlux,tRad,bRad)
!
! !DESCRIPTION:
!  This subroutine computes the buoyancy fluxes that are due
!  to
!  \begin{enumerate}
!    \item the surface heat flux,
!    \item the surface salinity flux caused by the value of
!          P-E (precipitation-evaporation),
!    \item and the short wave radiative flux.
!  \end{enumerate}
!  Additionally, it outputs the temperature flux ({\tt tFlux})
!  corresponding to the surface heat flux, the salinity flux
!  ({\tt sFlux})  corresponding to the value P-E, and the profile
!  of the temperature flux ({\tt tRad}) corresponding to the profile
!  of the radiative heat flux.
!
! This function is only called when the KPP turbulence model is used.
! When you call the KPP routines from another model outside GOTM, you
! are on your own in computing the  fluxes required by the KPP model, because
! they have to be consistent with the equation of state used in your model.
!
! !USES:
!   use density, only: rho0, cp
!   use density, only: get_alpha, get_beta
!   use density, only: alpha, beta
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
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

    !BOC
    !   ! alpha, beta at surface
    !   alpha0  = get_alpha(Ssrf,Tsrf,_ZERO_)
    !   beta0   = get_beta(Ssrf,Tsrf,_ZERO_)
    !
    !   ! temperature flux and associated buoyancy flux
    !   tFlux   = - shf/(rho0*cp)
    !   btFlux  = gravity*alpha0*tFlux
    !
    !   ! salinity flux and associated buoyancy flux
    !   sFlux   = - ssf
    !   bsFlux  = -gravity*beta0*sFlux
    !
    !   ! radiative temperature and buoyancy flux profiles
    !   tRad    =  rad/(rho0*cp)
    !   bRad    = gravity*alpha*tRad
    !EOC

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
