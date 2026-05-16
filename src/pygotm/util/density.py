r"""!-----------------------------------------------------------------------
!BOP
!
! !MODULE: density --- the equation of state \label{sec:eqstate}
!
! !INTERFACE:
!   MODULE density
!
! !DESCRIPTION:
!  Computes the density, $\mean{\rho}$, and buoyancy from the
!  salinity, $S$, the temperature, $\Theta$, and the thermodynamic
!  pressure, $P$, according to an \emph{equation of state},
!  \begin{equation}
!    \label{DefEOS}
!     \mean{\rho} = \hat{\rho} (S,\Theta,P)
!     \point
!  \end{equation}
!
!  The following remark on the thermodynamic interpretation of
!  density, temperature, and pressure is useful here.  If $\Theta$ is
!  identified with the in-situ temperature, and $P$ with the in-situ
!  pressure, then $\mean{\rho}$ will be the in-situ density.  On the
!  other hand, if $P$ is identified with the surface pressure, and
!  $\Theta$ with the potential temperature, the same equation of
!  state, \eq{DefEOS}, will yield $\mean{\rho}$ as the potential
!  density. Note that the quantity {\tt sigma\_t} found in the GOTM output
!  is simply computed from  $\mean{\rho}$ - 1000 kg m$^{-3}$, and may
!  therefore adopt different meanings.
!
!  At present, two different models for the equation of state ("modes"),
!  and four different "methods" how to evalute the equation of state
!  are implemented.
!
!  Modes:
!  \begin{enumerate}
!     \item The UNESCO equation of state according to \cite{FofonoffMillard83}
!     \item The \cite{JACKETTea05} equation of state
!  \end{enumerate}
!  Methods:
!  \begin{enumerate}
!     \item the full equation of state --- including pressure effects
!     \item the full equation of state --- without pressure effects
!     \item the linearised equation of state
!     \item a general linear form of the equation of state
!  \end{enumerate}
!
!USES:
!   use gsw_mod_teos10_constants, only: gsw_cp0
!   use gsw_mod_toolbox, only: gsw_rho,gsw_sigma0,gsw_alpha,gsw_beta
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_density, do_density, get_rho, get_alpha, get_beta
!
! !PUBLIC DATA MEMBERS:
!   integer, public :: density_method
!   REALTYPE, public :: T0,S0,p0,rho0=1027.,alpha0,beta0,cp
!   REALTYPE, public, allocatable :: alpha(:), beta(:)
!   REALTYPE, public, allocatable :: rho(:), rho_p(:)
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------

Note: The GOTM Fortran source (density.F90) uses the TEOS-10 Gibbs SeaWater
toolbox (gsw) rather than the original UNESCO 1983 polynomial described in the
module header. This Python translation wraps the same TEOS-10 library via the
`gsw` Python package (McDougall & Barker 2011; Roquet et al. 2015).

TEOS-10 reference: IOC, SCOR and IAPSO (2010). The international thermodynamic
equation of seawater 2010. Intergovernmental Oceanographic Commission,
Manuals and Guides No. 56.
"""

from __future__ import annotations

import gsw as _gsw
import numpy as np

# TEOS-10 standard specific heat of seawater at SA=0, CT=0, p=0.
# Source: IOC, SCOR and IAPSO (2010), TEOS-10 monograph, Appendix K.
# Matches Fortran gsw_mod_teos10_constants::gsw_cp0.
CP0: float = 3991.86795711963

__all__ = [
    "CP0",
    "DensityState",
    "METHOD_LINEAR_TEOS10",
    "METHOD_LINEAR_USER",
    "METHOD_TEOS10",
    "clean_density",
    "do_density",
    "get_alpha",
    "get_beta",
    "get_rho",
    "init_density",
]

# Density method codes matching GOTM Fortran density_method variable
METHOD_TEOS10 = 1  # Full TEOS-10 equation of state (gsw_rho / gsw_sigma0)
METHOD_LINEAR_TEOS10 = 2  # Linearised EOS; α, β computed from TEOS-10 at (S0,T0,p0)
METHOD_LINEAR_USER = 3  # Linearised EOS with user-supplied ρ₀, α₀, β₀


class DensityState:
    """State for the GOTM density / EOS module.

    Mirrors the module-level public and private variables in density.F90.
    Set configuration attributes, call :func:`init_density`, then call
    :func:`do_density` each time step.

    Attributes
    ----------
    density_method : int
        EOS method: METHOD_TEOS10 (1), METHOD_LINEAR_TEOS10 (2), or
        METHOD_LINEAR_USER (3).
    T0, S0, p0 : float
        Reference temperature [°C], salinity [g/kg], pressure [dbar].
        Used by methods 2 and 3 for the linearisation reference point.
    rho0 : float
        Reference density [kg/m³], default 1027.  Used by method 3.
    alpha0, beta0 : float
        Thermal expansion coefficient [1/K] and haline contraction coefficient
        [kg/g].  For method 2, computed from TEOS-10 at (S0, T0, p0).
        For method 3, supplied by the caller.
    cp : float
        Specific heat capacity of seawater [J/(kg·K)]; set by init_density.
    alpha, beta : np.ndarray or None, shape (nlev+1,)
        Per-interface thermal expansion and haline contraction.
    rho, rho_p : np.ndarray or None, shape (nlev+1,)
        In-situ and potential density at cell centres.  Index 0 is seabed
        (GOTM convention); index nlev is surface.
    """

    def __init__(self) -> None:
        self.density_method: int = METHOD_TEOS10
        self.T0: float = 10.0
        self.S0: float = 35.0
        self.p0: float = 0.0
        self.rho0: float = 1027.0
        self.alpha0: float = 0.0
        self.beta0: float = 0.0
        self.cp: float = 0.0

        # Internal reference density used in the linear EOS
        self._rhob: float = 1027.0

        # Arrays allocated by init_density
        self.alpha: np.ndarray | None = None
        self.beta: np.ndarray | None = None
        self.rho: np.ndarray | None = None
        self.rho_p: np.ndarray | None = None


def init_density(state: DensityState, nlev: int) -> None:
    """Initialise the density module state for a column of *nlev* layers.

    ! !IROUTINE: Configuring {\tt eqstate}
    !
    ! !DESCRIPTION:
    !   select case (density_method)
    !      case(1) ! use gsw_rho(S,T,p) - default p=0
    !         cp = gsw_cp0
    !      case(2) ! S0, T0, p0 are provided - alpha, beta, cp are calculated
    !         rhob   = gsw_sigma0(S0,T0) + 1000.
    !         alpha0 = gsw_alpha(S0,T0,p0)
    !         beta0  = gsw_beta(S0,T0,p0)
    !         cp     = gsw_cp0
    !      case(3) ! S0, T0, rho0, alpha, beta are all provided
    !         rhob = rho0

    Parameters
    ----------
    state : DensityState
        Pre-configured state (set density_method and reference values first).
    nlev : int
        Number of vertical layers.
    """
    if state.density_method == METHOD_TEOS10:
        state.cp = CP0
    elif state.density_method == METHOD_LINEAR_TEOS10:
        state._rhob = float(_gsw.sigma0(state.S0, state.T0)) + 1000.0
        state.alpha0 = float(_gsw.alpha(state.S0, state.T0, state.p0))
        state.beta0 = float(_gsw.beta(state.S0, state.T0, state.p0))
        state.cp = CP0
    elif state.density_method == METHOD_LINEAR_USER:
        state._rhob = state.rho0

    state.alpha = np.full(nlev + 1, state.alpha0)
    state.beta = np.full(nlev + 1, state.beta0)
    state.rho = np.zeros(nlev + 1)
    state.rho_p = np.zeros(nlev + 1)


def do_density(
    state: DensityState,
    nlev: int,
    S: np.ndarray,
    T: np.ndarray,
    p: np.ndarray,
    pi: np.ndarray,
) -> None:
    """Compute density, potential density, and expansion coefficients.

    ! !DESCRIPTION:
    !   ! compute interface salinity and temperature
    !   Si(1:nlev-1) = 0.5*(S(1:nlev-1) + S(2:nlev))
    !   Ti(1:nlev-1) = 0.5*(T(1:nlev-1) + T(2:nlev))
    !   Si(0) = S(0)  ;  Si(nlev) = S(nlev)
    !   Ti(0) = T(0)  ;  Ti(nlev) = T(nlev)
    !
    !   select case (density_method)
    !      case(1)
    !         rho(1:nlev)   = gsw_rho(S(1:nlev),T(1:nlev),p(1:nlev))
    !         rho_p(1:nlev) = gsw_sigma0(S(1:nlev),T(1:nlev)) + 1000.
    !         alpha         = gsw_alpha(Si,Ti,pi)
    !         beta          = gsw_beta(Si,Ti,pi)
    !      case(2,3)
    !         rho_p(1:nlev) = rhob*(1 - alpha0*(T(1:nlev)-T0)
    !                               + beta0*(S(1:nlev)-S0))
    !         rho = rho_p   ! Lars: here, we should implement some sort of
    !                       ! pressure dependency

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    nlev : int
        Number of vertical layers.
    S : np.ndarray, shape (nlev+1,)
        Absolute salinity at cell centres [g/kg].
    T : np.ndarray, shape (nlev+1,)
        Conservative temperature at cell centres [°C].
    p : np.ndarray, shape (nlev+1,)
        In-situ pressure at cell centres [dbar].
    pi : np.ndarray, shape (nlev+1,)
        In-situ pressure at cell interfaces [dbar].
    """
    assert state.rho is not None, "call init_density before do_density"
    assert state.rho_p is not None, "call init_density before do_density"
    assert state.alpha is not None, "call init_density before do_density"
    assert state.beta is not None, "call init_density before do_density"

    # Interface S and T: arithmetic mean of adjacent cell centres.
    # Boundary faces (k=0 seabed, k=nlev surface) use the single adjacent cell.
    si = np.empty(nlev + 1)
    ti = np.empty(nlev + 1)
    if nlev > 1:
        si[1:nlev] = 0.5 * (S[1:nlev] + S[2 : nlev + 1])
        ti[1:nlev] = 0.5 * (T[1:nlev] + T[2 : nlev + 1])
    si[0] = S[0]
    si[nlev] = S[nlev]
    ti[0] = T[0]
    ti[nlev] = T[nlev]

    if state.density_method == METHOD_TEOS10:
        state.rho[1 : nlev + 1] = _gsw.rho(
            S[1 : nlev + 1], T[1 : nlev + 1], p[1 : nlev + 1]
        )
        state.rho_p[1 : nlev + 1] = (
            _gsw.sigma0(S[1 : nlev + 1], T[1 : nlev + 1]) + 1000.0
        )
        state.alpha[:] = _gsw.alpha(si, ti, pi)
        state.beta[:] = _gsw.beta(si, ti, pi)
    else:
        # Linear EOS: ρ = ρ_b · (1 − α₀(T − T₀) + β₀(S − S₀))
        # Lars: here, we should implement some sort of pressure dependency
        state.rho_p[1 : nlev + 1] = state._rhob * (
            1.0
            - state.alpha0 * (T[1 : nlev + 1] - state.T0)
            + state.beta0 * (S[1 : nlev + 1] - state.S0)
        )
        state.rho[:] = state.rho_p


def get_rho(state: DensityState, S: float, T: float, p: float | None = None) -> float:
    """Compute density for a single water parcel.

    ! !DESCRIPTION:
    !   select case (density_method)
    !      case(1)
    !         if (present(p)) then
    !            get_rho = gsw_rho(S,T,p)
    !         else
    !            get_rho = gsw_sigma0(S,T) + 1000.
    !         end if
    !      case (2,3)
    !         get_rho = rhob*(1 - alpha0*(T-T0) + beta0*(S-S0))
    !         ! Lars: here, we should implement some pressure dependency

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    S : float
        Absolute salinity [g/kg].
    T : float
        Conservative temperature [°C].
    p : float, optional
        In-situ pressure [dbar].  If omitted under method 1, returns potential
        density (referenced to 0 dbar).
    """
    if state.density_method == METHOD_TEOS10:
        if p is not None:
            return float(_gsw.rho(S, T, p))
        return float(_gsw.sigma0(S, T)) + 1000.0
    return float(
        state._rhob
        * (1.0 - state.alpha0 * (T - state.T0) + state.beta0 * (S - state.S0))
    )


def get_alpha(state: DensityState, S: float, T: float, p: float) -> float:
    """Compute thermal expansion coefficient for a single water parcel.

    ! !DESCRIPTION:
    !   select case (density_method)
    !      case(1)
    !         get_alpha = gsw_alpha(S,T,p)
    !      case (2,3)
    !         get_alpha = alpha0

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    S : float
        Absolute salinity [g/kg].
    T : float
        Conservative temperature [°C].
    p : float
        In-situ pressure [dbar].
    """
    if state.density_method == METHOD_TEOS10:
        return float(_gsw.alpha(S, T, p))
    return state.alpha0


def get_beta(state: DensityState, S: float, T: float, p: float) -> float:
    """Compute haline contraction coefficient for a single water parcel.

    ! !DESCRIPTION:
    !   select case (density_method)
    !      case(1)
    !         get_beta = gsw_beta(S,T,p)
    !      case (2,3)
    !         get_beta = beta0

    Parameters
    ----------
    state : DensityState
        Initialised density state.
    S : float
        Absolute salinity [g/kg].
    T : float
        Conservative temperature [°C].
    p : float
        In-situ pressure [dbar].
    """
    if state.density_method == METHOD_TEOS10:
        return float(_gsw.beta(S, T, p))
    return state.beta0


def clean_density(state: DensityState) -> None:
    """Release the density arrays held by *state*.

    ! !DESCRIPTION:
    !   if (allocated(alpha)) deallocate(alpha)
    !   if (allocated(beta))  deallocate(beta)
    !   if (allocated(rho))   deallocate(rho)
    !   if (allocated(rho_p)) deallocate(rho_p)

    After clean_density, call init_density again before do_density.
    """
    state.alpha = None
    state.beta = None
    state.rho = None
    state.rho_p = None
