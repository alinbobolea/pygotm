"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: Mean Flow
!
! !INTERFACE:
!   module meanflow
!
! !DESCRIPTION:
!  This module provides all variables necessary for the meanflow
!  calculation and also makes the proper initialisations.
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_meanflow, post_init_meanflow, clean_meanflow
!
! !PUBLIC DATA MEMBERS:
!   logical, public                              :: grid_ready
!
!  coordinate z, layer thicknesses
!   REALTYPE, public, dimension(:), allocatable, target  :: ga,z,zi,h,ho
!
!  the sea surface elevation
!   REALTYPE, public, target  :: zeta=_ZERO_
!
!  the velocity components
!   REALTYPE, public, dimension(:), allocatable, target  :: u,v,w
!
!  velocity at old time step
!   REALTYPE, public, dimension(:), allocatable  :: uo,vo
!
!  potential temperature, salinity
!  T -> Tc - conservative temperature, S -> Sa - absolute salinity
!   REALTYPE, public, dimension(:), allocatable, target  :: T,S
!  Tp - potential temperature, Sp - practical salinity
!   REALTYPE, public, dimension(:), allocatable, target  :: Tp,Sp
!  Ti - in-situ temperature
!   REALTYPE, public, dimension(:), allocatable, target  :: Ti
!  Tobs, Sobs - observed profiles as conservative and absolute values
!   REALTYPE, public, dimension(:), allocatable, target  :: Tobs, Sobs
!
!  boyancy frequency squared
!  (total, from temperature only, from salinity only)
!   REALTYPE, public, dimension(:), allocatable  :: NN,NNT,NNS
!
!  shear-frequency squared
!  (total, from u only, from v only)
!   REALTYPE, public, dimension(:), allocatable  :: SS,SSU,SSV
!
!  Stokes-Eulerian cross-shear and Stokes shear squared
!   REALTYPE, public, dimension(:), allocatable  :: SSCSTK, SSSTK
!
!  buoyancy, short-wave radiation,
!  extra production of tke by see-grass etc
!   REALTYPE, public, dimension(:), allocatable  :: buoy,rad,xP
!
!  a dummy array
!  (most often used for diffusivities)
!   REALTYPE, public, dimension(:), allocatable  :: avh
!
!  extra friction terms due to e.g. seagrass
!   REALTYPE, public, dimension(:), allocatable  :: fric,drag
!
!  shading in the water column
!   REALTYPE, public, dimension(:), allocatable, target  :: bioshade
!
!  fake ice thickness - switch between 0 and 1 - see temperature.F90
!   REALTYPE, public  :: Hice
!
!  the 'meanflow' configuration
!   REALTYPE, public  :: h0b
!   logical,  public  :: calc_bottom_stress
!   REALTYPE, public  :: z0s_min
!   logical,  public  :: charnock
!   REALTYPE, public  :: charnock_val
!   REALTYPE, public  :: ddu
!   REALTYPE, public  :: ddl
!   integer,  public  :: grid_method
!   REALTYPE, public  :: c1ad, c2ad, c3ad, c4ad
!   REALTYPE, public  :: Tgrid
!   REALTYPE, public  :: NNnorm
!   REALTYPE, public  :: SSnorm
!   REALTYPE, public  :: dsurf
!   REALTYPE, public  :: dtgrid
!   character(LEN=PATH_MAX), public  :: grid_file
!   REALTYPE, public  :: gravity
!   REALTYPE, public  :: rotation_period
!   REALTYPE, public  :: avmolu
!   REALTYPE, public  :: avmolT
!   REALTYPE, public  :: avmolS
!   integer,  public  :: MaxItz0b
!   logical,  public  :: no_shear
!
!  the roughness lengths
!   REALTYPE, public  :: z0b,z0s,za
!
!  the coriolis parameter
!   REALTYPE, public  :: cori
!
!  the friction velocities
!   REALTYPE, public  :: u_taub,u_taubo,u_taus
!
!  bottom stress
!   REALTYPE, public, target  :: taub
!
!  other stuff
!   REALTYPE, public, target  :: depth0
!   REALTYPE, public, target  :: depth
!   REALTYPE, public          :: runtimeu, runtimev
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MeanflowState",
    "clean_meanflow",
    "init_meanflow",
    "post_init_meanflow",
]

_PI: float = float(np.float32(3.141592654))


class MeanflowState:
    """All module-level variables for the GOTM meanflow module.

    Mirrors the public and private data members declared in meanflow.F90.
    Call :func:`init_meanflow` to apply configuration, then set
    ``state.depth`` to the water column depth, then call
    :func:`post_init_meanflow` to allocate arrays and derive initial values.

    Array indexing follows the GOTM Fortran convention: index 0 is the
    seabed, index *nlev* is the surface, matching ``DIMENSION(0:nlev)``
    declarations.  All arrays therefore have shape ``(nlev + 1,)``.
    """

    def __init__(self) -> None:
        # grid readiness flag
        self.grid_ready: bool = False

        # --- grid coordinates and layer thicknesses, DIMENSION(0:nlev) ---
        self.ga: np.ndarray | None = None  # sigma-coordinate layer weights
        self.z: np.ndarray | None = None  # layer centre depths from surface [m]
        self.zi: np.ndarray | None = None  # layer interface depths [m]
        self.h: np.ndarray | None = None  # layer thicknesses [m]
        self.ho: np.ndarray | None = None  # old layer thicknesses [m]

        # sea surface elevation [m]
        self.zeta: float = 0.0

        # --- velocity components, DIMENSION(0:nlev) ---
        self.u: np.ndarray | None = None  # U-velocity [m/s]
        self.v: np.ndarray | None = None  # V-velocity [m/s]
        self.w: np.ndarray | None = None  # vertical velocity [m/s]
        self.uo: np.ndarray | None = None  # old U-velocity [m/s]
        self.vo: np.ndarray | None = None  # old V-velocity [m/s]

        # --- temperature and salinity, DIMENSION(0:nlev) ---
        # T -> Tc: conservative temperature; S -> Sa: absolute salinity
        self.T: np.ndarray | None = None  # conservative temperature [°C]
        self.S: np.ndarray | None = None  # absolute salinity [g/kg]
        # Tp: potential temperature [°C]; Sp: practical salinity [PSU]
        self.Tp: np.ndarray | None = None
        self.Sp: np.ndarray | None = None
        # Ti: in-situ temperature [°C]
        self.Ti: np.ndarray | None = None
        # observed profiles (conservative T, absolute S)
        self.Tobs: np.ndarray | None = None
        self.Sobs: np.ndarray | None = None

        # --- buoyancy frequency squared, DIMENSION(0:nlev) ---
        # total N², N² from temperature only, N² from salinity only
        self.NN: np.ndarray | None = None  # total N² [s⁻²]
        self.NNT: np.ndarray | None = None  # N² contribution from T [s⁻²]
        self.NNS: np.ndarray | None = None  # N² contribution from S [s⁻²]

        # --- shear-frequency squared, DIMENSION(0:nlev) ---
        # total M², M² from u only, M² from v only
        self.SS: np.ndarray | None = None  # total M² [s⁻²]
        self.SSU: np.ndarray | None = None  # M² from u [s⁻²]
        self.SSV: np.ndarray | None = None  # M² from v [s⁻²]

        # Stokes-Eulerian cross-shear and Stokes shear squared, DIMENSION(0:nlev)
        self.SSCSTK: np.ndarray | None = None  # Stokes-Eulerian cross-shear [s⁻²]
        self.SSSTK: np.ndarray | None = None  # Stokes shear squared [s⁻²]

        # --- buoyancy, radiation, extra TKE production, DIMENSION(0:nlev) ---
        self.buoy: np.ndarray | None = None  # buoyancy [m/s²]
        self.rad: np.ndarray | None = None  # short-wave radiation [W/m²]
        self.xP: np.ndarray | None = None  # extra TKE production [m²/s³]

        # dummy diffusivity work array, DIMENSION(0:nlev)
        self.avh: np.ndarray | None = None

        # extra friction terms (e.g. seagrass), DIMENSION(0:nlev)
        self.fric: np.ndarray | None = None  # extra friction [m/s²]
        self.drag: np.ndarray | None = None  # extra drag [1/s]

        # biological shading, DIMENSION(0:nlev); 1 = no shading
        self.bioshade: np.ndarray | None = None

        # fake ice thickness — switch between 0 and 1, see temperature.F90
        self.Hice: float = 0.0

        # --- configuration scalars (set by init_meanflow) ---
        # physical bottom roughness [m]; z0b = 0.03*h0b
        self.h0b: float = 0.05
        self.calc_bottom_stress: bool = True
        # hydrodynamic surface roughness minimum [m]
        self.z0s_min: float = 0.02
        # Charnock (1955) roughness adaptation
        self.charnock: bool = False
        # empirical constant for Charnock roughness adaptation [-]
        self.charnock_val: float = 1400.0
        # grid stretching parameters (upper and lower)
        self.ddu: float = 0.0
        self.ddl: float = 0.0
        # grid generation method selector
        self.grid_method: int = 1
        # adaptive grid advection coefficients
        self.c1ad: float = 0.0
        self.c2ad: float = 0.0
        self.c3ad: float = 0.0
        self.c4ad: float = 0.0
        # grid adaptation time scale [s]
        self.Tgrid: float = 0.0
        # N² normalisation factor for adaptive grid
        self.NNnorm: float = 0.0
        # M² normalisation factor for adaptive grid
        self.SSnorm: float = 0.0
        # surface layer depth [m]
        self.dsurf: float = 0.0
        # grid adaptation time step [s]
        self.dtgrid: float = 0.0
        # path to external grid file
        self.grid_file: str = ""
        # gravitational acceleration [m/s²]
        self.gravity: float = 9.81
        # Earth rotation period [s]; default = one sidereal day
        self.rotation_period: float = 86164.0
        # molecular viscosity for momentum [m²/s]
        self.avmolu: float = 1.3e-6
        # molecular diffusivity for temperature [m²/s]
        self.avmolT: float = 1.4e-7
        # molecular diffusivity for salinity [m²/s]
        self.avmolS: float = 1.1e-9
        # max iterations for hydrodynamic bottom roughness
        self.MaxItz0b: int = 1
        # suppress shear production term
        self.no_shear: bool = False

        # --- roughness lengths ---
        self.z0b: float = 0.0  # hydrodynamic bottom roughness [m]
        self.z0s: float = 0.0  # surface roughness [m]
        self.za: float = 0.0  # roughness from suspended sediment [m]

        # Coriolis parameter [s⁻¹]; set by post_init_meanflow from latitude
        self.cori: float = 0.0

        # --- friction velocities and bottom stress ---
        self.u_taub: float = 0.0  # bottom friction velocity [m/s]
        self.u_taubo: float = 0.0  # old bottom friction velocity [m/s]
        self.u_taus: float = 0.0  # surface friction velocity [m/s]
        self.taub: float = 0.0  # bottom stress [Pa]

        # --- water column depth ---
        # depth must be set by the caller before post_init_meanflow
        self.depth: float = 0.0  # current water column depth [m]
        self.depth0: float = 0.0  # initial water column depth [m]

        # cumulative run-time ramp timers [s]
        self.runtimeu: float = 0.0
        self.runtimev: float = 0.0

        # Cached Numba workspaces keyed by translated module name.
        # The cache is invalidated when the allocated vertical resolution changes.
        self._kernel_workspaces: dict[str, object] = {}
        self._kernel_nlev: int | None = None


def init_meanflow(
    state: MeanflowState,
    *,
    calc_bottom_stress: bool = True,
    h0b: float = 0.05,
    max_it_z0b: int = 1,
    charnock: bool = False,
    charnock_val: float = 1400.0,
    z0s_min: float = 0.02,
    gravity: float = 9.81,
    rotation_period: float = 86164.0,
    avmolu: float = 1.3e-6,
    avmolT: float = 1.4e-7,
    avmolS: float = 1.1e-9,
) -> None:
    """Read configuration into *state* for the meanflow module.

    ! !IROUTINE: Initialisation of the mean flow variables
    !
    ! !DESCRIPTION:
    !  Allocates memory and initialises everything related
    !  to the `meanflow' component of GOTM.
    !
    ! !REVISION HISTORY:
    !  Original FORTRAN author(s): Karsten Bolding & Hans Burchard

    Parameters
    ----------
    state:
        MeanflowState instance to configure.
    calc_bottom_stress:
        Compute bottom stress (default: True).
    h0b:
        Physical bottom roughness [m] (default 0.05).
        Relates to hydrodynamic roughness as z0b = 0.03*h0b + 0.1*nu/ustar.
    max_it_z0b:
        Number of iterations for hydrodynamic bottom roughness (default 1).
    charnock:
        Use Charnock (1955) surface roughness adaptation (default False).
    charnock_val:
        Empirical constant for Charnock roughness adaptation (default 1400).
    z0s_min:
        Minimum hydrodynamic surface roughness [m] (default 0.02).
    gravity:
        Gravitational acceleration [m/s²] (default 9.81).
    rotation_period:
        Earth rotation period [s] (default 86164, one sidereal day).
    avmolu:
        Molecular viscosity for momentum [m²/s] (default 1.3e-6).
    avmolT:
        Molecular diffusivity for temperature [m²/s] (default 1.4e-7).
    avmolS:
        Molecular diffusivity for salinity [m²/s] (default 1.1e-9).
    """
    state.calc_bottom_stress = calc_bottom_stress
    state.h0b = h0b
    state.MaxItz0b = max_it_z0b
    state.charnock = charnock
    state.charnock_val = charnock_val
    state.z0s_min = z0s_min
    state.gravity = gravity
    state.rotation_period = rotation_period
    state.avmolu = avmolu
    state.avmolT = avmolT
    state.avmolS = avmolS


def post_init_meanflow(
    state: MeanflowState,
    nlev: int,
    latitude: float,
) -> None:
    """Allocate arrays and initialise derived quantities for a column of *nlev* layers.

    ! !IROUTINE: Initialisation of the mean flow variables
    !
    ! !DESCRIPTION:
    !  Allocates memory and initialises everything related
    !  to the `meanflow' component of GOTM.
    !
    ! !INPUT PARAMETERS:
    !   integer, intent(in)                      :: nlev
    !   REALTYPE, intent(in)                     :: latitude
    !
    ! !DEFINED PARAMETERS:
    !   REALTYPE, parameter  :: pi=3.141592654
    !
    ! !REVISION HISTORY:
    !  Original FORTRAN author(s): Karsten Bolding & Hans Burchard

    Parameters
    ----------
    state:
        MeanflowState instance (must have ``state.depth`` set by caller).
    nlev:
        Number of model layers.  Arrays are allocated with shape ``(nlev+1,)``,
        index 0 = seabed, index nlev = surface.
    latitude:
        Geographic latitude [degrees N], used to compute the Coriolis parameter.

    Notes
    -----
    Important: ``state.depth`` must be set by the caller *before* this routine
    is called, because it has already been initialised by gotm.F90 in the
    Fortran reference code.
    """
    # Important: we do not initialize "depth" here, because it has already
    # been initialized by gotm.F90.

    # Initialize bottom and surface stress to zero.
    # They will be set in friction, but also used as input in the same routine.
    state.u_taub = 0.0
    state.u_taubo = 0.0
    state.u_taus = 0.0
    state.taub = 0.0

    # Store initial depth (actual depth will be a function of surface elevation)
    state.depth0 = state.depth

    # Initialize surface and bottom roughness
    state.z0b = 0.03 * state.h0b
    # lu (otherwise z0s is not initialized)
    state.z0s = state.z0s_min
    # roughness caused by suspended sediment
    state.za = 0.0

    # Calculate Coriolis parameter:
    # f = 2*Omega*sin(lat) where Omega = 2*pi/rotation_period
    # Fortran: cori = 2*2*pi/rotation_period * sin(2*pi*latitude/360.)
    state.cori = 4.0 * _PI / state.rotation_period * math.sin(_PI * latitude / 180.0)

    # Specify that the buoyancy profile and grid still need to be calculated.
    # Note: used only if a prognostic equation for buoyancy is used.
    state.grid_ready = False

    # Initialize cumulative run time used to detect u and v ramp.
    state.runtimeu = 0.0
    state.runtimev = 0.0
    state._kernel_workspaces.clear()
    state._kernel_nlev = nlev

    n = nlev + 1  # DIMENSION(0:nlev) → shape (nlev+1,)

    # grid arrays
    state.ga = np.zeros(n)
    state.z = np.zeros(n)
    state.zi = np.zeros(n)
    state.h = np.zeros(n)
    state.ho = np.zeros(n)

    # velocity arrays
    state.u = np.zeros(n)
    state.uo = np.zeros(n)
    state.v = np.zeros(n)
    state.vo = np.zeros(n)
    state.w = np.zeros(n)

    # extra friction terms
    state.fric = np.zeros(n)
    state.drag = np.zeros(n)

    # temperature arrays
    state.T = np.zeros(n)
    state.Tp = np.zeros(n)
    state.Ti = np.zeros(n)
    state.Tobs = np.zeros(n)

    # salinity arrays
    state.S = np.zeros(n)
    state.Sp = np.zeros(n)
    state.Sobs = np.zeros(n)

    # stratification arrays
    state.NN = np.zeros(n)
    state.NNT = np.zeros(n)
    state.NNS = np.zeros(n)

    # shear arrays
    state.SS = np.zeros(n)
    state.SSU = np.zeros(n)
    state.SSV = np.zeros(n)

    # Stokes shear arrays
    state.SSCSTK = np.zeros(n)
    state.SSSTK = np.zeros(n)

    # production and forcing arrays
    state.xP = np.zeros(n)
    state.buoy = np.zeros(n)
    state.rad = np.zeros(n)

    # diffusivity work array
    state.avh = np.zeros(n)

    # biological shading initialized to 1 (no shading = full transmission)
    state.bioshade = np.ones(n)


def clean_meanflow(state: MeanflowState) -> None:
    """Release all allocated arrays in *state*.

    ! !IROUTINE: Cleaning up the mean flow variables
    !
    ! !DESCRIPTION:
    !  De-allocates all memory allocated via init_meanflow()
    !
    ! !REVISION HISTORY:
    !  Original FORTRAN author(s): Karsten Bolding & Hans Burchard
    """
    state.ga = None
    state.z = None
    state.zi = None
    state.h = None
    state.ho = None
    state.u = None
    state.uo = None
    state.v = None
    state.vo = None
    state.w = None
    state.fric = None
    state.drag = None
    state.T = None
    state.Tp = None
    state.Ti = None
    state.Tobs = None
    state.S = None
    state.Sp = None
    state.Sobs = None
    state.NN = None
    state.NNT = None
    state.NNS = None
    state.SS = None
    state.SSU = None
    state.SSV = None
    state.SSCSTK = None
    state.SSSTK = None
    state.xP = None
    state.buoy = None
    state.rad = None
    state.avh = None
    state.bioshade = None
    state._kernel_workspaces.clear()
    state._kernel_nlev = None
