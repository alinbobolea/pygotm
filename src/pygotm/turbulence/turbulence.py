r"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: turbulence: its all in here \ldots \label{sec:turbulence}
!
! !INTERFACE:
!   module turbulence
!
! !DESCRIPTION:
! In this module, variables of the turbulence model and some
! member functions to manipulate them are defined. The key-functions
! are {\tt init\_turbulence()}, which initialises the model, and
! {\tt do\_turbulence()}, which manages the time step for the
! whole procedure. These two functions are the only `public' member
! functions i.e.\ they are callable from outside the module.
! There are many more internal functions, for
! which descriptions are provided seperately.
!
! It should be pointed out that the turbulence module of GOTM may be used in
! combination with virtually any shallow-wate 3-D circulation model
! using a structured grid in the vertical direction.
! To this end, a clear interface separating the mean flow and the turbulence
! part of GOTM is required. Vertical columns of the three-dimensional fields have
! to be copied into one-dimensional vectors, which are passed to GOTM. With the help
! of this information, GOTM updates the turbulent fields and returns one-dimensional
! vectors of the turbulent diffusivities and/or the turbulent fluxes to the 3-D model.
! The `door' between the 3-D model and GOTM is the function {\tt do\_turbulence()},
! which has been designed with these ideas in mind.
!
! !USES:
!   IMPLICIT NONE
!
!  default: all is private.
!   private
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_turbulence,post_init_turbulence,do_turbulence
!   public k_bc,q2over2_bc,epsilon_bc,omega_bc,psi_bc,q2l_bc
!   public clean_turbulence
!
! !PUBLIC DATA MEMBERS:
!  TKE, rate of dissipation, omega, turbulent length-scale
!   REALTYPE, public, dimension(:), allocatable, target :: tke,eps,omega,L
!
!  TKE at old time level
!   REALTYPE, public, dimension(:), allocatable, target   :: tkeo
!
!  buoyancy variance and its destruction
!   REALTYPE, public, dimension(:), allocatable   :: kb,epsb
!
!  shear and buoyancy production
!  of tke and buoyancy variance
!   REALTYPE, public, dimension(:), allocatable   :: P,B,Pb
!
!  extra production term
!   REALTYPE, public, dimension(:), allocatable   :: Px
!
!  Stokes production
!   REALTYPE, public, dimension(:), allocatable   :: PSTK
!
!  turbulent diffusivities
!  of momentum, temperature, salinity
!   REALTYPE, public, dimension(:), allocatable, target :: num
!   REALTYPE, public, dimension(:), allocatable, target :: nuh
!   REALTYPE, public, dimension(:), allocatable         :: nus
!
!  turbulent eddy coefficient for momentum flux down Stokes gradient
!  in second moment closures with Craik-Leibovich vortex force in the
!  algebraic Reynolds stress and flux models.
!   REALTYPE, public, dimension(:), allocatable         :: nucl
!
!  non-local fluxes of momentum
!   REALTYPE, public, dimension(:), allocatable   :: gamu,gamv
!
!  non-local fluxes
!  of buoyancy, temperature, salinity
!   REALTYPE, public, dimension(:), allocatable   :: gamb,gamh,gams
!
!  non-dimensional  stability functions
!   REALTYPE, public, dimension(:), allocatable   :: cmue1,cmue2, cmue3
!
!  spatially varying sq and sl for q2over2 and lengthscale
!   REALTYPE, public, dimension(:), allocatable   :: sq_var, sl_var
!
!  non-dimensional counter-gradient term
!   REALTYPE, public, dimension(:), allocatable   :: gam
!
!  alpha_M, alpha_N, and alpha_B
!   REALTYPE, public, dimension(:), allocatable   :: as,an,at
!
!  alpha_V, alpha_W
!  dimensionless Stokes-Eulerian cross-shear and Stokes shear^2
!   REALTYPE, public, dimension(:), allocatable   :: av,aw
!
!  surface proximity function
!   REALTYPE, public, dimension(:), allocatable   :: SPF
!
!  time scale ratio r
!   REALTYPE, public, dimension(:), allocatable   :: r
!
!  the gradient Richardson number
!   REALTYPE, public, dimension(:), allocatable   :: Rig
!
!  the flux Richardson number
!   REALTYPE, public, dimension(:), allocatable   :: xRf
!
!  turbulent velocity variances
!   REALTYPE, public, dimension(:), allocatable   :: uu,vv,ww
!
!  some additional constants
!   REALTYPE, public                              :: cm0,cmsf,cde,rcm, b1
!
!  Prandtl-number in neutrally stratified flow
!   REALTYPE, public                              :: Prandtl0
!
!  parameters for wave-breaking
!   REALTYPE, public                              :: craig_m,sig_e0
!
!  the 'turbulence' namelist
!   integer, public                               :: turb_method
!   integer, public                               :: tke_method
!   integer, public                               :: len_scale_method
!   integer, public                               :: stab_method
!
!  the 'bc' namelist
!   integer, public                               :: k_ubc
!   integer, public                               :: k_lbc
!   integer, public                               :: kb_ubc
!   integer, public                               :: kb_lbc
!   integer, public                               :: psi_ubc
!   integer, public                               :: psi_lbc
!   integer, public                               :: ubc_type
!   integer, public                               :: lbc_type
!
! !BUGS:
!        The algebraic equation for the TKE is not safe
!        to use at the moment. Use it only in connection
!        with the prescribed length-scale profiles. The
!        functions report_model() will report wrong things
!        for the algebraic TKE equation. To be fixed with
!        the next version.
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Karsten Bolding, Hans Burchard,
!                      Manuel Ruiz Villarreal,
!                      Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import numpy as np

__all__ = [
    "TurbulenceState",
    "init_turbulence",
    "post_init_turbulence",
    "do_turbulence",
    "run_variances",
    "clean_turbulence",
    "k_bc",
    "q2over2_bc",
    "epsilon_bc",
    "omega_bc",
    "psi_bc",
    "q2l_bc",
    "no_model",
    "algebraic",
    "first_order",
    "second_order",
    "tke_local_eq",
    "tke_keps",
    "tke_MY",
    "Constant",
    "Munk_Anderson",
    "Schumann_Gerz",
    "Parabolic",
    "Triangular",
    "Xing_Davies",
    "Robert_Ouellet",
    "Blackadar",
    "Bougeault_Andre",
    "diss_eq",
    "omega_eq",
    "length_eq",
    "generic_eq",
    "Dirichlet",
    "Neumann",
    "viscous",
    "logarithmic",
    "injection",
]

ScalarOverride = bool | int | float

# General outline of the turbulence model.
no_model = 0
algebraic = 1
first_order = 2
second_order = 3

# Method to update TKE.
tke_local_eq = 1
tke_keps = 2
tke_MY = 3

# Stability functions.
Constant = 1
Munk_Anderson = 2
Schumann_Gerz = 3

# Method to update length scale.
Parabolic = 1
Triangular = 2
Xing_Davies = 3
Robert_Ouellet = 4
Blackadar = 5
Bougeault_Andre = 6
diss_eq = 8
length_eq = 9
generic_eq = 10
omega_eq = 11

# Boundary conditions.
Dirichlet = 0
Neumann = 1
viscous = 0
logarithmic = 1
injection = 2

# Private second-order-closure selectors from turbulence.F90.
quasi_Eq = 1
weak_Eq_Kb_Eq = 2
weak_Eq_Kb = 3
quasi_Eq_H15 = 4

kb_algebraic = 1
kb_dynamic = 2

epsb_algebraic = 1
epsb_dynamic = 2

LIST = 0
GL78 = 1
MY82 = 2
KC94 = 3
LDOR96 = 4
CHCD01A = 5
CHCD01B = 6
CCH02 = 7

_CVMIX_TURB_METHOD = 100


def _f90_default_real(value: float) -> float:
    """Return the value GOTM gets from an unsuffixed Fortran real literal."""

    return float(np.float32(value))


_F90_TWO_THIRDS = float(np.float32(2.0) / np.float32(3.0))
_DEFAULT_EDDY_DIFFUSIVITY = 1.0e-6
_ARRAY_FIELD_NAMES = (
    "tke",
    "eps",
    "omega",
    "L",
    "tkeo",
    "kb",
    "epsb",
    "P",
    "B",
    "Pb",
    "Px",
    "PSTK",
    "num",
    "nuh",
    "nus",
    "nucl",
    "gamu",
    "gamv",
    "gamb",
    "gamh",
    "gams",
    "cmue1",
    "cmue2",
    "cmue3",
    "sq_var",
    "sl_var",
    "gam",
    "as_",
    "an",
    "at",
    "av",
    "aw",
    "SPF",
    "r",
    "Rig",
    "xRf",
    "uu",
    "vv",
    "ww",
)


def _fk_craig(u_tau: float, eta: float) -> float:
    r"""Return the Craig-Banner wave-breaking TKE flux.

    ! !ROUTINE: TKE flux from wave-breaking\label{sec:fkCraig}
    !
    ! !DESCRIPTION:
    ! This functions returns the flux of $k$ caused by breaking surface waves
    ! according to
    ! \begin{equation}
    !  \label{craig}
    !   F_k = \eta u_*^3
    !  \point
    ! \end{equation}
    ! This form has also been used by \cite{CraigBanner94}, who suggested
    ! $\eta \approx 100$.
    """
    return eta * u_tau**3


class TurbulenceState:
    """All module-level variables for the GOTM turbulence module.

    Mirrors the public and private data members declared in
    ``turbulence.F90``. Step 3.1 translates only the shared state and
    dispatcher scaffold. The physics routines called by
    :func:`do_turbulence` are translated in later Phase 3 steps.
    """

    def __init__(self) -> None:
        # Prognostic and diagnostic turbulence profiles, DIMENSION(0:nlev).
        self.tke: np.ndarray | None = None
        self.eps: np.ndarray | None = None
        self.omega: np.ndarray | None = None
        self.L: np.ndarray | None = None
        self.tkeo: np.ndarray | None = None
        self.kb: np.ndarray | None = None
        self.epsb: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.Pb: np.ndarray | None = None
        self.Px: np.ndarray | None = None
        self.PSTK: np.ndarray | None = None
        self.num: np.ndarray | None = None
        self.nuh: np.ndarray | None = None
        self.nus: np.ndarray | None = None
        self.nucl: np.ndarray | None = None
        self.gamu: np.ndarray | None = None
        self.gamv: np.ndarray | None = None
        self.gamb: np.ndarray | None = None
        self.gamh: np.ndarray | None = None
        self.gams: np.ndarray | None = None
        self.cmue1: np.ndarray | None = None
        self.cmue2: np.ndarray | None = None
        self.cmue3: np.ndarray | None = None
        self.sq_var: np.ndarray | None = None
        self.sl_var: np.ndarray | None = None
        self.gam: np.ndarray | None = None
        self.as_: np.ndarray | None = None  # Fortran name: as
        self.an: np.ndarray | None = None
        self.at: np.ndarray | None = None
        self.av: np.ndarray | None = None
        self.aw: np.ndarray | None = None
        self.SPF: np.ndarray | None = None
        self.r: np.ndarray | None = None
        self.Rig: np.ndarray | None = None
        self.xRf: np.ndarray | None = None
        self.uu: np.ndarray | None = None
        self.vv: np.ndarray | None = None
        self.ww: np.ndarray | None = None

        # Additional constants derived by later closure initialisation.
        self.cm0: float = 0.0
        self.cmsf: float = 0.0
        self.cde: float = 0.0
        self.rcm: float = 0.0
        self.b1: float = 0.0
        self.Prandtl0: float = 0.0
        self.craig_m: float = 0.0
        self.sig_e0: float = 0.0

        # Turbulence closure selection.
        self.turb_method: int = first_order
        self.tke_method: int = tke_keps
        self.len_scale_method: int = diss_eq
        self.stab_method: int = Schumann_Gerz

        # Boundary-condition selection.
        self.k_ubc: int = Neumann
        self.k_lbc: int = Neumann
        self.kb_ubc: int = Neumann
        self.kb_lbc: int = Neumann
        self.psi_ubc: int = Neumann
        self.psi_lbc: int = Neumann
        self.ubc_type: int = logarithmic
        self.lbc_type: int = logarithmic

        # General turbulence parameters.
        self.cm0_fix: float = 0.5477
        self.Prandtl0_fix: float = 0.74
        self.cw: float = 100.0
        self.compute_kappa: bool = True
        self.kappa: float = 0.4
        self.compute_c3: bool = True
        self.ri_st: float = 0.25
        self.length_lim: bool = True
        self.galp: float = 0.27
        self.const_num: float = 5.0e-4
        self.const_nuh: float = 5.0e-4
        self.k_min: float = 1.0e-8
        self.eps_min: float = 1.0e-12
        self.kb_min: float = 1.0e-8
        self.epsb_min: float = 1.0e-12

        # Generic two-equation model coefficients.
        self.compute_param: bool = False
        self.gen_m: float = 1.5
        self.gen_n: float = -1.0
        self.gen_p: float = 3.0
        self.cpsi1: float = 1.44
        self.cpsi2: float = 1.92
        self.cpsi3minus: float = 0.0
        self.cpsi3plus: float = 1.0
        self.cpsix: float = self.cpsi1
        self.cpsi4: float = 0.0
        self.sig_kpsi: float = 1.0
        self.sig_psi: float = 1.3
        self.gen_d: float = -1.2
        self.gen_alpha: float = -2.0
        self.gen_l: float = 0.2

        # k-epsilon coefficients.
        self.ce1: float = 1.44
        self.ce2: float = 1.92
        self.ce3minus: float = 0.0
        self.ce3plus: float = 1.5
        self.cex: float = self.ce1
        self.ce4: float = 0.0
        self.sig_k: float = 1.0
        self.sig_e: float = 1.3
        self.sig_peps: bool = False

        # k-omega coefficients.
        self.cw1: float = 0.555
        self.cw2: float = 0.833
        self.cw3minus: float = 0.0
        self.cw3plus: float = 0.5
        self.cwx: float = self.cw1
        self.cw4: float = 0.15
        self.sig_kw: float = 2.0
        self.sig_w: float = 2.0

        # Mellor-Yamada coefficients.
        self.e1: float = 1.8
        self.e2: float = 1.33
        self.e3: float = 1.8
        self.ex: float = self.e1
        self.e6: float = 4.0
        self.sq: float = 0.2
        self.sl: float = 0.2
        self.my_length: int = 1
        self.new_constr: bool = False

        # Second-order model switches and coefficients.
        self.scnd_method: int = 0
        self.kb_method: int = 0
        self.epsb_method: int = 0
        self.scnd_coeff: int = CHCD01A
        self.cc1: float = 0.0
        self.ct1: float = 0.0
        self.ctt: float = 0.0
        self.cc2: float = 0.0
        self.cc3: float = 0.0
        self.cc4: float = 0.0
        self.cc5: float = 0.0
        self.cc6: float = 0.0
        self.ct2: float = 0.0
        self.ct3: float = 0.0
        self.ct4: float = 0.0
        self.ct5: float = 0.0
        self.a1: float = 0.0
        self.a2: float = 0.0
        self.a3: float = 0.0
        self.a4: float = 0.0
        self.a5: float = 0.0
        self.at1: float = 0.0
        self.at2: float = 0.0
        self.at3: float = 0.0
        self.at4: float = 0.0
        self.at5: float = 0.0

        # Internal-wave model switches.
        self.iw_model: int = 0
        self.alpha: float = 0.0
        self.klimiw: float = 1.0e-6
        self.rich_cr: float = 0.7
        self.numiw: float = 1.0e-4
        self.nuhiw: float = 1.0e-5
        self.numshear: float = 5.0e-3

        # Lazily allocated NumPy workspaces reused across time steps.
        self._kernel_workspaces: dict[str, object] = {}
        self._kernel_nlev: int | None = None


def init_turbulence(
    state: TurbulenceState,
    nlev: int | None = None,
    *,
    overrides: Mapping[str, ScalarOverride] | None = None,
    **keyword_overrides: ScalarOverride,
) -> None:
    r"""Apply in-memory configuration overrides to ``state``.

    ! !IROUTINE: Initialise the turbulence module
    !
    ! !DESCRIPTION:
    ! Initialises all turbulence related stuff. This routine reads a number
    ! of namelists and allocates memory for turbulence related vectors.
    ! The core consists of calls to the the internal functions
    ! {\tt generate\_model()} and {\tt analyse\_model()}, discussed in
    ! great detail in \sect{sec:generate} and \sect{sec:analyse}, respectively.
    ! The former function computes the model coefficients for the generic
    ! two-equation model from physically motivated quantities. The latter
    ! function does the inverse.
    !
    ! !REVISION HISTORY:
    !  Original FORTRAN author(s): Karsten Bolding, Hans Burchard,
    !                      Manuel Ruiz Villarreal,
    !                      Lars Umlauf

    Parameters
    ----------
    state:
        Turbulence module state to configure.
    nlev:
        Optional number of levels. When provided, this routine also calls
        :func:`post_init_turbulence`.
    overrides:
        Optional mapping of attribute names to override values.
    **keyword_overrides:
        Additional attribute overrides. Keyword arguments take precedence over
        conflicting keys in ``overrides``.
    """
    merged_overrides: dict[str, ScalarOverride] = {}
    if overrides is not None:
        merged_overrides.update(overrides)
    merged_overrides.update(keyword_overrides)

    for name, value in merged_overrides.items():
        if name in _ARRAY_FIELD_NAMES:
            msg = f"{name!r} is an allocated profile field, not an init override"
            raise ValueError(msg)
        if not hasattr(state, name):
            msg = f"unknown turbulence configuration field {name!r}"
            raise AttributeError(msg)
        setattr(state, name, value)

    if nlev is not None:
        post_init_turbulence(state, nlev)


def post_init_turbulence(state: TurbulenceState, nlev: int) -> None:
    r"""Allocate turbulence arrays for ``0:nlev`` storage and seed defaults.

    ! !IROUTINE: Initialise the turbulence module
    !
    ! !DESCRIPTION:
    ! Initialises all turbulence related stuff. This routine reads a number
    ! of namelists and allocates memory for turbulence related vectors.
    !
    ! !INPUT PARAMETERS:
    !   integer, intent(in)        :: nlev
    !
    ! !REVISION HISTORY:
    !  Original FORTRAN author(s): Karsten Bolding, Hans Burchard,
    !                      Manuel Ruiz Villarreal,
    !                      Lars Umlauf
    """
    if nlev < 0:
        msg = "nlev must be non-negative"
        raise ValueError(msg)

    n = nlev + 1

    state.tke = np.full(n, state.k_min)
    state.tkeo = np.full(n, state.k_min)
    state.eps = np.full(n, state.eps_min)
    # omega is exposed at module scope in turbulence.F90, but its prognostic
    # updates are handled later by omegaeq.F90. Seed it deterministically here.
    state.omega = np.zeros(n)
    state.L = np.zeros(n)
    state.kb = np.full(n, state.kb_min)
    state.epsb = np.full(n, state.epsb_min)
    state.P = np.zeros(n)
    state.B = np.zeros(n)
    state.Pb = np.zeros(n)
    state.Px = np.zeros(n)
    state.PSTK = np.zeros(n)
    state.num = np.full(n, _DEFAULT_EDDY_DIFFUSIVITY)
    state.nuh = np.full(n, _DEFAULT_EDDY_DIFFUSIVITY)
    state.nus = np.full(n, _DEFAULT_EDDY_DIFFUSIVITY)
    state.nucl = np.zeros(n)
    state.gamu = np.zeros(n)
    state.gamv = np.zeros(n)
    state.gamb = np.zeros(n)
    state.gamh = np.zeros(n)
    state.gams = np.zeros(n)
    state.cmue1 = np.zeros(n)
    state.cmue2 = np.zeros(n)
    state.cmue3 = np.zeros(n)
    state.sq_var = np.full(n, state.sq)
    state.sl_var = np.full(n, state.sl)
    state.gam = np.zeros(n)
    state.an = np.zeros(n)
    state.as_ = np.zeros(n)
    state.at = np.zeros(n)
    state.av = np.zeros(n)
    state.aw = np.zeros(n)
    state.SPF = np.ones(n)
    state.r = np.zeros(n)
    state.Rig = np.zeros(n)
    state.xRf = np.zeros(n)
    state.uu = np.zeros(n)
    state.vv = np.zeros(n)
    state.ww = np.zeros(n)
    state._kernel_workspaces.clear()
    state._kernel_nlev = nlev

    if state.turb_method == no_model:
        state.nuh.fill(state.const_nuh)
        state.num.fill(state.const_num)
        return

    if state.turb_method == _CVMIX_TURB_METHOD:
        return

    if state.turb_method == algebraic:
        return

    if state.turb_method == second_order:
        _init_scnd(state)

    if state.turb_method not in (first_order, second_order):
        return

    _compute_cm0(state)
    state.cde = state.cm0**3
    state.rcm = state.cm0 / state.cmsf
    state.b1 = 2.0**1.5 / state.cde

    l_min = state.cde * state.k_min**1.5 / state.eps_min
    state.L.fill(l_min)

    if state.len_scale_method == generic_eq and state.compute_param:
        msg = (
            "compute_param=True requires generate_model(), which is not translated yet"
        )
        raise NotImplementedError(msg)

    _analyse_model(state)


def _state_array(state: TurbulenceState, name: str) -> np.ndarray:
    value = getattr(state, name)
    if value is None:
        msg = f"state.{name} is not allocated; call post_init_turbulence first"
        raise ValueError(msg)
    return cast(np.ndarray, value)


def _validate_profile(name: str, values: np.ndarray, nlev: int) -> None:
    expected = (nlev + 1,)
    if values.shape != expected:
        msg = f"{name} must have shape {expected}, got {values.shape}"
        raise ValueError(msg)


def _write_profile(field: np.ndarray, values: np.ndarray) -> None:
    if field.ndim == 1:
        field[:] = values
    else:
        field[0, :] = values


def _write_scalar(field: np.ndarray, value: float) -> None:
    field.fill(0.0)
    if field.ndim == 1:
        field[0] = value
    else:
        field[0, 0] = value


def _read_profile(field: np.ndarray) -> np.ndarray:
    if field.ndim == 1:
        return np.asarray(field, dtype=np.float64).copy()
    return np.asarray(field[0, :], dtype=np.float64).copy()


def _ensure_workspace(
    state: TurbulenceState,
    nlev: int,
    key: str,
    factory: Callable[[int], object],
) -> Any:
    if state._kernel_nlev != nlev:
        state._kernel_workspaces.clear()
        state._kernel_nlev = nlev

    workspace = state._kernel_workspaces.get(key)
    if workspace is None:
        workspace = factory(nlev)
        state._kernel_workspaces[key] = workspace
    return workspace


def _load_state_fields(
    state: TurbulenceState, workspace: object, names: tuple[str, ...]
) -> None:
    for name in names:
        _write_profile(getattr(workspace, name), _state_array(state, name))


def _load_input_fields(workspace: object, fields: Mapping[str, np.ndarray]) -> None:
    for name, values in fields.items():
        _write_profile(getattr(workspace, name), values)


def _write_scalar_fields(workspace: object, fields: Mapping[str, float]) -> None:
    for name, value in fields.items():
        _write_scalar(getattr(workspace, name), value)


def _zero_workspace_fields(workspace: object, names: tuple[str, ...]) -> None:
    for name in names:
        getattr(workspace, name).fill(0.0)


def _sync_state_fields(
    state: TurbulenceState, workspace: object, names: tuple[str, ...]
) -> None:
    for name in names:
        _state_array(state, name)[:] = _read_profile(getattr(workspace, name))


def _copy_neighbour_boundaries(array: np.ndarray, nlev: int) -> None:
    if nlev == 0:
        return
    array[0] = array[min(1, nlev)]
    array[nlev] = array[max(nlev - 1, 0)]


def _init_scnd(state: TurbulenceState) -> None:
    def preset(*values: float) -> tuple[float, ...]:
        return tuple(_f90_default_real(value) for value in values)

    presets = {
        GL78: preset(
            3.6, 0.8, 1.2, 1.2, 0.0, 0.5, 3.0, 0.3333, 0.3333, 0.0, 0.3333, 0.8
        ),
        MY82: preset(6.0, 0.32, 0.0, 0.0, 0.0, 0.0, 3.728, 0.0, 0.0, 0.0, 0.0, 0.6102),
        KC94: preset(6.0, 0.32, 0.0, 0.0, 0.0, 0.0, 3.728, 0.7, 0.7, 0.0, 0.2, 0.6102),
        LDOR96: preset(
            3.0, 0.8, 2.0, 1.118, 0.0, 0.5, 3.0, 0.3333, 0.3333, 0.0, 0.3333, 0.8
        ),
        CHCD01A: preset(
            5.0, 0.8, 1.968, 1.136, 0.0, 0.4, 5.95, 0.6, 1.0, 0.0, 0.3333, 0.72
        ),
        CHCD01B: preset(
            5.0,
            0.6983,
            1.9664,
            1.094,
            0.0,
            0.495,
            5.6,
            0.6,
            1.0,
            0.0,
            0.3333,
            0.477,
        ),
        CCH02: preset(
            5.0,
            0.7983,
            1.968,
            1.136,
            0.0,
            0.5,
            5.52,
            0.2134,
            0.357,
            0.0,
            0.3333,
            0.82,
        ),
    }

    if state.scnd_coeff != LIST:
        try:
            (
                state.cc1,
                state.cc2,
                state.cc3,
                state.cc4,
                state.cc5,
                state.cc6,
                state.ct1,
                state.ct2,
                state.ct3,
                state.ct4,
                state.ct5,
                state.ctt,
            ) = presets[state.scnd_coeff]
        except KeyError as exc:
            msg = f"invalid second-order coefficient set {state.scnd_coeff}"
            raise ValueError(msg) from exc

    state.a1 = _F90_TWO_THIRDS - state.cc2 / 2.0
    state.a2 = 1.0 - state.cc3 / 2.0
    state.a3 = 1.0 - state.cc4 / 2.0
    state.a4 = state.cc5 / 2.0
    state.a5 = 0.5 - state.cc6 / 2.0
    state.at1 = 1.0 - state.ct2
    state.at2 = 1.0 - state.ct3
    state.at3 = 2.0 * (1.0 - state.ct4)
    state.at4 = 2.0 * (1.0 - state.ct5)
    state.at5 = 2.0 * state.ctt * (1.0 - state.ct5)


def _compute_cm0(state: TurbulenceState) -> None:
    if state.turb_method == first_order:
        if state.stab_method not in (Constant, Munk_Anderson, Schumann_Gerz):
            msg = f"invalid first-order stability function {state.stab_method}"
            raise ValueError(msg)
        state.cm0 = state.cm0_fix
        state.cmsf = state.cm0_fix
        return

    if state.turb_method != second_order:
        msg = f"unsupported turb_method for cm0 initialisation: {state.turb_method}"
        raise ValueError(msg)

    a1 = _F90_TWO_THIRDS - state.cc2 / 2.0
    a3 = 1.0 - state.cc4 / 2.0
    n_val = state.cc1 / 2.0
    numerator = state.a2**2 - 3.0 * a3**2 + 3.0 * a1 * n_val
    if n_val == 0.0 or numerator <= 0.0:
        msg = "second-order closure coefficients are not initialised consistently"
        raise ValueError(msg)

    state.cm0 = (numerator / (3.0 * n_val * n_val)) ** 0.25
    state.cmsf = a1 / n_val / state.cm0**3


def _analyse_model(state: TurbulenceState) -> None:
    from pygotm.turbulence.compute_cpsi3 import compute_cpsi3
    from pygotm.turbulence.compute_rist import compute_rist

    if state.len_scale_method in (
        Parabolic,
        Triangular,
        Xing_Davies,
        Blackadar,
        Bougeault_Andre,
    ):
        if state.compute_kappa:
            state.kappa = 0.4
        state.gen_l = state.kappa
        state.gen_alpha = -np.sqrt(
            2.0 / 3.0 * state.cm0**2 * state.rcm * state.sig_k / state.gen_l**2
        )
        return

    if state.len_scale_method == Robert_Ouellet:
        if state.compute_kappa:
            state.kappa = 0.4
        return

    if state.len_scale_method == diss_eq:
        if state.compute_kappa:
            rad = state.sig_e * (state.ce2 - state.ce1)
            if rad <= 0.0:
                msg = "invalid k-epsilon coefficients: ce2 must exceed ce1"
                raise ValueError(msg)
            state.kappa = state.cm0 * np.sqrt(rad)
            if state.sig_peps:
                msg = "sig_peps requires compute_kappa=False, matching GOTM"
                raise ValueError(msg)
            state.sig_e0 = state.sig_e
        else:
            denom = state.ce2 - state.ce1
            if denom <= 0.0:
                msg = "invalid k-epsilon coefficients: ce2 must exceed ce1"
                raise ValueError(msg)
            state.sig_e = state.kappa**2 / denom / state.cm0**2
            if state.sig_peps:
                state.craig_m = np.sqrt(
                    1.5 * state.cmsf**2 * state.sig_k / state.kappa**2
                )
                state.sig_e0 = (
                    (4.0 / 3.0 * state.craig_m + 1.0)
                    * (state.craig_m + 1.0)
                    * state.kappa**2
                    / (state.ce2 * state.cmsf**2)
                )
            else:
                state.sig_e0 = state.sig_e

        state.gen_d = 1.0 / (1.0 - state.ce2)
        alpha_root_term = np.sqrt(state.sig_k + 24.0 * state.sig_e0 * state.ce2)
        l_root_term = np.sqrt(
            state.sig_k * (state.sig_k + 24.0 * state.sig_e0 * state.ce2)
        )
        state.gen_alpha = (
            4.0 * np.sqrt(state.sig_k) / (7.0 * np.sqrt(state.sig_k) - alpha_root_term)
        )
        state.gen_l = (
            state.cm0
            * np.sqrt(state.rcm)
            * np.sqrt(
                (
                    25.0 * state.sig_k
                    + 12.0 * state.sig_e0 * state.ce2
                    - 7.0 * l_root_term
                )
                / 12.0
            )
        )
        if state.compute_c3:
            state.ce3minus = compute_cpsi3(state, state.ce1, state.ce2, state.ri_st)
        else:
            state.ri_st = compute_rist(state, state.ce1, state.ce2, state.ce3minus)
        return

    if state.len_scale_method == omega_eq:
        if state.compute_kappa:
            rad = state.sig_w * (state.cw2 - state.cw1)
            if rad <= 0.0:
                msg = "invalid k-omega coefficients: cw2 must exceed cw1"
                raise ValueError(msg)
            state.kappa = state.cm0 * np.sqrt(rad)
        else:
            denom = state.cw2 - state.cw1
            if denom <= 0.0:
                msg = "invalid k-omega coefficients: cw2 must exceed cw1"
                raise ValueError(msg)
            state.sig_w = state.kappa**2 / denom / state.cm0**2

        state.sig_k = state.sig_kw
        state.gen_d = -1.0 / state.cw2
        alpha_root_term = np.sqrt(state.sig_k + 24.0 * state.sig_w * state.cw2)
        l_root_term = np.sqrt(
            state.sig_k * (state.sig_k + 24.0 * state.sig_w * state.cw2)
        )
        state.gen_alpha = (
            4.0 * np.sqrt(state.sig_k) / (3.0 * np.sqrt(state.sig_k) - alpha_root_term)
        )
        state.gen_l = (
            state.cm0
            * np.sqrt(state.rcm)
            * np.sqrt(
                (5.0 * state.sig_k + 12.0 * state.sig_w * state.cw2 - 3.0 * l_root_term)
                / 12.0
            )
        )
        if state.compute_c3:
            state.cw3minus = compute_cpsi3(state, state.cw1, state.cw2, state.ri_st)
        else:
            state.ri_st = compute_rist(state, state.cw1, state.cw2, state.cw3minus)
        return

    if state.len_scale_method == generic_eq:
        rad = state.sig_psi * (state.cpsi2 - state.cpsi1) / state.gen_n**2
        if rad <= 0.0:
            msg = "invalid GLS coefficients: cpsi2 must exceed cpsi1"
            raise ValueError(msg)
        state.kappa = state.cm0 * np.sqrt(rad)
        state.sig_k = state.sig_kpsi
        state.gen_d = (
            -2.0 * state.gen_n / (2.0 * state.gen_m + state.gen_n - 2.0 * state.cpsi2)
        )
        root_term = np.sqrt(state.sig_k + 24.0 * state.sig_psi * state.cpsi2)
        state.gen_alpha = (
            -4.0
            * state.gen_n
            * np.sqrt(state.sig_k)
            / ((1.0 + 4.0 * state.gen_m) * np.sqrt(state.sig_k) - root_term)
        )
        state.gen_l = (
            state.cm0
            * np.sqrt(state.rcm)
            * np.sqrt(
                (
                    (1.0 + 4.0 * state.gen_m + 8.0 * state.gen_m**2) * state.sig_k
                    + 12.0 * state.sig_psi * state.cpsi2
                    - (1.0 + 4.0 * state.gen_m)
                    * np.sqrt(
                        state.sig_k * (state.sig_k + 24.0 * state.sig_psi * state.cpsi2)
                    )
                )
                / (12.0 * state.gen_n**2)
            )
        )
        if state.compute_c3:
            state.cpsi3minus = compute_cpsi3(
                state,
                state.cpsi1,
                state.cpsi2,
                state.ri_st,
            )
        else:
            state.ri_st = compute_rist(
                state,
                state.cpsi1,
                state.cpsi2,
                state.cpsi3minus,
            )
        state.cpsi3plus = (1.5 - state.ce3plus) * state.gen_n + state.gen_m
        return

    if state.len_scale_method == length_eq:
        if state.compute_kappa:
            rad = (state.e2 - state.e1 + 1.0) / (state.sl * state.b1)
            if rad <= 0.0:
                msg = "invalid Mellor-Yamada coefficients: e2 must exceed e1 - 1"
                raise ValueError(msg)
            state.kappa = np.sqrt(rad)
        else:
            state.e2 = state.kappa**2 * state.b1 * state.sl + state.e1 - 1.0

        state.gen_d = -1.0
        root_term = np.sqrt(
            12.0 * state.e2 * (2.0 * state.sl - state.sq)
            + state.b1 * state.kappa**2 * state.sl * (state.sl + 12.0 * state.sq)
        )
        state.gen_alpha = (
            5.0 * state.kappa * np.sqrt(state.b1) * state.sl + root_term
        ) / (3.0 * state.kappa * np.sqrt(state.b1) * (state.sq - 2.0 * state.sl))
        state.gen_l = state.kappa * np.sqrt(
            (
                6.0 * state.e2 * (2.0 * state.sl - state.sq)
                + state.b1
                * state.kappa**2
                * state.sl
                * (13.0 * state.sl + 6.0 * state.sq)
                - 5.0 * np.sqrt(state.b1) * state.kappa * state.sl * root_term
            )
            / (6.0 * state.sq * (state.e2 - state.b1 * state.kappa**2 * state.sl) ** 2)
        )
        if state.compute_c3:
            state.e3 = compute_cpsi3(state, state.e1, 1.0, state.ri_st)
        else:
            state.ri_st = compute_rist(state, state.e1, 1.0, state.e3)
        return


def _surface_proximity_function(
    state: TurbulenceState,
    nlev: int,
    depth: float,
    h: np.ndarray,
) -> None:
    pstk = _state_array(state, "PSTK")
    length_scale = _state_array(state, "L")
    spf = _state_array(state, "SPF")

    l_s = 0.0
    znrm = 0.0
    for i in range(1, nlev):
        dz = 0.5 * (h[i] + h[i + 1])
        weight = max(0.0, pstk[i])
        znrm += weight * dz
        l_s += length_scale[i] * weight * dz

    surface_weight = max(0.0, pstk[nlev])
    znrm += 0.5 * surface_weight * h[nlev]
    l_s += 0.5 * length_scale[nlev] * surface_weight * h[nlev]
    l_s = max(l_s / max(np.finfo(np.float64).tiny, znrm), np.finfo(np.float64).tiny)

    zz = np.zeros(nlev + 1, dtype=np.float64)
    zz[0] = -depth
    for i in range(1, nlev + 1):
        zz[i] = zz[i - 1] + h[i]

    spf[:] = 1.0 - (1.0 + np.tanh(0.25 * zz / l_s))


def _kolpran(state: TurbulenceState, nlev: int) -> None:
    tke = _state_array(state, "tke")
    length_scale = _state_array(state, "L")
    num = _state_array(state, "num")
    nuh = _state_array(state, "nuh")
    nus = _state_array(state, "nus")
    nucl = _state_array(state, "nucl")
    cmue1 = _state_array(state, "cmue1")
    cmue2 = _state_array(state, "cmue2")
    cmue3 = _state_array(state, "cmue3")

    for i in range(nlev + 1):
        x = np.sqrt(tke[i]) * length_scale[i]
        num[i] = cmue1[i] * x
        nuh[i] = cmue2[i] * x
        nus[i] = cmue2[i] * x
        nucl[i] = cmue3[i] * x


def _run_production(
    state: TurbulenceState,
    nlev: int,
    NN: np.ndarray,
    SS: np.ndarray,
    xP: np.ndarray | None,
    SSCSTK: np.ndarray | None,
    SSSTK: np.ndarray | None,
) -> None:
    from pygotm.turbulence.production import ProductionWorkspace, step_production

    workspace = _ensure_workspace(
        state,
        nlev,
        "production",
        lambda n: ProductionWorkspace(n, n_cols=1),
    )
    _load_input_fields(workspace, {"NN": NN, "SS": SS})
    _load_state_fields(
        state,
        workspace,
        ("num", "nuh", "nucl", "P", "B", "Pb", "Px", "PSTK"),
    )
    _load_input_fields(
        workspace,
        {
            "xP": xP if xP is not None else np.zeros(nlev + 1, dtype=np.float64),
            "SSCSTK": (
                SSCSTK if SSCSTK is not None else np.zeros(nlev + 1, dtype=np.float64)
            ),
            "SSSTK": (
                SSSTK if SSSTK is not None else np.zeros(nlev + 1, dtype=np.float64)
            ),
        },
    )
    step_production(
        1,
        nlev,
        state.iw_model,
        state.alpha,
        int(xP is not None),
        int(SSCSTK is not None),
        int(SSSTK is not None),
        workspace.NN,
        workspace.SS,
        workspace.xP,
        workspace.SSCSTK,
        workspace.SSSTK,
        workspace.num,
        workspace.nuh,
        workspace.nucl,
        workspace.P,
        workspace.B,
        workspace.Pb,
        workspace.Px,
        workspace.PSTK,
    )
    _sync_state_fields(state, workspace, ("P", "B", "Pb", "Px", "PSTK"))


def _run_alpha_mnb(
    state: TurbulenceState,
    nlev: int,
    NN: np.ndarray,
    SS: np.ndarray,
    SSCSTK: np.ndarray | None,
    SSSTK: np.ndarray | None,
) -> None:
    from pygotm.turbulence.alpha_mnb import AlphaMNBWorkspace, step_alpha_mnb

    workspace = _ensure_workspace(
        state,
        nlev,
        "alpha_mnb",
        lambda n: AlphaMNBWorkspace(n, n_cols=1),
    )
    _load_state_fields(
        state, workspace, ("tke", "eps", "kb", "as_", "an", "at", "av", "aw")
    )
    _load_input_fields(workspace, {"NN": NN, "SS": SS})
    _load_input_fields(
        workspace,
        {
            "SSCSTK": (
                SSCSTK if SSCSTK is not None else np.zeros(nlev + 1, dtype=np.float64)
            ),
            "SSSTK": (
                SSSTK if SSSTK is not None else np.zeros(nlev + 1, dtype=np.float64)
            ),
        },
    )
    step_alpha_mnb(
        1,
        nlev,
        int(SSCSTK is not None),
        int(SSSTK is not None),
        workspace.tke,
        workspace.eps,
        workspace.kb,
        workspace.NN,
        workspace.SS,
        workspace.SSCSTK,
        workspace.SSSTK,
        workspace.as_,
        workspace.an,
        workspace.at,
        workspace.av,
        workspace.aw,
    )
    _sync_state_fields(state, workspace, ("as_", "an", "at", "av", "aw"))


def _run_first_order_stability(state: TurbulenceState, nlev: int) -> None:
    cmue1 = _state_array(state, "cmue1")
    cmue2 = _state_array(state, "cmue2")
    cmue3 = _state_array(state, "cmue3")
    cmue3.fill(0.0)

    if state.stab_method == Constant:
        cmue1.fill(state.cm0_fix)
        cmue2.fill(state.cm0_fix / state.Prandtl0_fix)
    elif state.stab_method == Munk_Anderson:
        from pygotm.turbulence.cmue_ma import CmueMAWorkspace, step_cmue_ma

        workspace = _ensure_workspace(
            state,
            nlev,
            "cmue_ma",
            lambda n: CmueMAWorkspace(n, n_cols=1),
        )
        _load_state_fields(state, workspace, ("as_", "an", "cmue1", "cmue2"))
        step_cmue_ma(
            1,
            nlev,
            state.cm0_fix,
            state.Prandtl0_fix,
            workspace.as_,
            workspace.an,
            workspace.cmue1,
            workspace.cmue2,
        )
        _sync_state_fields(state, workspace, ("cmue1", "cmue2"))
    elif state.stab_method == Schumann_Gerz:
        from pygotm.turbulence.cmue_sg import CmueSGWorkspace, step_cmue_sg

        workspace = _ensure_workspace(
            state,
            nlev,
            "cmue_sg",
            lambda n: CmueSGWorkspace(n, n_cols=1),
        )
        _load_state_fields(state, workspace, ("as_", "an", "cmue1", "cmue2"))
        step_cmue_sg(
            1,
            nlev,
            state.cm0_fix,
            state.Prandtl0_fix,
            workspace.as_,
            workspace.an,
            workspace.cmue1,
            workspace.cmue2,
        )
        _sync_state_fields(state, workspace, ("cmue1", "cmue2"))
    else:
        msg = f"invalid first-order stability function {state.stab_method}"
        raise ValueError(msg)

    _copy_neighbour_boundaries(cmue1, nlev)
    _copy_neighbour_boundaries(cmue2, nlev)


def _run_cmue_c(state: TurbulenceState, nlev: int) -> None:
    from pygotm.turbulence.cmue_c import CmueCWorkspace, step_cmue_c

    workspace = _ensure_workspace(
        state,
        nlev,
        "cmue_c",
        lambda n: CmueCWorkspace(n, n_cols=1),
    )
    _load_state_fields(state, workspace, ("as_", "an", "cmue1", "cmue2"))
    step_cmue_c(
        1,
        nlev,
        state.cm0,
        state.cc1,
        state.ct1,
        state.a1,
        state.a2,
        state.a3,
        state.a5,
        state.at1,
        state.at2,
        state.at3,
        state.at5,
        workspace.as_,
        workspace.an,
        workspace.cmue1,
        workspace.cmue2,
    )
    _sync_state_fields(state, workspace, ("cmue1", "cmue2"))
    _state_array(state, "cmue3").fill(0.0)


def _run_cmue_d(state: TurbulenceState, nlev: int) -> None:
    from pygotm.turbulence.cmue_d import CmueDWorkspace, step_cmue_d

    workspace = _ensure_workspace(
        state,
        nlev,
        "cmue_d",
        lambda n: CmueDWorkspace(n, n_cols=1),
    )
    _load_state_fields(state, workspace, ("as_", "an", "cmue1", "cmue2"))
    step_cmue_d(
        1,
        nlev,
        state.cm0,
        state.cc1,
        state.ct1,
        state.a1,
        state.a2,
        state.a3,
        state.a5,
        state.at1,
        state.at2,
        state.at3,
        state.at5,
        workspace.as_,
        workspace.an,
        workspace.cmue1,
        workspace.cmue2,
    )
    _sync_state_fields(state, workspace, ("cmue1", "cmue2"))
    _state_array(state, "cmue3").fill(0.0)


def _run_cmue_d_h15(state: TurbulenceState, nlev: int) -> None:
    from pygotm.turbulence.cmue_d_h15 import CmueDH15Workspace, step_cmue_d_h15

    workspace = _ensure_workspace(
        state,
        nlev,
        "cmue_d_h15",
        lambda n: CmueDH15Workspace(n, n_cols=1),
    )
    _load_state_fields(
        state,
        workspace,
        (
            "as_",
            "an",
            "av",
            "aw",
            "SPF",
            "cmue1",
            "cmue2",
            "cmue3",
            "sq_var",
            "sl_var",
        ),
    )
    step_cmue_d_h15(
        1,
        nlev,
        int(state.length_lim),
        state.sq,
        state.sl,
        workspace.as_,
        workspace.an,
        workspace.av,
        workspace.aw,
        workspace.SPF,
        workspace.cmue1,
        workspace.cmue2,
        workspace.cmue3,
        workspace.sq_var,
        workspace.sl_var,
    )
    _sync_state_fields(
        state,
        workspace,
        ("cmue1", "cmue2", "cmue3", "sq_var", "sl_var"),
    )
    _copy_neighbour_boundaries(_state_array(state, "sq_var"), nlev)
    _copy_neighbour_boundaries(_state_array(state, "sl_var"), nlev)


def _run_tke(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> None:
    if state.tke_method == tke_local_eq:
        from pygotm.turbulence.tkealgebraic import (
            TKEAlgebraicWorkspace,
            step_tkealgebraic,
        )

        workspace = _ensure_workspace(
            state,
            nlev,
            "tkealgebraic",
            lambda n: TKEAlgebraicWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "L", "cmue1", "cmue2"),
        )
        _load_input_fields(workspace, {"NN": NN, "SS": SS})
        _write_scalar_fields(
            workspace,
            {
                "u_taus": u_taus,
                "u_taub": u_taub,
            },
        )
        step_tkealgebraic(
            1,
            nlev,
            state.k_min,
            state.cm0,
            state.cde,
            workspace.tke,
            workspace.tkeo,
            workspace.L,
            workspace.NN,
            workspace.SS,
            workspace.cmue1,
            workspace.cmue2,
            workspace.u_taus,
            workspace.u_taub,
        )
        _sync_state_fields(state, workspace, ("tke", "tkeo"))
        return

    if state.tke_method == tke_keps:
        from pygotm.turbulence.tkeeq import TKEEquationWorkspace, step_tkeeq

        workspace = _ensure_workspace(
            state,
            nlev,
            "tkeeq",
            lambda n: TKEEquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "P", "B", "Px", "PSTK", "num", "eps"),
        )
        _load_input_fields(workspace, {"h": h})
        _write_scalar_fields(
            workspace,
            {
                "u_taus": u_taus,
                "u_taub": u_taub,
                "z0s": z0s,
                "z0b": z0b,
            },
        )
        _zero_workspace_fields(
            workspace,
            ("avh", "l_sour", "q_sour", "au", "bu", "cu", "du", "ru", "qu"),
        )
        step_tkeeq(
            1,
            nlev,
            dt,
            state.sig_k,
            state.k_min,
            state.k_ubc,
            state.k_lbc,
            state.ubc_type,
            state.lbc_type,
            state.cm0,
            state.cmsf,
            state.cw,
            state.gen_alpha,
            state.gen_l,
            workspace.tke,
            workspace.tkeo,
            workspace.h,
            workspace.P,
            workspace.B,
            workspace.Px,
            workspace.PSTK,
            workspace.num,
            workspace.eps,
            workspace.avh,
            workspace.l_sour,
            workspace.q_sour,
            workspace.u_taus,
            workspace.u_taub,
            workspace.z0s,
            workspace.z0b,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("tke", "tkeo"))
        return

    if state.tke_method == tke_MY:
        from pygotm.turbulence.q2over2eq import (
            Q2Over2EquationWorkspace,
            step_q2over2eq,
        )

        workspace = _ensure_workspace(
            state,
            nlev,
            "q2over2eq",
            lambda n: Q2Over2EquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "P", "B", "Px", "PSTK", "eps", "L", "sq_var"),
        )
        _load_input_fields(workspace, {"h": h})
        _write_scalar_fields(
            workspace,
            {
                "u_taus": u_taus,
                "u_taub": u_taub,
                "z0s": z0s,
                "z0b": z0b,
            },
        )
        _zero_workspace_fields(
            workspace,
            ("avh", "l_sour", "q_sour", "au", "bu", "cu", "du", "ru", "qu"),
        )
        step_q2over2eq(
            1,
            nlev,
            dt,
            state.k_min,
            state.b1,
            state.k_ubc,
            state.k_lbc,
            state.ubc_type,
            state.lbc_type,
            state.sq,
            state.cw,
            state.gen_alpha,
            state.gen_l,
            workspace.tke,
            workspace.tkeo,
            workspace.h,
            workspace.P,
            workspace.B,
            workspace.Px,
            workspace.PSTK,
            workspace.eps,
            workspace.L,
            workspace.sq_var,
            workspace.avh,
            workspace.l_sour,
            workspace.q_sour,
            workspace.u_taus,
            workspace.u_taub,
            workspace.z0s,
            workspace.z0b,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("tke", "tkeo"))
        return

    msg = f"invalid tke_method={state.tke_method}"
    raise ValueError(msg)


def _run_kb(state: TurbulenceState, nlev: int, dt: float, h: np.ndarray) -> None:
    if state.kb_method == kb_algebraic:
        from pygotm.turbulence.kbalgebraic import KBAlgebraicWorkspace, step_kbalgebraic

        workspace = _ensure_workspace(
            state,
            nlev,
            "kbalgebraic",
            lambda n: KBAlgebraicWorkspace(n, n_cols=1),
        )
        _load_state_fields(state, workspace, ("tke", "eps", "kb", "Pb"))
        step_kbalgebraic(
            1,
            nlev,
            state.ctt,
            state.kb_min,
            workspace.tke,
            workspace.eps,
            workspace.kb,
            workspace.Pb,
        )
        _sync_state_fields(state, workspace, ("kb",))
        return

    if state.kb_method == kb_dynamic:
        from pygotm.turbulence.kbeq import KBEquationWorkspace, step_kbeq

        workspace = _ensure_workspace(
            state,
            nlev,
            "kbeq",
            lambda n: KBEquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(state, workspace, ("kb", "Pb", "epsb", "nuh"))
        _load_input_fields(workspace, {"h": h})
        _zero_workspace_fields(
            workspace,
            ("avh", "l_sour", "q_sour", "au", "bu", "cu", "du", "ru", "qu"),
        )
        step_kbeq(
            1,
            nlev,
            dt,
            state.kb_min,
            state.kb_ubc,
            state.kb_lbc,
            workspace.kb,
            workspace.h,
            workspace.Pb,
            workspace.epsb,
            workspace.nuh,
            workspace.avh,
            workspace.l_sour,
            workspace.q_sour,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("kb",))
        return

    msg = f"invalid kb_method={state.kb_method}"
    raise ValueError(msg)


def _run_lengthscale(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    depth: float,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
) -> None:
    if state.len_scale_method == diss_eq:
        from pygotm.turbulence.dissipationeq import (
            DissipationEquationWorkspace,
            step_dissipationeq,
        )

        workspace = _ensure_workspace(
            state,
            nlev,
            "dissipationeq",
            lambda n: DissipationEquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "eps", "L", "P", "B", "Px", "PSTK", "num"),
        )
        _load_input_fields(workspace, {"h": h, "NN": NN, "SS": SS})
        _write_scalar_fields(
            workspace,
            {
                "u_taus": u_taus,
                "u_taub": u_taub,
                "z0s": z0s,
                "z0b": z0b,
            },
        )
        _zero_workspace_fields(
            workspace,
            (
                "avh",
                "sig_eff",
                "l_sour",
                "q_sour",
                "au",
                "bu",
                "cu",
                "du",
                "ru",
                "qu",
            ),
        )
        step_dissipationeq(
            1,
            nlev,
            dt,
            state.ce1,
            state.ce2,
            state.ce3plus,
            state.ce3minus,
            state.cex,
            state.ce4,
            state.cm0,
            state.cde,
            state.kappa,
            state.galp,
            state.sig_k,
            state.sig_e,
            state.sig_e0,
            int(state.sig_peps),
            int(state.length_lim),
            state.eps_min,
            state.psi_ubc,
            state.psi_lbc,
            state.ubc_type,
            state.lbc_type,
            state.cmsf,
            state.cw,
            state.gen_alpha,
            state.gen_l,
            workspace.tke,
            workspace.tkeo,
            workspace.eps,
            workspace.L,
            workspace.h,
            workspace.NN,
            workspace.SS,
            workspace.P,
            workspace.B,
            workspace.Px,
            workspace.PSTK,
            workspace.num,
            workspace.avh,
            workspace.sig_eff,
            workspace.l_sour,
            workspace.q_sour,
            workspace.u_taus,
            workspace.u_taub,
            workspace.z0s,
            workspace.z0b,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("eps", "L"))
        return

    if state.len_scale_method == omega_eq:
        from pygotm.turbulence.omegaeq import OmegaEquationWorkspace, step_omegaeq

        workspace = _ensure_workspace(
            state,
            nlev,
            "omegaeq",
            lambda n: OmegaEquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "eps", "L", "P", "B", "Px", "PSTK", "num"),
        )
        _load_input_fields(workspace, {"h": h, "NN": NN, "SS": SS})
        _write_scalar_fields(
            workspace,
            {
                "u_taus": u_taus,
                "u_taub": u_taub,
                "z0s": z0s,
                "z0b": z0b,
            },
        )
        _zero_workspace_fields(
            workspace,
            ("omega", "avh", "l_sour", "q_sour", "au", "bu", "cu", "du", "ru", "qu"),
        )
        step_omegaeq(
            1,
            nlev,
            dt,
            state.cw1,
            state.cw2,
            state.cw3plus,
            state.cw3minus,
            state.cwx,
            state.cw4,
            state.sig_w,
            state.cm0,
            state.kappa,
            state.cde,
            state.galp,
            int(state.length_lim),
            state.eps_min,
            state.psi_ubc,
            state.psi_lbc,
            state.ubc_type,
            state.lbc_type,
            state.sig_k,
            state.cmsf,
            state.cw,
            state.gen_alpha,
            state.gen_l,
            workspace.tke,
            workspace.tkeo,
            workspace.eps,
            workspace.L,
            workspace.h,
            workspace.NN,
            workspace.SS,
            workspace.P,
            workspace.B,
            workspace.Px,
            workspace.PSTK,
            workspace.num,
            workspace.u_taus,
            workspace.u_taub,
            workspace.z0s,
            workspace.z0b,
            workspace.omega,
            workspace.avh,
            workspace.l_sour,
            workspace.q_sour,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("omega", "eps", "L"))
        return

    if state.len_scale_method == generic_eq:
        from pygotm.turbulence.genericeq import GenericEquationWorkspace, step_genericeq

        workspace = _ensure_workspace(
            state,
            nlev,
            "genericeq",
            lambda n: GenericEquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "eps", "L", "P", "B", "Px", "PSTK", "num"),
        )
        _load_input_fields(workspace, {"h": h, "NN": NN, "SS": SS})
        _write_scalar_fields(
            workspace,
            {
                "u_taus": u_taus,
                "u_taub": u_taub,
                "z0s": z0s,
                "z0b": z0b,
            },
        )
        _zero_workspace_fields(
            workspace,
            ("psi", "avh", "l_sour", "q_sour", "au", "bu", "cu", "du", "ru", "qu"),
        )
        step_genericeq(
            1,
            nlev,
            dt,
            state.cpsi1,
            state.cpsi2,
            state.cpsi3plus,
            state.cpsi3minus,
            state.cpsix,
            state.cpsi4,
            state.sig_psi,
            state.cm0,
            state.kappa,
            state.cde,
            state.galp,
            int(state.length_lim),
            state.eps_min,
            state.psi_ubc,
            state.psi_lbc,
            state.ubc_type,
            state.lbc_type,
            state.sig_k,
            state.cmsf,
            state.cw,
            state.gen_m,
            state.gen_n,
            state.gen_p,
            state.gen_alpha,
            state.gen_l,
            workspace.tke,
            workspace.tkeo,
            workspace.eps,
            workspace.L,
            workspace.h,
            workspace.NN,
            workspace.SS,
            workspace.P,
            workspace.B,
            workspace.Px,
            workspace.PSTK,
            workspace.num,
            workspace.u_taus,
            workspace.u_taub,
            workspace.z0s,
            workspace.z0b,
            workspace.psi,
            workspace.avh,
            workspace.l_sour,
            workspace.q_sour,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("eps", "L"))
        return

    if state.len_scale_method == length_eq:
        from pygotm.turbulence.lengthscaleeq import (
            LengthScaleEquationWorkspace,
            step_lengthscaleeq,
        )

        workspace = _ensure_workspace(
            state,
            nlev,
            "lengthscaleeq",
            lambda n: LengthScaleEquationWorkspace(n, n_cols=1),
        )
        _load_state_fields(
            state,
            workspace,
            ("tke", "tkeo", "eps", "L", "P", "B", "Px", "PSTK", "sl_var"),
        )
        _load_input_fields(workspace, {"h": h, "NN": NN, "SS": SS})
        _write_scalar_fields(
            workspace,
            {
                "depth": depth,
                "u_taus": u_taus,
                "u_taub": u_taub,
                "z0s": z0s,
                "z0b": z0b,
            },
        )
        _zero_workspace_fields(
            workspace,
            ("q2l", "avh", "l_sour", "q_sour", "au", "bu", "cu", "du", "ru", "qu"),
        )
        step_lengthscaleeq(
            1,
            nlev,
            dt,
            state.k_min,
            state.eps_min,
            state.kappa,
            state.e1,
            state.e2,
            state.e3,
            state.ex,
            state.e6,
            state.b1,
            state.cde,
            state.my_length,
            state.galp,
            int(state.length_lim),
            state.psi_ubc,
            state.psi_lbc,
            state.ubc_type,
            state.lbc_type,
            state.sl,
            state.sq,
            state.cw,
            state.gen_alpha,
            state.gen_l,
            workspace.tke,
            workspace.tkeo,
            workspace.eps,
            workspace.L,
            workspace.h,
            workspace.NN,
            workspace.SS,
            workspace.P,
            workspace.B,
            workspace.Px,
            workspace.PSTK,
            workspace.sl_var,
            workspace.depth,
            workspace.u_taus,
            workspace.u_taub,
            workspace.z0s,
            workspace.z0b,
            workspace.q2l,
            workspace.avh,
            workspace.l_sour,
            workspace.q_sour,
            workspace.au,
            workspace.bu,
            workspace.cu,
            workspace.du,
            workspace.ru,
            workspace.qu,
        )
        _sync_state_fields(state, workspace, ("eps", "L"))
        return

    if state.len_scale_method == Bougeault_Andre:
        from pygotm.turbulence.potentialml import PotentialMLWorkspace, step_potentialml

        workspace = _ensure_workspace(
            state,
            nlev,
            "potentialml",
            lambda n: PotentialMLWorkspace(n, n_cols=1),
        )
        _load_state_fields(state, workspace, ("tke", "eps", "L"))
        _load_input_fields(workspace, {"h": h, "NN": NN})
        _write_scalar_fields(
            workspace,
            {"depth": depth, "z0b": z0b, "z0s": z0s},
        )
        step_potentialml(
            1,
            nlev,
            state.kappa,
            state.cde,
            state.galp,
            int(state.length_lim),
            state.eps_min,
            workspace.tke,
            workspace.eps,
            workspace.L,
            workspace.h,
            workspace.NN,
            workspace.depth,
            workspace.z0b,
            workspace.z0s,
        )
        _sync_state_fields(state, workspace, ("eps", "L"))
        return

    from pygotm.turbulence.algebraiclength import (
        AlgebraicLengthWorkspace,
        step_algebraiclength,
    )

    workspace = _ensure_workspace(
        state,
        nlev,
        "algebraiclength",
        lambda n: AlgebraicLengthWorkspace(n, n_cols=1),
    )
    _load_state_fields(state, workspace, ("tke", "eps", "L"))
    _load_input_fields(workspace, {"h": h, "NN": NN})
    _write_scalar_fields(
        workspace,
        {"depth": depth, "z0b": z0b, "z0s": z0s},
    )
    step_algebraiclength(
        1,
        state.len_scale_method,
        nlev,
        state.kappa,
        state.cde,
        state.galp,
        int(state.length_lim),
        state.eps_min,
        workspace.tke,
        workspace.eps,
        workspace.L,
        workspace.h,
        workspace.NN,
        workspace.depth,
        workspace.z0b,
        workspace.z0s,
    )
    _sync_state_fields(state, workspace, ("eps", "L"))


def _run_epsb(state: TurbulenceState, nlev: int) -> None:
    if state.epsb_method == epsb_algebraic:
        from pygotm.turbulence.epsbalgebraic import (
            EpsBAlgebraicWorkspace,
            step_epsbalgebraic,
        )

        workspace = _ensure_workspace(
            state,
            nlev,
            "epsbalgebraic",
            lambda n: EpsBAlgebraicWorkspace(n, n_cols=1),
        )
        _load_state_fields(state, workspace, ("tke", "eps", "kb", "epsb"))
        step_epsbalgebraic(
            1,
            nlev,
            state.ctt,
            state.epsb_min,
            workspace.tke,
            workspace.eps,
            workspace.kb,
            workspace.epsb,
        )
        _sync_state_fields(state, workspace, ("epsb",))
        return

    if state.epsb_method == epsb_dynamic:
        msg = "epsb_method=2 is not implemented in GOTM or pyGOTM"
        raise NotImplementedError(msg)

    msg = f"invalid epsb_method={state.epsb_method}"
    raise ValueError(msg)


def _run_internal_wave(
    state: TurbulenceState, nlev: int, NN: np.ndarray, SS: np.ndarray
) -> None:
    from pygotm.turbulence.internal_wave import (
        InternalWaveWorkspace,
        step_internal_wave,
    )

    workspace = _ensure_workspace(
        state,
        nlev,
        "internal_wave",
        lambda n: InternalWaveWorkspace(n, n_cols=1),
    )
    _load_state_fields(state, workspace, ("tke", "num", "nuh"))
    _load_input_fields(workspace, {"NN": NN, "SS": SS})
    step_internal_wave(
        1,
        nlev,
        state.iw_model,
        state.klimiw,
        state.rich_cr,
        state.numiw,
        state.nuhiw,
        state.numshear,
        workspace.tke,
        workspace.num,
        workspace.nuh,
        workspace.NN,
        workspace.SS,
    )
    _sync_state_fields(state, workspace, ("num", "nuh"))


def run_variances(
    state: TurbulenceState,
    nlev: int,
    SSU: np.ndarray,
    SSV: np.ndarray,
) -> None:
    r"""Update turbulent velocity variances from the current closure state."""

    _validate_profile("SSU", SSU, nlev)
    _validate_profile("SSV", SSV, nlev)

    from pygotm.turbulence.variances import VariancesWorkspace, step_variances

    workspace = _ensure_workspace(
        state,
        nlev,
        "variances",
        lambda n: VariancesWorkspace(n, n_cols=1),
    )
    _load_state_fields(
        state,
        workspace,
        ("tke", "eps", "P", "B", "Px", "num", "uu", "vv", "ww"),
    )
    _load_input_fields(workspace, {"SSU": SSU, "SSV": SSV})
    step_variances(
        1,
        nlev,
        state.cc1,
        state.ct1,
        state.a2,
        state.a3,
        state.a5,
        workspace.tke,
        workspace.eps,
        workspace.P,
        workspace.B,
        workspace.Px,
        workspace.num,
        workspace.SSU,
        workspace.SSV,
        workspace.uu,
        workspace.vv,
        workspace.ww,
    )
    _sync_state_fields(state, workspace, ("uu", "vv", "ww"))


def k_bc(
    state: TurbulenceState,
    bc: int,
    type_: int,
    zi: float,
    z0: float,
    u_tau: float,
) -> float:
    r"""Compute TKE boundary conditions for the k-equation.

    ! !DESCRIPTION:
    ! Computes prescribed and flux boundary conditions for  the transport
    ! equation \eq{tkeA}. The formal parameter {\tt bc} determines
    ! whether {\tt Dirichlet} or {\tt Neumann}-type boundary conditions
    ! are computed. Depending on the physical properties of the
    ! boundary-layer, the parameter {\tt type} relates either to a {\tt visous},
    ! a {\tt logarithmic}, or an {\tt injection}-type boundary-layer.
    ! In the latter case, the flux of TKE caused by breaking surface waves
    ! has to be specified. Presently, there is only one possibility
    ! to do so implemented in GOTM. It is described in \sect{sec:fkCraig}.
    ! All parameters that determine the boundary layer have to be
    ! set in {\tt gotmturb.nml}.
    !
    ! Note that  in this section, for brevity, $z$ denotes the distance
    ! from the wall (or the surface), and \emph{not} the standard
    ! coordinate of the same name used in GOTM.
    """
    if bc not in (Dirichlet, Neumann):
        msg = f"invalid k-equation boundary condition {bc}"
        raise ValueError(msg)

    if type_ == viscous:
        msg = "viscous boundary layers are not implemented for k_bc"
        raise NotImplementedError(msg)

    if type_ == logarithmic:
        if state.cm0 <= 0.0:
            msg = "state.cm0 must be initialised before using logarithmic k_bc"
            raise ValueError(msg)
        if bc == Dirichlet:
            return u_tau**2 / state.cm0**2
        return 0.0

    if type_ == injection:
        if state.cmsf <= 0.0:
            msg = "state.cmsf must be initialised before using injection k_bc"
            raise ValueError(msg)
        f_k = _fk_craig(u_tau, state.cw)
        capital_k = (
            -state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)
        ) ** (2.0 / 3.0) / z0**state.gen_alpha
        if bc == Dirichlet:
            return float(capital_k * (zi + z0) ** state.gen_alpha)
        return float(
            -state.cmsf
            / state.sig_k
            * capital_k**1.5
            * state.gen_alpha
            * state.gen_l
            * (zi + z0) ** (1.5 * state.gen_alpha)
        )

    msg = f"invalid k-equation boundary-layer type {type_}"
    raise ValueError(msg)


def q2over2_bc(
    state: TurbulenceState,
    bc: int,
    type_: int,
    zi: float,
    z0: float,
    u_tau: float,
) -> float:
    r"""Compute q2/2-equation boundary conditions.

    ! !DESCRIPTION:
    ! Computes prescribed and flux boundary conditions for  the transport
    ! equation \eq{tkeB}. The formal parameter {\tt bc} determines
    ! whether {\tt Dirichlet} or {\tt Neumann}-type boundary conditions
    ! are computed. Depending on the physical properties of the
    ! boundary-layer, the parameter {\tt type} relates either to a {\tt visous},
    ! a {\tt logarithmic}, or an {\tt injection}-type boundary-layer.
    ! In the latter case, the flux of TKE caused by breaking surface waves
    ! has to be specified. Presently, there is only one possibility
    ! to do so implemented in GOTM. It is described in \sect{sec:fkCraig}.
    ! All parameters that determine the boundary layer have to be
    ! set in {\tt gotmturb.nml}.
    !
    ! Note that  in this section, for brevity, $z$ denotes the distance
    ! from the wall (or the surface), and \emph{not} the standard
    ! coordinate of the same name used in GOTM.
    """
    if bc not in (Dirichlet, Neumann):
        msg = f"invalid q2/2-equation boundary condition {bc}"
        raise ValueError(msg)

    if type_ == viscous:
        msg = "viscous boundary layers are not implemented for q2over2_bc"
        raise NotImplementedError(msg)

    if type_ == logarithmic:
        if state.b1 <= 0.0:
            msg = "state.b1 must be initialised before using logarithmic q2over2_bc"
            raise ValueError(msg)
        if bc == Dirichlet:
            return float(u_tau**2 * state.b1 ** (2.0 / 3.0) / 2.0)
        return 0.0

    if type_ == injection:
        if state.sq <= 0.0:
            msg = "state.sq must be initialised before using injection q2over2_bc"
            raise ValueError(msg)
        if state.gen_alpha == 0.0:
            msg = "state.gen_alpha must be non-zero before using injection q2over2_bc"
            raise ValueError(msg)
        if state.gen_l <= 0.0:
            msg = "state.gen_l must be initialised before using injection q2over2_bc"
            raise ValueError(msg)

        f_k = _fk_craig(u_tau, state.cw)
        capital_k = (
            -f_k / (np.sqrt(2.0) * state.sq * state.gen_alpha * state.gen_l)
        ) ** (2.0 / 3.0) / z0**state.gen_alpha
        if bc == Dirichlet:
            return float(capital_k * (zi + z0) ** state.gen_alpha)
        return float(
            -np.sqrt(2.0)
            * state.sq
            * capital_k**1.5
            * state.gen_alpha
            * state.gen_l
            * (zi + z0) ** (1.5 * state.gen_alpha)
        )

    msg = f"invalid q2/2-equation boundary-layer type {type_}"
    raise ValueError(msg)


def epsilon_bc(
    state: TurbulenceState,
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
) -> float:
    r"""Compute epsilon-equation boundary conditions.

    ! !DESCRIPTION:
    ! Computes prescribed and flux boundary conditions for  the transport
    ! equation \eq{dissipation}. The formal parameter {\tt bc} determines
    ! whether {\tt Dirichlet} or {\tt Neumann}-type boundary conditions
    ! are computed. Depending on the physical properties of the
    ! boundary-layer, the parameter {\tt type} relates either to a {\tt visous},
    ! a {\tt logarithmic}, or an {\tt injection}-type boundary-layer.
    ! In the latter case, the flux of TKE caused by breaking surface waves
    ! has to be specified. Presently, there is only one possibility
    ! to do so implemented in GOTM. It is described in \sect{sec:fkCraig}.
    !
    ! Note that  in this section, for brevity, $z$ denotes the distance
    ! from the wall (or the surface), and \emph{not} the standard
    ! coordinate of the same name used in GOTM.
    """
    if bc not in (Dirichlet, Neumann):
        msg = f"invalid epsilon-equation boundary condition {bc}"
        raise ValueError(msg)

    if type_ == viscous:
        msg = "viscous boundary layers are not implemented for epsilon_bc"
        raise NotImplementedError(msg)

    if type_ == logarithmic:
        if state.cde <= 0.0:
            msg = "state.cde must be initialised before using logarithmic epsilon_bc"
            raise ValueError(msg)
        if state.kappa <= 0.0:
            msg = "state.kappa must be initialised before using logarithmic epsilon_bc"
            raise ValueError(msg)
        if bc == Dirichlet:
            return float(state.cde * ki**1.5 / (state.kappa * (zi + z0)))
        if state.cm0 <= 0.0:
            msg = "state.cm0 must be initialised before using logarithmic epsilon_bc"
            raise ValueError(msg)
        if state.sig_e <= 0.0:
            msg = "state.sig_e must be initialised before using logarithmic epsilon_bc"
            raise ValueError(msg)
        return float(state.cm0**4 * ki**2 / (state.sig_e * (zi + z0)))

    if type_ == injection:
        if state.cmsf <= 0.0:
            msg = "state.cmsf must be initialised before using injection epsilon_bc"
            raise ValueError(msg)
        if state.sig_e0 <= 0.0:
            msg = "state.sig_e0 must be initialised before using injection epsilon_bc"
            raise ValueError(msg)
        if state.cde <= 0.0:
            msg = "state.cde must be initialised before using injection epsilon_bc"
            raise ValueError(msg)
        if state.gen_l <= 0.0:
            msg = "state.gen_l must be initialised before using injection epsilon_bc"
            raise ValueError(msg)

        f_k = _fk_craig(u_tau, state.cw)
        capital_k = (
            -state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)
        ) ** (2.0 / 3.0) / z0**state.gen_alpha
        if bc == Dirichlet:
            return float(
                state.cde
                * capital_k**1.5
                / state.gen_l
                * (zi + z0) ** (1.5 * state.gen_alpha - 1.0)
            )
        return float(
            -state.cmsf
            * state.cde
            / state.sig_e0
            * capital_k**2
            * (1.5 * state.gen_alpha - 1.0)
            * (zi + z0) ** (2.0 * state.gen_alpha - 1.0)
        )

    msg = f"invalid epsilon-equation boundary-layer type {type_}"
    raise ValueError(msg)


def omega_bc(
    state: TurbulenceState,
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
) -> float:
    r"""Compute omega-equation boundary conditions.

    ! !DESCRIPTION:
    ! Under construction. Please refer to Umlauf et al. (2003) and Umlauf and
    ! Burchard (2003) for the basic documentation of the $k$-$\omega$ model and
    ! its boundary conditions.
    """
    if bc not in (Dirichlet, Neumann):
        msg = f"invalid omega-equation boundary condition {bc}"
        raise ValueError(msg)

    if type_ == viscous:
        msg = "viscous boundary layers are not implemented for omega_bc"
        raise NotImplementedError(msg)

    if type_ == logarithmic:
        if state.cm0 <= 0.0:
            msg = "state.cm0 must be initialised before using logarithmic omega_bc"
            raise ValueError(msg)
        if state.kappa <= 0.0:
            msg = "state.kappa must be initialised before using logarithmic omega_bc"
            raise ValueError(msg)
        if bc == Dirichlet:
            return float(ki**0.5 / (state.cm0 * state.kappa * (zi + z0)))
        if state.sig_w <= 0.0:
            msg = "state.sig_w must be initialised before using logarithmic omega_bc"
            raise ValueError(msg)
        return float(ki / (state.sig_w * (zi + z0)))

    if type_ == injection:
        if state.cmsf <= 0.0:
            msg = "state.cmsf must be initialised before using injection omega_bc"
            raise ValueError(msg)
        if state.cm0 <= 0.0:
            msg = "state.cm0 must be initialised before using injection omega_bc"
            raise ValueError(msg)
        if state.sig_w <= 0.0:
            msg = "state.sig_w must be initialised before using injection omega_bc"
            raise ValueError(msg)
        if state.gen_alpha == 0.0:
            msg = "state.gen_alpha must be non-zero before using injection omega_bc"
            raise ValueError(msg)
        if state.gen_l <= 0.0:
            msg = "state.gen_l must be initialised before using injection omega_bc"
            raise ValueError(msg)

        f_k = _fk_craig(u_tau, state.cw)
        capital_k = (
            -state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)
        ) ** (2.0 / 3.0) / z0**state.gen_alpha
        if bc == Dirichlet:
            return float(
                capital_k**0.5
                / (state.cm0 * state.gen_l)
                * (zi + z0) ** (0.5 * state.gen_alpha - 1.0)
            )
        return float(
            -state.cmsf
            * capital_k
            * (0.5 * state.gen_alpha - 1.0)
            / (state.sig_w * state.cm0)
            * (zi + z0) ** (state.gen_alpha - 1.0)
        )

    msg = f"invalid omega-equation boundary-layer type {type_}"
    raise ValueError(msg)


def psi_bc(
    state: TurbulenceState,
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
) -> float:
    r"""Compute generic-psi equation boundary conditions.

    ! !DESCRIPTION:
    ! Computes prescribed and flux boundary conditions for  the transport
    ! equation \eq{generic}. The formal parameter {\tt bc} determines
    ! whether {\tt Dirichlet} or {\tt Neumann}-type boundary conditions
    ! are computed. Depending on the physical properties of the
    ! boundary-layer, the parameter {\tt type} relates either to a {\tt visous},
    ! a {\tt logarithmic}, or an {\tt injection}-type boundary-layer.
    ! In the latter case, the flux of TKE caused by breaking surface waves
    ! has to be specified. Presently, there is only one possibility
    ! to do so implemented in GOTM. It is described in \sect{sec:fkCraig}.
    !
    ! Note that  in this section, for brevity, $z$ denotes the distance
    ! from the wall (or the surface), and \emph{not} the standard
    ! coordinate of the same name used in GOTM.
    """
    if bc not in (Dirichlet, Neumann):
        msg = f"invalid psi-equation boundary condition {bc}"
        raise ValueError(msg)

    if type_ == viscous:
        msg = "viscous boundary layers are not implemented for psi_bc"
        raise NotImplementedError(msg)

    if type_ == logarithmic:
        if state.cm0 <= 0.0:
            msg = "state.cm0 must be initialised before using logarithmic psi_bc"
            raise ValueError(msg)
        if state.kappa <= 0.0:
            msg = "state.kappa must be initialised before using logarithmic psi_bc"
            raise ValueError(msg)
        if bc == Dirichlet:
            return float(
                state.cm0**state.gen_p
                * state.kappa**state.gen_n
                * ki**state.gen_m
                * (zi + z0) ** state.gen_n
            )
        if state.sig_psi <= 0.0:
            msg = "state.sig_psi must be initialised before using logarithmic psi_bc"
            raise ValueError(msg)
        return float(
            -state.gen_n
            * state.cm0 ** (state.gen_p + 1.0)
            * state.kappa ** (state.gen_n + 1.0)
            / state.sig_psi
            * ki ** (state.gen_m + 0.5)
            * (zi + z0) ** state.gen_n
        )

    if type_ == injection:
        if state.cmsf <= 0.0:
            msg = "state.cmsf must be initialised before using injection psi_bc"
            raise ValueError(msg)
        if state.cm0 <= 0.0:
            msg = "state.cm0 must be initialised before using injection psi_bc"
            raise ValueError(msg)
        if state.sig_psi <= 0.0:
            msg = "state.sig_psi must be initialised before using injection psi_bc"
            raise ValueError(msg)
        if state.gen_alpha == 0.0:
            msg = "state.gen_alpha must be non-zero before using injection psi_bc"
            raise ValueError(msg)
        if state.gen_l <= 0.0:
            msg = "state.gen_l must be initialised before using injection psi_bc"
            raise ValueError(msg)

        f_k = _fk_craig(u_tau, state.cw)
        capital_k = (
            -state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)
        ) ** (2.0 / 3.0) / z0**state.gen_alpha
        if bc == Dirichlet:
            return float(
                state.cm0**state.gen_p
                * capital_k**state.gen_m
                * state.gen_l**state.gen_n
                * (zi + z0) ** (state.gen_m * state.gen_alpha + state.gen_n)
            )
        return float(
            -(state.gen_m * state.gen_alpha + state.gen_n)
            * state.cmsf
            * state.cm0**state.gen_p
            / state.sig_psi
            * capital_k ** (state.gen_m + 0.5)
            * state.gen_l ** (state.gen_n + 1.0)
            * (zi + z0) ** ((state.gen_m + 0.5) * state.gen_alpha + state.gen_n)
        )

    msg = f"invalid psi-equation boundary-layer type {type_}"
    raise ValueError(msg)


def q2l_bc(
    state: TurbulenceState,
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
) -> float:
    r"""Compute q2l-equation boundary conditions.

    ! !DESCRIPTION:
    ! Computes prescribed and flux boundary conditions for the transport
    ! equation \eq{MY}. The formal parameter {\tt bc} determines
    ! whether {\tt Dirichlet} or {\tt Neumann}-type boundary conditions
    ! are computed. Depending on the physical properties of the
    ! boundary-layer, the parameter {\tt type} relates either to a {\tt visous},
    ! a {\tt logarithmic}, or an {\tt injection}-type boundary-layer.
    ! In the latter case, the flux of TKE caused by breaking surface waves
    ! has to be specified. Presently, there is only one possibility
    ! to do so implemented in GOTM. It is described in \sect{sec:fkCraig}.
    !
    ! Note that in this section, for brevity, $z$ denotes the distance
    ! from the wall (or the surface), and \emph{not} the standard
    ! coordinate of the same name used in GOTM.
    """
    if bc not in (Dirichlet, Neumann):
        msg = f"invalid q2l-equation boundary condition {bc}"
        raise ValueError(msg)

    if type_ == viscous:
        msg = "viscous boundary layers are not implemented for q2l_bc"
        raise NotImplementedError(msg)

    if type_ == logarithmic:
        if state.kappa <= 0.0:
            msg = "state.kappa must be initialised before using logarithmic q2l_bc"
            raise ValueError(msg)
        if bc == Dirichlet:
            return float(2.0 * state.kappa * ki * (zi + z0))
        if state.sl <= 0.0:
            msg = "state.sl must be initialised before using logarithmic q2l_bc"
            raise ValueError(msg)
        return float(
            -2.0 * np.sqrt(2.0) * state.sl * state.kappa**2 * ki**1.5 * (zi + z0)
        )

    if type_ == injection:
        if state.sq <= 0.0:
            msg = "state.sq must be initialised before using injection q2l_bc"
            raise ValueError(msg)
        if state.sl <= 0.0:
            msg = "state.sl must be initialised before using injection q2l_bc"
            raise ValueError(msg)
        if state.gen_alpha == 0.0:
            msg = "state.gen_alpha must be non-zero before using injection q2l_bc"
            raise ValueError(msg)
        if state.gen_l <= 0.0:
            msg = "state.gen_l must be initialised before using injection q2l_bc"
            raise ValueError(msg)

        f_k = _fk_craig(u_tau, state.cw)
        capital_k = (
            -f_k / (np.sqrt(2.0) * state.sq * state.gen_alpha * state.gen_l)
        ) ** (2.0 / 3.0) / z0**state.gen_alpha
        if bc == Dirichlet:
            return float(
                2.0 * capital_k * state.gen_l * (zi + z0) ** (state.gen_alpha + 1.0)
            )
        return float(
            -2.0
            * np.sqrt(2.0)
            * state.sl
            * (state.gen_alpha + 1.0)
            * capital_k**1.5
            * state.gen_l**2
            * (zi + z0) ** (1.5 * state.gen_alpha + 1.0)
        )

    msg = f"invalid q2l-equation boundary-layer type {type_}"
    raise ValueError(msg)


def do_turbulence(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    depth: float,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    *,
    xP: np.ndarray | None = None,
    SSCSTK: np.ndarray | None = None,
    SSSTK: np.ndarray | None = None,
) -> None:
    r"""Dispatch one turbulence time step.

    ! !DESCRIPTION: This routine is the central point of the
    ! turbulence scheme. It determines the order, in which
    ! turbulence variables are updated, and calls
    ! other member functions updating
    ! the TKE, the length-scale, the dissipation rate, the ASM etc.
    !
    ! !INPUT PARAMETERS:
    !  number of vertical layers
    !   integer,  intent(in)                :: nlev
    !  time step (s)
    !   REALTYPE, intent(in)                :: dt
    !  distance between surface and bottom (m)
    !   REALTYPE, intent(in)                :: depth
    !  surface and bottom friction velocity (m/s)
    !   REALTYPE, intent(in)                :: u_taus,u_taub
    !  surface and bottom roughness length (m)
    !   REALTYPE, intent(in)                :: z0s,z0b
    !  layer thickness (m)
    !   REALTYPE, intent(in)                :: h(0:nlev)
    !  boyancy frequency squared (1/s^2)
    !   REALTYPE, intent(in)                :: NN(0:nlev)
    !  shear-frequency squared (1/s^2)
    !   REALTYPE, intent(in)                :: SS(0:nlev)

    """
    for name, values in (("h", h), ("NN", NN), ("SS", SS)):
        _validate_profile(name, values, nlev)
    if xP is not None:
        _validate_profile("xP", xP, nlev)
    if SSCSTK is not None:
        _validate_profile("SSCSTK", SSCSTK, nlev)
    if SSSTK is not None:
        _validate_profile("SSSTK", SSSTK, nlev)

    if state.turb_method == no_model:
        return

    if state.turb_method == _CVMIX_TURB_METHOD:
        return

    if state.turb_method == algebraic:
        msg = "turb_method=1 algebraic diffusivity is not translated yet"
        raise NotImplementedError(msg)

    if state.turb_method == first_order:
        _run_production(state, nlev, NN, SS, xP, SSCSTK, SSSTK)
        _run_alpha_mnb(state, nlev, NN, SS, None, None)
        _run_first_order_stability(state, nlev)
        _run_tke(state, nlev, dt, u_taus, u_taub, z0s, z0b, h, NN, SS)
        _run_lengthscale(state, nlev, dt, depth, u_taus, u_taub, z0s, z0b, h, NN, SS)
        _kolpran(state, nlev)
        _run_internal_wave(state, nlev, NN, SS)
        return

    if state.turb_method == second_order:
        _run_production(state, nlev, NN, SS, xP, SSCSTK, SSSTK)

        if state.scnd_method == quasi_Eq:
            _run_alpha_mnb(state, nlev, NN, SS, None, None)
            _run_cmue_d(state, nlev)
        elif state.scnd_method == weak_Eq_Kb_Eq:
            _run_alpha_mnb(state, nlev, NN, SS, None, None)
            _run_cmue_c(state, nlev)
        elif state.scnd_method == weak_Eq_Kb:
            msg = "scnd_method=3 is not tested in GOTM or pyGOTM"
            raise NotImplementedError(msg)
        elif state.scnd_method == quasi_Eq_H15:
            if SSCSTK is None or SSSTK is None:
                msg = "quasi_Eq_H15 requires SSCSTK and SSSTK inputs"
                raise ValueError(msg)
            _run_alpha_mnb(state, nlev, NN, SS, SSCSTK, SSSTK)
            _surface_proximity_function(state, nlev, depth, h)
            _run_cmue_d_h15(state, nlev)
        else:
            msg = f"invalid scnd_method={state.scnd_method}"
            raise ValueError(msg)

        _run_tke(state, nlev, dt, u_taus, u_taub, z0s, z0b, h, NN, SS)
        _run_kb(state, nlev, dt, h)
        _run_lengthscale(state, nlev, dt, depth, u_taus, u_taub, z0s, z0b, h, NN, SS)
        _run_epsb(state, nlev)
        if state.scnd_method == quasi_Eq_H15:
            _run_alpha_mnb(state, nlev, NN, SS, SSCSTK, SSSTK)
        else:
            _run_alpha_mnb(state, nlev, NN, SS, None, None)
        _kolpran(state, nlev)
        return

    msg = f"invalid turb_method={state.turb_method}; expected 0, 1, 2, 3, or 100"
    raise ValueError(msg)


def clean_turbulence(state: TurbulenceState) -> None:
    r"""Release all allocated turbulence profile arrays.

    ! !DESCRIPTION:
    !  De-allocate all memory allocated in init\_turbulence().
    !
    ! !REVISION HISTORY:
    !  Original FORTRAN author(s): Karsten Bolding
    """
    for name in _ARRAY_FIELD_NAMES:
        setattr(state, name, None)
    state._kernel_workspaces.clear()
    state._kernel_nlev = None
