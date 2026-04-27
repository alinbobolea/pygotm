# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: observations --- the 'real' world \label{sec:observations}
!
! !INTERFACE:
!   module observations
!
! !DESCRIPTION:
!  This module provides the necessary subroutines for communicating
!  `observations' to GOTM.
!  The module operates according to the general philosophy used in GOTM,
!  i.e.\ it provides {\tt init\_observ\-ations()} to be called in the overall
!  initialisation routine and {\tt get\_all\_obs()} to be called in the time
!  loop to actually obtain the `observations'.
!  In addition to these subroutines the module also provides two routines
!  for reading scalar-type observations and profile-type observations.
!  Each observation has a date stamp with the format {\tt yyyy-mm-dd hh:dd:mm}.
!  The module uses the {\tt time} module (see \sect{sec:time})
!  to convert the time string to the
!  internal time representation of GOTM.
!  Profiles are interpolated to the actual GOTM model grid.
!  Free format is used for reading-in the actual data.
!
! !USES:
!   use input
!   use settings
!   IMPLICIT NONE
!  default: all is private.
!   private
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_observations, post_init_observations, get_all_obs, clean_observations
!
! !PUBLIC DATA MEMBERS:
!
!  'observed' salinity profile
!   integer, public :: initial_salinity_type
!   type (type_profile_input), public, target :: sprof_input
!
!  'observed' temperature profile
!   integer, public :: initial_temperature_type
!   type (type_profile_input), public, target :: tprof_input
!
!  'observed' oxygen profile
!   type (type_profile_input), public, target :: o2_prof_input
!
!  'observed' horizontal salinity gradients
!   type (type_profile_input), public, target :: dsdx_input,dsdy_input
!
!  'observed' horizontal temperature gradients
!   type (type_profile_input), public, target :: dtdx_input,dtdy_input
!
!  internal horizontal pressure gradients
!   REALTYPE, public, dimension(:), allocatable :: idpdx,idpdy
!
!  horizontal velocity profiles
!   type (type_profile_input), public, target :: uprof_input,vprof_input
!
!  observed profile of turbulent dissipation rates
!   type (type_profile_input), public, target :: epsprof_input
!
!  relaxation times for salinity and temperature
!   REALTYPE, public, dimension(:), allocatable, target :: SRelaxTau
!   REALTYPE, public, dimension(:), allocatable         :: TRelaxTau
!
!  sea surface elevation, sea surface gradients and height of velocity obs.
!   type (type_scalar_input), public, target :: zeta_input,dpdx_input,dpdy_input,h_press_input
!
!  vertical advection velocity
!   type (type_scalar_input), public, target :: w_adv_input,w_height_input
!
!  Parameters for water classification - default Jerlov type I
!   type (type_scalar_input), public, target :: A_input, g1_input, g2_input
!
! !DEFINED PARAMETERS:
!
!  pre-defined parameters
!   integer, parameter        :: NOTHING=0
!   integer, parameter        :: ANALYTICAL=1
!   integer, parameter        :: CONSTANT=1
!   integer, parameter        :: FROMFILE=2
!   integer, parameter        :: CONST_PROF=1
!   integer, parameter        :: TWO_LAYERS=2
!   integer, parameter        :: CONST_NN=3
!   integer, parameter        :: ANALYTICAL_OFFSET=10
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from pygotm.config.settings import GotmSettings
from pygotm.input.input import (
    ProfileInput,
    ScalarInput,
    register_input,
)
from pygotm.observations.analytical_profile import analytical_profile
from pygotm.observations.const_nns import const_NNS
from pygotm.observations.const_nnt import const_NNT
from pygotm.util.density import DensityState
from pygotm.util.util import MUSCL, P2, P2_PDM, UPSTREAM, Superbee

__all__ = [
    "ANALYTICAL",
    "ANALYTICAL_OFFSET",
    "CONST_NN",
    "CONST_PROF",
    "CONSTANT",
    "FROMFILE",
    "NOTHING",
    "TWO_LAYERS",
    "ObservationsState",
    "clean_observations",
    "get_all_obs",
    "init_observations",
    "post_init_observations",
]

NOTHING = 0
ANALYTICAL = 1
CONSTANT = 1
FROMFILE = 2
CONST_PROF = 1
TWO_LAYERS = 2
CONST_NN = 3
ANALYTICAL_OFFSET = 10
PI = math.pi

_TEMPERATURE_TYPE = {"in_situ": 1, "potential": 2, "conservative": 3}
_SALINITY_TYPE = {"practical": 1, "absolute": 2}
_EXT_PRESS_MODE = {"elevation": 0, "velocity": 1, "average_velocity": 2}
_INT_PRESS_TYPE = {"none": 0, "gradients": 1, "plume": 2}
_PLUME_TYPE = {"surface": 1, "bottom": 2}
_W_ADV_DISCR = {
    "upstream": UPSTREAM,
    "p2": P2,
    "superbee": Superbee,
    "muscl": MUSCL,
    "p2_pdm": P2_PDM,
}
_EXTINCTION_METHOD = {
    "jerlov_i": 1,
    "jerlov_1_50m": 2,
    "jerlov_ia": 3,
    "jerlov_ib": 4,
    "jerlov_ii": 5,
    "jerlov_iii": 6,
    "custom": 7,
}


def _profile_method(name: str, *, analytical_constant: bool) -> int:
    if name == "off":
        return NOTHING
    if name == "constant":
        return (
            ANALYTICAL_OFFSET + CONST_PROF if analytical_constant else CONSTANT
        )
    if name == "file":
        return FROMFILE
    if analytical_constant and name == "two_layer":
        return ANALYTICAL_OFFSET + TWO_LAYERS
    if analytical_constant and name == "buoyancy":
        return ANALYTICAL_OFFSET + CONST_NN
    raise ValueError(f"unsupported profile method {name!r}")


def _scalar_method(name: str) -> int:
    if name == "constant":
        return 0
    if name == "file":
        return FROMFILE
    if name == "off":
        return NOTHING
    if name == "tidal":
        return ANALYTICAL
    raise ValueError(f"unsupported scalar method {name!r}")


def _profile_input(
    *,
    name: str,
    method: int,
    path: str = "",
    index: int = 1,
    constant_value: float = 0.0,
    scale_factor: float = 1.0,
    add_offset: float = 0.0,
    minimum: float = -math.inf,
    maximum: float = math.inf,
    method_off: int = NOTHING,
    method_constant: int = CONSTANT,
    method_file: int = FROMFILE,
) -> ProfileInput:
    return ProfileInput(
        name=name,
        method=method,
        path=path,
        index=index,
        constant_value=constant_value,
        scale_factor=scale_factor,
        add_offset=add_offset,
        minimum=minimum,
        maximum=maximum,
        method_off=method_off,
        method_constant=method_constant,
        method_file=method_file,
    )


def _scalar_input(
    *,
    name: str,
    method: int,
    path: str = "",
    index: int = 1,
    constant_value: float = 0.0,
    scale_factor: float = 1.0,
    add_offset: float = 0.0,
    minimum: float = -math.inf,
    maximum: float = math.inf,
    method_off: int = NOTHING,
    method_constant: int = 0,
    method_file: int = FROMFILE,
) -> ScalarInput:
    return ScalarInput(
        name=name,
        method=method,
        path=path,
        index=index,
        constant_value=constant_value,
        scale_factor=scale_factor,
        add_offset=add_offset,
        minimum=minimum,
        maximum=maximum,
        method_off=method_off,
        method_constant=method_constant,
        method_file=method_file,
    )


@dataclass
class ObservationsState:
    """Mutable state for the translated GOTM observations module."""

    initial_salinity_type: int = 1
    sprof_input: ProfileInput = field(
        default_factory=lambda: _profile_input(
            name="salinity",
            method=NOTHING,
            method_constant=ANALYTICAL_OFFSET + CONST_PROF,
        )
    )
    initial_temperature_type: int = 1
    tprof_input: ProfileInput = field(
        default_factory=lambda: _profile_input(
            name="temperature",
            method=NOTHING,
            method_constant=ANALYTICAL_OFFSET + CONST_PROF,
        )
    )
    o2_prof_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="o2", method=NOTHING)
    )
    dsdx_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="dsdx", method=NOTHING)
    )
    dsdy_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="dsdy", method=NOTHING)
    )
    dtdx_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="dtdx", method=NOTHING)
    )
    dtdy_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="dtdy", method=NOTHING)
    )
    idpdx: np.ndarray | None = None
    idpdy: np.ndarray | None = None
    uprof_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="u_obs", method=NOTHING)
    )
    vprof_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="v_obs", method=NOTHING)
    )
    epsprof_input: ProfileInput = field(
        default_factory=lambda: _profile_input(name="eps_obs", method=NOTHING)
    )
    SRelaxTau: np.ndarray | None = None
    TRelaxTau: np.ndarray | None = None
    zeta_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="zeta", method=0)
    )
    dpdx_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="dpdx", method=0)
    )
    dpdy_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="dpdy", method=0)
    )
    h_press_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="h_press", method=0)
    )
    w_adv_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="w_adv", method=NOTHING)
    )
    w_height_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="w_height", method=0)
    )
    A_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="A", method=0, constant_value=0.7)
    )
    g1_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="g1", method=0, constant_value=0.4)
    )
    g2_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="g2", method=0, constant_value=8.0)
    )
    z_s1: float = 0.0
    s_1: float = 0.0
    z_s2: float = 0.0
    s_2: float = 0.0
    s_obs_NN: float = 0.0
    SRelaxTauM: float = 1.0e15
    SRelaxTauS: float = 1.0e15
    SRelaxTauB: float = 1.0e15
    SRelaxSurf: float = 0.0
    SRelaxBott: float = 0.0
    z_t1: float = 0.0
    t_1: float = 0.0
    z_t2: float = 0.0
    t_2: float = 0.0
    t_obs_NN: float = 0.0
    TRelaxTauM: float = 1.0e15
    TRelaxTauS: float = 1.0e15
    TRelaxTauB: float = 1.0e15
    TRelaxSurf: float = 0.0
    TRelaxBott: float = 0.0
    ext_press_mode: int = 0
    PeriodM: float = 44714.0
    AmpMu: float = 0.0
    AmpMv: float = 0.0
    PhaseMu: float = 0.0
    PhaseMv: float = 0.0
    PeriodS: float = 43200.0
    AmpSu: float = 0.0
    AmpSv: float = 0.0
    PhaseSu: float = 0.0
    PhaseSv: float = 0.0
    int_press_type: int = 0
    s_adv: bool = False
    t_adv: bool = False
    plume_type: int = 2
    plume_slope_x: float = 0.0
    plume_slope_y: float = 0.0
    extinct_method: int = 1
    w_adv_discr: int = P2_PDM
    period_1: float = 44714.0
    amp_1: float = 0.0
    phase_1: float = 0.0
    period_2: float = 43200.0
    amp_2: float = 0.0
    phase_2: float = 0.0
    Hs_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="Hs", method=0)
    )
    Tz_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="Tz", method=0)
    )
    phiw_input: ScalarInput = field(
        default_factory=lambda: _scalar_input(name="phiw", method=0)
    )
    vel_relax_tau: float = 1.0e15
    vel_relax_ramp: float = 1.0e15
    b_obs_surf: float = 0.0
    b_obs_NN: float = 0.0
    b_obs_sbf: float = 0.0


def init_observations(
    state: ObservationsState,
    settings: GotmSettings | Mapping[str, object] | None = None,
) -> None:
    """Initialise observation/input descriptors from GOTM settings."""

    if settings is None:
        parsed = GotmSettings()
    elif isinstance(settings, GotmSettings):
        parsed = settings
    else:
        parsed = GotmSettings.model_validate(settings)

    state.initial_temperature_type = _TEMPERATURE_TYPE[parsed.temperature.type]
    state.initial_salinity_type = _SALINITY_TYPE[parsed.salinity.type]

    state.z_t1 = parsed.temperature.two_layer.z_s
    state.t_1 = parsed.temperature.two_layer.t_s
    state.z_t2 = parsed.temperature.two_layer.z_b
    state.t_2 = parsed.temperature.two_layer.t_b
    state.t_obs_NN = parsed.temperature.NN
    state.TRelaxTauM = parsed.temperature.relax.tau
    state.TRelaxTauS = parsed.temperature.relax.tau_s
    state.TRelaxTauB = parsed.temperature.relax.tau_b
    state.TRelaxSurf = parsed.temperature.relax.h_s
    state.TRelaxBott = parsed.temperature.relax.h_b

    state.z_s1 = parsed.salinity.two_layer.z_s
    state.s_1 = parsed.salinity.two_layer.s_s
    state.z_s2 = parsed.salinity.two_layer.z_b
    state.s_2 = parsed.salinity.two_layer.s_b
    state.s_obs_NN = parsed.salinity.NN
    state.SRelaxTauM = parsed.salinity.relax.tau
    state.SRelaxTauS = parsed.salinity.relax.tau_s
    state.SRelaxTauB = parsed.salinity.relax.tau_b
    state.SRelaxSurf = parsed.salinity.relax.h_s
    state.SRelaxBott = parsed.salinity.relax.h_b

    state.extinct_method = _EXTINCTION_METHOD[parsed.light_extinction.method]
    state.ext_press_mode = _EXT_PRESS_MODE[parsed.mimic_3d.ext_pressure.type]
    state.int_press_type = _INT_PRESS_TYPE[parsed.mimic_3d.int_pressure.type]
    state.s_adv = parsed.mimic_3d.int_pressure.s_adv
    state.t_adv = parsed.mimic_3d.int_pressure.t_adv
    state.plume_type = _PLUME_TYPE[parsed.mimic_3d.int_pressure.plume.type]
    state.plume_slope_x = parsed.mimic_3d.int_pressure.plume.x_slope
    state.plume_slope_y = parsed.mimic_3d.int_pressure.plume.y_slope
    state.w_adv_discr = _W_ADV_DISCR[parsed.w.adv_discr]

    state.AmpMu = parsed.mimic_3d.ext_pressure.dpdx.tidal.amp_1
    state.PhaseMu = parsed.mimic_3d.ext_pressure.dpdx.tidal.phase_1
    state.AmpSu = parsed.mimic_3d.ext_pressure.dpdx.tidal.amp_2
    state.PhaseSu = parsed.mimic_3d.ext_pressure.dpdx.tidal.phase_2
    state.AmpMv = parsed.mimic_3d.ext_pressure.dpdy.tidal.amp_1
    state.PhaseMv = parsed.mimic_3d.ext_pressure.dpdy.tidal.phase_1
    state.AmpSv = parsed.mimic_3d.ext_pressure.dpdy.tidal.amp_2
    state.PhaseSv = parsed.mimic_3d.ext_pressure.dpdy.tidal.phase_2
    state.PeriodM = parsed.mimic_3d.ext_pressure.period_1
    state.PeriodS = parsed.mimic_3d.ext_pressure.period_2
    state.period_1 = parsed.mimic_3d.zeta.period_1
    state.period_2 = parsed.mimic_3d.zeta.period_2
    state.amp_1 = parsed.mimic_3d.zeta.tidal.amp_1
    state.phase_1 = parsed.mimic_3d.zeta.tidal.phase_1
    state.amp_2 = parsed.mimic_3d.zeta.tidal.amp_2
    state.phase_2 = parsed.mimic_3d.zeta.tidal.phase_2

    state.tprof_input = _profile_input(
        name="temperature",
        method=_profile_method(parsed.temperature.method, analytical_constant=True),
        path=parsed.temperature.path,
        index=parsed.temperature.column,
        constant_value=parsed.temperature.constant_value,
        scale_factor=parsed.temperature.scale_factor,
        add_offset=parsed.temperature.add_offset,
        minimum=-2.0,
        maximum=40.0,
        method_off=NOTHING,
        method_constant=ANALYTICAL_OFFSET + CONST_PROF,
    )
    state.sprof_input = _profile_input(
        name="salinity",
        method=_profile_method(parsed.salinity.method, analytical_constant=True),
        path=parsed.salinity.path,
        index=parsed.salinity.column,
        constant_value=parsed.salinity.constant_value,
        scale_factor=parsed.salinity.scale_factor,
        add_offset=parsed.salinity.add_offset,
        minimum=0.0,
        maximum=40.0,
        method_off=NOTHING,
        method_constant=ANALYTICAL_OFFSET + CONST_PROF,
    )

    gradients = parsed.mimic_3d.int_pressure.gradients
    state.dtdx_input = _profile_input(
        name="dtdx",
        method=_profile_method(gradients.dtdx.method, analytical_constant=False),
        path=gradients.dtdx.path,
        index=gradients.dtdx.column,
        constant_value=gradients.dtdx.constant_value,
        scale_factor=gradients.dtdx.scale_factor,
        add_offset=gradients.dtdx.add_offset,
    )
    state.dtdy_input = _profile_input(
        name="dtdy",
        method=_profile_method(gradients.dtdy.method, analytical_constant=False),
        path=gradients.dtdy.path,
        index=gradients.dtdy.column,
        constant_value=gradients.dtdy.constant_value,
        scale_factor=gradients.dtdy.scale_factor,
        add_offset=gradients.dtdy.add_offset,
    )
    state.dsdx_input = _profile_input(
        name="dsdx",
        method=_profile_method(gradients.dsdx.method, analytical_constant=False),
        path=gradients.dsdx.path,
        index=gradients.dsdx.column,
        constant_value=gradients.dsdx.constant_value,
        scale_factor=gradients.dsdx.scale_factor,
        add_offset=gradients.dsdx.add_offset,
    )
    state.dsdy_input = _profile_input(
        name="dsdy",
        method=_profile_method(gradients.dsdy.method, analytical_constant=False),
        path=gradients.dsdy.path,
        index=gradients.dsdy.column,
        constant_value=gradients.dsdy.constant_value,
        scale_factor=gradients.dsdy.scale_factor,
        add_offset=gradients.dsdy.add_offset,
    )

    state.uprof_input = _profile_input(
        name="u_obs",
        method=_profile_method(parsed.velocities.u.method, analytical_constant=False),
        path=parsed.velocities.u.path,
        index=parsed.velocities.u.column,
        constant_value=parsed.velocities.u.constant_value,
        scale_factor=parsed.velocities.u.scale_factor,
        add_offset=parsed.velocities.u.add_offset,
    )
    state.vprof_input = _profile_input(
        name="v_obs",
        method=_profile_method(parsed.velocities.v.method, analytical_constant=False),
        path=parsed.velocities.v.path,
        index=parsed.velocities.v.column,
        constant_value=parsed.velocities.v.constant_value,
        scale_factor=parsed.velocities.v.scale_factor,
        add_offset=parsed.velocities.v.add_offset,
    )
    state.vel_relax_tau = parsed.velocities.relax.tau
    state.vel_relax_ramp = parsed.velocities.relax.ramp

    state.epsprof_input = _profile_input(
        name="eps_obs",
        method=_profile_method(
            parsed.turbulence.epsprof.method, analytical_constant=False
        ),
        path=parsed.turbulence.epsprof.path,
        index=parsed.turbulence.epsprof.column,
        constant_value=parsed.turbulence.epsprof.constant_value,
        scale_factor=parsed.turbulence.epsprof.scale_factor,
        add_offset=parsed.turbulence.epsprof.add_offset,
        method_off=NOTHING,
        method_constant=NOTHING,
    )
    state.o2_prof_input = _profile_input(name="o2", method=NOTHING)

    state.dpdx_input = _scalar_input(
        name="dpdx",
        method=_scalar_method(parsed.mimic_3d.ext_pressure.dpdx.method),
        path=parsed.mimic_3d.ext_pressure.dpdx.path,
        index=parsed.mimic_3d.ext_pressure.dpdx.column,
        constant_value=parsed.mimic_3d.ext_pressure.dpdx.constant_value,
        scale_factor=parsed.mimic_3d.ext_pressure.dpdx.scale_factor,
        add_offset=parsed.mimic_3d.ext_pressure.dpdx.add_offset,
    )
    state.dpdy_input = _scalar_input(
        name="dpdy",
        method=_scalar_method(parsed.mimic_3d.ext_pressure.dpdy.method),
        path=parsed.mimic_3d.ext_pressure.dpdy.path,
        index=parsed.mimic_3d.ext_pressure.dpdy.column,
        constant_value=parsed.mimic_3d.ext_pressure.dpdy.constant_value,
        scale_factor=parsed.mimic_3d.ext_pressure.dpdy.scale_factor,
        add_offset=parsed.mimic_3d.ext_pressure.dpdy.add_offset,
    )
    state.h_press_input = _scalar_input(
        name="h_press",
        method=_scalar_method(parsed.mimic_3d.ext_pressure.h.method),
        path=parsed.mimic_3d.ext_pressure.h.path,
        index=parsed.mimic_3d.ext_pressure.h.column,
        constant_value=parsed.mimic_3d.ext_pressure.h.constant_value,
        scale_factor=parsed.mimic_3d.ext_pressure.h.scale_factor,
        add_offset=parsed.mimic_3d.ext_pressure.h.add_offset,
    )
    state.zeta_input = _scalar_input(
        name="zeta",
        method=_scalar_method(parsed.mimic_3d.zeta.method),
        path=parsed.mimic_3d.zeta.path,
        index=parsed.mimic_3d.zeta.column,
        constant_value=parsed.mimic_3d.zeta.constant_value,
        scale_factor=parsed.mimic_3d.zeta.scale_factor,
        add_offset=parsed.mimic_3d.zeta.add_offset,
    )
    state.w_adv_input = _scalar_input(
        name="w_adv",
        method=_scalar_method(parsed.w.max.method),
        path=parsed.w.max.path,
        index=parsed.w.max.column,
        constant_value=parsed.w.max.constant_value,
        scale_factor=parsed.w.max.scale_factor,
        add_offset=parsed.w.max.add_offset,
    )
    state.w_height_input = _scalar_input(
        name="w_height",
        method=_scalar_method(parsed.w.height.method),
        path=parsed.w.height.path,
        index=parsed.w.height.column,
        constant_value=parsed.w.height.constant_value,
        scale_factor=parsed.w.height.scale_factor,
        add_offset=parsed.w.height.add_offset,
    )
    state.A_input = _scalar_input(
        name="A",
        method=_scalar_method(parsed.light_extinction.A.method),
        path=parsed.light_extinction.A.path,
        index=parsed.light_extinction.A.column,
        constant_value=parsed.light_extinction.A.constant_value,
        scale_factor=parsed.light_extinction.A.scale_factor,
        add_offset=parsed.light_extinction.A.add_offset,
    )
    state.g1_input = _scalar_input(
        name="g1",
        method=_scalar_method(parsed.light_extinction.g1.method),
        path=parsed.light_extinction.g1.path,
        index=parsed.light_extinction.g1.column,
        constant_value=parsed.light_extinction.g1.constant_value,
        scale_factor=parsed.light_extinction.g1.scale_factor,
        add_offset=parsed.light_extinction.g1.add_offset,
    )
    state.g2_input = _scalar_input(
        name="g2",
        method=_scalar_method(parsed.light_extinction.g2.method),
        path=parsed.light_extinction.g2.path,
        index=parsed.light_extinction.g2.column,
        constant_value=parsed.light_extinction.g2.constant_value,
        scale_factor=parsed.light_extinction.g2.scale_factor,
        add_offset=parsed.light_extinction.g2.add_offset,
    )
    state.Hs_input = _scalar_input(
        name="Hs",
        method=_scalar_method(parsed.waves.Hs.method),
        path=parsed.waves.Hs.path,
        index=parsed.waves.Hs.column,
        constant_value=parsed.waves.Hs.constant_value,
        scale_factor=parsed.waves.Hs.scale_factor,
        add_offset=parsed.waves.Hs.add_offset,
    )
    state.Tz_input = _scalar_input(
        name="Tz",
        method=_scalar_method(parsed.waves.Tz.method),
        path=parsed.waves.Tz.path,
        index=parsed.waves.Tz.column,
        constant_value=parsed.waves.Tz.constant_value,
        scale_factor=parsed.waves.Tz.scale_factor,
        add_offset=parsed.waves.Tz.add_offset,
    )
    state.phiw_input = _scalar_input(
        name="phiw",
        method=_scalar_method(parsed.waves.phiw.method),
        path=parsed.waves.phiw.path,
        index=parsed.waves.phiw.column,
        constant_value=parsed.waves.phiw.constant_value,
        scale_factor=parsed.waves.phiw.scale_factor,
        add_offset=parsed.waves.phiw.add_offset,
    )


def post_init_observations(
    state: ObservationsState,
    depth: float,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    h: np.ndarray,
    gravity: float,
    density_state: DensityState,
) -> None:
    """Allocate observation work arrays and prepare analytical/profile inputs."""

    state.idpdx = np.zeros(nlev + 1, dtype=np.float64)
    state.idpdy = np.zeros(nlev + 1, dtype=np.float64)
    state.SRelaxTau = np.zeros(nlev + 1, dtype=np.float64)
    state.TRelaxTau = np.zeros(nlev + 1, dtype=np.float64)

    db = 0.0
    ds = depth
    state.SRelaxTau[0] = state.SRelaxTauB
    state.TRelaxTau[0] = state.TRelaxTauB
    for i in range(1, nlev + 1):
        state.TRelaxTau[i] = state.TRelaxTauM
        state.SRelaxTau[i] = state.SRelaxTauM
        db += 0.5 * h[i]
        ds -= 0.5 * h[i]
        if db <= state.SRelaxBott:
            state.SRelaxTau[i] = state.SRelaxTauB
        if ds <= state.SRelaxSurf:
            state.SRelaxTau[i] = state.SRelaxTauS
        if db <= state.TRelaxBott:
            state.TRelaxTau[i] = state.TRelaxTauB
        if ds <= state.TRelaxSurf:
            state.TRelaxTau[i] = state.TRelaxTauS
        db += 0.5 * h[i]
        ds -= 0.5 * h[i]
        if state.sprof_input.method != 0 and state.SRelaxTau[i] <= 0.0:
            raise ValueError("SRelaxTau must be positive for active salinity inputs")
        if state.tprof_input.method != 0 and state.TRelaxTau[i] <= 0.0:
            raise ValueError("TRelaxTau must be positive for active temperature inputs")

    register_input(state.sprof_input)
    assert state.sprof_input.data is not None
    state.sprof_input.data.fill(state.sprof_input.constant_value)
    if state.sprof_input.method == ANALYTICAL_OFFSET + TWO_LAYERS:
        state.sprof_input.data = analytical_profile(
            nlev,
            z,
            state.z_s1,
            state.s_1,
            state.z_s2,
            state.s_2,
        )
    elif state.sprof_input.method == ANALYTICAL_OFFSET + CONST_NN:
        if state.tprof_input.method != ANALYTICAL_OFFSET + CONST_PROF:
            msg = "salinity buoyancy profile requires constant temperature"
            raise ValueError(msg)
        state.sprof_input.data = const_NNS(
            density_state,
            nlev,
            z,
            zi,
            state.s_1,
            state.tprof_input.constant_value,
            state.s_obs_NN,
            gravity,
        )

    register_input(state.tprof_input)
    assert state.tprof_input.data is not None
    state.tprof_input.data.fill(state.tprof_input.constant_value)
    if state.tprof_input.method == ANALYTICAL_OFFSET + TWO_LAYERS:
        state.tprof_input.data = analytical_profile(
            nlev,
            z,
            state.z_t1,
            state.t_1,
            state.z_t2,
            state.t_2,
        )
    elif state.tprof_input.method == ANALYTICAL_OFFSET + CONST_NN:
        if state.sprof_input.method != ANALYTICAL_OFFSET + CONST_PROF:
            msg = "temperature buoyancy profile requires constant salinity"
            raise ValueError(msg)
        state.tprof_input.data = const_NNT(
            density_state,
            nlev,
            z,
            zi,
            state.t_1,
            state.sprof_input.constant_value,
            state.t_obs_NN,
            gravity,
        )

    register_input(state.h_press_input)
    register_input(state.dpdx_input)
    register_input(state.dpdy_input)
    register_input(state.dsdx_input)
    register_input(state.dsdy_input)
    register_input(state.dtdx_input)
    register_input(state.dtdy_input)

    if state.extinct_method == 1:
        state.A_input.value = 0.58
        state.g1_input.value = 0.35
        state.g2_input.value = 23.0
    elif state.extinct_method == 2:
        state.A_input.value = 0.68
        state.g1_input.value = 1.20
        state.g2_input.value = 28.0
    elif state.extinct_method == 3:
        state.A_input.value = 0.62
        state.g1_input.value = 0.60
        state.g2_input.value = 20.0
    elif state.extinct_method == 4:
        state.A_input.value = 0.67
        state.g1_input.value = 1.00
        state.g2_input.value = 17.0
    elif state.extinct_method == 5:
        state.A_input.value = 0.77
        state.g1_input.value = 1.50
        state.g2_input.value = 14.0
    elif state.extinct_method == 6:
        state.A_input.value = 0.78
        state.g1_input.value = 1.40
        state.g2_input.value = 7.9
    else:
        register_input(state.A_input)
        register_input(state.g1_input)
        register_input(state.g2_input)

    register_input(state.w_height_input)
    register_input(state.w_adv_input)
    register_input(state.zeta_input)
    register_input(state.Hs_input)
    register_input(state.Tz_input)
    register_input(state.phiw_input)
    register_input(state.uprof_input)
    register_input(state.vprof_input)
    register_input(state.epsprof_input)
    register_input(state.o2_prof_input)


def get_all_obs(
    state: ObservationsState,
    julday: int,
    secs: int,
    nlev: int,
    z: np.ndarray,
    *,
    fsecs: float | None = None,
) -> None:
    """Update analytical observations for the current model time."""

    del julday, nlev, z
    model_seconds = float(secs) if fsecs is None else fsecs
    if state.dpdx_input.method == ANALYTICAL:
        state.dpdx_input.value = (
            state.AmpMu
            * math.sin(2.0 * PI * (model_seconds - state.PhaseMu) / state.PeriodM)
            + state.AmpSu
            * math.sin(2.0 * PI * (model_seconds - state.PhaseSu) / state.PeriodS)
            + state.dpdx_input.constant_value
        )
    if state.dpdy_input.method == ANALYTICAL:
        state.dpdy_input.value = (
            state.AmpMv
            * math.sin(2.0 * PI * (model_seconds - state.PhaseMv) / state.PeriodM)
            + state.AmpSv
            * math.sin(2.0 * PI * (model_seconds - state.PhaseSv) / state.PeriodS)
            + state.dpdy_input.constant_value
        )
    if state.zeta_input.method == ANALYTICAL:
        state.zeta_input.value = (
            state.amp_1
            * math.sin(2.0 * PI * (model_seconds - state.phase_1) / state.period_1)
            + state.amp_2
            * math.sin(2.0 * PI * (model_seconds - state.phase_2) / state.period_2)
            + state.zeta_input.constant_value
        )


def clean_observations(state: ObservationsState) -> None:
    """Release observation work arrays."""

    state.idpdx = None
    state.idpdy = None
    state.SRelaxTau = None
    state.TRelaxTau = None
