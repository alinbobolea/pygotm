# ruff: noqa: E501
"""
GOTM–FABM coupling layer — translation of ``gotm_fabm.F90``.

Provides the interface between the General Ocean Turbulence Model (GOTM) and
the Framework for Aquatic Biogeochemical Models (FABM).  The Fortran source
delegates biogeochemistry to the external FABM library (``use fabm``,
``use fabm_types``, ``use fabm_config``, ``use field_manager``); this module
mirrors that boundary and keeps pyGOTM physics independent of any specific
``pyfabm`` installation.

Fortran state arrays (``cc``, ``cc_diag``, ``rhs``) and observation records
are held in :class:`FabmState` and :class:`FabmObservation` respectively.

Public interface: :func:`configure_gotm_fabm`, :func:`gotm_fabm_create_model`,
:func:`init_gotm_fabm`, :func:`init_var_gotm_fabm`,
:func:`init_gotm_fabm_state`, :func:`start_gotm_fabm`,
:func:`set_env_gotm_fabm`, :func:`do_gotm_fabm`, :func:`clean_gotm_fabm`,
:func:`register_observation`, :func:`register_scalar_observation`,
:func:`register_bulk_observation`, :func:`register_horizontal_observation`,
:func:`register_field`, :func:`calculate_derived_input`,
:func:`calculate_conserved_quantities`, :func:`right_hand_side_rhs`,
:func:`right_hand_side_ppdd`, :func:`do_repair_state`, :func:`light`,
:func:`save_diagnostics`, :func:`calendar_date_interface`,
:class:`FabmState`, :class:`FabmObservation`.

Original FORTRAN authors: Jorn Bruggeman.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numba
import numpy as np

from pygotm.util.adv_center import CONSERVATIVE, FLUX, P2_PDM, adv_center
from pygotm.util.diff_center import NEUMANN, diff_center

__all__ = [
    "FabmObservation",
    "FabmState",
    "calculate_conserved_quantities",
    "calculate_derived_input",
    "calendar_date_interface",
    "center_depths_single",
    "clean_gotm_fabm",
    "configure_gotm_fabm",
    "do_gotm_fabm",
    "do_repair_state",
    "gotm_driver_fatal_error",
    "gotm_driver_log_message",
    "gotm_fabm_create_model",
    "init_gotm_fabm",
    "init_gotm_fabm_state",
    "init_var_gotm_fabm",
    "light",
    "par_from_background_single",
    "par_with_bioext_from_attenuation_single",
    "register_bulk_observation",
    "register_field",
    "register_horizontal_observation",
    "register_observation",
    "register_scalar_observation",
    "right_hand_side_ppdd",
    "right_hand_side_rhs",
    "save_diagnostics",
    "set_env_gotm_fabm",
    "start_gotm_fabm",
    "step_fabm_post_rates_single",
    "step_fabm_transport_single",
]


@dataclass
class FabmObservation:
    """Registered observed FABM variable and optional relaxation data."""

    kind: str
    variable_id: Any
    data: np.ndarray | float
    relax_tau: np.ndarray | float | None = None


@dataclass
class FabmState:
    """pyGOTM-owned FABM coupling state."""

    fabm_calc: bool = False
    model: Any | None = None
    freshwater_impact: bool = True
    repair_state: bool = False
    nlev: int = 0
    dt: float = 0.0
    cc: np.ndarray | None = None
    rhs: np.ndarray | None = None
    pp: np.ndarray | None = None
    dd: np.ndarray | None = None
    bioshade: np.ndarray | None = None
    observations: list[FabmObservation] = field(default_factory=list)
    registered_fields: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, np.ndarray | float] = field(default_factory=dict)


def calendar_date_interface(julian: int) -> tuple[int, int, int]:
    """Convert a Julian day count to a calendar date."""

    current = date(1, 1, 1) + timedelta(days=int(julian) - 1)
    return current.year, current.month, current.day


@numba.njit(cache=True)
def center_depths_single(nlev: int, h: np.ndarray, depth: np.ndarray) -> None:
    """Calculate local depth below surface from layer height.

    Used internally to compute light field, and may be used by
    biogeochemical models as well.
    """

    if nlev <= 0:
        return
    depth[nlev - 1] = 0.5 * h[nlev]
    for idx in range(nlev - 2, -1, -1):
        depth[idx] = depth[idx + 1] + 0.5 * (h[idx + 1] + h[idx + 2])


@numba.njit(cache=True)
def par_from_background_single(
    nlev: int,
    h: np.ndarray,
    rad: np.ndarray,
    light_A: float,
    light_g2: float,
    depth: np.ndarray,
    par_col: np.ndarray,
) -> float:
    """Calculate photosynthetically active radiation (PAR)."""

    surface_par = float(rad[nlev] * (1.0 - light_A))
    if nlev <= 0 or light_g2 <= 0.0:
        for idx in range(nlev):
            par_col[idx] = 0.0
        return surface_par
    center_depths_single(nlev, h, depth)
    for idx in range(nlev):
        par_value = surface_par * math.exp(-depth[idx] / light_g2)
        if par_value < 0.0:
            par_value = 0.0
        par_col[idx] = par_value
    return surface_par


@numba.njit(cache=True)
def par_with_bioext_from_attenuation_single(
    nlev: int,
    attenuation: np.ndarray,
    h: np.ndarray,
    rad: np.ndarray,
    light_A: float,
    light_g2: float,
    depth: np.ndarray,
    par_col: np.ndarray,
) -> float:
    """Calculate PAR using background and biotic extinction."""

    surface_par = float(rad[nlev] * (1.0 - light_A))
    if nlev <= 0 or light_g2 <= 0.0:
        for idx in range(nlev):
            par_col[idx] = 0.0
        return surface_par
    center_depths_single(nlev, h, depth)
    bioext = 0.0
    for idx in range(nlev - 1, -1, -1):
        local_ext = attenuation[idx]
        if local_ext < 0.0:
            local_ext = 0.0

        # Add the extinction of the first half of the grid box.
        bioext += local_ext * h[idx + 1] * 0.5

        # Calculate photosynthetically active radiation (PAR).
        par_value = surface_par * math.exp(-depth[idx] / light_g2 - bioext)
        if par_value < 0.0:
            par_value = 0.0
        par_col[idx] = par_value

        # Add the extinction of the second half of the grid box.
        bioext += local_ext * h[idx + 1] * 0.5
    return surface_par


@numba.njit(cache=True)
def step_fabm_transport_single(
    nlev: int,
    dt: float,
    cnpar: float,
    precip: float,
    has_vert_move: int,
    n_interior: int,
    vert_move: np.ndarray,
    h_step: np.ndarray,
    nuh_step: np.ndarray,
    cc: np.ndarray,
    y: np.ndarray,
    ws: np.ndarray,
    adv_cu: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    tau_r: np.ndarray,
    y_obs: np.ndarray,
) -> None:
    """Vertical advection and residual movement; Vertical diffusion."""

    if has_vert_move != 0 and nlev >= 2:
        for var in range(n_interior):
            has_motion = False
            for k in range(nlev):
                if vert_move[var, k] != 0.0:
                    has_motion = True
                    break
            if not has_motion:
                continue

            for k in range(nlev + 1):
                ws[k] = 0.0
            for k in range(1, nlev):
                h_sum = h_step[k] + h_step[k + 1]
                iweight = 0.5
                if h_sum != 0.0:
                    iweight = h_step[k + 1] / h_sum
                ws[k] = (
                    iweight * vert_move[var, k - 1]
                    + (1.0 - iweight) * vert_move[var, k]
                )

            for k in range(nlev):
                y[k + 1] = cc[var, k]
            adv_center(
                nlev,
                dt,
                h_step,
                h_step,
                ws,
                FLUX,
                FLUX,
                0.0,
                0.0,
                P2_PDM,
                CONSERVATIVE,
                y,
                adv_cu,
            )
            for k in range(nlev):
                cc[var, k] = y[k + 1]

    for var in range(n_interior):
        for k in range(nlev):
            y[k + 1] = cc[var, k]
        surface_flux = -cc[var, nlev - 1] * precip
        diff_center(
            nlev,
            dt,
            cnpar,
            0,
            h_step,
            NEUMANN,
            NEUMANN,
            surface_flux,
            0.0,
            nuh_step,
            l_sour,
            q_sour,
            tau_r,
            y_obs,
            y,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )
        for k in range(nlev):
            cc[var, k] = y[k + 1]


@numba.njit(cache=True)
def step_fabm_post_rates_single(
    nlev: int,
    dt: float,
    n_interior: int,
    n_surface: int,
    n_bottom: int,
    bulk_rates: np.ndarray,
    surf_rates: np.ndarray,
    bot_rates: np.ndarray,
    cc: np.ndarray,
) -> None:
    """Add pelagic sink and source terms for all depth levels."""

    for var in range(n_interior):
        for k in range(nlev):
            bulk_rate = bulk_rates[var, k]
            cc[var, k] += dt * bulk_rate
        cc[var, nlev - 1] += dt * (
            surf_rates[var, nlev - 1] - bulk_rates[var, nlev - 1]
        )
        cc[var, 0] += dt * (bot_rates[var, 0] - bulk_rates[var, 0])

    surface_start = n_interior
    surface_stop = surface_start + n_surface
    for var in range(surface_start, surface_stop):
        value = cc[var, nlev - 1] + dt * surf_rates[var, nlev - 1]
        for k in range(nlev):
            cc[var, k] = value

    bottom_start = n_interior + n_surface
    bottom_stop = bottom_start + n_bottom
    for var in range(bottom_start, bottom_stop):
        value = cc[var, 0] + dt * bot_rates[var, 0]
        for k in range(nlev):
            cc[var, k] = value


def configure_gotm_fabm(
    state: FabmState,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Configure the FABM coupling from a YAML-like mapping."""

    raw = {} if cfg is None else cfg
    state.fabm_calc = bool(raw.get("use", state.fabm_calc))
    state.freshwater_impact = bool(
        raw.get("freshwater_impact", state.freshwater_impact)
    )
    state.repair_state = bool(raw.get("repair_state", state.repair_state))


def gotm_fabm_create_model(
    state: FabmState,
    namlst: str | None = None,
    *,
    model_factory: Callable[[str | None], Any] | None = None,
) -> Any:
    """Create the external FABM model object."""

    if model_factory is None:
        try:
            import pyfabm
        except ImportError as exc:
            msg = "pyfabm is required for live FABM coupling"
            raise RuntimeError(msg) from exc
        model_factory = pyfabm.Model
    state.model = model_factory(namlst)
    state.fabm_calc = True
    return state.model


def init_gotm_fabm(
    state: FabmState,
    nlev: int,
    dt: float,
    field_manager: Any | None = None,
) -> None:
    """Initialise FABM coupling arrays."""

    state.nlev = nlev
    state.dt = float(dt)
    if not state.fabm_calc:
        return
    init_var_gotm_fabm(state, nlev)
    init_gotm_fabm_state(state, nlev)
    if field_manager is not None and state.model is not None:
        start_gotm_fabm(state, nlev, field_manager)


def init_var_gotm_fabm(state: FabmState, nlev: int) -> None:
    """Allocate FABM state/rate arrays."""

    nstate = int(getattr(state.model, "state_variable_count", 1) or 1)
    state.cc = np.zeros((nstate, nlev + 1), dtype=np.float64)
    state.rhs = np.zeros_like(state.cc)
    state.pp = np.zeros_like(state.cc)
    state.dd = np.zeros_like(state.cc)
    state.bioshade = np.ones(nlev + 1, dtype=np.float64)


def register_field(
    state: FabmState,
    variable: Any,
    prefix: str = "",
    dimensions: tuple[str, ...] = (),
    data0d: float | None = None,
    data1d: np.ndarray | None = None,
    part_of_state: bool = False,
    used: bool = True,
) -> None:
    """Register a FABM field with a field-manager-like registry."""

    if not used:
        return
    name = prefix + str(getattr(variable, "name", variable))
    state.registered_fields[name] = {
        "variable": variable,
        "dimensions": dimensions,
        "data0d": data0d,
        "data1d": data1d,
        "part_of_state": part_of_state,
    }


def register_scalar_observation(
    state: FabmState,
    scalar_id: Any,
    data: float,
) -> None:
    """Register an observed scalar FABM variable."""

    state.observations.append(FabmObservation("scalar", scalar_id, float(data)))


def register_bulk_observation(
    state: FabmState,
    variable_id: Any,
    data: np.ndarray,
    relax_tau: np.ndarray,
) -> None:
    """Register an observed depth-varying FABM variable."""

    state.observations.append(FabmObservation("bulk", variable_id, data, relax_tau))


def register_horizontal_observation(
    state: FabmState,
    horizontal_id: Any,
    data: float,
    relax_tau: float,
) -> None:
    """Register an observed horizontal-slice FABM variable."""

    state.observations.append(
        FabmObservation("horizontal", horizontal_id, float(data), float(relax_tau))
    )


def register_observation(
    state: FabmState, variable_id: Any, data: Any, relax_tau: Any = None
) -> None:
    """Dispatch observation registration by data shape."""

    if isinstance(data, np.ndarray):
        if relax_tau is None:
            relax_tau = np.full_like(data, 1.0e15)
        register_bulk_observation(state, variable_id, data, relax_tau)
    elif relax_tau is None:
        register_scalar_observation(state, variable_id, float(data))
    else:
        register_horizontal_observation(
            state, variable_id, float(data), float(relax_tau)
        )


def init_gotm_fabm_state(state: FabmState, nlev: int) -> None:
    """Initialise FABM state variables from the external model when available."""

    if state.cc is None:
        init_var_gotm_fabm(state, nlev)
    if state.model is not None and hasattr(state.model, "initialize_state"):
        assert state.cc is not None
        state.model.initialize_state(state.cc)


def start_gotm_fabm(
    state: FabmState,
    nlev: int,
    field_manager: Any | None = None,
) -> None:
    """Start FABM after GOTM fields have been registered."""

    del nlev
    if state.model is not None and hasattr(state.model, "start"):
        state.model.start()
    if field_manager is not None:
        for name, record in state.registered_fields.items():
            if hasattr(field_manager, "register"):
                field_manager.register(
                    name, provider=lambda r=record: r.get("data1d", r.get("data0d"))
                )


def set_env_gotm_fabm(state: FabmState, **environment: Any) -> None:
    """Store environmental arrays/scalars passed from GOTM."""

    state.environment.update(environment)
    if state.model is not None and hasattr(state.model, "set_environment"):
        state.model.set_environment(**environment)


def calculate_derived_input(state: FabmState, nlev: int) -> None:
    """Ask FABM to calculate derived environmental inputs."""

    del nlev
    if state.model is not None and hasattr(state.model, "calculate_derived_input"):
        state.model.calculate_derived_input()


def right_hand_side_rhs(
    state: FabmState,
    first: int,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    rhs: np.ndarray,
) -> None:
    """Calculate FABM right-hand-side source terms."""

    del first, numc, nlev
    rhs.fill(0.0)
    if state.model is not None and hasattr(state.model, "get_rates"):
        rates = state.model.get_rates(cc)
        rhs[:] = np.asarray(rates, dtype=np.float64)


def right_hand_side_ppdd(
    state: FabmState,
    first: int,
    numc: int,
    nlev: int,
    cc: np.ndarray,
    pp: np.ndarray,
    dd: np.ndarray,
) -> None:
    """Calculate split production and destruction terms."""

    del first, numc, nlev
    pp.fill(0.0)
    dd.fill(0.0)
    if state.model is not None and hasattr(state.model, "get_sources"):
        prod, dest = state.model.get_sources(cc)
        pp[:] = np.asarray(prod, dtype=np.float64)
        dd[:] = np.asarray(dest, dtype=np.float64)


def do_gotm_fabm(state: FabmState, nlev: int, itime: int) -> None:
    """Advance FABM one GOTM time step."""

    del itime
    if not state.fabm_calc:
        return
    assert state.cc is not None
    assert state.rhs is not None
    right_hand_side_rhs(state, 1, state.cc.shape[0], nlev, state.cc, state.rhs)
    state.cc[:, 1 : nlev + 1] += state.dt * state.rhs[:, 1 : nlev + 1]
    if state.repair_state:
        do_repair_state(state, nlev, "do_gotm_fabm")


def do_repair_state(state: FabmState, nlev: int, location: str = "") -> None:
    """Repair invalid FABM state values."""

    del nlev, location
    if state.model is not None and hasattr(state.model, "repair_state"):
        assert state.cc is not None
        state.model.repair_state(state.cc)
    elif state.cc is not None:
        np.maximum(state.cc, 0.0, out=state.cc)


def light(state: FabmState, nlev: int, attenuation: float = 0.04) -> None:
    """Compute a simple exponentially decaying light/shading profile."""

    if state.bioshade is None:
        state.bioshade = np.ones(nlev + 1, dtype=np.float64)
    for k in range(nlev + 1):
        state.bioshade[k] = np.exp(-attenuation * (nlev - k))


def save_diagnostics(state: FabmState) -> dict[str, np.ndarray | float]:
    """Return a copy of current FABM diagnostics."""

    copied: dict[str, np.ndarray | float] = {}
    for name, value in state.diagnostics.items():
        copied[name] = value.copy() if isinstance(value, np.ndarray) else float(value)
    return copied


def clean_gotm_fabm(state: FabmState) -> None:
    """Release FABM-owned arrays and observations."""

    state.cc = None
    state.rhs = None
    state.pp = None
    state.dd = None
    state.bioshade = None
    state.observations.clear()
    state.registered_fields.clear()
    state.environment.clear()
    state.diagnostics.clear()


def gotm_driver_fatal_error(self: Any, location: str, message: str) -> None:
    """FABM driver fatal-error callback."""

    del self
    raise RuntimeError(f"{location}: {message}")


def gotm_driver_log_message(self: Any, message: str) -> None:
    """FABM driver log callback."""

    del self, message


def calculate_conserved_quantities(
    state: FabmState,
    nlev: int,
    h: np.ndarray,
    total: np.ndarray,
) -> None:
    """Calculate vertically integrated conserved quantities."""

    assert state.cc is not None
    total.fill(0.0)
    for variable in range(state.cc.shape[0]):
        for k in range(1, nlev + 1):
            total[variable] += state.cc[variable, k] * h[k]
