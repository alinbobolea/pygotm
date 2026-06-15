"""Core runtime containers, builders, and time-loop selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from pygotm.gotm.runtime_forcing import RuntimeForcing, allocate_runtime_forcing
from pygotm.gotm.runtime_output import RuntimeOutput, allocate_runtime_output
from pygotm.gotm.runtime_params import RuntimeParams, make_runtime_params
from pygotm.gotm.runtime_state import RuntimeState, allocate_runtime_state
from pygotm.gotm.runtime_work import RuntimeWork, allocate_runtime_work
from pygotm.gotm.time_loop import run_compiled_time_loop
from pygotm.turbulence.turbulence import (
    Constant,
    Munk_Anderson,
    Schumann_Gerz,
    diss_eq,
    epsb_algebraic,
    first_order,
    kb_algebraic,
    length_eq,
    omega_eq,
    quasi_Eq,
    second_order,
    tke_keps,
    tke_MY,
    weak_Eq_Kb_Eq,
)


class UnsupportedConfigurationError(RuntimeError):
    """Raised when setup requests physics not yet supported by compiled loops."""


class TimeLoopRunner(Protocol):
    """Callable wrapper that crosses into the compiled timestep loop."""

    def __call__(
        self,
        params: RuntimeParams,
        state: RuntimeState,
        work: RuntimeWork,
        forcing: RuntimeForcing,
        output: RuntimeOutput,
        step_offset: int = 0,
        out_slot_base: int = 0,
        write_ic: int = 1,
        init_int_precip: float = 0.0,
        init_int_evap: float = 0.0,
        init_int_swr: float = 0.0,
        init_int_heat: float = 0.0,
        init_int_total: float = 0.0,
        output_reduce_mode: int = 0,
        hydro_store: int = 0,
        hydro_T: np.ndarray | None = None,
        hydro_S: np.ndarray | None = None,
        hydro_rho: np.ndarray | None = None,
        hydro_h: np.ndarray | None = None,
        hydro_nuh: np.ndarray | None = None,
        hydro_rad: np.ndarray | None = None,
        hydro_taub: np.ndarray | None = None,
        hydro_Vc: np.ndarray | None = None,
        hydro_Vco: np.ndarray | None = None,
        hydro_Afo: np.ndarray | None = None,
        hydro_wq: np.ndarray | None = None,
        hydro_Qres: np.ndarray | None = None,
        hydro_stream_Q: np.ndarray | None = None,
    ) -> int: ...


@dataclass(slots=True)
class RuntimePhaseTimings:
    """Optional wall-clock timing accumulator for compiled runtime phases."""

    runtime_build_s: float = 0.0
    force_build_s: float = 0.0
    integration_s: float = 0.0
    compiled_integration_s: float = 0.0
    fabm_chunk_s: float = 0.0
    copy_back_s: float = 0.0


@dataclass(slots=True)
class RuntimeBundle:
    """All setup-time containers needed to invoke a compiled runtime."""

    params: RuntimeParams
    state: RuntimeState
    work: RuntimeWork
    forcing: RuntimeForcing
    output: RuntimeOutput
    runner: TimeLoopRunner
    output_reduce_mode: int = 0

    def run(self) -> int:
        """Execute the selected compiled time-loop wrapper."""

        return self.runner(
            self.params,
            self.state,
            self.work,
            self.forcing,
            self.output,
            output_reduce_mode=self.output_reduce_mode,
        )


def _unsupported_feature_names(params: RuntimeParams) -> tuple[str, ...]:
    unsupported: list[str] = []
    if params.calc_bottom_stress != 1:
        unsupported.append("calc_bottom_stress")
    if params.int_press_type not in (0, 1, 2):
        unsupported.append("int_press_type")
    if params.airsea_fluxes_method not in (0, 1, 2):
        unsupported.append("airsea.fluxes_method")
    if params.turb_method not in (first_order, second_order):
        unsupported.append("turb_method")
    if params.tke_method not in (tke_keps, tke_MY):
        unsupported.append("tke_method")
    if params.len_scale_method not in (omega_eq, diss_eq, length_eq):
        unsupported.append("len_scale_method")
    if params.stab_method not in (Constant, Munk_Anderson, Schumann_Gerz):
        unsupported.append("stab_method")
    if params.turb_method == second_order and params.tke_method == tke_keps:
        if params.scnd_method not in (weak_Eq_Kb_Eq, quasi_Eq):
            unsupported.append("scnd_method")
        if params.kb_method != kb_algebraic:
            unsupported.append("kb_method")
        if params.epsb_method != epsb_algebraic:
            unsupported.append("epsb_method")
    if params.iw_model not in (0, 1, 2):
        unsupported.append("iw_model")
    return tuple(unsupported)


def select_time_loop(params: RuntimeParams) -> TimeLoopRunner:
    """Return the compiled loop wrapper for the currently supported setup."""

    if _matches_supported_path(params):
        return run_compiled_time_loop

    unsupported = _unsupported_feature_names(params)
    if not unsupported:
        unsupported = (
            "turb_method",
            "tke_method",
            "len_scale_method",
            "stab_method",
            "scnd_method",
        )
    names = ", ".join(unsupported)
    msg = (
        "compiled GOTM runtime does not yet support the requested "
        f"configuration settings: {names}"
    )
    raise UnsupportedConfigurationError(msg)


def _matches_supported_path(params: RuntimeParams) -> bool:
    if not (
        params.calc_bottom_stress == 1
        and params.int_press_type in (0, 1, 2)
        and params.airsea_fluxes_method in (0, 1, 2)
        and params.turb_method in (first_order, second_order)
        and params.tke_method in (tke_keps, tke_MY)
        and params.len_scale_method in (omega_eq, diss_eq, length_eq)
        and params.stab_method in (Constant, Munk_Anderson, Schumann_Gerz)
        and params.iw_model in (0, 1, 2)
    ):
        return False
    if params.turb_method == second_order and params.tke_method == tke_keps:
        return (
            params.scnd_method in (weak_Eq_Kb_Eq, quasi_Eq)
            and params.kb_method == kb_algebraic
            and params.epsb_method == epsb_algebraic
        )
    return True  # first_order or tke_MY — scnd/kb/epsb fields are irrelevant


def build_runtime_state(nlev: int) -> RuntimeState:
    """Allocate runtime state arrays."""

    return allocate_runtime_state(nlev)


build_runtime_params = make_runtime_params


def build_runtime_work(nlev: int) -> RuntimeWork:
    """Allocate persistent runtime work arrays."""

    return allocate_runtime_work(nlev)


def build_runtime_forcing(nlev: int, nt: int, *, nstreams: int = 0) -> RuntimeForcing:
    """Allocate dense forcing arrays for a compiled runtime."""

    return allocate_runtime_forcing(nlev, nt, nstreams=nstreams)


def build_runtime_output(
    nlev: int,
    nt: int,
    *,
    output: bool = True,
    output_every: int = 1,
    force_final: bool = True,
) -> RuntimeOutput:
    """Allocate dense output buffers."""

    return allocate_runtime_output(
        nlev,
        nt,
        enabled=output,
        output_every=output_every,
        force_final=force_final,
    )


def build_runtime(
    params: RuntimeParams,
    *,
    output: bool = True,
    output_every: int = 1,
    force_final: bool = True,
    output_reduce_mode: int = 0,
) -> RuntimeBundle:
    """Allocate containers and select the compiled loop for *params*."""

    runner = select_time_loop(params)
    return RuntimeBundle(
        params=params,
        state=build_runtime_state(params.nlev),
        work=build_runtime_work(params.nlev),
        forcing=build_runtime_forcing(
            params.nlev,
            params.nt,
            nstreams=params.nstreams,
        ),
        output=build_runtime_output(
            params.nlev,
            params.nt,
            output=output,
            output_every=output_every,
            force_final=force_final,
        ),
        runner=runner,
        output_reduce_mode=output_reduce_mode,
    )


def _output_reduce_mode(time_method: str) -> int:
    if time_method == "mean":
        return 1
    if time_method == "integrated":
        return 2
    return 0


def _profile(value: object, name: str, nlev: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        msg = f"{name} is not allocated"
        raise ValueError(msg)
    if value.shape != (nlev + 1,):
        msg = f"{name} must have shape {(nlev + 1,)}, got {value.shape}"
        raise ValueError(msg)
    return value


def _copy_profiles(source: object, target: RuntimeState, names: Sequence[str]) -> None:
    for name in names:
        source_array = _profile(getattr(source, name), name, target.nlev)
        np.copyto(getattr(target, name), source_array)
