"""Setup helpers for the compiled single-column GOTM runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import gsw
import numpy as np
import xarray as xr

from pygotm.gotm.runtime_forcing import RuntimeForcing, allocate_runtime_forcing
from pygotm.gotm.runtime_output import RuntimeOutput, allocate_runtime_output
from pygotm.gotm.runtime_params import RuntimeParams, make_runtime_params
from pygotm.gotm.runtime_state import RuntimeState, allocate_runtime_state
from pygotm.gotm.runtime_work import RuntimeWork, allocate_runtime_work
from pygotm.gotm.time_loop import run_compiled_time_loop
from pygotm.input.input import do_input
from pygotm.observations.observations import get_all_obs
from pygotm.stokes_drift.stokes_drift import do_stokes_drift
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

__all__ = [
    "RuntimeBundle",
    "TimeLoopRunner",
    "UnsupportedConfigurationError",
    "build_runtime",
    "build_runtime_forcing",
    "build_runtime_forcing_from_run",
    "build_runtime_from_run",
    "build_runtime_output",
    "build_runtime_params",
    "build_runtime_state",
    "build_runtime_work",
    "runtime_output_to_dataset",
    "select_time_loop",
]


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
    ) -> int: ...


@dataclass(slots=True)
class RuntimeBundle:
    """All setup-time containers needed to invoke a compiled runtime."""

    params: RuntimeParams
    state: RuntimeState
    work: RuntimeWork
    forcing: RuntimeForcing
    output: RuntimeOutput
    runner: TimeLoopRunner

    def run(self) -> int:
        """Execute the selected compiled time-loop wrapper."""

        return self.runner(
            self.params,
            self.state,
            self.work,
            self.forcing,
            self.output,
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


def build_runtime_forcing(nlev: int, nt: int) -> RuntimeForcing:
    """Allocate dense forcing arrays for a compiled runtime."""

    return allocate_runtime_forcing(nlev, nt)


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
) -> RuntimeBundle:
    """Allocate containers and select the compiled loop for *params*."""

    runner = select_time_loop(params)
    return RuntimeBundle(
        params=params,
        state=build_runtime_state(params.nlev),
        work=build_runtime_work(params.nlev),
        forcing=build_runtime_forcing(params.nlev, params.nt),
        output=build_runtime_output(
            params.nlev,
            params.nt,
            output=output,
            output_every=output_every,
            force_final=force_final,
        ),
        runner=runner,
    )


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


def _surface_input_value(input_: Any | None, fallback: float = 0.0) -> float:
    if input_ is None:
        return fallback
    return float(input_.value)


def _surface_input_optional_value(input_: Any | None) -> float:
    if input_ is None:
        return float("nan")
    return float(input_.value)


def _input_method(input_: Any | None) -> int:
    if input_ is None:
        return 0
    return int(getattr(input_, "method", 0))


def _stokes_runtime_active(stokes: Any) -> bool:
    return any(
        getattr(stokes, name) != 0
        for name in (
            "us0_method",
            "vs0_method",
            "ds_method",
            "uwnd_method",
            "vwnd_method",
            "usprof_method",
            "vsprof_method",
            "dusdz_method",
            "dvsdz_method",
        )
    )


def _validate_run_supported_by_compiled_runtime(run: Any, *, output: bool) -> None:
    unsupported: list[str] = []

    if output and str(run.output_schedule.time_method) != "point":
        unsupported.append(f"output.time_method={run.output_schedule.time_method!r}")
    if not bool(run.output_schedule.capture_initial):
        unsupported.append("output.capture_initial=False")

    if unsupported:
        names = ", ".join(unsupported)
        msg = f"compiled GOTM runtime does not yet support: {names}"
        raise UnsupportedConfigurationError(msg)


def _make_runtime_params_from_run(run: Any, nt: int) -> RuntimeParams:
    meanflow = run.meanflow
    turbulence = run.turbulence
    density = run.density
    observations = run.observations
    seagrass = run.seagrass
    stokes = run.stokes_drift
    diagnostics = run.diagnostics

    rho0 = float(density.rho0)
    tx = _surface_input_value(run.surface_inputs.tx, float(run.airsea.tx)) / rho0
    ty = _surface_input_value(run.surface_inputs.ty, float(run.airsea.ty)) / rho0

    return make_runtime_params(
        nlev=int(run.nlev),
        nt=nt,
        dt=float(run.dt),
        cnpar=float(run.cnpar),
        latitude=float(run.latitude),
        longitude=float(run.longitude),
        depth=float(run.depth),
        gravity=float(meanflow.gravity),
        rho0=rho0,
        cori=float(meanflow.cori),
        avmolu=float(meanflow.avmolu),
        avmolT=float(meanflow.avmolT),
        avmolS=float(meanflow.avmolS),
        cp=float(density.cp),
        h0b=float(meanflow.h0b),
        z0s_min=float(meanflow.z0s_min),
        calc_bottom_stress=1 if meanflow.calc_bottom_stress else 0,
        charnock=1 if meanflow.charnock else 0,
        charnock_val=float(meanflow.charnock_val),
        max_it_z0b=int(meanflow.MaxItz0b),
        plume_active=(
            1
            if observations.int_press_type == 2 and observations.plume_type == 1
            else 0
        ),
        int_press_type=int(observations.int_press_type),
        plume_type=int(observations.plume_type),
        plume_slope_x=float(observations.plume_slope_x),
        plume_slope_y=float(observations.plume_slope_y),
        seagrass_active=1 if seagrass.seagrass_calc else 0,
        seagrass_alpha=float(seagrass.alpha),
        seagrass_grassind=int(seagrass.grassind),
        seagrass_grassn=int(seagrass.grassn),
        stokes_active=1 if _stokes_runtime_active(stokes) else 0,
        w_adv_active=int(observations.w_adv_input.method),
        w_adv_discr=int(observations.w_adv_discr),
        s_adv=1 if observations.s_adv else 0,
        t_adv=1 if observations.t_adv else 0,
        sprof_input_active=1 if observations.sprof_input.method != 0 else 0,
        tprof_input_active=1 if observations.tprof_input.method != 0 else 0,
        uprof_input_active=1 if observations.uprof_input.method != 0 else 0,
        vprof_input_active=1 if observations.vprof_input.method != 0 else 0,
        zeta_input_active=1 if _input_method(observations.zeta_input) != 0 else 0,
        grid_method=int(meanflow.grid_method),
        airsea_fluxes_method=int(run.airsea.fluxes_method),
        airsea_hum_method=int(run.airsea.hum_method),
        airsea_shortwave_method=int(run.airsea.shortwave_method),
        airsea_shortwave_type=int(run.airsea.shortwave_type),
        airsea_longwave_method=int(run.airsea.longwave_method),
        airsea_longwave_type=int(run.airsea.longwave_type),
        airsea_albedo_method=int(run.airsea.albedo_method),
        airsea_ssuv_method=int(run.airsea.ssuv_method),
        airsea_sst_obs_method=_input_method(run.surface_inputs.sst_obs),
        airsea_shortwave_scale_factor=float(run.airsea.shortwave_scale_factor),
        airsea_heat_scale_factor=float(run.airsea.heat_scale_factor),
        airsea_const_albedo=float(run.airsea.const_albedo),
        ice_model=(
            int(run.ice_params.model)
            if getattr(run, "ice_params", None) is not None
            else 1
        ),
        turb_method=int(turbulence.turb_method),
        tke_method=int(turbulence.tke_method),
        len_scale_method=int(turbulence.len_scale_method),
        my_b1=float(turbulence.b1),
        my_sq=float(turbulence.sq),
        my_sl=float(turbulence.sl),
        my_e1=float(turbulence.e1),
        my_e2=float(turbulence.e2),
        my_e3=float(turbulence.e3),
        my_ex=float(turbulence.ex),
        my_e6=float(turbulence.e6),
        my_length=int(turbulence.my_length),
        stab_method=int(turbulence.stab_method),
        scnd_method=int(turbulence.scnd_method),
        kb_method=int(turbulence.kb_method),
        epsb_method=int(turbulence.epsb_method),
        iw_model=int(turbulence.iw_model),
        prandtl0_fix=float(turbulence.Prandtl0_fix),
        mld_method=int(diagnostics.mld_method),
        mld_diff_k=float(diagnostics.diff_k),
        mld_ri_crit=float(diagnostics.Ri_crit),
        kappa=float(turbulence.kappa),
        cm0=float(turbulence.cm0),
        cmsf=float(turbulence.cmsf),
        cde=float(turbulence.cde),
        k_min=float(turbulence.k_min),
        eps_min=float(turbulence.eps_min),
        kb_min=float(turbulence.kb_min),
        epsb_min=float(turbulence.epsb_min),
        tx=tx,
        ty=ty,
        dzetadx=float(observations.dpdx_input.value),
        dzetady=float(observations.dpdy_input.value),
        ext_press_mode=int(observations.ext_press_mode),
        vel_relax_ramp=float(observations.vel_relax_ramp),
        k_ubc=int(turbulence.k_ubc),
        k_lbc=int(turbulence.k_lbc),
        psi_ubc=int(turbulence.psi_ubc),
        psi_lbc=int(turbulence.psi_lbc),
        ubc_type=int(turbulence.ubc_type),
        lbc_type=int(turbulence.lbc_type),
        length_lim=1 if turbulence.length_lim else 0,
        sig_k=float(turbulence.sig_k),
        sig_w=float(turbulence.sig_w),
        cw=float(turbulence.cw),
        gen_alpha=float(turbulence.gen_alpha),
        gen_l=float(turbulence.gen_l),
        galp=float(turbulence.galp),
        cc1=float(turbulence.cc1),
        ct1=float(turbulence.ct1),
        ctt=float(turbulence.ctt),
        a1=float(turbulence.a1),
        a2=float(turbulence.a2),
        a3=float(turbulence.a3),
        a5=float(turbulence.a5),
        at1=float(turbulence.at1),
        at2=float(turbulence.at2),
        at3=float(turbulence.at3),
        at5=float(turbulence.at5),
        cw1=float(turbulence.cw1),
        cw2=float(turbulence.cw2),
        cw3plus=float(turbulence.cw3plus),
        cw3minus=float(turbulence.cw3minus),
        cwx=float(turbulence.cwx),
        cw4=float(turbulence.cw4),
        ce1=float(turbulence.ce1),
        ce2=float(turbulence.ce2),
        ce3plus=float(turbulence.ce3plus),
        ce3minus=float(turbulence.ce3minus),
        cex=float(turbulence.cex),
        ce4=float(turbulence.ce4),
        sig_e=float(turbulence.sig_e),
        sig_e0=float(turbulence.sig_e0),
        sig_peps=1 if turbulence.sig_peps else 0,
        iw_alpha=float(turbulence.alpha),
        klimiw=float(turbulence.klimiw),
        rich_cr=float(turbulence.rich_cr),
        numiw=float(turbulence.numiw),
        nuhiw=float(turbulence.nuhiw),
        numshear=float(turbulence.numshear),
        light_A=float(observations.A_input.value),
        light_g1=float(observations.g1_input.value),
        light_g2=float(observations.g2_input.value),
        density_method=int(density.density_method),
        rhob=float(density._rhob),
        alpha0=float(density.alpha0),
        beta0=float(density.beta0),
        T0=float(density.T0),
        S0=float(density.S0),
    )


def _copy_profile_data(input_: Any | None, target: np.ndarray) -> None:
    if input_ is not None and isinstance(getattr(input_, "data", None), np.ndarray):
        np.copyto(target, input_.data)
    else:
        target.fill(0.0)


def _update_runtime_relaxation_targets(run: Any) -> None:
    meanflow = run.meanflow
    observations = run.observations

    assert meanflow.z is not None
    assert meanflow.Sobs is not None
    assert meanflow.Tobs is not None
    assert meanflow.S is not None

    z = meanflow.z[1 : run.nlev + 1]
    pressure = -z

    if (
        observations.sprof_input.method != 0
        and observations.sprof_input.data is not None
    ):
        if observations.initial_salinity_type == 1:
            meanflow.Sobs[1 : run.nlev + 1] = gsw.SA_from_SP(
                observations.sprof_input.data[1 : run.nlev + 1],
                pressure,
                run.longitude,
                run.latitude,
            )
        else:
            meanflow.Sobs[1 : run.nlev + 1] = observations.sprof_input.data[
                1 : run.nlev + 1
            ]

    if (
        observations.tprof_input.method != 0
        and observations.tprof_input.data is not None
    ):
        if observations.initial_temperature_type == 1:
            meanflow.Tobs[1 : run.nlev + 1] = gsw.CT_from_t(
                meanflow.Sobs[1 : run.nlev + 1],
                observations.tprof_input.data[1 : run.nlev + 1],
                pressure,
            )
        elif observations.initial_temperature_type == 2:
            meanflow.Tobs[1 : run.nlev + 1] = gsw.CT_from_pt(
                meanflow.Sobs[1 : run.nlev + 1],
                observations.tprof_input.data[1 : run.nlev + 1],
            )
        else:
            meanflow.Tobs[1 : run.nlev + 1] = observations.tprof_input.data[
                1 : run.nlev + 1
            ]


def _record_forcing_step(run: Any, forcing: RuntimeForcing, step: int) -> None:
    observations = run.observations
    surface_inputs = run.surface_inputs

    forcing.yearday[step] = int(run.time.yearday)
    forcing.time[step] = float(step) * float(run.dt)
    forcing.secondsofday[step] = float(run.time.fsecondsofday)
    forcing.zeta[step] = _surface_input_value(observations.zeta_input)
    forcing.dpdx[step] = _surface_input_value(observations.dpdx_input)
    forcing.dpdy[step] = _surface_input_value(observations.dpdy_input)
    forcing.h_press[step] = _surface_input_value(observations.h_press_input)

    forcing.tx[step] = _surface_input_value(surface_inputs.tx)
    forcing.ty[step] = _surface_input_value(surface_inputs.ty)
    forcing.heat[step] = _surface_input_value(surface_inputs.heat)
    forcing.swr[step] = _surface_input_value(surface_inputs.swr)
    forcing.airp[step] = _surface_input_value(surface_inputs.airp)
    forcing.airt[step] = _surface_input_value(surface_inputs.airt)
    forcing.hum[step] = _surface_input_value(surface_inputs.hum)
    forcing.cloud[step] = _surface_input_value(surface_inputs.cloud)
    forcing.u10[step] = _surface_input_value(surface_inputs.u10)
    forcing.v10[step] = _surface_input_value(surface_inputs.v10)
    forcing.precip[step] = _surface_input_value(surface_inputs.precip)
    forcing.longwave[step] = _surface_input_value(surface_inputs.longwave)
    forcing.sst_obs[step] = _surface_input_optional_value(surface_inputs.sst_obs)
    forcing.sss_obs[step] = _surface_input_optional_value(surface_inputs.sss_obs)
    forcing.w_adv[step] = _surface_input_value(observations.w_adv_input)
    forcing.w_height[step] = _surface_input_value(observations.w_height_input)

    np.copyto(forcing.Tobs[step], run.meanflow.Tobs)
    np.copyto(forcing.Sobs[step], run.meanflow.Sobs)
    _copy_profile_data(observations.uprof_input, forcing.uprof[step])
    _copy_profile_data(observations.vprof_input, forcing.vprof[step])
    _copy_profile_data(observations.dtdx_input, forcing.dtdx[step])
    _copy_profile_data(observations.dtdy_input, forcing.dtdy[step])
    _copy_profile_data(observations.dsdx_input, forcing.dsdx[step])
    _copy_profile_data(observations.dsdy_input, forcing.dsdy[step])
    stokes = run.stokes_drift
    forcing.us0[step] = float(stokes.us0)
    forcing.vs0[step] = float(stokes.vs0)
    forcing.ds[step] = float(stokes.ds)
    if stokes.usprof is not None:
        np.copyto(forcing.us[step], stokes.usprof)
    if stokes.vsprof is not None:
        np.copyto(forcing.vs[step], stokes.vsprof)
    if stokes.dusdz is not None:
        np.copyto(forcing.dusdz[step], stokes.dusdz)
    if stokes.dvsdz is not None:
        np.copyto(forcing.dvsdz[step], stokes.dvsdz)


def _populate_runtime_forcing_from_run(
    run: Any,
    forcing: RuntimeForcing,
) -> None:
    stokes = run.stokes_drift
    stokes_active = _stokes_runtime_active(stokes)

    if forcing.nt == 0:
        _record_forcing_step(run, forcing, 0)
        forcing.validate()
        return

    _record_forcing_step(run, forcing, 0)
    for step in range(1, forcing.nt + 1):
        run.time.update_time(step)
        do_input(
            run.time.julianday,
            run.time.secondsofday,
            run.nlev,
            run.meanflow.z,
        )
        get_all_obs(
            run.observations,
            run.time.julianday,
            run.time.secondsofday,
            run.nlev,
            run.meanflow.z,
            fsecs=run.time.fsecs,
        )
        if stokes_active:
            do_stokes_drift(
                stokes,
                run.nlev,
                run.meanflow.z,
                run.meanflow.zi,
                run.meanflow.gravity,
                (
                    float(run.surface_inputs.u10.value)
                    if run.surface_inputs.u10 is not None
                    else 0.0
                ),
                (
                    float(run.surface_inputs.v10.value)
                    if run.surface_inputs.v10 is not None
                    else 0.0
                ),
            )
        _update_runtime_relaxation_targets(run)
        _record_forcing_step(run, forcing, step)

    run.time.update_time(0)
    forcing.validate()


def _populate_initial_runtime_forcing_from_run(
    run: Any,
    forcing: RuntimeForcing,
) -> None:
    _record_forcing_step(run, forcing, 0)
    forcing.validate()


def build_runtime_forcing_from_run(
    run: Any,
    *,
    max_steps: int | None = None,
) -> RuntimeForcing:
    """Precompute observation and surface forcing arrays from an initialized run."""

    if not bool(run.initialized):
        msg = "run has not been initialised"
        raise RuntimeError(msg)
    last_step = (
        int(run.time.MaxN) if max_steps is None else min(run.time.MaxN, max_steps)
    )
    forcing = build_runtime_forcing(int(run.nlev), int(last_step))
    _populate_runtime_forcing_from_run(run, forcing)
    return forcing


def build_runtime_from_run(
    run: Any,
    *,
    max_steps: int | None = None,
    output: bool = False,
) -> RuntimeBundle:
    """Copy an initialized GotmRun object graph into flat runtime containers."""

    if not bool(run.initialized):
        msg = "run has not been initialised"
        raise RuntimeError(msg)

    _validate_run_supported_by_compiled_runtime(run, output=output)

    last_step = (
        int(run.time.MaxN) if max_steps is None else min(run.time.MaxN, max_steps)
    )
    params = _make_runtime_params_from_run(run, int(last_step))
    output_every = int(run.output_schedule.interval_steps)
    bundle = build_runtime(
        params,
        output=output,
        output_every=output_every,
        force_final=max_steps is not None or last_step < output_every,
    )

    state = bundle.state
    _copy_profiles(
        run.meanflow,
        state,
        (
            "ga",
            "z",
            "zi",
            "h",
            "ho",
            "u",
            "uo",
            "v",
            "vo",
            "w",
            "T",
            "S",
            "Tp",
            "Sp",
            "Ti",
            "Tobs",
            "Sobs",
            "NN",
            "NNT",
            "NNS",
            "SS",
            "SSU",
            "SSV",
            "SSCSTK",
            "SSSTK",
            "buoy",
            "rad",
            "xP",
            "avh",
            "fric",
            "drag",
            "bioshade",
        ),
    )
    _copy_profiles(
        run.turbulence,
        state,
        (
            "tke",
            "tkeo",
            "eps",
            "omega",
            "L",
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
        ),
    )

    if run.density.alpha is not None:
        np.copyto(state.alpha, run.density.alpha)
    if run.density.beta is not None:
        np.copyto(state.beta, run.density.beta)
    if run.density.rho_p is not None:
        np.copyto(state.rho_p, run.density.rho_p)
    if run.density.rho is not None:
        np.copyto(state.rho, run.density.rho)

    state.z0b[0] = float(run.meanflow.z0b)
    state.z0s[0] = float(run.meanflow.z0s)
    state.za[0] = float(run.meanflow.za)
    state.u_taub[0] = float(run.meanflow.u_taub)
    state.u_taubo[0] = float(run.meanflow.u_taubo)
    state.u_taus[0] = float(run.meanflow.u_taus)
    state.taub[0] = float(run.meanflow.taub)
    state.tx[0] = params.tx
    state.ty[0] = params.ty
    ice_state = getattr(run, "ice_state", None)
    if ice_state is not None:
        for name in (
            "Hice",
            "Hsnow",
            "Hfrazil",
            "T1",
            "T2",
            "Tice_surface",
            "fdd",
            "ice_cover",
            "Tf",
            "albedo_ice",
            "transmissivity",
            "ocean_ice_flux",
            "ocean_ice_heat_flux",
            "ocean_ice_salt_flux",
            "surface_ice_energy",
            "bottom_ice_energy",
            "melt_rate",
            "T_melt",
            "S_melt",
        ):
            getattr(state, name)[0] = float(getattr(ice_state, name)[0])

    if run.stokes_drift.dusdz is not None:
        np.copyto(bundle.work.dusdz, run.stokes_drift.dusdz)
    if run.stokes_drift.dvsdz is not None:
        np.copyto(bundle.work.dvsdz, run.stokes_drift.dvsdz)

    bundle.work.vel_relax_tau[:] = float(run.observations.vel_relax_tau)
    bundle.work.vel_relax_tau_eff[:] = bundle.work.vel_relax_tau
    if run.observations.SRelaxTau is not None:
        np.copyto(bundle.work.s_relax_tau, run.observations.SRelaxTau)
    if run.observations.TRelaxTau is not None:
        np.copyto(bundle.work.t_relax_tau, run.observations.TRelaxTau)

    seagrass = run.seagrass
    if seagrass.seagrass_calc:
        for source, target in (
            (seagrass.grassz, bundle.work.seagrass_z),
            (seagrass.exc, bundle.work.seagrass_exc),
            (seagrass.vfric, bundle.work.seagrass_vfric),
            (seagrass.xx, bundle.work.seagrass_xx),
            (seagrass.yy, bundle.work.seagrass_yy),
            (seagrass.xxP, bundle.work.seagrass_xxP),
            (seagrass.excur, bundle.work.seagrass_excur),
            (seagrass.grassfric, bundle.work.seagrass_grassfric),
        ):
            if source is not None:
                target[: source.shape[0]] = source
    _populate_runtime_forcing_from_run(run, bundle.forcing)
    bundle.state.validate()
    bundle.work.validate()
    return bundle


_ICE_REFERENCE_SCALARS = (
    "Hfrazil",
    "Hice",
    "Tf",
    "Tice_surface",
    "bottom_ice_energy",
    "ocean_ice_flux",
    "ocean_ice_heat_flux",
    "ocean_ice_salt_flux",
    "surface_ice_energy",
)
_WINTON_REFERENCE_SCALARS = ("T1", "T2")
_FABM_REFERENCE_SCALARS = (
    "surface_albedo",
    "surface_drag_coefficient_in_air",
)
_JRC_MED_ERGOM_REFERENCE_SCALARS = (
    "jrc_med_ergom_DNB",
    "jrc_med_ergom_OFL",
    "jrc_med_ergom_PBR",
    "jrc_med_ergom_SBR",
    "jrc_med_ergom_fl",
    "jrc_med_ergom_pb",
)
_BSEM_REFERENCE_PROFILES = (
    "bsem_PAR",
    "bsem_PPR",
    "bsem_am",
    "bsem_dn",
    "bsem_hs",
    "bsem_ni",
    "bsem_o2",
    "bsem_pl",
    "bsem_ps",
    "bsem_zg",
    "bsem_zl",
    "bsem_zn",
    "bsem_zs",
)
_JRC_MED_ERGOM_REFERENCE_PROFILES = (
    "jrc_med_ergom_Amm",
    "jrc_med_ergom_DNP",
    "jrc_med_ergom_DO_mg",
    "jrc_med_ergom_GPP",
    "jrc_med_ergom_NCP",
    "jrc_med_ergom_NFX",
    "jrc_med_ergom_NPR",
    "jrc_med_ergom_Nit",
    "jrc_med_ergom_PAR",
    "jrc_med_ergom_PPR",
    "jrc_med_ergom_Pho",
    "jrc_med_ergom_TN",
    "jrc_med_ergom_TP",
    "jrc_med_ergom_aa",
    "jrc_med_ergom_bb",
    "jrc_med_ergom_bb_chla",
    "jrc_med_ergom_dd",
    "jrc_med_ergom_ff",
    "jrc_med_ergom_ff_chla",
    "jrc_med_ergom_nn",
    "jrc_med_ergom_o2",
    "jrc_med_ergom_po",
    "jrc_med_ergom_pp",
    "jrc_med_ergom_pp_chla",
    "jrc_med_ergom_pw",
    "jrc_med_ergom_tot_chla",
    "jrc_med_ergom_zz",
)
_NPZD_REFERENCE_PROFILES = (
    "npzd_NPR",
    "npzd_PAR",
    "npzd_PPR",
    "npzd_det",
    "npzd_nut",
    "npzd_phy",
    "npzd_zoo",
)


def _document_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _document_token(value: object, default: str) -> str:
    if value is None:
        return default
    return str(value).strip().lower().replace("-", "_")


def _fabm_yaml_text(run: Any) -> str:
    fabm_path = Path(run.yaml_path).with_name("fabm.yaml")
    if not fabm_path.exists():
        return ""
    return fabm_path.read_text(encoding="utf-8").lower()


def _reference_scalar_output_names(run: Any, output: RuntimeOutput) -> tuple[str, ...]:
    names: list[str] = []
    document = _document_mapping(run.document)
    surface = _document_mapping(document.get("surface"))
    ice = _document_mapping(surface.get("ice"))
    ice_model = _document_token(ice.get("model"), "simple")
    if ice_model in {"simple", "basal_melt", "lebedev", "mylake", "winton"}:
        names.extend(_ICE_REFERENCE_SCALARS)
        if ice_model == "winton":
            names.extend(_WINTON_REFERENCE_SCALARS)

    fabm = _document_mapping(document.get("fabm"))
    if bool(fabm.get("use", False)):
        names.extend(_FABM_REFERENCE_SCALARS)
        fabm_text = _fabm_yaml_text(run)
        if "jrc_med_ergom" in fabm_text:
            names.extend(_JRC_MED_ERGOM_REFERENCE_SCALARS)

    available = output.reference_scalars
    return tuple(name for name in names if name in available)


def _reference_z_profile_output_names(
    run: Any,
    output: RuntimeOutput,
) -> tuple[str, ...]:
    names: list[str] = []
    eps_input = getattr(run.observations, "epsprof_input", None)
    if (
        getattr(eps_input, "method", 0) != 0
        and getattr(eps_input, "data", None) is not None
    ):
        names.append("eps_obs")

    document = _document_mapping(run.document)
    fabm = _document_mapping(document.get("fabm"))
    if bool(fabm.get("use", False)):
        names.append("attenuation_coefficient_of_photosynthetic_radiative_flux")
        fabm_text = _fabm_yaml_text(run)
        if "bsem" in fabm_text:
            names.extend(_BSEM_REFERENCE_PROFILES)
            names.append("total_nitrogen")
        if "jrc_med_ergom" in fabm_text:
            names.extend(_JRC_MED_ERGOM_REFERENCE_PROFILES)
        if "npzd" in fabm_text:
            names.extend(_NPZD_REFERENCE_PROFILES)
            names.append("total_nitrogen")
        if "bb/passive" in fabm_text:
            names.append("sed_c")

    available = output.reference_z_profiles
    return tuple(name for name in names if name in available)


def runtime_output_to_dataset(run: Any, bundle: RuntimeBundle) -> xr.Dataset:
    """Convert dense compiled output buffers to an xarray dataset after a run."""

    output = bundle.output
    if not output.enabled:
        msg = "runtime output buffers are disabled"
        raise ValueError(msg)

    nlev = bundle.params.nlev
    output.validate(nlev)

    time = np.asarray(output.time, dtype=np.float64)
    time_attrs = {
        "long_name": "time",
        "units": f"seconds since {run.time.start}",
        "calendar": "standard",
    }

    z_start = min(max(int(getattr(run.output_schedule, "k_start", 1)), 1), nlev)
    zi_start = min(max(int(getattr(run.output_schedule, "k1_start", 1)) - 1, 0), nlev)

    z_profiles = output.z[:, z_start:]
    zi_profiles = output.zi[:, zi_start:]

    def z_profile(values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        return (
            ("time", "z", "lat", "lon"),
            np.asarray(values[:, z_start:], dtype=np.float64)[:, :, None, None],
        )

    def zi_profile(values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        return (
            ("time", "zi", "lat", "lon"),
            np.asarray(values[:, zi_start:], dtype=np.float64)[:, :, None, None],
        )

    def scalar(values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        return (
            ("time", "lat", "lon"),
            np.asarray(values, dtype=np.float64)[:, None, None],
        )

    def diagnostic_z_profile(values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        return (
            ("time", "z", "lat", "lon"),
            np.asarray(values, dtype=np.float64)[:, :, None, None],
        )

    coords: dict[str, Any] = {
        "time": ("time", time, time_attrs),
        "z": (("time", "z", "lat", "lon"), z_profiles[:, :, None, None]),
        "zi": (("time", "zi", "lat", "lon"), zi_profiles[:, :, None, None]),
        "lat": ("lat", np.asarray([float(run.latitude)], dtype=np.float64)),
        "lon": ("lon", np.asarray([float(run.longitude)], dtype=np.float64)),
    }

    data_vars: dict[str, Any] = {
        "rho_p": z_profile(output.rho_p),
        "zeta": scalar(output.zeta),
        "u_taus": scalar(output.u_taus),
        "u10": scalar(output.u10),
        "v10": scalar(output.v10),
        "airt": scalar(output.airt),
        "airp": scalar(output.airp),
        "hum": scalar(output.hum),
        "es": scalar(output.es),
        "ea": scalar(output.ea),
        "qs": scalar(output.qs),
        "qa": scalar(output.qa),
        "rhoa": scalar(output.rhoa),
        "cloud": scalar(output.cloud),
        "albedo": scalar(output.albedo),
        "precip": scalar(output.precip),
        "evap": scalar(output.evap),
        "int_precip": scalar(output.int_precip),
        "int_evap": scalar(output.int_evap),
        "int_swr": scalar(output.int_swr),
        "int_heat": scalar(output.int_heat),
        "int_total": scalar(output.int_total),
        "I_0": scalar(output.I_0),
        "qh": scalar(output.qh),
        "qe": scalar(output.qe),
        "ql": scalar(output.ql),
        "heat": scalar(output.heat),
        "tx": scalar(output.tx),
        "ty": scalar(output.ty),
        "sst": scalar(output.sst),
        "sst_obs": scalar(output.sst_obs),
        "sss": scalar(output.sss),
        "mld_surf": scalar(output.mld_surf),
        "u": z_profile(output.u),
        "v": z_profile(output.v),
        "temp": z_profile(output.T),
        "salt": z_profile(output.S),
        "temp_obs": z_profile(output.Tobs),
        "salt_obs": z_profile(output.Sobs),
        "u_obs": z_profile(output.u_obs),
        "v_obs": z_profile(output.v_obs),
        "idpdx": z_profile(output.idpdx),
        "idpdy": z_profile(output.idpdy),
        "tke": zi_profile(output.tke),
        "eps": zi_profile(output.eps),
        "num": zi_profile(output.num),
        "nuh": zi_profile(output.nuh),
        "h": z_profile(output.h),
        "xP": z_profile(output.xP),
        "fric": z_profile(output.fric),
        "drag": z_profile(output.drag),
        "avh": z_profile(output.avh),
        "bioshade": z_profile(output.bioshade),
        "ga": z_profile(output.ga),
        "uu": zi_profile(output.uu),
        "vv": zi_profile(output.vv),
        "ww": zi_profile(output.ww),
        "NN": zi_profile(output.NN),
        "NNT": zi_profile(output.NNT),
        "NNS": zi_profile(output.NNS),
        "buoy": z_profile(output.buoy),
        "SS": zi_profile(output.SS),
        "P": zi_profile(output.P),
        "G": zi_profile(output.B),
        "Pb": zi_profile(output.Pb),
        "kb": zi_profile(output.kb),
        "epsb": zi_profile(output.epsb),
        "L": zi_profile(output.L),
        "PSTK": zi_profile(output.PSTK),
        "cmue1": zi_profile(output.cmue1),
        "cmue2": zi_profile(output.cmue2),
        "gamu": zi_profile(output.gamu),
        "gamv": zi_profile(output.gamv),
        "gamh": zi_profile(output.gamh),
        "gams": zi_profile(output.gams),
        "Rig": zi_profile(output.Rig),
        "gamb": zi_profile(output.gamb),
        "gam": zi_profile(output.gam),
        "as": zi_profile(output.as_),
        "an": zi_profile(output.an),
        "at": zi_profile(output.at),
        "r": zi_profile(output.r),
        "taux": zi_profile(output.taux),
        "tauy": zi_profile(output.tauy),
        "u_taub": scalar(output.u_taub),
        "taub": scalar(output.taub),
        "mld_bott": scalar(output.mld_bott),
        "rad": zi_profile(output.rad),
        "us": z_profile(output.us),
        "vs": z_profile(output.vs),
        "dusdz": zi_profile(output.dusdz),
        "dvsdz": zi_profile(output.dvsdz),
        "us0": scalar(output.us0),
        "vs0": scalar(output.vs0),
        "ds": scalar(output.ds),
        "Ekin": scalar(output.Ekin),
        "Epot": scalar(output.Epot),
        "Eturb": scalar(output.Eturb),
        "nus": zi_profile(output.nus),
        "nucl": zi_profile(output.nucl),
    }
    for name in _reference_scalar_output_names(run, output):
        data_vars[name] = scalar(output.reference_scalars[name])
    for name in _reference_z_profile_output_names(run, output):
        data_vars[name] = z_profile(output.reference_z_profiles[name])

    if int(bundle.params.density_method) == 1:
        conservative_temperature = np.asarray(output.T[:, z_start:], dtype=np.float64)
        absolute_salinity = np.asarray(output.S[:, z_start:], dtype=np.float64)
        pressure = np.asarray(-z_profiles, dtype=np.float64)
        data_vars.update(
            {
                "rho": z_profile(output.rho),
                "temp_p": diagnostic_z_profile(
                    gsw.pt_from_CT(absolute_salinity, conservative_temperature)
                ),
                "temp_i": diagnostic_z_profile(
                    gsw.t_from_CT(
                        absolute_salinity,
                        conservative_temperature,
                        pressure,
                    )
                ),
                "salt_p": diagnostic_z_profile(
                    gsw.SP_from_SA(
                        absolute_salinity,
                        pressure,
                        float(run.longitude),
                        float(run.latitude),
                    )
                ),
            }
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "title": str(run.settings.title),
            "source_yaml": str(run.yaml_path),
            "nlev": int(nlev),
            "dt": float(bundle.params.dt),
            "runtime": "compiled",
        },
    )
