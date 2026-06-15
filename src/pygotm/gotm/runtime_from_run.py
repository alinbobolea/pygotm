"""Assemble a full compiled RuntimeBundle from a high-level GotmRun."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from pygotm.gotm.runtime_core import (
    RuntimeBundle,
    RuntimePhaseTimings,
    _copy_profiles,
    _output_reduce_mode,
    build_runtime,
)
from pygotm.gotm.runtime_forcing_from_run import (
    _populate_runtime_forcing_from_run,
)
from pygotm.gotm.runtime_params_from_run import (
    _make_runtime_params_from_run,
    _validate_run_supported_by_compiled_runtime,
)


def build_runtime_from_run(
    run: Any,
    *,
    max_steps: int | None = None,
    output: bool = False,
    timings: RuntimePhaseTimings | None = None,
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
    requested_output_every = int(run.output_schedule.interval_steps)
    output_every = requested_output_every
    bundle = build_runtime(
        params,
        output=output,
        output_every=output_every,
        force_final=max_steps is not None or last_step < requested_output_every,
        output_reduce_mode=_output_reduce_mode(run.output_schedule.time_method),
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
            "Vc",
            "Vco",
            "Af",
            "Afo",
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

    _copy_profiles(
        run.observations,
        state,
        (
            "Qs",
            "Qt",
            "Ls",
            "Lt",
            "Qlayer",
            "Qres",
            "FQ",
            "wq",
        ),
    )

    state.z0b[0] = float(run.meanflow.z0b)
    state.z0s[0] = float(run.meanflow.z0s)
    state.za[0] = float(run.meanflow.za)
    state.u_taub[0] = float(run.meanflow.u_taub)
    state.u_taubo[0] = float(run.meanflow.u_taubo)
    state.u_taus[0] = float(run.meanflow.u_taus)
    state.taub[0] = float(run.meanflow.taub)
    state.tx[0] = params.tx
    state.ty[0] = params.ty
    state.net_water_balance[0] = float(run.meanflow.net_water_balance)
    state.int_water_balance[0] = float(run.meanflow.int_water_balance)
    state.int_fwf[0] = float(run.meanflow.int_fwf)
    state.int_flows[0] = float(run.meanflow.int_flows)
    state.stream_int_inflow[0] = float(getattr(run.streams, "int_inflow", 0.0))
    state.stream_int_outflow[0] = float(getattr(run.streams, "int_outflow", 0.0))
    ice_state = getattr(run, "ice_state", None)
    if ice_state is not None:
        for name in (
            "Hice",
            "Hsnow",
            "Hfrazil",
            "dHis",
            "dHib",
            "T1",
            "T2",
            "Tice_surface",
            "fdd",
            "ice_cover",
            "Tf",
            "albedo_ice",
            "attenuation_ice",
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
    force_t0 = time.perf_counter()
    try:
        _populate_runtime_forcing_from_run(run, bundle.forcing)
    finally:
        if timings is not None:
            timings.force_build_s += time.perf_counter() - force_t0
    bundle.state.validate()
    bundle.work.validate()
    return bundle
