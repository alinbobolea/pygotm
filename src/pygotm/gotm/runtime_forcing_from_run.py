"""Populate compiled runtime forcing arrays from a high-level GotmRun."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gsw
import numpy as np

from pygotm.gotm.runtime_core import (
    build_runtime_forcing,
)
from pygotm.gotm.runtime_forcing import RuntimeForcing
from pygotm.gotm.runtime_params_from_run import (
    _run_uses_ice_zeta,
    _stokes_runtime_active,
    _surface_input_optional_value,
    _surface_input_value,
)
from pygotm.input.input import do_input
from pygotm.meanflow.updategrid import updategrid
from pygotm.observations.observations import get_all_obs
from pygotm.stokes_drift.stokes_drift import do_stokes_drift
from pygotm.util.gsw import gsw_sa_from_sp, gsw_saar
from pygotm.util.gsw.modules.gsw_mod_teos10_constants import gsw_sso, gsw_ups
from pygotm.util.gsw.toolbox.gsw_sa_from_sp import _is_baltic


def _copy_profile_data(input_: Any | None, target: np.ndarray) -> None:
    if input_ is not None and isinstance(getattr(input_, "data", None), np.ndarray):
        np.copyto(target, input_.data)
    else:
        target.fill(0.0)


def _copy_stream_static(run: Any, forcing: RuntimeForcing) -> None:
    streams = getattr(run, "streams", None)
    if streams is None or forcing.nstreams == 0:
        return
    for source_name, target_name in (
        ("methods", "stream_method"),
        ("has_T", "stream_has_T"),
        ("has_S", "stream_has_S"),
        ("zl", "stream_zl"),
        ("zu", "stream_zu"),
    ):
        source = getattr(streams, source_name, None)
        if source is not None:
            np.copyto(getattr(forcing, target_name), source)
    forcing.stream_concentrations.clear()
    forcing.stream_concentration_masks.clear()
    for name in getattr(streams, "concentration_names", ()):
        values = getattr(streams, "concentration_values", {}).get(name)
        mask = getattr(streams, "concentration_has", {}).get(name)
        if values is None or mask is None:
            continue
        forcing.stream_concentrations[name] = np.zeros(
            (forcing.nt + 1, forcing.nstreams),
            dtype=np.float64,
        )
        forcing.stream_concentration_masks[name] = np.ascontiguousarray(
            mask,
            dtype=np.int64,
        )


@dataclass(slots=True)
class _SalinityConversionCache:
    pressure: np.ndarray
    saar_factor: np.ndarray
    pressure_valid: bool
    baltic: bool


def _make_salinity_conversion_cache(
    nlev: int,
    longitude: float,
    latitude: float,
) -> _SalinityConversionCache:
    return _SalinityConversionCache(
        pressure=np.empty(nlev, dtype=np.float64),
        saar_factor=np.empty(nlev, dtype=np.float64),
        pressure_valid=False,
        baltic=_is_baltic(float(longitude), float(latitude)),
    )


def _copy_absolute_salinity_from_practical(
    practical_salinity: np.ndarray,
    pressure: np.ndarray,
    longitude: float,
    latitude: float,
    target: np.ndarray,
    cache: _SalinityConversionCache | None,
) -> None:
    if cache is None:
        np.copyto(
            target,
            np.asarray(
                gsw_sa_from_sp(practical_salinity, pressure, longitude, latitude),
                dtype=np.float64,
            ),
        )
        return

    if cache.baltic:
        np.multiply((gsw_sso - 0.087) / 35.0, practical_salinity, out=target)
        target += 0.087
        return

    if not cache.pressure_valid or not np.array_equal(cache.pressure, pressure):
        np.copyto(cache.pressure, pressure)
        np.copyto(
            cache.saar_factor,
            1.0 + np.asarray(gsw_saar(cache.pressure, longitude, latitude)),
        )
        cache.pressure_valid = True

    np.multiply(gsw_ups, practical_salinity, out=target)
    target *= cache.saar_factor


def _update_runtime_relaxation_targets(
    run: Any,
    salinity_cache: _SalinityConversionCache | None = None,
) -> None:
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
        if bool(getattr(meanflow, "lake", False)):
            meanflow.Sobs[1 : run.nlev + 1] = observations.sprof_input.data[
                1 : run.nlev + 1
            ]
        elif observations.initial_salinity_type == 1:
            _copy_absolute_salinity_from_practical(
                observations.sprof_input.data[1 : run.nlev + 1],
                pressure,
                float(run.longitude),
                float(run.latitude),
                meanflow.Sobs[1 : run.nlev + 1],
                salinity_cache,
            )
        else:
            meanflow.Sobs[1 : run.nlev + 1] = observations.sprof_input.data[
                1 : run.nlev + 1
            ]

    if (
        observations.tprof_input.method != 0
        and observations.tprof_input.data is not None
    ):
        if bool(getattr(meanflow, "lake", False)):
            meanflow.Tobs[1 : run.nlev + 1] = observations.tprof_input.data[
                1 : run.nlev + 1
            ]
        elif observations.initial_temperature_type == 1:
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
    forcing.zeta[step] = (
        float(run.meanflow.zeta)
        if _run_uses_ice_zeta(run)
        else _surface_input_value(observations.zeta_input)
    )
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

    np.copyto(forcing.h[step], run.meanflow.h)
    np.copyto(forcing.ho[step], run.meanflow.ho)
    np.copyto(forcing.Vc[step], run.meanflow.Vc)
    np.copyto(forcing.Vco[step], run.meanflow.Vco)
    np.copyto(forcing.Af[step], run.meanflow.Af)
    np.copyto(forcing.Afo[step], run.meanflow.Afo)
    np.copyto(forcing.z[step], run.meanflow.z)
    np.copyto(forcing.zi[step], run.meanflow.zi)

    streams = getattr(run, "streams", None)
    if streams is not None and forcing.nstreams != 0:
        streams.update_values_from_inputs()
        if streams.flow_values is not None:
            np.copyto(forcing.stream_flow[step], streams.flow_values)
        if streams.temp_values is not None:
            np.copyto(forcing.stream_temp[step], streams.temp_values)
        if streams.salt_values is not None:
            np.copyto(forcing.stream_salt[step], streams.salt_values)
        for name, values in getattr(streams, "concentration_values", {}).items():
            target = forcing.stream_concentrations.get(name)
            if target is not None:
                np.copyto(target[step], values)

    np.copyto(forcing.Tobs[step], run.meanflow.Tobs)
    np.copyto(forcing.Sobs[step], run.meanflow.Sobs)
    _copy_profile_data(observations.tprof_input, forcing.Tprof[step])
    _copy_profile_data(observations.sprof_input, forcing.Sprof[step])
    if bool(getattr(run.meanflow, "lake", False)):
        if observations.tprof_input.method == 0:
            forcing.Tprof[step, :] = 0.0
        if observations.sprof_input.method == 0:
            forcing.Sprof[step, :] = 0.0
    _copy_profile_data(observations.epsprof_input, forcing.epsprof[step])
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
    forcing.light_A[step] = _surface_input_value(observations.A_input)
    forcing.light_g1[step] = _surface_input_value(observations.g1_input)
    forcing.light_g2[step] = _surface_input_value(observations.g2_input)
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
    meanflow = run.meanflow

    if forcing.nt == 0:
        _copy_stream_static(run, forcing)
        _record_forcing_step(run, forcing, 0)
        forcing.validate()
        return

    initial_zeta = float(meanflow.zeta)
    initial_grid = (
        np.array(meanflow.h, copy=True),
        np.array(meanflow.ho, copy=True),
        np.array(meanflow.Vc, copy=True),
        np.array(meanflow.Vco, copy=True),
        np.array(meanflow.Af, copy=True),
        np.array(meanflow.Afo, copy=True),
        np.array(meanflow.z, copy=True),
        np.array(meanflow.zi, copy=True),
    )
    salinity_cache = _make_salinity_conversion_cache(
        int(run.nlev),
        float(run.longitude),
        float(run.latitude),
    )
    _copy_stream_static(run, forcing)
    _record_forcing_step(run, forcing, 0)
    try:
        for step in range(1, forcing.nt + 1):
            run.time.update_time(step)
            do_input(
                run.time.julianday,
                run.time.secondsofday,
                run.nlev,
                meanflow.z,
            )
            get_all_obs(
                run.observations,
                run.time.julianday,
                run.time.secondsofday,
                run.nlev,
                meanflow.z,
                fsecs=run.time.fsecs,
            )
            if stokes_active:
                do_stokes_drift(
                    stokes,
                    run.nlev,
                    meanflow.z,
                    meanflow.zi,
                    meanflow.gravity,
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
            if not _run_uses_ice_zeta(run):
                meanflow.zeta = float(run.observations.zeta_input.value)
            updategrid(meanflow, run.nlev, run.dt, meanflow.zeta)
            _update_runtime_relaxation_targets(run, salinity_cache)
            _record_forcing_step(run, forcing, step)
    finally:
        meanflow.zeta = initial_zeta
        np.copyto(meanflow.h, initial_grid[0])
        np.copyto(meanflow.ho, initial_grid[1])
        np.copyto(meanflow.Vc, initial_grid[2])
        np.copyto(meanflow.Vco, initial_grid[3])
        np.copyto(meanflow.Af, initial_grid[4])
        np.copyto(meanflow.Afo, initial_grid[5])
        np.copyto(meanflow.z, initial_grid[6])
        np.copyto(meanflow.zi, initial_grid[7])
        run.time.update_time(0)

    forcing.validate()


def _populate_initial_runtime_forcing_from_run(
    run: Any,
    forcing: RuntimeForcing,
) -> None:
    _copy_stream_static(run, forcing)
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
    nstreams = int(getattr(getattr(run, "streams", None), "nstreams", 0))
    forcing = build_runtime_forcing(
        int(run.nlev),
        int(last_step),
        nstreams=nstreams,
    )
    _populate_runtime_forcing_from_run(run, forcing)
    return forcing
