"""Convert compiled runtime output buffers into an xarray Dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gsw
import numpy as np
import xarray as xr

from pygotm.gotm.runtime_core import (
    RuntimeBundle,
)
from pygotm.gotm.runtime_output import RuntimeOutput
from pygotm.util.density import (
    METHOD_JACKETT_FULL,
    METHOD_JACKETT_POTENTIAL,
    METHOD_TEOS10,
    METHOD_UNESCO_FULL,
    METHOD_UNESCO_POTENTIAL,
)
from pygotm.util.gsw import gsw_sp_from_sa

_ICE_EXTRA_SCALARS = (
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
_WINTON_EXTRA_SCALARS = ("T1", "T2")
_FABM_FEEDBACK_EXTRA_SCALARS = (
    "surface_albedo",
    "surface_drag_coefficient_in_air",
)


def _document_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _document_token(value: object, default: str) -> str:
    if value is None:
        return default
    return str(value).strip().lower().replace("-", "_")


def _extra_scalar_output_names(run: Any, output: RuntimeOutput) -> tuple[str, ...]:
    names: list[str] = []
    document = _document_mapping(run.document)
    surface = _document_mapping(document.get("surface"))
    ice = _document_mapping(surface.get("ice"))
    ice_model = _document_token(ice.get("model"), "simple")
    if ice_model in {"simple", "basal_melt", "lebedev", "mylake", "winton"}:
        names.extend(_ICE_EXTRA_SCALARS)
        if ice_model == "winton":
            names.extend(_WINTON_EXTRA_SCALARS)

    fabm = _document_mapping(document.get("fabm"))
    if bool(fabm.get("use", False)):
        names.extend(_FABM_FEEDBACK_EXTRA_SCALARS)

    available = output.extra_scalars
    return tuple(name for name in names if name in available)


def _extra_z_profile_output_names(
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

    available = output.extra_z_profiles
    return tuple(name for name in names if name in available)


def runtime_output_to_dataset(
    run: Any,
    bundle: RuntimeBundle,
    *,
    attrs: Mapping[str, str | int | float] | None = None,
) -> xr.Dataset:
    """Convert dense compiled output buffers to an xarray dataset after a run."""

    output = bundle.output
    if not output.enabled:
        msg = "runtime output buffers are disabled"
        raise ValueError(msg)

    nlev = bundle.params.nlev
    output.validate(nlev)

    time_method = str(getattr(run.output_schedule, "time_method", "point"))
    interval_steps = max(int(getattr(run.output_schedule, "interval_steps", 1)), 1)
    nt = int(bundle.params.nt)

    if time_method in {"mean", "integrated"} and output.output_every == 1:
        steps: list[int] = [0]
        windows: list[tuple[int, int]] = [(0, 0)]
        previous = 0
        for step in range(interval_steps, nt + 1, interval_steps):
            steps.append(step)
            windows.append((previous + 1, step))
            previous = step
        if output.force_final and previous < nt:
            steps.append(nt)
            windows.append((previous + 1, nt))

        step_indices = np.asarray(steps, dtype=np.int64)

        def output_values(values: np.ndarray) -> np.ndarray:
            raw = np.asarray(values, dtype=np.float64)
            reduced = np.empty((len(windows),) + raw.shape[1:], dtype=np.float64)
            reduced[0] = raw[0]
            for i, (start, stop) in enumerate(windows[1:], start=1):
                window = raw[start : stop + 1]
                if time_method == "mean":
                    reduced[i] = np.mean(window, axis=0)
                else:
                    reduced[i] = np.sum(window, axis=0)
            return reduced

        time = np.asarray(output.time[step_indices], dtype=np.float64)
    else:

        def output_values(values: np.ndarray) -> np.ndarray:
            return np.asarray(values, dtype=np.float64)

        time = np.asarray(output.time, dtype=np.float64)

    def output_window_values(values: np.ndarray) -> np.ndarray:
        raw = np.asarray(values, dtype=np.float64)
        if time_method not in {"mean", "integrated"}:
            return output_values(raw)
        steps = np.asarray(output.output_step, dtype=np.int64)
        reduced = np.empty((steps.size,) + raw.shape[1:], dtype=np.float64)
        if steps.size == 0:
            return reduced
        first_step = min(max(int(steps[0]), 0), raw.shape[0] - 1)
        reduced[0] = raw[first_step]
        previous = int(steps[0])
        for i in range(1, steps.size):
            step = int(steps[i])
            if step < 0:
                reduced[i] = np.nan
                continue
            start = max(previous + 1, 0)
            stop = min(step, raw.shape[0] - 1)
            if stop < start:
                reduced[i] = raw[min(max(step, 0), raw.shape[0] - 1)]
            else:
                window = raw[start : stop + 1]
                if time_method == "mean":
                    reduced[i] = np.mean(window, axis=0)
                else:
                    reduced[i] = np.sum(window, axis=0)
            previous = step
        return reduced

    time_attrs = {
        "long_name": "time",
        "units": f"seconds since {run.time.start}",
        "calendar": "standard",
    }

    z_start = min(max(int(getattr(run.output_schedule, "k_start", 1)), 1), nlev)
    zi_start = min(max(int(getattr(run.output_schedule, "k1_start", 1)) - 1, 0), nlev)

    z_profiles = output_values(output.z)[:, z_start:]
    zi_profiles = output_values(output.zi)[:, zi_start:]

    def z_profile(
        values: np.ndarray,
        var_attrs: Mapping[str, str] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
        return (
            ("time", "z", "lat", "lon"),
            output_values(values)[:, z_start:][:, :, None, None],
            dict(var_attrs or {}),
        )

    def zi_profile(values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        return (
            ("time", "zi", "lat", "lon"),
            output_values(values)[:, zi_start:][:, :, None, None],
        )

    def scalar(
        values: np.ndarray,
        var_attrs: Mapping[str, str] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
        return (
            ("time", "lat", "lon"),
            output_values(values)[:, None, None],
            dict(var_attrs or {}),
        )

    def scalar_series(
        values: np.ndarray,
        var_attrs: Mapping[str, str] | None = None,
    ) -> tuple[tuple[str, ...], np.ndarray, dict[str, str]]:
        return (
            ("time", "lat", "lon"),
            np.asarray(values, dtype=np.float64)[:, None, None],
            dict(var_attrs or {}),
        )

    def diagnostic_z_profile(values: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
        return (
            ("time", "z", "lat", "lon"),
            output_values(values)[:, :, None, None],
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
    if int(bundle.params.ice_model) != 0:
        data_vars["dHis"] = scalar(output.dHis)
        data_vars["dHib"] = scalar(output.dHib)
    if int(bundle.params.lake) != 0:
        data_vars.update(
            {
                "qlobs": scalar(output.qlobs),
                "int_flow": scalar(output.int_flow),
                "int_water_balance": scalar(output.int_water_balance),
                "int_inflow": scalar(output.int_inflow),
                "int_outflow": scalar(output.int_outflow),
                "Af": z_profile(output.Af),
                "Qlayer": z_profile(output.Qlayer),
                "Qs": z_profile(output.Qs),
                "Qt": z_profile(output.Qt),
                "wq": z_profile(output.wq),
                "FQ": z_profile(output.FQ),
                "Qres": z_profile(output.Qres),
                "xRf": zi_profile(output.xRf),
            }
        )
    if int(bundle.params.nstreams) > 0:
        data_vars["Q_Kristine"] = scalar(output.Q_Kristine)
        data_vars["T_Kristine"] = scalar(output.T_Kristine)
    if int(bundle.params.nstreams) > 1:
        data_vars["Q_Unguaged"] = scalar(output.Q_Unguaged)
        data_vars["T_Unguaged"] = scalar(output.T_Unguaged)
    if int(bundle.params.nstreams) > 2:
        data_vars["Q_Stensta"] = scalar(output.Q_Stensta)
    for name in _extra_scalar_output_names(run, output):
        data_vars[name] = scalar(output.extra_scalars[name])
    for name in _extra_z_profile_output_names(run, output):
        data_vars[name] = z_profile(output.extra_z_profiles[name])
    for name, values in output.fabm_scalars.items():
        if name in data_vars:
            msg = f"FABM output {name!r} collides with an existing output variable"
            raise ValueError(msg)
        data_vars[name] = scalar(values, output.fabm_attrs.get(name))
    for name, values in output.fabm_z_profiles.items():
        if name in data_vars:
            msg = f"FABM output {name!r} collides with an existing output variable"
            raise ValueError(msg)
        data_vars[name] = z_profile(values, output.fabm_attrs.get(name))

    if int(bundle.params.lake) != 0:
        for name in (
            "selmaprotbas_fl_c",
            "selmaprotbas_fl_p",
            "selmaprotbas_fl_n",
            "selmaprotbas_fl_si",
            "selmaprotbas_pb",
        ):
            if name in output.fabm_scalars:
                values = output_values(output.fabm_scalars[name])[:, None, None, None]
                data_vars[name] = (
                    ("time", "z", "lat", "lon"),
                    np.broadcast_to(
                        values,
                        (values.shape[0], z_profiles.shape[1], 1, 1),
                    ).copy(),
                    dict(output.fabm_attrs.get(name) or {}),
                )

    calculator_aliases = [
        ("total_silica_calculator_result", "total_silicon"),
        ("total_carbon_calculator_result", "total_carbon"),
        ("total_phosphorus_calculator_result", "total_phosphorus"),
        ("total_nitrogen_calculator_result", "total_nitrogen"),
        ("total_chlorophyll_calculator_result", "total_chlorophyll"),
        (
            "total_phosphorus_at_interfaces_calculator_result",
            "total_phosphorus_at_interfaces",
        ),
    ]
    if int(bundle.params.lake) != 0:
        calculator_aliases.append(
            (
                "attenuation_coefficient_of_photosynthetic_radiative_flux_calculator_result",
                "attenuation_coefficient_of_photosynthetic_radiative_flux",
            )
        )
    for target, source in calculator_aliases:
        if target not in data_vars and source in data_vars:
            data_vars[target] = data_vars[source]

    if int(bundle.params.lake) != 0:
        data_vars["SS"] = z_profile(output.SS)
        data_vars["NN"] = z_profile(output.NN)
        data_vars["NNT"] = z_profile(output.NNT)
        data_vars["NNS"] = z_profile(output.NNS)

    if time_method == "mean":
        forcing = bundle.forcing
        for name, values in (
            ("u10", forcing.u10),
            ("v10", forcing.v10),
            ("airt", forcing.airt),
            ("airp", forcing.airp),
            ("hum", forcing.hum),
            ("cloud", forcing.cloud),
            ("precip", forcing.precip),
            ("us0", forcing.us0),
            ("vs0", forcing.vs0),
            ("ds", forcing.ds),
        ):
            data_vars[name] = scalar_series(output_window_values(values))
        data_vars["sst_obs"] = scalar_series(
            np.nan_to_num(output_window_values(forcing.sst_obs), nan=0.0)
        )
        data_vars["sss"] = scalar_series(
            np.nan_to_num(output_window_values(forcing.sss_obs), nan=0.0)
        )
        if int(bundle.params.nstreams) > 0:
            data_vars["Q_Kristine"] = scalar_series(
                output_window_values(forcing.stream_flow[:, 0])
            )
            data_vars["T_Kristine"] = scalar_series(
                output_window_values(forcing.stream_temp[:, 0])
            )
        if int(bundle.params.nstreams) > 1:
            data_vars["Q_Unguaged"] = scalar_series(
                output_window_values(forcing.stream_flow[:, 1])
            )
            data_vars["T_Unguaged"] = scalar_series(
                output_window_values(forcing.stream_temp[:, 1])
            )
        if int(bundle.params.nstreams) > 2:
            data_vars["Q_Stensta"] = scalar_series(
                output_window_values(forcing.stream_flow[:, 2])
            )

    density_method = int(bundle.params.density_method)
    if density_method in {
        METHOD_TEOS10,
        METHOD_UNESCO_FULL,
        METHOD_UNESCO_POTENTIAL,
        METHOD_JACKETT_FULL,
        METHOD_JACKETT_POTENTIAL,
    }:
        data_vars["rho"] = z_profile(output.rho)

    if density_method == METHOD_TEOS10:
        conservative_temperature = output_values(output.T)[:, z_start:]
        absolute_salinity = output_values(output.S)[:, z_start:]
        pressure = np.asarray(-z_profiles, dtype=np.float64)
        data_vars.update(
            {
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
                    np.asarray(
                        gsw_sp_from_sa(
                            absolute_salinity,
                            pressure,
                            float(run.longitude),
                            float(run.latitude),
                        ),
                        dtype=np.float64,
                    )
                ),
            }
        )

    dataset_attrs = (
        dict(attrs)
        if attrs is not None
        else {
            "title": str(run.settings.title),
            "source_yaml": str(run.yaml_path),
            "nlev": int(nlev),
            "dt": float(bundle.params.dt),
            "runtime": "compiled",
        }
    )

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=dataset_attrs,
    )
