"""NetCDF comparison utilities for pyGOTM validation using Frechet distance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr

from pygotm.validate import numeric_variable_names, open_validation_dataset
from pygotm.validation.frechet import frechet_raw_and_normalized
from pygotm.validation.tolerances import (
    DEFAULT_FRECHET_CONFIG,
    FrechetConfig,
    classify_section,
)

__all__ = ["VarResult", "ValidationError", "compare_nc"]

FloatArray = npt.NDArray[np.float64]
Section = Literal["pygotm", "pyfabm"]
Status = Literal["PASS", "MARGINAL", "DISCREPANT", "BROKEN"]
Color = Literal["green", "yellow", "orange", "red"]
MetricMode = Literal["d_norm", "d_rel"]

_TIME_DIM_NAMES = {"time", "t"}


class ValidationError(Exception):
    """Raised for unrecoverable comparison failures."""


@dataclass
class VarResult:
    """Per-variable result for Frechet validation."""

    name: str
    section: Section

    status: Status
    color: Color

    reference_at_worst: float
    calculated_at_worst: float

    d_raw: float
    d_norm: float

    plot_html: str | None
    metric_mode: MetricMode = "d_norm"
    score: float | None = None

    @property
    def primary_score(self) -> float:
        """Compatibility alias for report rendering and old JSON payloads."""

        return self.d_norm if self.score is None else self.score


def _classify_status(score: float, config: FrechetConfig) -> tuple[Status, Color]:
    if not np.isfinite(score):
        return "BROKEN", "red"
    if score < config.pass_tol:
        return "PASS", "green"
    if score < config.marginal_tol:
        return "MARGINAL", "yellow"
    if score < config.discrepant_tol:
        return "DISCREPANT", "orange"
    return "BROKEN", "red"


def _make_plot_html(
    ref_arr: FloatArray,
    calc_arr: FloatArray,
    *,
    case_name: str,
    section: str,
    variable: str,
    status: str,
    d_raw: float,
    d_norm: float,
    score: float,
    metric_mode: str,
) -> str:
    section_label = "PyGOTM" if section == "pygotm" else "PyFABM"
    title = (
        f"Case: {case_name} | Section: {section_label} | Variable: {variable}<br>"
        f"Status: {status} | Raw Frechet: {d_raw:.3e} | "
        f"Normalized Frechet: {d_norm:.3e} | "
        f"Score: {score:.3e} ({metric_mode})"
    )
    x = list(range(len(ref_arr.ravel())))
    fig = go.Figure(
        [
            go.Scatter(x=x, y=ref_arr.ravel().tolist(), mode="lines", name="Reference"),
            go.Scatter(
                x=x,
                y=calc_arr.ravel().tolist(),
                mode="lines",
                name="Calculated",
            ),
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Aligned sample index",
        yaxis_title=variable,
        height=300,
        hovermode="x unified",
    )
    return str(pio.to_html(fig, include_plotlyjs=False, full_html=False))


def _broken_result(name: str) -> VarResult:
    section = classify_section(name)
    return VarResult(
        name=name,
        section=section,
        status="BROKEN",
        color="red",
        reference_at_worst=float("nan"),
        calculated_at_worst=float("nan"),
        d_raw=float("inf"),
        d_norm=float("inf"),
        plot_html=None,
    )


def _time_dim(data_array: xr.DataArray) -> str | None:
    for dim in data_array.dims:
        if str(dim).lower() in _TIME_DIM_NAMES:
            return str(dim)
    return None


def _time_values(data_array: xr.DataArray, dim: str) -> FloatArray:
    if dim in data_array.coords:
        values = np.asarray(data_array.coords[dim].values)
    else:
        values = np.arange(data_array.sizes[dim], dtype=np.float64)
    if np.issubdtype(values.dtype, np.datetime64):
        values = values.astype("datetime64[ns]").astype(np.int64)
    elif np.issubdtype(values.dtype, np.timedelta64):
        values = values.astype("timedelta64[ns]").astype(np.int64)
    return np.asarray(values, dtype=np.float64)


def _sort_unique_time(
    times: FloatArray,
    values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    finite = np.isfinite(times)
    times = times[finite]
    values = np.take(values, np.flatnonzero(finite), axis=0)
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_values = np.take(values, order, axis=0)
    unique_times, unique_indices = np.unique(sorted_times, return_index=True)
    unique_values = np.take(sorted_values, unique_indices, axis=0)
    return unique_times, unique_values


def _interp_time_axis(
    values: FloatArray,
    source_times: FloatArray,
    target_times: FloatArray,
    axis: int,
) -> FloatArray:
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    times, moved = _sort_unique_time(source_times, moved)
    if times.size == 0:
        return np.full_like(np.moveaxis(moved, 0, axis), np.nan, dtype=np.float64)

    flat = moved.reshape((moved.shape[0], -1))
    out = np.full((target_times.size, flat.shape[1]), np.nan, dtype=np.float64)
    for col in range(flat.shape[1]):
        series = flat[:, col]
        valid = np.isfinite(series)
        if not np.any(valid):
            continue
        xp = times[valid]
        fp = series[valid]
        if xp.size == 1:
            out[np.isclose(target_times, xp[0]), col] = fp[0]
            continue
        out[:, col] = np.interp(target_times, xp, fp, left=np.nan, right=np.nan)

    reshaped = out.reshape((target_times.size, *moved.shape[1:]))
    return np.moveaxis(reshaped, 0, axis)


def _aligned_arrays(
    ref_da: xr.DataArray, calc_da: xr.DataArray
) -> tuple[FloatArray, FloatArray]:
    ref_values = np.asarray(ref_da.values, dtype=np.float64)
    calc_values = np.asarray(calc_da.values, dtype=np.float64)
    ref_time_dim = _time_dim(ref_da)
    calc_time_dim = _time_dim(calc_da)

    if ref_time_dim is None and calc_time_dim is None:
        if ref_values.shape != calc_values.shape:
            msg = "shape mismatch for non-time variable"
            raise ValidationError(msg)
        return ref_values.ravel(), calc_values.ravel()

    if ref_time_dim is None or calc_time_dim is None:
        msg = "time dimension mismatch"
        raise ValidationError(msg)

    ref_axis = ref_da.get_axis_num(ref_time_dim)
    calc_axis = calc_da.get_axis_num(calc_time_dim)
    ref_other_shape = ref_values.shape[:ref_axis] + ref_values.shape[ref_axis + 1 :]
    calc_other_shape = (
        calc_values.shape[:calc_axis] + calc_values.shape[calc_axis + 1 :]
    )
    if ref_other_shape != calc_other_shape:
        msg = "non-time dimension shape mismatch"
        raise ValidationError(msg)

    ref_times = _time_values(ref_da, ref_time_dim)
    calc_times = _time_values(calc_da, calc_time_dim)
    finite_times = np.concatenate(
        [ref_times[np.isfinite(ref_times)], calc_times[np.isfinite(calc_times)]]
    )
    if finite_times.size == 0:
        msg = "no finite time coordinates"
        raise ValidationError(msg)
    union_times = np.unique(finite_times)

    ref_interp = _interp_time_axis(ref_values, ref_times, union_times, ref_axis)
    calc_interp = _interp_time_axis(calc_values, calc_times, union_times, calc_axis)
    ref_flat = np.moveaxis(ref_interp, ref_axis, 0).ravel()
    calc_flat = np.moveaxis(calc_interp, calc_axis, 0).ravel()
    return ref_flat, calc_flat


def _at_worst(ref_arr: FloatArray, calc_arr: FloatArray) -> tuple[float, float]:
    valid = np.isfinite(ref_arr) & np.isfinite(calc_arr)
    if not np.any(valid):
        msg = "no finite overlapping data points"
        raise ValidationError(msg)
    valid_indices = np.flatnonzero(valid)
    errors = np.abs(calc_arr[valid] - ref_arr[valid])
    worst_flat_idx = int(valid_indices[int(np.argmax(errors))])
    return float(ref_arr[worst_flat_idx]), float(calc_arr[worst_flat_idx])


def _compute_var_result(
    name: str,
    ref_arr: FloatArray,
    calc_arr: FloatArray,
    case_name: str,
    config: FrechetConfig,
) -> VarResult:
    """Compute Frechet validation metrics for one aligned variable."""

    ref_at_worst, calc_at_worst = _at_worst(ref_arr, calc_arr)
    frechet = frechet_raw_and_normalized(
        ref_arr,
        calc_arr,
        abs_tolerance=config.frechet_abs_tol,
        robust=config.robust,
        q_low=config.q_low,
        q_high=config.q_high,
        switch_oom=config.switch_oom,
        eps_floor=config.eps_floor,
        frechet_k=config.frechet_k,
    )
    d_raw = float(frechet["d_raw"])
    d_norm = float(frechet["d_norm"])
    if d_raw < config.frechet_abs_tol:
        d_raw = 0.0
        d_norm = 0.0

    finite_ref = ref_arr[np.isfinite(ref_arr)]
    finite_calc = calc_arr[np.isfinite(calc_arr)]
    signal_scale = max(
        float(np.max(np.abs(finite_ref))) if finite_ref.size else 0.0,
        float(np.max(np.abs(finite_calc))) if finite_calc.size else 0.0,
        config.eps_floor,
    )
    score, metric_mode = config.effective_score(name, d_raw, d_norm, signal_scale)

    section = classify_section(name)
    status, color = _classify_status(score, config)

    plot_html: str | None = None
    if status in ("MARGINAL", "DISCREPANT"):
        plot_html = _make_plot_html(
            ref_arr,
            calc_arr,
            case_name=case_name,
            section=section,
            variable=name,
            status=status,
            d_raw=d_raw,
            d_norm=d_norm,
            score=score,
            metric_mode=metric_mode,
        )

    return VarResult(
        name=name,
        section=section,
        status=status,
        color=color,
        reference_at_worst=ref_at_worst,
        calculated_at_worst=calc_at_worst,
        d_raw=d_raw,
        d_norm=d_norm,
        plot_html=plot_html,
        metric_mode=metric_mode,
        score=score,
    )


def compare_nc(
    py_path: Path,
    ref_path: Path,
    case_name: str,
    config: FrechetConfig = DEFAULT_FRECHET_CONFIG,
) -> list[VarResult]:
    """Compare *py_path* against *ref_path* using Frechet distance."""

    py_ds = open_validation_dataset(py_path)
    try:
        ref_ds = open_validation_dataset(ref_path)
        try:
            ref_vars = numeric_variable_names(ref_ds)
            py_vars = {
                str(name)
                for name, da in py_ds.data_vars.items()
                if np.issubdtype(da.dtype, np.number)
            }

            results: list[VarResult] = []
            for name in ref_vars:
                if name not in py_vars:
                    results.append(_broken_result(name))
                    continue

                try:
                    ref_arr, calc_arr = _aligned_arrays(ref_ds[name], py_ds[name])
                    results.append(
                        _compute_var_result(
                            name,
                            ref_arr,
                            calc_arr,
                            case_name,
                            config,
                        )
                    )
                except ValidationError:
                    results.append(_broken_result(name))

            for name in sorted(py_vars.difference(set(ref_vars))):
                results.append(_broken_result(name))

        finally:
            ref_ds.close()
    finally:
        py_ds.close()

    return results
