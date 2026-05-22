"""Debug artifacts for parity validation residual analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from pygotm.validation.reference import open_validation_dataset

# Diagnostic-only fallback tolerances used to compute the "passes" field in
# debug dumps.  These are not parity-decision values; real parity uses the
# per-variable tolerance registry in pygotm.validation.tolerances.
_ATOL: float = 1e-12
_RTOL: float = 5e-6

__all__ = ["TURBULENCE_DEBUG_VARIABLES", "write_turbulence_debug_dump"]

TURBULENCE_DEBUG_VARIABLES: tuple[str, ...] = (
    "tke",
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
    "cmue1",
    "cmue2",
    "cmue3",
    "as",
    "an",
    "at",
    "av",
    "aw",
    "uu",
    "vv",
    "ww",
    "SS",
    "SSU",
    "SSV",
    "SSCSTK",
    "SSSTK",
    "Rig",
    "gamu",
    "gamv",
    "gamh",
    "gams",
    "gamb",
    "gam",
    "r",
)


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _move_time_axis(
    data: xr.DataArray,
) -> tuple[np.ndarray, list[str], tuple[int, ...]]:
    arr: np.ndarray = np.asarray(data.values, dtype=np.float64)
    dims = [str(dim) for dim in data.dims]
    if arr.ndim == 0:
        reshaped: np.ndarray = arr.reshape(1, 1)
        return reshaped, [], ()
    if "time" in dims:
        time_axis = dims.index("time")
        arr = np.moveaxis(arr, time_axis, 0)
        spatial_dims = [dim for i, dim in enumerate(dims) if i != time_axis]
        spatial_shape = tuple(arr.shape[1:])
    else:
        spatial_dims = dims
        spatial_shape = tuple(arr.shape)
        arr = arr.reshape(1, arr.size)
    if arr.ndim == 1:
        arr = arr.reshape(arr.shape[0], 1)
    else:
        arr = arr.reshape(arr.shape[0], int(np.prod(arr.shape[1:])))
    return arr, spatial_dims, spatial_shape


def _spatial_index(
    flat_index: int,
    dims: list[str],
    shape: tuple[int, ...],
) -> dict[str, int]:
    if not dims or not shape:
        return {}
    unraveled = np.unravel_index(flat_index, shape)
    return {dim: int(index) for dim, index in zip(dims, unraveled, strict=True)}


def _metric_record(
    *,
    variable: str,
    py_values: np.ndarray,
    ref_values: np.ndarray,
    spatial_dims: list[str],
    spatial_shape: tuple[int, ...],
    time_index: int | None,
) -> dict[str, object]:
    abs_err = np.abs(py_values - ref_values)
    worst_metric = np.where(np.isfinite(abs_err), abs_err, np.inf)
    worst_i = int(np.argmax(worst_metric))
    ref_range = float(np.nanmax(ref_values) - np.nanmin(ref_values))
    rmse = float(np.sqrt(np.nanmean(abs_err * abs_err)))
    nonzero = np.abs(ref_values) > 0.0
    rel_err = np.zeros_like(abs_err)
    rel_err[nonzero] = abs_err[nonzero] / np.abs(ref_values[nonzero])
    atol_var = max(1.0e-7 * ref_range, _ATOL)
    passes = bool(np.all(abs_err <= atol_var + _RTOL * np.abs(ref_values)))

    record: dict[str, object] = {
        "variable": variable,
        "status": "PASS" if passes else "FAIL",
        "max_abs_err": _json_float(float(abs_err[worst_i])),
        "max_rel_err": _json_float(float(np.nanmax(rel_err))),
        "rmse": _json_float(rmse),
        "nrmse": _json_float(rmse / ref_range if ref_range > 0.0 else float("nan")),
        "ref_range": _json_float(ref_range),
        "ref_at_worst": _json_float(float(ref_values[worst_i])),
        "calc_at_worst": _json_float(float(py_values[worst_i])),
        "flat_index": worst_i,
        "spatial_index": _spatial_index(worst_i, spatial_dims, spatial_shape),
    }
    if time_index is not None:
        record["time_index"] = time_index
    return record


def write_turbulence_debug_dump(
    py_path: Path,
    ref_path: Path,
    output_path: Path,
    *,
    variables: tuple[str, ...] = TURBULENCE_DEBUG_VARIABLES,
) -> Path:
    """Write per-variable and per-time turbulence comparison metrics as JSON."""
    py_ds = open_validation_dataset(py_path)
    try:
        ref_ds = open_validation_dataset(ref_path)
        try:
            summary: list[dict[str, object]] = []
            per_time: list[dict[str, object]] = []

            for name in variables:
                if name not in py_ds or name not in ref_ds:
                    continue
                py_da = py_ds[name]
                ref_da = ref_ds[name]
                if (
                    py_da.dims != ref_da.dims
                    or py_da.shape != ref_da.shape
                    or not np.issubdtype(py_da.dtype, np.number)
                    or not np.issubdtype(ref_da.dtype, np.number)
                ):
                    summary.append(
                        {
                            "variable": name,
                            "status": "STRUCTURE_MISMATCH",
                            "py_dims": list(py_da.dims),
                            "ref_dims": list(ref_da.dims),
                            "py_shape": list(py_da.shape),
                            "ref_shape": list(ref_da.shape),
                        }
                    )
                    continue

                py_values, spatial_dims, spatial_shape = _move_time_axis(py_da)
                ref_values, _ref_spatial_dims, _ref_spatial_shape = _move_time_axis(
                    ref_da
                )
                summary.append(
                    _metric_record(
                        variable=name,
                        py_values=py_values.ravel(),
                        ref_values=ref_values.ravel(),
                        spatial_dims=["time", *spatial_dims],
                        spatial_shape=(py_values.shape[0], *spatial_shape),
                        time_index=None,
                    )
                )
                for time_index in range(py_values.shape[0]):
                    per_time.append(
                        _metric_record(
                            variable=name,
                            py_values=py_values[time_index],
                            ref_values=ref_values[time_index],
                            spatial_dims=spatial_dims,
                            spatial_shape=spatial_shape,
                            time_index=time_index,
                        )
                    )
        finally:
            ref_ds.close()
    finally:
        py_ds.close()

    payload = {
        "py_nc_path": str(py_path),
        "ref_nc_path": str(ref_path),
        "variables": list(variables),
        "summary": summary,
        "per_time": per_time,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
