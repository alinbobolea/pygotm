"""NetCDF comparison utilities for pyGOTM validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

__all__ = ["ATOL", "RTOL", "VarResult", "compare_nc"]

RTOL: float = 1e-6
ATOL: float = 1e-12


@dataclass
class VarResult:
    name: str
    status: Literal["PASS", "FAIL", "SKIP"]
    ref_at_worst: float
    calc_at_worst: float
    max_abs_err: float
    max_rel_err: float
    rmse: float
    nrmse: float


def _open_nc(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="scipy")
    except Exception:
        return xr.open_dataset(path, engine="netcdf4")


def compare_nc(
    py_path: Path,
    ref_path: Path,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> list[VarResult]:
    """Return per-variable metrics comparing *py_path* against *ref_path*."""
    py_ds = _open_nc(py_path)
    try:
        ref_ds = _open_nc(ref_path)
        try:
            py_ds = py_ds.squeeze(drop=True)
            ref_ds = ref_ds.squeeze(drop=True)

            ref_vars = [
                str(name)
                for name, da in ref_ds.data_vars.items()
                if np.issubdtype(da.dtype, np.number)
            ]
            py_vars = {
                str(name)
                for name, da in py_ds.data_vars.items()
                if np.issubdtype(da.dtype, np.number)
            }

            results: list[VarResult] = []
            for name in ref_vars:
                if name not in py_vars:
                    results.append(VarResult(
                        name=name, status="SKIP",
                        ref_at_worst=float("nan"), calc_at_worst=float("nan"),
                        max_abs_err=float("nan"), max_rel_err=float("nan"),
                        rmse=float("nan"), nrmse=float("nan"),
                    ))
                    continue

                ref_arr = np.asarray(ref_ds[name].values, dtype=np.float64).ravel()
                py_arr  = np.asarray(py_ds[name].values,  dtype=np.float64).ravel()

                if ref_arr.shape != py_arr.shape:
                    results.append(VarResult(
                        name=name, status="FAIL",
                        ref_at_worst=float("nan"), calc_at_worst=float("nan"),
                        max_abs_err=float("inf"), max_rel_err=float("inf"),
                        rmse=float("inf"), nrmse=float("inf"),
                    ))
                    continue

                abs_err = np.abs(py_arr - ref_arr)
                worst_i = int(np.argmax(abs_err))
                max_abs = float(abs_err[worst_i])
                ref_rng = float(np.max(ref_arr) - np.min(ref_arr))
                rmse    = float(np.sqrt(np.mean(abs_err ** 2)))

                nonzero = np.abs(ref_arr) > 0.0
                rel_err = np.zeros_like(abs_err)
                rel_err[nonzero] = abs_err[nonzero] / np.abs(ref_arr[nonzero])
                max_rel = float(np.max(rel_err))

                nrmse = rmse / ref_rng if ref_rng > 0.0 else float("nan")

                atol_var = max(2e-6 * ref_rng, atol) if ref_rng > 0.0 else atol
                passes = bool(np.all(
                    np.abs(py_arr - ref_arr) <= atol_var + rtol * np.abs(ref_arr)
                ))
                results.append(VarResult(
                    name=name, status="PASS" if passes else "FAIL",
                    ref_at_worst=float(ref_arr[worst_i]),
                    calc_at_worst=float(py_arr[worst_i]),
                    max_abs_err=max_abs, max_rel_err=max_rel,
                    rmse=rmse, nrmse=nrmse,
                ))
        finally:
            ref_ds.close()
    finally:
        py_ds.close()

    return results
