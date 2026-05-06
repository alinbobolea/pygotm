"""NetCDF comparison utilities for pyGOTM validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from pygotm.validate import numeric_variable_names, open_validation_dataset

__all__ = ["ATOL", "RTOL", "VarResult", "compare_nc"]

RTOL: float = 5.0e-6
ATOL: float = 1.0e-12


@dataclass
class VarResult:
    name: str
    status: Literal["PASS", "FAIL"]
    ref_at_worst: float
    calc_at_worst: float
    max_abs_err: float
    max_rel_err: float
    rmse: float
    nrmse: float


def _failed_structure_result(name: str) -> VarResult:
    return VarResult(
        name=name,
        status="FAIL",
        ref_at_worst=float("nan"),
        calc_at_worst=float("nan"),
        max_abs_err=float("inf"),
        max_rel_err=float("inf"),
        rmse=float("inf"),
        nrmse=float("inf"),
    )


def compare_nc(
    py_path: Path,
    ref_path: Path,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> list[VarResult]:
    """Return per-variable metrics comparing *py_path* against *ref_path*."""
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
                    results.append(_failed_structure_result(name))
                    continue

                if (
                    py_ds[name].dims != ref_ds[name].dims
                    or py_ds[name].shape != ref_ds[name].shape
                ):
                    results.append(_failed_structure_result(name))
                    continue

                ref_arr = np.asarray(ref_ds[name].values, dtype=np.float64).ravel()
                py_arr = np.asarray(py_ds[name].values, dtype=np.float64).ravel()

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

                atol_var = max(1.0e-7 * ref_rng, atol)
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

            for name in sorted(py_vars.difference(ref_vars)):
                results.append(_failed_structure_result(name))
        finally:
            ref_ds.close()
    finally:
        py_ds.close()

    return results
