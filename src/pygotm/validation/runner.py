"""Per-case execution and validation logic for pyGOTM validation."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path

from pygotm.validation.compare import compare_nc
from pygotm.validation.report import CaseResult

__all__ = ["run_case", "validate_case"]


def _ref_path(case_name: str) -> Path:
    from pygotm.validate import resolve_reference_case
    return resolve_reference_case(case_name).reference_path


def _yaml_path(case_name: str) -> Path:
    from pygotm.validate import resolve_reference_case
    return resolve_reference_case(case_name).yaml_path


def run_case(
    case_name: str,
    runs_dir: Path,
    on_step: Callable[[int, int], None] | None = None,
) -> tuple[Path, float]:
    """Run GotmDriver for *case_name*, write NetCDF, return (path, elapsed_s)."""
    import time
    from pygotm.driver import GotmDriver

    case_dir = runs_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    nc_path = case_dir / f"{case_name}.nc"

    t0 = time.monotonic()
    GotmDriver(_yaml_path(case_name)).run(output_path=nc_path, on_step=on_step)
    return nc_path, time.monotonic() - t0


def validate_case(
    case_name: str,
    runs_dir: Path,
    *,
    skip_run: bool = False,
    on_step: Callable[[int, int], None] | None = None,
) -> CaseResult:
    """Run (optionally) and validate a single GOTM case."""
    ref_path = _ref_path(case_name)

    if skip_run:
        py_path = runs_dir / case_name / f"{case_name}.nc"
        elapsed = 0.0
        if not py_path.is_file():
            return CaseResult(
                case_name=case_name, status="ERROR",
                error=f"NetCDF not found: {py_path}",
                py_nc_path=str(py_path), ref_nc_path=str(ref_path),
                wall_time_s=0.0,
            )
    else:
        try:
            py_path, elapsed = run_case(case_name, runs_dir, on_step=on_step)
        except Exception as exc:
            return CaseResult(
                case_name=case_name, status="ERROR",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                py_nc_path="", ref_nc_path=str(ref_path),
                wall_time_s=0.0,
            )

    var_results = compare_nc(py_path, ref_path)
    n_pass = sum(1 for v in var_results if v.status == "PASS")
    n_fail = sum(1 for v in var_results if v.status == "FAIL")
    n_skip = sum(1 for v in var_results if v.status == "SKIP")

    return CaseResult(
        case_name=case_name,
        status="PASS" if n_fail == 0 else "FAIL",
        error=None,
        py_nc_path=str(py_path),
        ref_nc_path=str(ref_path),
        wall_time_s=elapsed,
        variables=var_results,
        n_pass=n_pass,
        n_fail=n_fail,
        n_skip=n_skip,
    )
