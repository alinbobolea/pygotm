"""Per-case execution and validation logic for pyGOTM validation."""

from __future__ import annotations

import time
import traceback
from dataclasses import replace
from pathlib import Path

from pygotm.validation.compare import compare_nc
from pygotm.validation.debug import write_turbulence_debug_dump
from pygotm.validation.report import CaseResult, Report, write_case_html

__all__ = [
    "main",
    "run_case",
    "validate_case",
    "validate_case_to_html",
]


def _ref_path(case_name: str) -> Path:
    from pygotm.validate import resolve_reference_case

    return resolve_reference_case(case_name).reference_path


def _yaml_path(case_name: str) -> Path:
    from pygotm.validate import resolve_reference_case

    return resolve_reference_case(case_name).yaml_path


def run_case(
    case_name: str,
    runs_dir: Path,
) -> tuple[Path, float]:
    """Run a compiled parity case, write NetCDF, return (path, elapsed_s)."""
    from pygotm.driver import GotmDriver
    from pygotm.validate import resolve_reference_case

    case = resolve_reference_case(case_name)
    case_dir = runs_dir / case.run_name
    case_dir.mkdir(parents=True, exist_ok=True)
    nc_path = case_dir / f"{case.run_name}.nc"

    t0 = time.monotonic()
    dataset = GotmDriver(case.yaml_path).run(output_path=nc_path)
    try:
        if dataset.attrs.get("runtime") != "compiled":
            msg = (
                f"parity case {case.run_name!r} did not use the Numba compiled runtime"
            )
            raise RuntimeError(msg)
    finally:
        dataset.close()
    elapsed = time.monotonic() - t0
    return nc_path, elapsed


def validate_case(
    case_name: str,
    runs_dir: Path,
    *,
    skip_run: bool = False,
    debug_turbulence: bool = False,
) -> CaseResult:
    """Run (optionally) and validate a single GOTM case."""
    from pygotm.validate import resolve_reference_case

    case = resolve_reference_case(case_name)
    ref_path = case.reference_path

    if skip_run:
        py_path = runs_dir / case.run_name / f"{case.run_name}.nc"
        elapsed = 0.0
        if not py_path.is_file():
            return CaseResult(
                case_name=case.run_name,
                status="ERROR",
                error=f"NetCDF not found: {py_path}",
                py_nc_path=str(py_path),
                ref_nc_path=str(ref_path),
                wall_time_s=0.0,
                task_name=case.task_name,
            )
    else:
        try:
            py_path, elapsed = run_case(case_name, runs_dir)
        except Exception as exc:
            return CaseResult(
                case_name=case.run_name,
                status="ERROR",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                py_nc_path="",
                ref_nc_path=str(ref_path),
                wall_time_s=0.0,
                task_name=case.task_name,
            )

    var_results = compare_nc(py_path, ref_path, case_name=case.run_name)
    if debug_turbulence:
        write_turbulence_debug_dump(
            py_path,
            ref_path,
            runs_dir / case.run_name / "turbulence_debug.json",
        )
    n_pass = sum(1 for v in var_results if v.status == "PASS")
    n_marginal = sum(1 for v in var_results if v.status == "MARGINAL")
    n_discrepant = sum(1 for v in var_results if v.status == "DISCREPANT")
    n_broken = sum(1 for v in var_results if v.status == "BROKEN")
    case_pass = n_marginal == 0 and n_discrepant == 0 and n_broken == 0

    return CaseResult(
        case_name=case.run_name,
        status="PASS" if case_pass else "FAIL",
        error=None,
        py_nc_path=str(py_path),
        ref_nc_path=str(ref_path),
        wall_time_s=elapsed,
        task_name=case.task_name,
        variables=var_results,
        n_pass=n_pass,
        n_marginal=n_marginal,
        n_discrepant=n_discrepant,
        n_broken=n_broken,
    )


def _case_verdict(result: CaseResult) -> str:
    if result.status == "PASS":
        return "FULL PARITY"
    if result.status == "ERROR":
        return "FAILED VALIDATION"
    return "PARTIAL PARITY"


def _summary_case(result: CaseResult) -> CaseResult:
    """Drop per-variable plot payloads after the case HTML has been written."""

    return replace(result, variables=[])


def validate_case_to_html(
    case_name: str,
    runs_dir: Path,
    output_dir: Path,
    *,
    generated_at: str,
    hardware: dict[str, str],
    skip_run: bool = False,
    debug_turbulence: bool = False,
) -> CaseResult:
    """Run or compare one case, write its HTML page, and return a summary."""

    result = validate_case(
        case_name,
        runs_dir,
        skip_run=skip_run,
        debug_turbulence=debug_turbulence,
    )
    report = Report(
        generated_at=generated_at,
        hardware=hardware,
        cases=[result],
        verdict=_case_verdict(result),
    )
    write_case_html(report, result, output_dir)
    return _summary_case(result)


def main() -> None:
    """CLI compatibility shim for ``python -m pygotm.validation.runner``."""

    from pygotm.validation.run_validation import cli

    cli()


if __name__ == "__main__":
    main()
