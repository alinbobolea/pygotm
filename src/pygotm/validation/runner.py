"""Per-case execution and validation logic for pyGOTM validation."""

from __future__ import annotations

import shutil
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from pygotm.validation.compare import compare_nc
from pygotm.validation.debug import write_turbulence_debug_dump
from pygotm.validation.report import CaseResult, Report, write_case_html

__all__ = [
    "run_case",
    "summary_case",
    "strip_case_plots",
    "validate_case",
    "validate_case_to_html",
]


def run_case(
    case_name: str,
    runs_dir: Path,
    *,
    cases_root: Path | None = None,
) -> tuple[Path, float]:
    """Run a compiled parity case, write NetCDF, return (path, elapsed_s)."""
    from pygotm.driver import GotmDriver
    from pygotm.validation.reference import (
        open_reference_dataset,
        resolve_reference_case,
    )

    case = resolve_reference_case(case_name, cases_root=cases_root)
    case_dir = runs_dir / case.run_name
    case_dir.mkdir(parents=True, exist_ok=True)
    nc_path = case_dir / f"{case.run_name}.nc"

    t0 = time.monotonic()
    driver = GotmDriver(case.yaml_path)
    dataset = driver.run()
    try:
        if dataset.attrs.get("runtime") != "compiled":
            msg = (
                f"parity case {case.run_name!r} did not use the Numba compiled runtime"
            )
            raise RuntimeError(msg)
        reference = open_reference_dataset(case)
        try:
            keep = [name for name in reference.data_vars if name in dataset.data_vars]
            dataset = dataset[keep]
            GotmDriver.write_dataset(dataset, nc_path)
        finally:
            reference.close()
        # Stage the exact configs the run consumed so validation/runs/<case> is a
        # self-consistent, re-runnable reference bundle for downstream ingestion.
        _stage_bundle_configs(driver.config, case_dir)
    finally:
        dataset.close()
    elapsed = time.monotonic() - t0
    return nc_path, elapsed


def _fabm_config_filename(document: dict[str, Any]) -> str:
    """Return the FABM config filename a GOTM document references (default)."""

    raw_fabm = document.get("fabm")
    raw_fabm = raw_fabm if isinstance(raw_fabm, dict) else {}
    return str(
        raw_fabm.get("config")
        or raw_fabm.get("config_file")
        or raw_fabm.get("yaml")
        or raw_fabm.get("file")
        or "fabm.yaml"
    )


def _stage_bundle_configs(config: Any, case_dir: Path) -> None:
    """Stage the exact configs a run consumed into its bundle directory.

    Writes the GOTM YAML that was run and the **materialized** FABM YAML that
    pyfabm actually loaded — the legacy GOTM-lake ``selmaprotbas`` phytoplankton
    ``alpha``/``beta`` parameters are stripped by
    :func:`pygotm.fabm.config.resolve_fabm_config_path` because conda
    ``pyfabm`` cannot parse them. Staging this materialized config keeps the
    bundle's ``fabm.yaml`` byte-identical to the NetCDF ``fabm_yaml_sha256``
    attribute, so re-running ``validation/runs/<case>`` reproduces the recorded
    config hashes (the bundle is self-consistent for downstream ingestion).
    """
    from pygotm.fabm.config import fabm_enabled, resolve_fabm_config_path

    source_path = getattr(config, "source_path", None)
    if source_path is None:
        return
    source_path = Path(source_path)
    document = config.resolved_document()
    if not isinstance(document, dict):
        return

    staged_gotm = case_dir / source_path.name
    if source_path.resolve() != staged_gotm.resolve():
        shutil.copyfile(source_path, staged_gotm)

    if not fabm_enabled(document):
        return
    config_path = resolve_fabm_config_path(source_path, document)
    relative = Path(_fabm_config_filename(document))
    if relative.is_absolute():
        return
    dest = case_dir / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if config_path.resolve() != dest.resolve():
        shutil.copyfile(config_path, dest)


def validate_case(
    case_name: str,
    runs_dir: Path,
    *,
    skip_run: bool = False,
    debug_turbulence: bool = False,
    cases_root: Path | None = None,
) -> CaseResult:
    """Run (optionally) and validate a single GOTM case."""
    from pygotm.validation.reference import resolve_reference_case

    case = resolve_reference_case(case_name, cases_root=cases_root)
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
            py_path, elapsed = run_case(case_name, runs_dir, cases_root=cases_root)
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


def summary_case(result: CaseResult) -> CaseResult:
    """Return a lightweight case summary without per-variable result rows."""

    return replace(result, variables=[])


def strip_case_plots(result: CaseResult) -> CaseResult:
    """Return per-variable results without embedded Plotly HTML payloads."""

    return replace(
        result,
        variables=[replace(variable, plot_html=None) for variable in result.variables],
    )


def validate_case_to_html(
    case_name: str,
    runs_dir: Path,
    output_dir: Path,
    *,
    generated_at: str,
    hardware: dict[str, str],
    skip_run: bool = False,
    debug_turbulence: bool = False,
    cases_root: Path | None = None,
) -> CaseResult:
    """Run or compare one case, write its HTML page, and return JSON-safe data."""

    result = validate_case(
        case_name,
        runs_dir,
        skip_run=skip_run,
        debug_turbulence=debug_turbulence,
        cases_root=cases_root,
    )
    report = Report(
        generated_at=generated_at,
        hardware=hardware,
        cases=[result],
        verdict=_case_verdict(result),
    )
    write_case_html(report, result, output_dir)
    return strip_case_plots(result)
