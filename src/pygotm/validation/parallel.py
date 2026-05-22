"""Dask-powered parallel validation runner for pyGOTM."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from dask.distributed import Client, LocalCluster, as_completed

from pygotm.validation.reference import resolve_reference_case
from pygotm.validation.report import CaseResult
from pygotm.validation.runner import validate_case, validate_case_to_html

__all__ = ["run_cases_parallel"]


def _run_case_worker(
    case_name: str,
    runs_dir: Path,
    arch_name: str,
    skip_run: bool,
    debug_turbulence: bool = False,
    report_dir: Path | None = None,
    report_generated_at: str | None = None,
    report_hardware: dict[str, str] | None = None,
) -> CaseResult:
    """Worker function executed in each Dask subprocess.

    Runs and validates a single GOTM case. *arch_name* is retained for
    compatibility with the validation CLI; Numba kernels currently execute on CPU.
    """
    _ = arch_name

    if report_dir is not None:
        if report_generated_at is None:
            msg = "report_generated_at is required when report_dir is set"
            raise ValueError(msg)
        return validate_case_to_html(
            case_name,
            runs_dir,
            report_dir,
            generated_at=report_generated_at,
            hardware=report_hardware or {},
            skip_run=skip_run,
            debug_turbulence=debug_turbulence,
        )

    return validate_case(
        case_name,
        runs_dir,
        skip_run=skip_run,
        debug_turbulence=debug_turbulence,
    )


def run_cases_parallel(
    case_names: list[str],
    runs_dir: Path,
    arch_name: str,
    n_workers: int,
    *,
    dashboard_port: int = 8787,
    skip_run: bool = False,
    debug_turbulence: bool = False,
    report_dir: Path | None = None,
    report_generated_at: str | None = None,
    report_hardware: dict[str, str] | None = None,
    processes: bool = True,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run *case_names* in parallel using a Dask LocalCluster.

    Returns results in the same order as *case_names*.
    Calls *on_result* as each case completes (arrival order).
    """
    n_workers = max(1, min(n_workers, len(case_names)))
    cases = [resolve_reference_case(name) for name in case_names]

    logging.getLogger("distributed").setLevel(logging.CRITICAL)

    with (
        LocalCluster(  # type: ignore[no-untyped-call]
            n_workers=n_workers,
            threads_per_worker=1,
            processes=processes,
            # Disable Dask nanny memory kills; validation owns failure reporting.
            memory_limit=0,
            silence_logs=logging.CRITICAL,
            dashboard_address=f":{dashboard_port}",
        ) as cluster,
        Client(cluster, timeout=120) as client,  # type: ignore[no-untyped-call]
    ):
        print(f"  Dask dashboard : {cluster.dashboard_link}")

        futures = {
            client.submit(
                _run_case_worker,
                case_names[index],
                runs_dir,
                arch_name,
                skip_run,
                debug_turbulence,
                report_dir,
                report_generated_at,
                report_hardware,
                key=case.task_name,
            ): index
            for index, case in enumerate(cases)
        }

        result_map: dict[int, CaseResult] = {}
        for future in as_completed(futures):  # type: ignore[no-untyped-call]
            index = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                case = cases[index]
                result = CaseResult(
                    case_name=case.run_name,
                    status="ERROR",
                    error=f"{type(exc).__name__}: {exc}",
                    py_nc_path="",
                    ref_nc_path=str(case.reference_path),
                    wall_time_s=0.0,
                    task_name=case.task_name,
                )
            result_map[index] = result
            if on_result is not None:
                on_result(result)

    return [result_map[index] for index in range(len(case_names))]
