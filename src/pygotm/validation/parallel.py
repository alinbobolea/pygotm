"""Dask-powered parallel validation runner for pyGOTM."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from dask.distributed import Client, LocalCluster, as_completed

from pygotm.validate import resolve_reference_case
from pygotm.validation.report import CaseResult
from pygotm.validation.runner import validate_case

__all__ = ["run_cases_parallel"]


def _make_step_writer(progress_path: Path) -> Callable[[int, int], None]:
    """Return a throttled on_step callback that overwrites a progress file at ~1 Hz."""
    last_write = 0.0
    t0 = time.monotonic()

    def on_step(current: int, total: int) -> None:
        nonlocal last_write
        now = time.monotonic()
        if current < total and now - last_write < 1.0:
            return
        last_write = now
        try:
            progress_path.write_text(f"{current} {total} {now - t0:.1f}")
        except OSError:
            pass

    return on_step


def _run_case_worker(
    case_name: str,
    runs_dir: Path,
    arch_name: str,
    skip_run: bool,
    debug_turbulence: bool = False,
) -> CaseResult:
    """Worker function executed in each Dask subprocess.

    Runs and validates a single GOTM case. *arch_name* is retained for
    compatibility with the validation CLI; Numba kernels currently execute on CPU.
    """
    _ = arch_name

    case = resolve_reference_case(case_name)
    progress_path = runs_dir / case.run_name / ".progress"
    on_step = _make_step_writer(progress_path) if not skip_run else None
    try:
        return validate_case(
            case_name,
            runs_dir,
            skip_run=skip_run,
            debug_turbulence=debug_turbulence,
            on_step=on_step,
        )
    finally:
        progress_path.unlink(missing_ok=True)


def run_cases_parallel(
    case_names: list[str],
    runs_dir: Path,
    arch_name: str,
    n_workers: int,
    *,
    dashboard_port: int = 8787,
    skip_run: bool = False,
    debug_turbulence: bool = False,
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
            processes=True,
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
