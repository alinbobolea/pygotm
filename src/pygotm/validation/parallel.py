"""Dask-powered parallel validation runner for pyGOTM."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from dask.distributed import Client, LocalCluster, as_completed

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
) -> CaseResult:
    """Worker function executed in each Dask subprocess.

    Initialises Taichi from the offline kernel cache, then runs and
    validates a single GOTM case.
    """
    import taichi as ti
    from taichi.lang import impl as ti_impl

    arch = getattr(ti, arch_name)
    if ti_impl.get_runtime().prog is None:
        ti.init(arch=arch, default_fp=ti.f64, offline_cache=True)

    progress_path = runs_dir / case_name / ".progress"
    on_step = _make_step_writer(progress_path) if not skip_run else None
    try:
        return validate_case(case_name, runs_dir, skip_run=skip_run, on_step=on_step)
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
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    """Run *case_names* in parallel using a Dask LocalCluster.

    Returns results in the same order as *case_names*.
    Calls *on_result* as each case completes (arrival order).
    """
    n_workers = max(1, min(n_workers, len(case_names)))

    logging.getLogger("distributed").setLevel(logging.CRITICAL)

    with LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        silence_logs=logging.CRITICAL,
        dashboard_address=f":{dashboard_port}",
    ) as cluster, Client(cluster, timeout=120) as client:

        print(f"  Dask dashboard : {cluster.dashboard_link}")

        futures = {
            client.submit(
                _run_case_worker,
                name,
                runs_dir,
                arch_name,
                skip_run,
                key=f"validate-{name}",
            ): name
            for name in case_names
        }

        result_map: dict[str, CaseResult] = {}
        for future in as_completed(futures):
            case_name = futures[future]
            result = future.result()
            result_map[case_name] = result
            if on_result is not None:
                on_result(result)

    return [result_map[name] for name in case_names]
