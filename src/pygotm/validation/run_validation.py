"""pyGOTM validation — parity check against Fortran GOTM reference.

Workflow:
  1. Detect platform (CPU count, GPU availability)
  2. Warm up Numba kernels once before timed runs
  4. Run validation cases in parallel via Dask (dashboard at --dashboard-port)
  5. Compare each run against Fortran reference NetCDF
  6. Write validation/results.json + validation/report.html

Usage
-----
    uv run python -m pygotm.validation.run_validation
    uv run python -m pygotm.validation.run_validation --cases couette,channel
    uv run python -m pygotm.validation.run_validation --all
    uv run python -m pygotm.validation.run_validation --no-run
    uv run python -m pygotm.validation.run_validation --workers 4
    uv run python -m pygotm.validation.run_validation --dashboard-port 8788
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np

from pygotm.validate import REFERENCE_CASE_NAMES
from pygotm.validation.compare import ATOL, RTOL
from pygotm.validation.hardware import detect_platform
from pygotm.validation.parallel import run_cases_parallel
from pygotm.validation.report import CaseResult, Report, render_html, save_json
from pygotm.validation.runner import validate_case
from pygotm.validation.warmup import trigger_numba_jit

ALL_CASES: tuple[str, ...] = REFERENCE_CASE_NAMES
DEFAULT_CASES: tuple[str, ...] = ("couette", "channel", "entrainment")


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.0f}s"


class _ProgressMonitor:
    """Background thread that redraws an in-place progress block from .progress files.

    Workers write  `current total elapsed_s`  to  runs_dir/<case>/.progress  at ~1 Hz.
    This class polls those files every second and redraws the block using ANSI cursor
    moves.  Disabled automatically when stdout is not a TTY (piped/redirected output).
    """

    _POLL_S = 1.0

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._tty = sys.stdout.isatty()
        self._lock = threading.Lock()
        self._prev_lines = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if self._tty:
            self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._tty and self._thread.is_alive():
            self._thread.join(timeout=2)
        with self._lock:
            self._erase()

    def print_line(self, text: str) -> None:
        """Print a completion line, keeping the progress block below it."""
        if self._tty:
            with self._lock:
                self._erase()
                print(text, flush=True)
                self._draw()
        else:
            print(text, flush=True)

    # ------------------------------------------------------------------
    # internal

    def _loop(self) -> None:
        while not self._stop:
            time.sleep(self._POLL_S)
            with self._lock:
                self._erase()
                self._draw()

    def _erase(self) -> None:
        if self._prev_lines:
            # Move cursor up N lines then erase from cursor to end of screen.
            print(f"\033[{self._prev_lines}A\033[J", end="", flush=True)
            self._prev_lines = 0

    def _draw(self) -> None:
        entries = self._read_progress()
        if not entries:
            return
        print("  In progress:", flush=True)
        for case_name, current, total, elapsed in entries:
            pct = int(100 * current / total) if total else 0
            filled = pct // 5
            bar = "█" * filled + "░" * (20 - filled)
            print(
                f"    {case_name:<20} [{bar}] {pct:>3}%  {_fmt_time(elapsed)}",
                flush=True,
            )
        self._prev_lines = 1 + len(entries)  # header + case lines

    def _read_progress(self) -> list[tuple[str, int, int, float]]:
        result = []
        for path in sorted(self._runs_dir.glob("*/.progress")):
            try:
                parts = path.read_text().split()
                result.append((
                    path.parent.name,
                    int(parts[0]),
                    int(parts[1]),
                    float(parts[2]),
                ))
            except (OSError, ValueError, IndexError):
                pass
        return result


def _make_on_result(
    total: int, monitor: _ProgressMonitor | None = None
) -> Callable[[CaseResult], None]:
    """Return a callback that prints each result with a [N/total] counter."""
    completed = [0]
    w = len(str(total))

    def on_result(result: CaseResult) -> None:
        completed[0] += 1
        counter = f"[{completed[0]:>{w}}/{total}]"
        name = f"{result.case_name:<20}"

        if result.status == "ERROR":
            msg = (result.error or "").splitlines()[0][:60]
            line = f"  {counter} ERROR  {name}  {msg}"
        elif result.status == "PASS":
            line = (
                f"  {counter} PASS   {name}  "
                f"{result.n_pass} vars  ({_fmt_time(result.wall_time_s)})"
            )
        else:
            worst = max(
                (v for v in result.variables if v.status == "FAIL"),
                key=lambda v: v.nrmse if np.isfinite(v.nrmse) else -1,
                default=None,
            )
            worst_str = (
                f"  worst={worst.name} nrmse={worst.nrmse:.2e}" if worst else ""
            )
            line = (
                f"  {counter} FAIL   {name}  "
                f"{result.n_fail}/{result.n_pass + result.n_fail} vars"
                f"{worst_str}  ({_fmt_time(result.wall_time_s)})"
            )

        if monitor is not None:
            monitor.print_line(line)
        else:
            print(line, flush=True)

    return on_result


@click.command()
@click.option(
    "--cases", default=None, show_default=False,
    help=(
        "Comma-separated cases or case/input-yaml-base specs. "
        "Defaults to couette,channel,entrainment."
    ),
)
@click.option(
    "--all", "run_all", is_flag=True,
    help="Run all 22 GOTM reference cases.",
)
@click.option(
    "--device", default=None, metavar="ARCH",
    help="Execution backend label. Numba validation currently supports cpu.",
)
@click.option(
    "--workers", default=None, type=int, metavar="N",
    help="Dask worker count. Defaults to detected CPU count.",
)
@click.option(
    "--dashboard-port", default=8787, show_default=True, type=int,
    help="Port for the Dask dashboard.",
)
@click.option(
    "--output-dir", default="validation", show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for run outputs and report.",
)
@click.option(
    "--no-run", "skip_run", is_flag=True,
    help="Skip re-running pyGOTM; compare existing NetCDFs only.",
)
@click.option(
    "--no-warmup", "skip_warmup", is_flag=True,
    help="Skip the Numba kernel warm-up step.",
)
@click.option("--rtol", default=RTOL, show_default=True, type=float)
@click.option("--atol", default=ATOL, show_default=True, type=float)
def cli(
    cases: str | None,
    run_all: bool,
    device: str | None,
    workers: int | None,
    dashboard_port: int,
    output_dir: Path,
    skip_run: bool,
    skip_warmup: bool,
    rtol: float,
    atol: float,
) -> None:
    """Run pyGOTM validation and produce JSON + HTML report."""
    # 1. Detect platform
    platform_info = detect_platform()
    arch_choices = platform_info.available_archs
    n_cpus = platform_info.cpu_count
    n_workers = workers if workers is not None else n_cpus

    print(f"pyGOTM Validation  rtol={rtol:.0e}")
    print(f"  CPU count : {n_cpus}  (Dask workers: {n_workers})")
    print(f"  Available archs: {', '.join(arch_choices)}")

    selected_arch = device or "cpu"
    if selected_arch not in arch_choices:
        print(
            f"ERROR: --device {selected_arch!r} not available. "
            f"Available: {', '.join(arch_choices)}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"  Execution backend: {selected_arch}")

    # 3. Build case list
    if run_all:
        case_list = list(ALL_CASES)
    elif cases is not None:
        case_list = [c.strip() for c in cases.split(",") if c.strip()]
    else:
        case_list = list(DEFAULT_CASES)

    runs_dir = output_dir / "runs"
    print(f"  Cases ({len(case_list)}): {', '.join(case_list)}")
    print("=" * 60)

    # 4. Warm up Numba kernels
    if not skip_run and not skip_warmup:
        print("  Warming up Numba kernels ...", end=" ", flush=True)
        warmup_elapsed = trigger_numba_jit()
        print(f"done ({_fmt_time(warmup_elapsed)})")
    elif skip_warmup:
        print("  Skipping warm-up (--no-warmup)")

    # 5. Run cases
    hw = dict(platform_info.hardware)
    hw["execution_backend"] = selected_arch

    monitor = _ProgressMonitor(runs_dir)
    on_result = _make_on_result(len(case_list), monitor)

    if skip_run:
        results: list[CaseResult] = []
        for name in case_list:
            result = validate_case(name, runs_dir, skip_run=True)
            on_result(result)
            results.append(result)
    else:
        print(f"  Running {len(case_list)} cases on {n_workers} workers ...")
        monitor.start()
        try:
            results = run_cases_parallel(
                case_names=case_list,
                runs_dir=runs_dir,
                arch_name=selected_arch,
                n_workers=n_workers,
                dashboard_port=dashboard_port,
                skip_run=False,
                on_result=on_result,
            )
        finally:
            monitor.stop()

    # 6. Build and write report
    cases_passed = sum(1 for r in results if r.status == "PASS")
    n_cases = len(results)
    if all(r.status == "PASS" for r in results):
        verdict = "FULL PARITY"
    elif cases_passed > 0:
        verdict = "PARTIAL PARITY"
    else:
        verdict = "FAILED VALIDATION"

    report = Report(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        rtol=rtol,
        atol=atol,
        hardware=hw,
        cases=results,
        verdict=verdict,
    )

    json_path = output_dir / "results.json"
    html_path = output_dir / "report.html"
    save_json(report, json_path)
    html_path.write_text(render_html(report), encoding="utf-8")

    total_wall = sum(r.wall_time_s for r in results)
    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}  ({cases_passed}/{n_cases} cases passed)")
    print(f"  Total wall time : {_fmt_time(total_wall)}")
    print(f"  JSON  : {json_path}")
    print(f"  HTML  : {html_path}")

    raise SystemExit(0 if verdict == "FULL PARITY" else 1)


if __name__ == "__main__":
    cli()
