"""pyGOTM validation — parity check against Fortran GOTM reference.

Workflow:
  1. Detect platform (CPU count, GPU availability)
  2. Warm up Numba kernels once before timed runs
  4. Run validation cases serially or through Dask when --workers > 1
  5. Compare each run against Fortran reference NetCDF
  6. Write validation/report.html + per-case reports

Usage
-----
    python -m pygotm.validation.run_validation
    python -m pygotm.validation.run_validation --cases couette,channel
    python -m pygotm.validation.run_validation --all
    python -m pygotm.validation.run_validation --group non-stim
    python -m pygotm.validation.run_validation --exclude plume,resolute
    python -m pygotm.validation.run_validation --no-run
    python -m pygotm.validation.run_validation --workers 4
    python -m pygotm.validation.run_validation --dashboard-port 8788
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import click

from pygotm.validation.hardware import detect_platform
from pygotm.validation.parallel import run_cases_parallel
from pygotm.validation.reference import REFERENCE_CASE_NAMES, resolve_reference_case
from pygotm.validation.report import CaseResult, Report, save_json, write_html_index
from pygotm.validation.runner import validate_case_to_html
from pygotm.validation.warmup import trigger_numba_jit

ALL_CASES: tuple[str, ...] = REFERENCE_CASE_NAMES
DEFAULT_CASES: tuple[str, ...] = ("couette", "channel", "entrainment")
NON_STIM_CASES: tuple[str, ...] = tuple(
    case for case in ALL_CASES if case not in {"plume", "resolute"}
)
CASE_GROUPS: dict[str, tuple[str, ...]] = {
    "default": DEFAULT_CASES,
    "non-stim": NON_STIM_CASES,
    "all": ALL_CASES,
}


def _split_case_names(value: str | None) -> list[str]:
    if value is None:
        return []
    return [case.strip() for case in value.split(",") if case.strip()]


def _select_case_list(
    *,
    cases: str | None,
    run_all: bool,
    group: str | None,
    exclude: str | None,
) -> list[str]:
    if group is not None:
        case_list = list(CASE_GROUPS[group])
    elif run_all:
        case_list = list(ALL_CASES)
    elif cases is not None:
        case_list = _split_case_names(cases)
    else:
        case_list = list(DEFAULT_CASES)

    excluded = set(_split_case_names(exclude))
    if excluded:
        case_list = [case for case in case_list if case not in excluded]
    return case_list


def _validate_case_list(case_list: list[str]) -> None:
    """Fail before warmup/execution if any requested case cannot be resolved."""

    for case_name in case_list:
        resolve_reference_case(case_name)


def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.0f}s"


def _case_display_name(result: CaseResult) -> str:
    if result.task_name is not None:
        return result.task_name
    return f"{result.case_name}-gotm"


def _make_on_result(total: int) -> Callable[[CaseResult], None]:
    """Return a callback that prints case completion counts."""
    completed = [0]

    def on_result(result: CaseResult) -> None:
        completed[0] += 1
        print(
            f"Complete case: {_case_display_name(result)} "
            f"[{_fmt_duration(result.wall_time_s)}] | {completed[0]}/{total} cases",
            flush=True,
        )

    return on_result


def _run_cases(
    *,
    case_list: list[str],
    runs_dir: Path,
    output_dir: Path,
    selected_arch: str,
    n_workers: int,
    dashboard_port: int,
    generated_at: str,
    hardware: dict[str, str],
    skip_run: bool,
    debug_turbulence: bool,
    on_result: Callable[[CaseResult], None],
) -> list[CaseResult]:
    if not case_list:
        return []

    if len(case_list) == 1 or n_workers <= 1:
        results: list[CaseResult] = []
        for case_name in case_list:
            result = validate_case_to_html(
                case_name,
                runs_dir,
                output_dir,
                generated_at=generated_at,
                hardware=hardware,
                skip_run=skip_run,
                debug_turbulence=debug_turbulence,
            )
            on_result(result)
            results.append(result)
        return results

    return run_cases_parallel(
        case_names=case_list,
        runs_dir=runs_dir,
        arch_name=selected_arch,
        n_workers=n_workers,
        dashboard_port=dashboard_port,
        skip_run=skip_run,
        debug_turbulence=debug_turbulence,
        report_dir=output_dir,
        report_generated_at=generated_at,
        report_hardware=hardware,
        on_result=on_result,
    )


@click.command()
@click.option(
    "--cases",
    default=None,
    show_default=False,
    help=(
        "Comma-separated cases or case/input-yaml-base specs. "
        "Defaults to couette,channel,entrainment."
    ),
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    help="Run all 22 GOTM reference cases.",
)
@click.option(
    "--group",
    type=click.Choice(sorted(CASE_GROUPS)),
    default=None,
    help="Named validation case group.",
)
@click.option(
    "--exclude",
    default=None,
    show_default=False,
    help="Comma-separated case names to omit from the selected set.",
)
@click.option(
    "--device",
    default=None,
    metavar="ARCH",
    help="Execution backend label. Numba validation currently supports cpu.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=int,
    metavar="N",
    help="Validation worker count. Use N>1 to run cases through Dask.",
)
@click.option(
    "--dashboard-port",
    default=8787,
    show_default=True,
    type=int,
    help="Port for the Dask dashboard.",
)
@click.option(
    "--output-dir",
    default="validation",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for run outputs and report.",
)
@click.option(
    "--no-run",
    "skip_run",
    is_flag=True,
    help="Skip re-running pyGOTM; compare existing NetCDFs only.",
)
@click.option(
    "--no-warmup",
    "skip_warmup",
    is_flag=True,
    help="Skip the Numba kernel warm-up step.",
)
@click.option(
    "--debug-turbulence",
    is_flag=True,
    help="Write per-time turbulence comparison dumps under each run directory.",
)
def cli(
    cases: str | None,
    run_all: bool,
    group: str | None,
    exclude: str | None,
    device: str | None,
    workers: int,
    dashboard_port: int,
    output_dir: Path,
    skip_run: bool,
    skip_warmup: bool,
    debug_turbulence: bool,
) -> None:
    """Run pyGOTM validation and produce HTML reports."""
    # 1. Detect platform
    platform_info = detect_platform()
    arch_choices = platform_info.available_archs
    n_workers = max(1, workers)

    selected_arch = device or "cpu"
    if selected_arch not in arch_choices:
        print(
            f"ERROR: --device {selected_arch!r} not available. "
            f"Available: {', '.join(arch_choices)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # 3. Build case list
    case_list = _select_case_list(
        cases=cases,
        run_all=run_all,
        group=group,
        exclude=exclude,
    )
    try:
        _validate_case_list(case_list)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    runs_dir = output_dir / "runs"
    print(f"pyGOTM validation starting ({len(case_list)} cases)")

    # 4. Warm up Numba kernels
    if not skip_run and not skip_warmup:
        trigger_numba_jit()

    # 5. Run cases
    hw = dict(platform_info.hardware)
    hw["execution_backend"] = selected_arch

    on_result = _make_on_result(len(case_list))
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = _run_cases(
        case_list=case_list,
        runs_dir=runs_dir,
        output_dir=output_dir,
        selected_arch=selected_arch,
        n_workers=n_workers,
        dashboard_port=dashboard_port,
        generated_at=generated_at,
        hardware=hw,
        skip_run=skip_run,
        debug_turbulence=debug_turbulence,
        on_result=on_result,
    )

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
        generated_at=generated_at,
        hardware=hw,
        cases=results,
        verdict=verdict,
    )

    html_path = output_dir / "report.html"
    json_path = output_dir / "report.json"
    write_html_index(report, output_dir)
    save_json(report, json_path)

    print()
    print("pyGOTM validation complete")
    print(f"  Cases completed: {n_cases}/{len(case_list)}")
    print(f"  HTML  : {html_path}")
    print(f"  JSON  : {json_path}")

    raise SystemExit(0 if verdict == "FULL PARITY" else 1)


if __name__ == "__main__":
    cli()
