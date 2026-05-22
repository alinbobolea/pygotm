"""Render validation HTML reports directly from existing NetCDF outputs.

Usage
-----
    python -m pygotm.validation.render_report
    python -m pygotm.validation.render_report --cases couette,channel
    python -m pygotm.validation.render_report --all --workers 4
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import click

from pygotm.validation.hardware import detect_platform
from pygotm.validation.report import Report, write_html_index
from pygotm.validation.run_validation import (
    CASE_GROUPS,
    _make_on_result,
    _run_cases,
    _select_case_list,
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
@click.option("--all", "run_all", is_flag=True, help="Render all reference cases.")
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
    "--workers",
    default=None,
    type=int,
    metavar="N",
    help="Dask worker count for multi-case report rendering.",
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
    help="Directory containing runs/ and receiving report HTML.",
)
def cli(
    cases: str | None,
    run_all: bool,
    group: str | None,
    exclude: str | None,
    workers: int | None,
    dashboard_port: int,
    output_dir: Path,
) -> None:
    """Regenerate validation HTML from existing run/reference NetCDF files."""

    platform_info = detect_platform()
    n_workers = workers if workers is not None else platform_info.cpu_count
    case_list = _select_case_list(
        cases=cases,
        run_all=run_all,
        group=group,
        exclude=exclude,
    )
    hardware = dict(platform_info.hardware)
    hardware["execution_backend"] = "cpu"
    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = _run_cases(
        case_list=case_list,
        runs_dir=output_dir / "runs",
        output_dir=output_dir,
        selected_arch="cpu",
        n_workers=n_workers,
        dashboard_port=dashboard_port,
        generated_at=generated_at,
        hardware=hardware,
        skip_run=True,
        debug_turbulence=False,
        on_result=_make_on_result(len(case_list)),
    )

    cases_passed = sum(1 for result in results if result.status == "PASS")
    if all(result.status == "PASS" for result in results):
        verdict = "FULL PARITY"
    elif cases_passed > 0:
        verdict = "PARTIAL PARITY"
    else:
        verdict = "FAILED VALIDATION"

    report = Report(
        generated_at=generated_at,
        hardware=hardware,
        cases=results,
        verdict=verdict,
    )
    html_path = write_html_index(report, output_dir)
    click.echo(f"Report written to: {html_path}")

    raise SystemExit(0 if verdict == "FULL PARITY" else 1)


if __name__ == "__main__":
    cli()
