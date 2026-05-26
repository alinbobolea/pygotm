"""Regression tests for validation report.json generation."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from pygotm.validation import run_validation
from pygotm.validation.hardware import PlatformInfo
from pygotm.validation.report import CaseResult, load_json


def test_validate_cli_writes_report_json_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_validation,
        "detect_platform",
        lambda: PlatformInfo(
            cpu_count=1,
            gpu_count=0,
            available_archs=["cpu"],
            hardware={"cpu_count": "1"},
        ),
    )

    def fake_run_cases(**kwargs: object) -> list[CaseResult]:
        del kwargs
        return [
            CaseResult(
                case_name="couette",
                status="PASS",
                error=None,
                py_nc_path="py.nc",
                ref_nc_path="ref.nc",
                wall_time_s=0.1,
                n_pass=1,
            )
        ]

    monkeypatch.setattr(run_validation, "_run_cases", fake_run_cases)

    result = CliRunner().invoke(
        run_validation.cli,
        ["--cases", "couette", "--no-run", "--output-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    report = load_json(tmp_path / "report.json")
    assert report.verdict == "FULL PARITY"
    assert report.cases[0].case_name == "couette"
