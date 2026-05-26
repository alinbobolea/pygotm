"""Regression tests for validation JSON artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from pygotm.validation import run_validation
from pygotm.validation.compare import VarResult
from pygotm.validation.hardware import PlatformInfo
from pygotm.validation.report import CaseResult, load_json


def test_validate_cli_writes_summary_and_results_json(
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
                variables=[
                    VarResult(
                        name="temp",
                        section="pygotm",
                        status="PASS",
                        color="green",
                        reference_at_worst=1.0,
                        calculated_at_worst=1.0,
                        d_raw=0.0,
                        d_norm=0.0,
                        plot_html="<div>plot payload</div>",
                    )
                ],
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
    assert report.cases[0].variables == []

    results = load_json(tmp_path / "results.json")
    assert results.cases[0].variables[0].name == "temp"
    assert results.cases[0].variables[0].plot_html is None

    results_payload = json.loads((tmp_path / "results.json").read_text())
    assert results_payload["cases"][0]["variables"][0]["plot_html"] is None
