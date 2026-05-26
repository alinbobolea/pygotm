"""Tests for validation run selection helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from pygotm.validation import run_validation
from pygotm.validation.hardware import PlatformInfo
from pygotm.validation.report import CaseResult, Report
from pygotm.validation.run_validation import (
    ALL_CASES,
    DEFAULT_CASES,
    NON_STIM_CASES,
    _make_on_result,
    _run_cases,
    _select_case_list,
    _validate_case_list,
)


def _case_result(status: str = "PASS") -> CaseResult:
    return CaseResult(
        case_name="couette",
        status=status,
        error=None,
        py_nc_path="validation/runs/couette/couette.nc",
        ref_nc_path="validation/reference/couette/couette.nc",
        wall_time_s=0.0,
        task_name="couette-gotm",
        n_pass=1 if status == "PASS" else 0,
    )


def test_non_stim_group_excludes_stim_cases() -> None:
    selected = _select_case_list(
        cases=None,
        run_all=False,
        group="non-stim",
        exclude=None,
    )

    assert tuple(selected) == NON_STIM_CASES
    assert "plume" not in selected
    assert "resolute" not in selected


def test_exclude_filters_selected_cases() -> None:
    selected = _select_case_list(
        cases=None,
        run_all=True,
        group=None,
        exclude="plume,resolute",
    )

    assert selected == [case for case in ALL_CASES if case not in {"plume", "resolute"}]


def test_explicit_cases_take_precedence_over_default_group() -> None:
    selected = _select_case_list(
        cases="couette,channel",
        run_all=False,
        group=None,
        exclude=None,
    )

    assert selected == ["couette", "channel"]
    assert tuple(selected) != DEFAULT_CASES


def test_validate_case_list_accepts_case_yaml_specs() -> None:
    _validate_case_list(["entrainment/gotm_keps"])


def test_on_result_reports_completed_case_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    on_result = _make_on_result(2)

    on_result(
        CaseResult(
            case_name="couette",
            status="FAIL",
            error=None,
            py_nc_path="couette.nc",
            ref_nc_path="ref.nc",
            wall_time_s=10.0,
            task_name="couette-gotm",
            n_pass=4,
            n_broken=1,
        )
    )
    on_result(
        CaseResult(
            case_name="channel",
            status="PASS",
            error=None,
            py_nc_path="channel.nc",
            ref_nc_path="ref.nc",
            wall_time_s=11.0,
            task_name="channel-gotm",
            n_pass=5,
        )
    )

    assert capsys.readouterr().out.splitlines() == [
        "Complete case: couette-gotm [10.0s] | 1/2 cases",
        "Complete case: channel-gotm [11.0s] | 2/2 cases",
    ]


def test_run_cases_renders_multi_case_no_run_with_dask_when_workers_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_cases_parallel(**kwargs: object) -> list[CaseResult]:
        assert kwargs["case_names"] == ["couette", "channel"]
        assert kwargs["runs_dir"] == tmp_path / "runs"
        assert kwargs["arch_name"] == "cpu"
        assert kwargs["n_workers"] == 4
        assert kwargs["skip_run"] is True
        assert kwargs["debug_turbulence"] is False
        assert kwargs["report_dir"] == tmp_path
        assert kwargs["report_generated_at"] == "2026-05-22T00:00:00Z"
        assert kwargs["report_hardware"] == {"execution_backend": "cpu"}
        return [_case_result(), _case_result()]

    monkeypatch.setattr(
        run_validation,
        "run_cases_parallel",
        fake_run_cases_parallel,
    )

    results = _run_cases(
        case_list=["couette", "channel"],
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path,
        selected_arch="cpu",
        n_workers=4,
        dashboard_port=8787,
        generated_at="2026-05-22T00:00:00Z",
        hardware={"execution_backend": "cpu"},
        skip_run=True,
        debug_turbulence=False,
        on_result=lambda result: None,
    )

    assert results == [_case_result(), _case_result()]


def test_run_cases_renders_multi_case_serially_with_one_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_run_cases_parallel(**kwargs: object) -> list[CaseResult]:
        raise AssertionError("one-worker validation should not enter Dask")

    def fake_validate_case_to_html(
        case_name: str,
        runs_dir: Path,
        output_dir: Path,
        *,
        generated_at: str,
        hardware: dict[str, str],
        skip_run: bool = False,
        debug_turbulence: bool = False,
    ) -> CaseResult:
        assert runs_dir == tmp_path / "runs"
        assert output_dir == tmp_path
        assert generated_at == "2026-05-22T00:00:00Z"
        assert hardware == {"execution_backend": "cpu"}
        assert skip_run
        assert not debug_turbulence
        calls.append(case_name)
        return _case_result()

    monkeypatch.setattr(
        run_validation,
        "run_cases_parallel",
        fail_run_cases_parallel,
    )
    monkeypatch.setattr(
        run_validation,
        "validate_case_to_html",
        fake_validate_case_to_html,
    )

    results = _run_cases(
        case_list=["couette", "channel"],
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path,
        selected_arch="cpu",
        n_workers=1,
        dashboard_port=8787,
        generated_at="2026-05-22T00:00:00Z",
        hardware={"execution_backend": "cpu"},
        skip_run=True,
        debug_turbulence=False,
        on_result=lambda result: None,
    )

    assert results == [_case_result(), _case_result()]
    assert calls == ["couette", "channel"]


def test_run_cases_renders_single_case_no_run_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail_run_cases_parallel(**kwargs: object) -> list[CaseResult]:
        raise AssertionError("single-case report rendering should not enter Dask")

    def fake_validate_case_to_html(
        case_name: str,
        runs_dir: Path,
        output_dir: Path,
        *,
        generated_at: str,
        hardware: dict[str, str],
        skip_run: bool = False,
        debug_turbulence: bool = False,
    ) -> CaseResult:
        assert runs_dir == tmp_path / "runs"
        assert output_dir == tmp_path
        assert generated_at == "2026-05-22T00:00:00Z"
        assert hardware == {"execution_backend": "cpu"}
        assert skip_run
        assert not debug_turbulence
        calls.append(case_name)
        return _case_result()

    monkeypatch.setattr(
        run_validation,
        "run_cases_parallel",
        fail_run_cases_parallel,
    )
    monkeypatch.setattr(
        run_validation,
        "validate_case_to_html",
        fake_validate_case_to_html,
    )

    results = _run_cases(
        case_list=["couette"],
        runs_dir=tmp_path / "runs",
        output_dir=tmp_path,
        selected_arch="cpu",
        n_workers=4,
        dashboard_port=8787,
        generated_at="2026-05-22T00:00:00Z",
        hardware={"execution_backend": "cpu"},
        skip_run=True,
        debug_turbulence=False,
        on_result=lambda result: None,
    )

    assert results == [_case_result()]
    assert calls == ["couette"]


def test_run_validation_cli_writes_html_and_report_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_validation,
        "detect_platform",
        lambda: PlatformInfo(
            cpu_count=2,
            gpu_count=0,
            available_archs=["cpu"],
            hardware={"cpu_count": "2"},
        ),
    )

    def fake_run_cases(**kwargs: object) -> list[CaseResult]:
        assert kwargs["skip_run"] is True
        assert kwargs["n_workers"] == 1
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        (output_dir / "couette-gotm.html").write_text("case", encoding="utf-8")
        return [_case_result()]

    def fake_write_html_index(report: Report, output_dir: Path) -> Path:
        path = output_dir / "report.html"
        path.write_text(report.verdict, encoding="utf-8")
        return path

    monkeypatch.setattr(run_validation, "_run_cases", fake_run_cases)
    monkeypatch.setattr(run_validation, "write_html_index", fake_write_html_index)

    result = CliRunner().invoke(
        run_validation.cli,
        [
            "--cases",
            "couette",
            "--no-run",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report.json").is_file()
    assert not (tmp_path / "results.json").exists()
    assert f"JSON  : {tmp_path / 'report.json'}" in result.output


def test_run_validation_cli_reports_unknown_case_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_validation,
        "detect_platform",
        lambda: PlatformInfo(
            cpu_count=2,
            gpu_count=0,
            available_archs=["cpu"],
            hardware={"cpu_count": "2"},
        ),
    )
    monkeypatch.setattr(
        run_validation,
        "trigger_numba_jit",
        lambda: pytest.fail("unknown cases should fail before warmup"),
    )

    result = CliRunner().invoke(
        run_validation.cli,
        [
            "--cases",
            "does_not_exist",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "ERROR: unknown GOTM reference case 'does_not_exist'" in result.output
    assert "Traceback" not in result.output
    assert "pyGOTM validation starting" not in result.output
