"""Tests for validation/report.py — data classes, serialization, and HTML rendering."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pygotm.validation.compare import VarResult
from pygotm.validation.report import (
    CaseResult,
    Report,
    _fmt,
    _fmt_full,
    _fmt_time,
    load_json,
    render_html,
    save_json,
)


def test_fmt_full_round_trips_exactly() -> None:
    values = [
        1.2345678901234567e-05,
        -9.876543210987654e+10,
        0.0,
        1.0,
        1.7976931348623157e+308,
    ]
    for v in values:
        assert float(_fmt_full(v)) == v, f"round-trip failed for {v!r}"


def test_fmt_full_handles_nan() -> None:
    assert _fmt_full(float("nan")) == "—"


def test_fmt_full_handles_inf() -> None:
    assert _fmt_full(float("inf")) == "—"
    assert _fmt_full(float("-inf")) == "—"


def test_fmt_full_handles_none() -> None:
    assert _fmt_full(None) == "—"


def test_fmt_full_differs_from_three_digit_scientific() -> None:
    v = 1.2345678901234567e-05
    assert _fmt_full(v) != _fmt(v, precision=3)


def test_fmt_precision() -> None:
    assert _fmt(1.23456e-5, precision=3) == "1.235e-05"


def test_fmt_handles_nan_inf_none() -> None:
    assert _fmt(float("nan")) == "—"
    assert _fmt(float("inf")) == "—"
    assert _fmt(None) == "—"


def test_fmt_time_seconds() -> None:
    assert _fmt_time(45.7) == "45.7s"


def test_fmt_time_minutes() -> None:
    assert _fmt_time(90.0) == "1m 30s"


def _make_case(name: str, status: str = "PASS") -> CaseResult:
    return CaseResult(
        case_name=name, status=status, error=None,
        py_nc_path=f"/runs/{name}.nc", ref_nc_path=f"/ref/{name}.nc",
        wall_time_s=1.5,
        variables=[
            VarResult(
                name="u", status="PASS",
                ref_at_worst=1.2345678901234567e-05,
                calc_at_worst=1.2345678901234568e-05,
                max_abs_err=1e-20, max_rel_err=1e-14,
                rmse=5e-21, nrmse=2e-14,
            )
        ],
        n_pass=1, n_fail=0, n_skip=0,
    )


def _make_report(cases: list[CaseResult]) -> Report:
    return Report(
        generated_at="2026-04-26T10:00:00Z",
        rtol=1e-6,
        atol=1e-12,
        hardware={"cpu_model": "Test CPU", "cpu_count": "8", "taichi_version": "1.7.4",
                  "python_version": "3.12.0", "platform": "Linux", "execution_backend": "cpu"},
        cases=cases,
        verdict="FULL PARITY",
    )


def test_save_load_json_round_trips(tmp_path: Path) -> None:
    report = _make_report([_make_case("couette")])
    path = tmp_path / "results.json"
    save_json(report, path)
    loaded = load_json(path)
    assert loaded.verdict == "FULL PARITY"
    assert len(loaded.cases) == 1
    assert loaded.cases[0].case_name == "couette"
    assert loaded.cases[0].variables[0].name == "u"


def test_save_json_writes_null_for_nan(tmp_path: Path) -> None:
    case = _make_case("couette")
    case.variables[0].nrmse = float("nan")
    report = _make_report([case])
    path = tmp_path / "results.json"
    save_json(report, path)
    raw = json.loads(path.read_text())
    assert raw["cases"][0]["variables"][0]["nrmse"] is None


def test_load_json_handles_missing_hardware(tmp_path: Path) -> None:
    report = _make_report([_make_case("couette")])
    path = tmp_path / "results.json"
    save_json(report, path)
    data = json.loads(path.read_text())
    del data["hardware"]
    path.write_text(json.dumps(data))
    loaded = load_json(path)
    assert loaded.hardware == {}


def test_var_result_ref_calc_values_preserve_full_precision(tmp_path: Path) -> None:
    v_orig = 1.2345678901234567e-05
    case = _make_case("couette")
    case.variables[0].ref_at_worst = v_orig
    report = _make_report([case])
    path = tmp_path / "results.json"
    save_json(report, path)
    loaded = load_json(path)
    assert loaded.cases[0].variables[0].ref_at_worst == v_orig


def test_render_html_returns_string() -> None:
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    assert isinstance(html, str)
    assert len(html) > 100


def test_render_html_contains_verdict() -> None:
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    assert "FULL PARITY" in html


def test_render_html_contains_case_name() -> None:
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    assert "couette" in html


def test_render_html_contains_full_precision_values() -> None:
    v = 1.2345678901234567e-05
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    assert repr(v) in html


def test_render_html_contains_hardware_section() -> None:
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    assert "Test CPU" in html
    assert "Execution backend" in html


def test_render_html_shows_gpu_info_when_present() -> None:
    report = _make_report([_make_case("couette")])
    report.hardware["gpu_info"] = "CUDA: RTX 4090"
    html = render_html(report)
    assert "RTX 4090" in html


def test_render_html_fail_case_is_open_by_default() -> None:
    fail_case = _make_case("channel", status="FAIL")
    fail_case.variables[0].status = "FAIL"
    fail_case.n_fail = 1
    fail_case.n_pass = 0
    report = _make_report([fail_case])
    html = render_html(report)
    assert 'data-status="FAIL" open' in html


def test_render_html_pass_case_is_closed_by_default() -> None:
    report = _make_report([_make_case("couette", status="PASS")])
    html = render_html(report)
    assert 'data-status="PASS" open' not in html


def test_render_html_multiple_cases() -> None:
    report = _make_report([_make_case("couette"), _make_case("channel")])
    html = render_html(report)
    assert "couette" in html
    assert "channel" in html
