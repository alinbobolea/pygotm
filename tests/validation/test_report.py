"""Tests for validation/report.py — new three-indicator report structure."""

from __future__ import annotations

import json
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


def _make_var(
    name: str = "temp",
    section: str = "pygotm",
    status: str = "PASS",
    color: str = "green",
    primary_score: float = 0.1,
    birge_ratio: float = 0.05,
    bias: float = 0.0,
    plot_html: str | None = None,
) -> VarResult:
    return VarResult(
        name=name,
        section=section,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        color=color,  # type: ignore[arg-type]
        reference_at_worst=1.2345678901234567e-05,
        calculated_at_worst=1.2345678901234568e-05,
        primary_score=primary_score,
        birge_ratio=birge_ratio,
        normalized_signed_bias=bias,
        plot_html=plot_html,
    )


def _make_case(
    name: str = "couette",
    status: str = "PASS",
    variables: list[VarResult] | None = None,
) -> CaseResult:
    vars_ = variables or [_make_var()]
    n_pass = sum(1 for v in vars_ if v.status == "PASS")
    n_marginal = sum(1 for v in vars_ if v.status == "MARGINAL")
    n_discrepant = sum(1 for v in vars_ if v.status == "DISCREPANT")
    n_broken = sum(1 for v in vars_ if v.status == "BROKEN")
    return CaseResult(
        case_name=name,
        status=status,
        error=None,
        py_nc_path=f"/runs/{name}.nc",
        ref_nc_path=f"/ref/{name}.nc",
        wall_time_s=1.5,
        variables=vars_,
        n_pass=n_pass,
        n_marginal=n_marginal,
        n_discrepant=n_discrepant,
        n_broken=n_broken,
    )


def _make_report(cases: list[CaseResult]) -> Report:
    return Report(
        generated_at="2026-05-11T10:00:00Z",
        hardware={"cpu_model": "Test CPU", "cpu_count": "8"},
        cases=cases,
        verdict="FULL PARITY",
    )


# ---------------------------------------------------------------------------
# CaseResult structure
# ---------------------------------------------------------------------------


def test_case_result_has_four_status_counts() -> None:
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(CaseResult)}
    assert "n_pass" in field_names
    assert "n_marginal" in field_names
    assert "n_discrepant" in field_names
    assert "n_broken" in field_names
    assert "n_fail" not in field_names  # deprecated


def test_report_has_no_global_atol_rtol() -> None:
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(Report)}
    assert "atol" not in field_names
    assert "rtol" not in field_names


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_save_load_json_round_trips(tmp_path: Path) -> None:
    report = _make_report([_make_case("couette")])
    path = tmp_path / "results.json"
    save_json(report, path)
    loaded = load_json(path)
    assert loaded.verdict == "FULL PARITY"
    assert len(loaded.cases) == 1
    assert loaded.cases[0].case_name == "couette"
    assert loaded.cases[0].variables[0].name == "temp"
    assert loaded.cases[0].variables[0].primary_score == pytest.approx(0.1)


def test_json_preserves_full_precision_ref_calc(tmp_path: Path) -> None:
    v_orig = 1.2345678901234567e-05
    case = _make_case("couette")
    case.variables[0] = VarResult(
        name="temp", section="pygotm", status="PASS", color="green",
        reference_at_worst=v_orig, calculated_at_worst=v_orig,
        primary_score=0.1, birge_ratio=0.05, normalized_signed_bias=0.0,
        plot_html=None,
    )
    report = _make_report([case])
    path = tmp_path / "results.json"
    save_json(report, path)
    loaded = load_json(path)
    assert loaded.cases[0].variables[0].reference_at_worst == v_orig


def test_json_nan_written_as_null(tmp_path: Path) -> None:
    case = _make_case("couette")
    case.variables[0] = VarResult(
        name="temp", section="pygotm", status="PASS", color="green",
        reference_at_worst=float("nan"), calculated_at_worst=float("nan"),
        primary_score=float("nan"), birge_ratio=float("nan"),
        normalized_signed_bias=float("nan"), plot_html=None,
    )
    report = _make_report([case])
    path = tmp_path / "results.json"
    save_json(report, path)
    raw = json.loads(path.read_text())
    assert raw["cases"][0]["variables"][0]["primary_score"] is None


# ---------------------------------------------------------------------------
# HTML rendering — columns
# ---------------------------------------------------------------------------


def test_render_html_contains_required_column_headers() -> None:
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    for col in ("Status", "Variable", "Reference", "Calculated",
                "Primary score", "Birge ratio", "Normalized signed bias"):
        assert col in html, f"missing column header: {col}"


def test_render_html_does_not_contain_deprecated_column_headers() -> None:
    report = _make_report([_make_case("couette")])
    html = render_html(report)
    for deprecated in ("max_abs_err", "max_rel_err", "RMSE", "NRMSE",
                       "mean_abs_err", "mean_rel_err", "R2", "correlation"):
        assert deprecated not in html, f"deprecated column found: {deprecated}"


# ---------------------------------------------------------------------------
# HTML rendering — sections
# ---------------------------------------------------------------------------


def test_render_html_has_pygotm_and_pyfabm_sections() -> None:
    vars_ = [
        _make_var(name="temp", section="pygotm"),
        _make_var(name="oxygen", section="pyfabm"),
    ]
    report = _make_report([_make_case("seagrass", variables=vars_)])
    html = render_html(report)
    assert "PyGOTM" in html
    assert "PyFABM" in html


def test_render_html_separates_pygotm_from_pyfabm() -> None:
    """PyGOTM variables must appear before the PyFABM section header."""
    vars_ = [
        _make_var(name="temp", section="pygotm"),
        _make_var(name="oxygen", section="pyfabm"),
    ]
    report = _make_report([_make_case("seagrass", variables=vars_)])
    html = render_html(report)
    # Use section-specific headers to avoid matching methodology text earlier in the page
    pos_pygotm_header = html.find("PyGOTM variables")
    pos_pyfabm_header = html.find("PyFABM variables")
    pos_temp = html.find(">temp<")
    pos_oxygen = html.find(">oxygen<")
    assert pos_pygotm_header < pos_pyfabm_header
    assert pos_temp < pos_pyfabm_header
    assert pos_oxygen > pos_pygotm_header


# ---------------------------------------------------------------------------
# HTML rendering — plots
# ---------------------------------------------------------------------------


def test_render_html_embeds_plot_for_marginal_variable() -> None:
    FAKE_PLOT = "<div>PLOTLY_DIV_CONTENT_HERE</div>"
    vars_ = [_make_var(name="temp", section="pygotm", status="MARGINAL",
                       color="yellow", primary_score=2.0, plot_html=FAKE_PLOT)]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_html(report)
    assert "PLOTLY_DIV_CONTENT_HERE" in html


def test_render_html_no_plot_for_pass_variable() -> None:
    vars_ = [_make_var(name="temp", status="PASS", color="green", primary_score=0.1)]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_html(report)
    assert "PLOTLY_DIV_CONTENT_HERE" not in html


def test_render_html_includes_plotlyjs_cdn() -> None:
    vars_ = [_make_var(name="temp", section="pygotm", status="MARGINAL",
                       color="yellow", primary_score=2.0, plot_html="<div>x</div>")]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_html(report)
    assert "plotly" in html.lower()


# ---------------------------------------------------------------------------
# HTML rendering — color coding
# ---------------------------------------------------------------------------


def test_render_html_pass_status_green() -> None:
    vars_ = [_make_var(status="PASS", color="green")]
    html = render_html(_make_report([_make_case(variables=vars_)]))
    assert "green" in html or "#2e7d32" in html


def test_render_html_marginal_status_yellow() -> None:
    vars_ = [_make_var(status="MARGINAL", color="yellow", primary_score=2.0)]
    html = render_html(_make_report([_make_case(variables=vars_)]))
    assert "yellow" in html or "#f9a825" in html or "#fbc02d" in html or "MARGINAL" in html


def test_render_html_broken_status_red() -> None:
    vars_ = [_make_var(status="BROKEN", color="red", primary_score=50.0)]
    html = render_html(_make_report([_make_case(variables=vars_)]))
    assert "red" in html or "#c62828" in html or "BROKEN" in html


# ---------------------------------------------------------------------------
# HTML rendering — general
# ---------------------------------------------------------------------------


def test_render_html_contains_full_precision_ref_value() -> None:
    v = 1.2345678901234567e-05
    vars_ = [VarResult(
        name="temp", section="pygotm", status="PASS", color="green",
        reference_at_worst=v, calculated_at_worst=v,
        primary_score=0.1, birge_ratio=0.05, normalized_signed_bias=0.0,
        plot_html=None,
    )]
    report = _make_report([_make_case(variables=vars_)])
    html = render_html(report)
    assert repr(v) in html or format(v, ".17g") in html


def test_render_html_contains_case_name() -> None:
    report = _make_report([_make_case("seagrass")])
    html = render_html(report)
    assert "seagrass" in html


def test_render_html_contains_verdict() -> None:
    report = _make_report([_make_case()])
    html = render_html(report)
    assert "FULL PARITY" in html


def test_render_html_contains_hardware_section() -> None:
    report = _make_report([_make_case()])
    html = render_html(report)
    assert "Test CPU" in html


def test_render_html_multiple_cases() -> None:
    report = _make_report([_make_case("couette"), _make_case("channel")])
    html = render_html(report)
    assert "couette" in html
    assert "channel" in html


def test_fmt_time_seconds() -> None:
    assert _fmt_time(45.7) == "45.7s"


def test_fmt_time_minutes() -> None:
    assert _fmt_time(90.0) == "1m 30s"


def test_fmt_handles_nan_inf_none() -> None:
    assert _fmt(float("nan")) == "—"
    assert _fmt(float("inf")) == "—"
    assert _fmt(None) == "—"


def test_fmt_full_handles_nan() -> None:
    assert _fmt_full(float("nan")) == "—"


def test_fmt_full_handles_inf() -> None:
    assert _fmt_full(float("inf")) == "—"


def test_fmt_full_handles_none() -> None:
    assert _fmt_full(None) == "—"
