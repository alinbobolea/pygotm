"""Tests for validation/report.py Frechet report structure."""

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
    render_case_html,
    render_html,
    save_json,
    write_html_reports,
)


def _make_var(
    name: str = "temp",
    section: str = "pygotm",
    status: str = "PASS",
    color: str = "green",
    d_raw: float = 1.0e-13,
    d_norm: float = 0.0,
    plot_html: str | None = None,
    metric_mode: str = "d_norm",
    score: float | None = None,
    peak_d_norm: float | None = None,
) -> VarResult:
    return VarResult(
        name=name,
        section=section,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        color=color,  # type: ignore[arg-type]
        reference_at_worst=1.2345678901234567e-05,
        calculated_at_worst=1.2345678901234568e-05,
        d_raw=d_raw,
        d_norm=d_norm,
        plot_html=plot_html,
        metric_mode=metric_mode,  # type: ignore[arg-type]
        score=score,
        peak_d_norm=peak_d_norm,
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
        task_name=f"{name}-gotm",
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


def test_case_result_has_four_status_counts() -> None:
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(CaseResult)}
    assert "n_pass" in field_names
    assert "n_marginal" in field_names
    assert "n_discrepant" in field_names
    assert "n_broken" in field_names
    assert "n_fail" not in field_names


def test_report_has_no_global_atol_rtol() -> None:
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(Report)}
    assert "atol" not in field_names
    assert "rtol" not in field_names


def test_save_load_json_round_trips(tmp_path: Path) -> None:
    report = _make_report([_make_case("couette")])
    path = tmp_path / "results.json"
    save_json(report, path)
    loaded = load_json(path)
    assert loaded.verdict == "FULL PARITY"
    assert len(loaded.cases) == 1
    assert loaded.cases[0].case_name == "couette"
    assert loaded.cases[0].task_name == "couette-gotm"
    assert loaded.cases[0].variables[0].name == "temp"
    assert loaded.cases[0].variables[0].d_norm == pytest.approx(0.0)


def test_load_json_migrates_old_primary_score_results(tmp_path: Path) -> None:
    path = tmp_path / "old-results.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-11T10:00:00Z",
                "hardware": {},
                "verdict": "PARTIAL PARITY",
                "cases": [
                    {
                        "case_name": "couette",
                        "status": "FAIL",
                        "error": None,
                        "py_nc_path": "py.nc",
                        "ref_nc_path": "ref.nc",
                        "wall_time_s": 1.0,
                        "task_name": "couette-gotm",
                        "n_pass": 0,
                        "n_marginal": 1,
                        "n_discrepant": 0,
                        "n_broken": 0,
                        "variables": [
                            {
                                "name": "temp",
                                "section": "pygotm",
                                "status": "MARGINAL",
                                "color": "yellow",
                                "reference_at_worst": 1.0,
                                "calculated_at_worst": 1.01,
                                "primary_score": 0.02,
                                "birge_ratio": 0.01,
                                "normalized_signed_bias": 0.0,
                                "plot_html": None,
                            }
                        ],
                    }
                ],
            }
        )
    )
    loaded = load_json(path)
    assert loaded.cases[0].variables[0].d_norm == pytest.approx(0.02)
    assert loaded.cases[0].variables[0].metric_mode == "d_norm"
    assert loaded.cases[0].variables[0].primary_score == pytest.approx(0.02)


def test_var_result_from_json_defaults_metric_mode_to_dnorm() -> None:
    from pygotm.validation.report import _var_result_from_json

    result = _var_result_from_json(
        {
            "name": "temp",
            "section": "pygotm",
            "status": "PASS",
            "color": "green",
            "reference_at_worst": 1.0,
            "calculated_at_worst": 1.0,
            "d_raw": 0.0,
            "d_norm": 0.0,
            "plot_html": None,
        }
    )
    assert result.metric_mode == "d_norm"
    assert result.primary_score == pytest.approx(0.0)


def test_var_result_from_json_preserves_drel_metric_mode_and_score() -> None:
    from pygotm.validation.report import _var_result_from_json

    result = _var_result_from_json(
        {
            "name": "NN",
            "section": "pygotm",
            "status": "PASS",
            "color": "green",
            "reference_at_worst": 1.3e-6,
            "calculated_at_worst": 1.3039e-6,
            "d_raw": 3.9e-9,
            "d_norm": 0.013,
            "plot_html": None,
            "metric_mode": "d_rel",
            "score": 0.003,
        }
    )
    assert result.metric_mode == "d_rel"
    assert result.primary_score == pytest.approx(0.003)


def test_json_preserves_full_precision_ref_calc(tmp_path: Path) -> None:
    v_orig = 1.2345678901234567e-05
    case = _make_case("couette")
    case.variables[0] = VarResult(
        name="temp",
        section="pygotm",
        status="PASS",
        color="green",
        reference_at_worst=v_orig,
        calculated_at_worst=v_orig,
        d_raw=0.0,
        d_norm=0.0,
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
        name="temp",
        section="pygotm",
        status="PASS",
        color="green",
        reference_at_worst=float("nan"),
        calculated_at_worst=float("nan"),
        d_raw=float("nan"),
        d_norm=float("nan"),
        plot_html=None,
    )
    report = _make_report([case])
    path = tmp_path / "results.json"
    save_json(report, path)
    raw = json.loads(path.read_text())
    assert raw["cases"][0]["variables"][0]["d_norm"] is None


def test_render_html_contains_required_column_headers() -> None:
    report = _make_report([_make_case("couette")])
    html = render_case_html(report, report.cases[0])
    for col in (
        "Status",
        "Variable",
        "Reference",
        "Calculated",
        "Raw Frechet",
        "Normalized Frechet",
        "d_rel",
        "Peak-sensitive d_norm",
    ):
        assert col in html, f"missing column header: {col}"


def test_render_html_does_not_contain_old_metric_headers() -> None:
    report = _make_report([_make_case("couette")])
    html = render_case_html(report, report.cases[0])
    for old in ("Primary score", "Birge ratio", "Normalized signed bias"):
        assert old not in html, f"old column found: {old}"


def test_var_rows_html_shows_drel_score_and_metric_mode() -> None:
    from pygotm.validation.report import _var_rows_html

    v = _make_var(
        name="NN",
        d_raw=4.2e-9,
        d_norm=0.013,
        metric_mode="d_rel",
        score=4.2e-9 / 1.3e-6,
    )
    html = _var_rows_html([v])
    assert "3.231e-03" in html
    assert "d_rel" in html
    assert "1.300e-02" not in html


def test_var_rows_html_omits_default_dnorm_metric_label() -> None:
    from pygotm.validation.report import _var_rows_html

    html = _var_rows_html([_make_var(d_norm=0.002, metric_mode="d_norm")])
    assert "2.000e-03" in html
    assert "(d_norm)" not in html


def test_var_rows_html_shows_peak_sensitive_dnorm() -> None:
    from pygotm.validation.report import _var_rows_html

    html = _var_rows_html([_make_var(d_norm=0.002, peak_d_norm=0.08)])

    assert "8.000e-02" in html


def test_render_html_has_pygotm_and_pyfabm_sections() -> None:
    vars_ = [
        _make_var(name="temp", section="pygotm"),
        _make_var(name="oxygen", section="pyfabm"),
    ]
    report = _make_report([_make_case("seagrass", variables=vars_)])
    html = render_case_html(report, report.cases[0])
    assert "PyGOTM" in html
    assert "PyFABM" in html


def test_render_html_separates_pygotm_from_pyfabm() -> None:
    vars_ = [
        _make_var(name="temp", section="pygotm"),
        _make_var(name="oxygen", section="pyfabm"),
    ]
    report = _make_report([_make_case("seagrass", variables=vars_)])
    html = render_case_html(report, report.cases[0])
    pos_pygotm_header = html.find("PyGOTM variables")
    pos_pyfabm_header = html.find("PyFABM variables")
    pos_temp = html.find(">temp<")
    pos_oxygen = html.find(">oxygen<")
    assert pos_pygotm_header < pos_pyfabm_header
    assert pos_temp < pos_pyfabm_header
    assert pos_oxygen > pos_pygotm_header


def test_render_html_embeds_plot_for_marginal_variable() -> None:
    fake_plot = "<div>PLOTLY_DIV_CONTENT_HERE</div>"
    vars_ = [
        _make_var(
            name="temp",
            section="pygotm",
            status="MARGINAL",
            color="yellow",
            d_norm=0.02,
            plot_html=fake_plot,
        )
    ]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_case_html(report, report.cases[0])
    assert "PLOTLY_DIV_CONTENT_HERE" in html


def test_render_html_no_plot_for_pass_variable() -> None:
    vars_ = [_make_var(name="temp", status="PASS", color="green", d_norm=0.0)]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_case_html(report, report.cases[0])
    assert "PLOTLY_DIV_CONTENT_HERE" not in html


def test_render_html_includes_plotlyjs_cdn() -> None:
    vars_ = [
        _make_var(
            name="temp",
            section="pygotm",
            status="MARGINAL",
            color="yellow",
            d_norm=0.02,
            plot_html="<div>x</div>",
        )
    ]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_case_html(report, report.cases[0])
    assert "plotly" in html.lower()


def test_render_html_pass_status_green() -> None:
    vars_ = [_make_var(status="PASS", color="green")]
    html = render_html(_make_report([_make_case(variables=vars_)]))
    assert "green" in html or "#2e7d32" in html


def test_render_html_marginal_status_yellow() -> None:
    vars_ = [_make_var(status="MARGINAL", color="yellow", d_norm=0.02)]
    html = render_html(_make_report([_make_case(variables=vars_)]))
    assert "yellow" in html or "#f9a825" in html or "MARGINAL" in html


def test_render_html_broken_status_red() -> None:
    vars_ = [_make_var(status="BROKEN", color="red", d_norm=0.25)]
    html = render_html(_make_report([_make_case(variables=vars_)]))
    assert "red" in html or "#c62828" in html or "BROKEN" in html


def test_render_html_contains_full_precision_ref_value() -> None:
    v = 1.2345678901234567e-05
    vars_ = [
        VarResult(
            name="temp",
            section="pygotm",
            status="PASS",
            color="green",
            reference_at_worst=v,
            calculated_at_worst=v,
            d_raw=0.0,
            d_norm=0.0,
            plot_html=None,
        )
    ]
    report = _make_report([_make_case(variables=vars_)])
    html = render_case_html(report, report.cases[0])
    assert repr(v) in html or format(v, ".17g") in html


def test_render_html_is_lightweight_case_index() -> None:
    vars_ = [
        _make_var(
            name="temp",
            section="pygotm",
            status="MARGINAL",
            color="yellow",
            d_norm=0.02,
            plot_html="<div>PLOTLY_DIV_CONTENT_HERE</div>",
        )
    ]
    report = _make_report([_make_case("couette", variables=vars_)])
    html = render_html(report)
    assert 'name="case-frame"' in html
    assert "couette-gotm.html" in html
    assert "PLOTLY_DIV_CONTENT_HERE" not in html


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


def test_write_html_reports_writes_index_and_case_pages(tmp_path: Path) -> None:
    report = _make_report([_make_case("couette"), _make_case("channel")])
    index_path = write_html_reports(report, tmp_path)
    assert index_path == tmp_path / "report.html"
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "couette-gotm.html").is_file()
    assert (tmp_path / "channel-gotm.html").is_file()


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
