"""Validation report data classes, JSON serialization, and HTML rendering."""

# ruff: noqa: E501

from __future__ import annotations

import html as _html
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from pygotm.validation.compare import VarResult

__all__ = [
    "CaseResult",
    "Report",
    "load_json",
    "render_case_html",
    "render_html",
    "save_json",
    "write_case_html",
    "write_html_index",
    "write_html_reports",
]

_STATUS_COLORS: dict[str, str] = {
    "PASS": "#2e7d32",
    "FAIL": "#c62828",
    "MARGINAL": "#f9a825",
    "DISCREPANT": "#e65100",
    "BROKEN": "#c62828",
    "ERROR": "#c62828",
}

_BADGE_CLASSES: dict[str, str] = {
    "PASS": "badge-pass",
    "FAIL": "badge-broken",
    "MARGINAL": "badge-marginal",
    "DISCREPANT": "badge-discrepant",
    "BROKEN": "badge-broken",
    "ERROR": "badge-error",
}


@dataclass
class CaseResult:
    case_name: str
    status: str  # "PASS" | "FAIL" | "ERROR"
    error: str | None
    py_nc_path: str
    ref_nc_path: str
    wall_time_s: float
    task_name: str | None = None
    variables: list[VarResult] = field(default_factory=list)
    n_pass: int = 0
    n_marginal: int = 0
    n_discrepant: int = 0
    n_broken: int = 0


@dataclass
class Report:
    generated_at: str
    hardware: dict[str, str]
    cases: list[CaseResult]
    verdict: str  # "FULL PARITY" | "PARTIAL PARITY" | "FAILED VALIDATION"


def _sanitise(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


def save_json(report: Report, path: Path) -> None:
    data = _sanitise(asdict(report))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Report:
    with open(path) as f:
        data = json.load(f)
    cases = []
    for cd in data["cases"]:
        case_data = dict(cd)
        vars_data = case_data.pop("variables", [])
        vars_ = [_var_result_from_json(v) for v in vars_data]
        case_data.setdefault("n_marginal", 0)
        case_data.setdefault("n_discrepant", 0)
        case_data.setdefault("n_broken", 0)
        case_data.setdefault("task_name", None)
        case_data.pop("n_fail", None)
        case_data.pop("n_skip", None)
        cases.append(CaseResult(**case_data, variables=vars_))
    data["cases"] = cases
    data.setdefault("hardware", {})
    data.pop("atol", None)
    data.pop("rtol", None)
    return Report(**data)


def _json_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    return float(value)


def _var_result_from_json(data: dict[str, Any]) -> VarResult:
    raw = dict(data)
    if "d_norm" not in raw and "primary_score" in raw:
        raw["d_norm"] = raw["primary_score"]
    raw.setdefault("d_raw", float("nan"))
    raw.pop("primary_score", None)
    raw.pop("birge_ratio", None)
    raw.pop("normalized_signed_bias", None)
    raw["reference_at_worst"] = _json_float(raw.get("reference_at_worst"))
    raw["calculated_at_worst"] = _json_float(raw.get("calculated_at_worst"))
    raw["d_raw"] = _json_float(raw.get("d_raw"))
    raw["d_norm"] = _json_float(raw.get("d_norm"))
    raw.setdefault("plot_html", None)
    raw.setdefault("metric_mode", "d_norm")
    raw["score"] = _json_float(raw.get("score"), raw["d_norm"])
    raw["peak_d_norm"] = (
        None if raw.get("peak_d_norm") is None else _json_float(raw.get("peak_d_norm"))
    )
    return VarResult(**raw)


def _fmt(v: float | None, precision: int = 3) -> str:
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return "—"
    return f"{v:.{precision}e}"


def _fmt_full(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return "—"
    return format(v, ".17g")


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.0f}s"


def _status_cell(status: str, label: str | None = None) -> str:
    colour = _STATUS_COLORS.get(status, "#333")
    text = label or status
    return f'<td style="color:{colour};font-weight:bold">{text}</td>'


def _hardware_section(hw: dict[str, str]) -> str:
    if not hw:
        return ""
    rows: list[tuple[str, str]] = [
        ("CPU", hw.get("cpu_model", "unknown")),
        ("Logical CPUs", hw.get("cpu_count", "?")),
        ("RAM", hw.get("ram_total", "unknown")),
        ("Execution backend", hw.get("execution_backend", "cpu")),
        ("Numba", hw.get("numba_version", "unknown")),
        ("Python", hw.get("python_version", "unknown")),
        ("OS", hw.get("platform", "unknown")),
    ]
    if hw.get("gpu_info"):
        rows.insert(2, ("GPU", hw["gpu_info"]))
    cells = "".join(
        f"<tr><th>{label}</th><td>{_html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"""
<div class="hardware-box">
  <h3 style="margin:0 0 .5em">Hardware</h3>
  <table class="hw-table"><tbody>{cells}</tbody></table>
</div>"""


def _var_rows_html(variables: list[VarResult]) -> str:
    rows = ""
    for v in variables:
        colour = _STATUS_COLORS.get(v.status, "#333")
        metric_label = (
            f" <small>({_html.escape(v.metric_mode)})</small>"
            if v.metric_mode == "d_rel"
            else ""
        )
        rows += (
            "<tr>"
            f'<td style="color:{colour};font-weight:bold">{v.status}</td>'
            f"<td>{_html.escape(v.name)}</td>"
            f"<td><code>{_fmt_full(v.reference_at_worst)}</code></td>"
            f"<td><code>{_fmt_full(v.calculated_at_worst)}</code></td>"
            f"<td>{_fmt(v.d_raw)}</td>"
            f"<td>{_fmt(v.primary_score)}{metric_label}</td>"
            f"<td>{_fmt(v.peak_d_norm)}</td>"
            f"<td>{'—' if v.plot_html is None else '↓ see below'}</td>"
            f"</tr>\n"
        )
        if v.plot_html is not None:
            rows += (
                '<tr><td colspan="8" style="padding:0;border-top:none">'
                f'<div class="plot-container">{v.plot_html}</div>'
                "</td></tr>\n"
            )
    return rows


def _section_html(section_label: str, variables: list[VarResult]) -> str:
    if not variables:
        return (
            f'<p style="color:#888;font-style:italic">No {section_label} variables.</p>'
        )
    table_header = (
        "<table>"
        "<thead><tr>"
        "<th>Status</th><th>Variable</th>"
        "<th>Reference (full precision)</th>"
        "<th>Calculated (full precision)</th>"
        "<th>Raw Frechet</th>"
        "<th>Score (Normalized Frechet / d_rel)</th>"
        "<th>Peak-sensitive d_norm</th>"
        "<th>Parameter plot</th>"
        "</tr></thead>"
        f"<tbody>{_var_rows_html(variables)}</tbody></table>"
    )
    return f'<h4 style="margin:1em 0 .4em">{section_label} variables</h4>{table_header}'


def _case_report_stem(case: CaseResult) -> str:
    if case.task_name:
        return case.task_name
    if "-" in case.case_name:
        return case.case_name
    return f"{case.case_name}-gotm"


def _case_report_filename(case: CaseResult) -> str:
    return f"{_case_report_stem(case)}.html"


def _case_counts(case: CaseResult) -> tuple[int, int, int, int, int]:
    n_total = case.n_pass + case.n_marginal + case.n_discrepant + case.n_broken
    return (
        case.n_pass,
        case.n_marginal,
        case.n_discrepant,
        case.n_broken,
        n_total,
    )


def _summary_rows_html(report: Report) -> str:
    rows = ""
    for c in report.cases:
        n_pass, n_marginal, n_discrepant, n_broken, n_total = _case_counts(c)
        time_str = _fmt_time(c.wall_time_s) if c.wall_time_s > 0 else "—"
        badge_cls = _BADGE_CLASSES.get(c.status, "badge-error")
        case_link = _html.escape(quote(_case_report_filename(c), safe=""))
        rows += (
            f"<tr>"
            f'<td><a href="{case_link}" target="case-frame">{_html.escape(c.case_name)}</a></td>'
            f'<td><span class="case-badge {badge_cls}">{c.status}</span></td>'
            f"<td>{n_pass}</td><td>{n_marginal}</td>"
            f"<td>{n_discrepant}</td><td>{n_broken}</td>"
            f"<td>{n_total}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{time_str}</td>"
            f"</tr>\n"
        )
    total_wall = sum(c.wall_time_s for c in report.cases)
    rows += (
        f"<tr style='font-weight:bold;border-top:2px solid #ccc'>"
        f"<td>TOTAL ({len(report.cases)} cases)</td><td>—</td>"
        f"<td>{sum(c.n_pass for c in report.cases)}</td>"
        f"<td>{sum(c.n_marginal for c in report.cases)}</td>"
        f"<td>{sum(c.n_discrepant for c in report.cases)}</td>"
        f"<td>{sum(c.n_broken for c in report.cases)}</td>"
        f"<td>—</td>"
        f"<td style='font-variant-numeric:tabular-nums'>{_fmt_time(total_wall)}</td>"
        f"</tr>\n"
    )
    return rows


def _case_nav_html(report: Report) -> str:
    links = ""
    for index, case in enumerate(report.cases):
        filename = _case_report_filename(case)
        href = _html.escape(quote(filename, safe=""))
        badge_cls = _BADGE_CLASSES.get(case.status, "badge-error")
        selected_class = " is-selected" if index == 0 else ""
        links += f"""
  <a class="case-link{selected_class}" href="{href}" target="case-frame" data-case-link>
    <span class="case-name">{_html.escape(_case_report_stem(case))}</span>
    <span class="case-badge {badge_cls}">{case.status}</span>
  </a>"""
    return links


def _case_section_html(case: CaseResult) -> str:
    time_str = _fmt_time(case.wall_time_s) if case.wall_time_s > 0 else "—"
    badge_cls = _BADGE_CLASSES.get(case.status, "badge-error")

    if case.status == "ERROR":
        return f"""
<section class="case-section" data-status="ERROR">
  <header class="case-summary">
    <h1>{_html.escape(case.case_name)}</h1>
    <span class="case-badge badge-error">ERROR</span>
    <span class="case-meta">wall time: {time_str}</span>
  </header>
  <p style="color:#c62828;margin:.6em 0 0">ERROR: {_html.escape(case.error or "unknown error")}</p>
</section>
"""

    n_pass, n_marginal, n_discrepant, n_broken, _ = _case_counts(case)
    pygotm_vars = [v for v in case.variables if v.section == "pygotm"]
    pyfabm_vars = [v for v in case.variables if v.section == "pyfabm"]

    return f"""
<section class="case-section" data-status="{case.status}">
  <header class="case-summary">
    <h1>{_html.escape(case.case_name)}</h1>
    <span class="case-badge {badge_cls}">{case.status}</span>
    <span class="case-meta">
      PASS: {n_pass} &nbsp;·&nbsp;
      MARGINAL: {n_marginal} &nbsp;·&nbsp;
      DISCREPANT: {n_discrepant} &nbsp;·&nbsp;
      BROKEN: {n_broken} &nbsp;·&nbsp;
      wall time: {time_str}
    </span>
  </header>
  <div class="case-body">
    <p style="font-size:0.85em;color:#555;margin:.4em 0">
      Python: <code>{_html.escape(case.py_nc_path)}</code><br>
      Reference: <code>{_html.escape(case.ref_nc_path)}</code>
    </p>
    {_section_html("1. PyGOTM", pygotm_vars)}
    {_section_html("2. PyFABM", pyfabm_vars)}
  </div>
</section>
"""


def render_case_html(report: Report, case: CaseResult) -> str:
    """Render one validation case as a standalone HTML report page."""
    total_wall = sum(c.wall_time_s for c in report.cases)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pyGOTM Validation Case: {_html.escape(case.case_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body   {{ font-family: system-ui, sans-serif; margin: 1.25rem; color: #222; }}
  h1     {{ margin: 0; font-size: 1.35rem; }}
  h4     {{ color: #444; }}
  table  {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #ddd; padding: .35em .6em; text-align: left; white-space: nowrap; }}
  th     {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover td {{ background: #fafafa; }}
  code   {{ background: #f5f5f5; padding: .1em .4em; border-radius: 3px; font-size: 0.9em; }}
  .case-section {{ max-width: 1600px; margin: 0 auto; }}
  .case-summary {{ display: flex; align-items: center; gap: .7em; border-bottom: 1px solid #ddd; padding-bottom: .7em; }}
  .case-meta  {{ font-size: 0.8em; color: #666; margin-left: auto; }}
  .case-badge {{ font-size: 0.8em; font-weight: 700; padding: .15em .5em; border-radius: 3px; }}
  .badge-pass       {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-marginal   {{ background: #fff9c4; color: #f57f17; }}
  .badge-discrepant {{ background: #ffe0b2; color: #e65100; }}
  .badge-broken     {{ background: #ffebee; color: #c62828; }}
  .badge-error      {{ background: #ffebee; color: #c62828; }}
  .case-body  {{ padding-bottom: 1em; }}
  .plot-container {{ margin: .5em 0; }}
  .report-meta {{ color:#666; font-size:0.8em; margin:.4em 0 1em; }}
</style>
</head>
<body>
<p class="report-meta">Generated: {report.generated_at} &nbsp;·&nbsp; Total validation wall time: {_fmt_time(total_wall)}</p>
{_case_section_html(case)}
</body>
</html>
"""


def render_html(report: Report) -> str:
    """Render the lightweight validation index with links to case reports."""
    cases_passed = sum(1 for c in report.cases if c.status == "PASS")
    total_cases = len(report.cases)
    verdict_colour = (
        "#2e7d32"
        if report.verdict == "FULL PARITY"
        else ("#e65100" if report.verdict == "PARTIAL PARITY" else "#c62828")
    )
    total_wall = sum(c.wall_time_s for c in report.cases)
    first_case_src = (
        _html.escape(quote(_case_report_filename(report.cases[0]), safe=""))
        if report.cases
        else "about:blank"
    )
    hw_html = _hardware_section(report.hardware)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pyGOTM Validation Report</title>
<script>
MathJax = {{ tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  *      {{ box-sizing: border-box; }}
  body   {{ font-family: system-ui, sans-serif; margin: 0; color: #1f2937; background: #f7f8fb; }}
  .page  {{ max-width: 1800px; margin: 0 auto; padding: 1.25rem 1.5rem 1.5rem; }}
  h1     {{ font-size: 1.25rem; margin: 0; }}
  h2     {{ margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: .3em; font-size: 1rem; }}
  h4     {{ color: #444; }}
  table  {{ border-collapse: collapse; width: 100%; margin: .8em 0; font-size: 0.85em; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: .35em .6em; text-align: left; white-space: nowrap; }}
  th     {{ background: #f1f5f9; font-weight: 600; }}
  tr:hover td {{ background: #f8fafc; }}
  a      {{ color: #2563eb; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .topbar {{ display: flex; align-items: center; gap: .8em; padding: .9em 1.1em; background: white; border: 1px solid #d9e2ec; border-radius: 6px; }}
  .topbar-meta {{ color: #64748b; font-size: .85em; }}
  .verdict {{ font-size: .9em; font-weight: bold; padding: .25em .55em;
               border: 1px solid; border-radius: 4px; margin-left: auto; }}
  code   {{ background: #f5f5f5; padding: .1em .4em; border-radius: 3px; font-size: 0.9em; }}
  .hardware-box {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px;
                   padding: .8em 1.2em; margin: 1em 0; display: inline-block; min-width: 420px; }}
  .hw-table {{ margin: 0; font-size: 0.88em; border: none; }}
  .hw-table th, .hw-table td {{ border: none; padding: .2em .7em .2em 0; background: transparent; }}
  .hw-table th {{ width: 140px; color: #555; font-weight: 600; }}
  .case-badge {{ font-size: 0.8em; font-weight: 700; padding: .15em .5em; border-radius: 3px; }}
  .badge-pass       {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-marginal   {{ background: #fff9c4; color: #f57f17; }}
  .badge-discrepant {{ background: #ffe0b2; color: #e65100; }}
  .badge-broken     {{ background: #ffebee; color: #c62828; }}
  .badge-error      {{ background: #ffebee; color: #c62828; }}
  .methodology {{ background: white; border: 1px solid #d9e2ec; border-radius: 6px;
                  padding: 1em 1.4em; margin: 1em 0; }}
  .status-band {{ display: grid; grid-template-columns: auto auto auto 1fr; gap: .3em 1em;
                  font-size: 0.88em; margin-top: .6em; }}
  .case-browser {{ display: grid; grid-template-columns: 280px minmax(0, 1fr); min-height: 680px; margin-top: 1rem; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; background: white; }}
  .case-nav {{ border-right: 1px solid #cbd5e1; background: #f8fafc; overflow: auto; }}
  .case-nav h2 {{ margin: 0; padding: .8em 1em; border-bottom: 1px solid #cbd5e1; font-size: .85em; letter-spacing: .04em; text-transform: uppercase; color: #64748b; }}
  .case-link {{ display: flex; align-items: center; justify-content: space-between; gap: .7em; padding: .65em .85em; border-bottom: 1px solid #e2e8f0; color: #1f2937; }}
  .case-link:hover {{ background: #eaf2ff; text-decoration: none; }}
  .case-link.is-selected {{ background: #dbeafe; border-left: 4px solid #2563eb; padding-left: calc(.85em - 4px); }}
  .case-name {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .case-frame {{ width: 100%; height: 100%; min-height: 680px; border: 0; background: white; }}
  @media (max-width: 900px) {{
    .page {{ padding: .8rem; }}
    .topbar {{ align-items: flex-start; flex-direction: column; }}
    .verdict {{ margin-left: 0; }}
    .hardware-box {{ min-width: 0; width: 100%; }}
    .case-browser {{ grid-template-columns: 1fr; }}
    .case-nav {{ border-right: 0; border-bottom: 1px solid #cbd5e1; max-height: 260px; }}
  }}
</style>
</head>
<body>
<div class="page">
<div class="topbar">
  <h1>pyGOTM Validation Report</h1>
  <span class="topbar-meta">Generated: {report.generated_at} &nbsp;·&nbsp; Total wall time: {_fmt_time(total_wall)}</span>
  <div class="verdict" style="color:{verdict_colour};border-color:{verdict_colour}">
    {report.verdict} &nbsp; ({cases_passed}/{total_cases} cases passed)
  </div>
</div>

{hw_html}

<h2>Validation Methodology</h2>
<div class="methodology">
  <p style="font-size:0.85em;color:#555;margin:.2em 0 .8em">
    Discrete Frechet distance is computed per numeric variable after aligning both outputs onto a shared time grid.
    Status is driven by the score: normalized Frechet distance for physically meaningful signal magnitudes, or relative raw distance below the variable magnitude floor.
  </p>
  <p style="font-size:0.85em">
    <strong>Raw distance:</strong> \\(d_{{raw}} = F(\\text{{ref}}, \\text{{calc}})\\), the discrete Frechet distance on original values.<br>
    <strong>Normalized distance:</strong> \\(d_{{norm}} = F(N(\\text{{ref}}), N(\\text{{calc}}))\\), where \\(N\\) is dynamic linear/log range normalization.<br>
    <strong>Relative raw distance:</strong> \\(d_{{rel}} = d_{{raw}} / \\max(|\\text{{ref}}|, |\\text{{calc}}|)\\), used when the signal magnitude is below its variable floor.
  </p>
  <p style="font-size:0.85em">
    If \\(d_{{raw}} &lt; 10^{{-12}}\\), both distances are reported as zero.
    Reference and calculated values are shown at the largest absolute aligned difference.
  </p>
  <div class="status-band">
    <span style="color:#2e7d32;font-weight:bold">PASS</span>     <span>green</span>  <span>score \\(&lt; 0.01\\)</span>       <span>Shape-equivalent within threshold.</span>
    <span style="color:#f9a825;font-weight:bold">MARGINAL</span> <span>yellow</span> <span>score \\(0.01\\) to \\(&lt; 0.05\\)</span>  <span>Small shape deviation. Plot generated.</span>
    <span style="color:#e65100;font-weight:bold">DISCREPANT</span><span>orange</span><span>score \\(0.05\\) to \\(&lt; 0.20\\)</span> <span>Deterministic implementation difference likely. Plot generated.</span>
    <span style="color:#c62828;font-weight:bold">BROKEN</span>    <span>red</span>   <span>score \\(\\geq 0.20\\)</span>          <span>Severe mismatch or structural comparison failure.</span>
  </div>
  <p style="font-size:0.85em;margin-top:.6em">
    Comparison plots are generated only for <strong>MARGINAL</strong> and <strong>DISCREPANT</strong> variables.
    Each case is split into <strong>PyGOTM</strong> and <strong>PyFABM</strong> sections.
  </p>
</div>

<h2>Summary</h2>
<table>
  <thead>
    <tr>
      <th>Case</th><th>Status</th>
      <th>PASS</th><th>MARGINAL</th><th>DISCREPANT</th><th>BROKEN</th>
      <th>Total vars</th><th>Wall time</th>
    </tr>
  </thead>
  <tbody>{_summary_rows_html(report)}</tbody>
</table>

<div class="case-browser">
  <nav class="case-nav" aria-label="Case reports">
    <h2>Cases</h2>
    {_case_nav_html(report)}
  </nav>
  <iframe class="case-frame" name="case-frame" src="{first_case_src}" title="Selected validation case report"></iframe>
</div>

<script>
document.querySelectorAll('[data-case-link]').forEach(link => {{
  link.addEventListener('click', () => {{
    document.querySelectorAll('[data-case-link]').forEach(item => item.classList.remove('is-selected'));
    link.classList.add('is-selected');
  }});
}});
</script>
</div>
</body>
</html>
"""


def write_html_reports(
    report: Report,
    output_dir: Path,
    *,
    index_filename: str = "report.html",
) -> Path:
    """Write the report index and one standalone HTML page per validation case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in report.cases:
        write_case_html(report, case, output_dir)

    return write_html_index(report, output_dir, index_filename=index_filename)


def write_case_html(report: Report, case: CaseResult, output_dir: Path) -> Path:
    """Write one standalone validation case HTML page."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / _case_report_filename(case)
    case_path.write_text(render_case_html(report, case), encoding="utf-8")
    return case_path


def write_html_index(
    report: Report,
    output_dir: Path,
    *,
    index_filename: str = "report.html",
) -> Path:
    """Write the validation report index without rewriting case pages."""

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / index_filename
    index_path.write_text(render_html(report), encoding="utf-8")
    return index_path
