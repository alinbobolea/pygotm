"""Validation report data classes, JSON serialization, and HTML rendering."""

# ruff: noqa: E501

from __future__ import annotations

import html as _html
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pygotm.validation.compare import VarResult

__all__ = [
    "CaseResult",
    "Report",
    "load_json",
    "render_html",
    "save_json",
]

_STATUS_COLORS: dict[str, str] = {
    "PASS":       "#2e7d32",
    "MARGINAL":   "#f9a825",
    "DISCREPANT": "#e65100",
    "BROKEN":     "#c62828",
    "ERROR":      "#c62828",
}

_BADGE_CLASSES: dict[str, str] = {
    "PASS":       "badge-pass",
    "MARGINAL":   "badge-marginal",
    "DISCREPANT": "badge-discrepant",
    "BROKEN":     "badge-broken",
    "ERROR":      "badge-error",
}


@dataclass
class CaseResult:
    case_name: str
    status: str           # "PASS" | "FAIL" | "ERROR"
    error: str | None
    py_nc_path: str
    ref_nc_path: str
    wall_time_s: float
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
    verdict: str           # "FULL PARITY" | "PARTIAL PARITY" | "FAILED VALIDATION"


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
        vars_data = cd.pop("variables", [])
        vars_ = [VarResult(**v) for v in vars_data]
        cd.setdefault("n_marginal", 0)
        cd.setdefault("n_discrepant", 0)
        cd.setdefault("n_broken", 0)
        cd.pop("n_fail", None)
        cd.pop("n_skip", None)
        cases.append(CaseResult(**cd, variables=vars_))
    data["cases"] = cases
    data.setdefault("hardware", {})
    data.pop("atol", None)
    data.pop("rtol", None)
    return Report(**data)


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
        rows += (
            "<tr>"
            f'<td style="color:{colour};font-weight:bold">{v.status}</td>'
            f"<td>{_html.escape(v.name)}</td>"
            f"<td><code>{_fmt_full(v.reference_at_worst)}</code></td>"
            f"<td><code>{_fmt_full(v.calculated_at_worst)}</code></td>"
            f"<td>{_fmt(v.primary_score)}</td>"
            f"<td>{_fmt(v.birge_ratio)}</td>"
            f"<td>{_fmt(v.normalized_signed_bias)}</td>"
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
        return f'<p style="color:#888;font-style:italic">No {section_label} variables.</p>'
    table_header = (
        "<table>"
        "<thead><tr>"
        "<th>Status</th><th>Variable</th>"
        "<th>Reference (full precision)</th>"
        "<th>Calculated (full precision)</th>"
        "<th>Primary score (p99)</th>"
        "<th>Birge ratio</th>"
        "<th>Normalized signed bias</th>"
        "<th>Parameter plot</th>"
        "</tr></thead>"
        f"<tbody>{_var_rows_html(variables)}</tbody></table>"
    )
    return f'<h4 style="margin:1em 0 .4em">{section_label} variables</h4>{table_header}'


def render_html(report: Report) -> str:
    """Render the validation report as a standalone HTML page."""
    cases_passed = sum(1 for c in report.cases if c.status == "PASS")
    total_cases = len(report.cases)
    verdict_colour = (
        "#2e7d32"
        if report.verdict == "FULL PARITY"
        else ("#e65100" if report.verdict == "PARTIAL PARITY" else "#c62828")
    )
    total_wall = sum(c.wall_time_s for c in report.cases)

    rows_summary = ""
    for c in report.cases:
        n_total = c.n_pass + c.n_marginal + c.n_discrepant + c.n_broken
        time_str = _fmt_time(c.wall_time_s) if c.wall_time_s > 0 else "—"
        badge_cls = _BADGE_CLASSES.get(c.status, "badge-error")
        rows_summary += (
            f"<tr>"
            f"<td>{c.case_name}</td>"
            f'<td><span class="case-badge {badge_cls}">{c.status}</span></td>'
            f"<td>{c.n_pass}</td><td>{c.n_marginal}</td>"
            f"<td>{c.n_discrepant}</td><td>{c.n_broken}</td>"
            f"<td>{n_total}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{time_str}</td>"
            f"</tr>\n"
        )
    rows_summary += (
        f"<tr style='font-weight:bold;border-top:2px solid #ccc'>"
        f"<td>TOTAL ({total_cases} cases)</td><td>—</td>"
        f"<td>{sum(c.n_pass for c in report.cases)}</td>"
        f"<td>{sum(c.n_marginal for c in report.cases)}</td>"
        f"<td>{sum(c.n_discrepant for c in report.cases)}</td>"
        f"<td>{sum(c.n_broken for c in report.cases)}</td>"
        f"<td>—</td>"
        f"<td style='font-variant-numeric:tabular-nums'>{_fmt_time(total_wall)}</td>"
        f"</tr>\n"
    )

    case_sections = ""
    for c in report.cases:
        is_open = c.status in ("FAIL", "ERROR") or c.n_marginal + c.n_discrepant + c.n_broken > 0
        open_attr = " open" if is_open else ""
        time_str = _fmt_time(c.wall_time_s) if c.wall_time_s > 0 else "—"
        badge_cls = _BADGE_CLASSES.get(c.status, "badge-error")

        if c.status == "ERROR":
            case_sections += f"""
<details class="case-section" data-status="ERROR"{open_attr}>
  <summary class="case-summary">
    <span class="case-title">{_html.escape(c.case_name)}</span>
    <span class="case-badge badge-error">ERROR</span>
    <span class="case-meta">wall time: {time_str}</span>
  </summary>
  <p style="color:#c62828;margin:.6em 0 0 1.2em">ERROR: {_html.escape(c.error or "unknown error")}</p>
</details>
"""
            continue

        pygotm_vars = [v for v in c.variables if v.section == "pygotm"]
        pyfabm_vars = [v for v in c.variables if v.section == "pyfabm"]

        case_sections += f"""
<details class="case-section" data-status="{c.status}"{open_attr}>
  <summary class="case-summary">
    <span class="case-title">{_html.escape(c.case_name)}</span>
    <span class="case-badge {badge_cls}">{c.status}</span>
    <span class="case-meta">
      PASS: {c.n_pass} &nbsp;·&nbsp;
      MARGINAL: {c.n_marginal} &nbsp;·&nbsp;
      DISCREPANT: {c.n_discrepant} &nbsp;·&nbsp;
      BROKEN: {c.n_broken} &nbsp;·&nbsp;
      wall time: {time_str}
    </span>
  </summary>
  <div class="case-body">
    <p style="font-size:0.85em;color:#555;margin:.4em 0">
      Python: <code>{_html.escape(c.py_nc_path)}</code><br>
      Reference: <code>{_html.escape(c.ref_nc_path)}</code>
    </p>
    {_section_html("1. PyGOTM", pygotm_vars)}
    {_section_html("2. PyFABM", pyfabm_vars)}
  </div>
</details>
"""

    hw_html = _hardware_section(report.hardware)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pyGOTM Validation Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
MathJax = {{ tex: {{ inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
  body   {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1     {{ border-bottom: 2px solid #ccc; padding-bottom: .4em; }}
  h2     {{ margin-top: 2.5em; border-bottom: 1px solid #ddd; padding-bottom: .3em; }}
  h4     {{ color: #444; }}
  table  {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #ddd; padding: .35em .6em; text-align: left; white-space: nowrap; }}
  th     {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover td {{ background: #fafafa; }}
  .verdict {{ font-size: 1.4em; font-weight: bold; padding: .5em 1em;
               border-left: 6px solid; margin: 1.2em 0; }}
  code   {{ background: #f5f5f5; padding: .1em .4em; border-radius: 3px; font-size: 0.9em; }}
  .hardware-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
                   padding: .8em 1.2em; margin: 1.2em 0; display: inline-block; min-width: 420px; }}
  .hw-table {{ margin: 0; font-size: 0.88em; border: none; }}
  .hw-table th, .hw-table td {{ border: none; padding: .2em .7em .2em 0; background: transparent; }}
  .hw-table th {{ width: 140px; color: #555; font-weight: 600; }}
  .case-section {{ border: 1px solid #e0e0e0; border-radius: 6px; margin: .8em 0; }}
  .case-summary {{ display: flex; align-items: center; gap: .7em; cursor: pointer;
                   padding: .6em 1em; list-style: none; user-select: none; }}
  .case-summary::-webkit-details-marker {{ display: none; }}
  .case-summary::before {{ content: "▶"; font-size: .75em; color: #888;
                            transition: transform .2s; min-width: 1em; }}
  details[open] > .case-summary::before {{ transform: rotate(90deg); }}
  .case-title {{ font-size: 1.1em; font-weight: 600; }}
  .case-meta  {{ font-size: 0.8em; color: #666; margin-left: auto; }}
  .case-badge {{ font-size: 0.8em; font-weight: 700; padding: .15em .5em; border-radius: 3px; }}
  .badge-pass       {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-marginal   {{ background: #fff9c4; color: #f57f17; }}
  .badge-discrepant {{ background: #ffe0b2; color: #e65100; }}
  .badge-broken     {{ background: #ffebee; color: #c62828; }}
  .badge-error      {{ background: #ffebee; color: #c62828; }}
  .case-body  {{ padding: 0 1em 1em; }}
  .plot-container {{ margin: .5em 0; }}
  .controls {{ display: flex; gap: .5em; margin: 1em 0; }}
  .btn {{ padding: .35em .9em; border: 1px solid #bbb; border-radius: 4px;
          background: #f5f5f5; cursor: pointer; font-size: 0.85em; }}
  .btn:hover {{ background: #e8e8e8; }}
  .methodology {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
                  padding: 1em 1.4em; margin: 1.2em 0; }}
  .status-band {{ display: grid; grid-template-columns: auto auto auto 1fr; gap: .3em 1em;
                  font-size: 0.88em; margin-top: .6em; }}
</style>
</head>
<body>
<h1>pyGOTM Validation Report</h1>
<p style="color:#666;font-size:0.9em">Generated: {report.generated_at} &nbsp;·&nbsp; Total wall time: {_fmt_time(total_wall)}</p>

<div class="verdict" style="color:{verdict_colour};border-color:{verdict_colour}">
  {report.verdict} &nbsp; ({cases_passed}/{total_cases} cases passed)
</div>

{hw_html}

<h2>Validation Methodology</h2>
<div class="methodology">
  <p style="font-size:0.85em;color:#555;margin:.2em 0 .8em">
    Three indicators are used. Status is driven exclusively by the primary tolerance-normalized score.
  </p>
  <p style="font-size:0.85em">
    <strong>Pointwise normalized error:</strong>
    \\[E_i = \\frac{{|\\text{{calc}}_i - \\text{{ref}}_i|}}{{a_{{tol}} + r_{{tol}} \\cdot \\max(|\\text{{ref}}_i|,\\, s_{{floor}})}}\\]
  </p>
  <p style="font-size:0.85em">
    <strong>Primary score (p99):</strong> \\(P_{{99}} = \\text{{percentile}}(E_i, 99)\\) — drives status classification.<br>
    <strong>Birge ratio:</strong> \\(B = \\sqrt{{\\text{{mean}}(E_i^2)}}\\) — RMS diagnostic only, not used for status.<br>
    <strong>Normalized signed bias:</strong> \\(\\text{{NSB}} = \\bar{{(\\text{{calc}} - \\text{{ref}})}} \\;/\\; (a_{{tol}} + r_{{tol}} \\cdot \\max(\\bar{{|\\text{{ref}}|}}, s_{{floor}}))\\)
  </p>
  <div class="status-band">
    <span style="color:#2e7d32;font-weight:bold">PASS</span>     <span>green</span>  <span>\\(P_{{99}} \\leq 1\\)</span>       <span>Within tolerance.</span>
    <span style="color:#f9a825;font-weight:bold">MARGINAL</span> <span>yellow</span> <span>\\(1 &lt; P_{{99}} \\leq 3\\)</span>  <span>Slightly outside. Possible float32 sensitivity.</span>
    <span style="color:#e65100;font-weight:bold">DISCREPANT</span><span>orange</span><span>\\(3 &lt; P_{{99}} \\leq 10\\)</span> <span>Deterministic implementation difference likely.</span>
    <span style="color:#c62828;font-weight:bold">BROKEN</span>    <span>red</span>   <span>\\(P_{{99}} > 10\\)</span>          <span>Severe mismatch. Debug before inspecting plots.</span>
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
  <tbody>{rows_summary}</tbody>
</table>

<div class="controls">
  <button class="btn" onclick="expandAll()">Expand All</button>
  <button class="btn" onclick="collapseAll()">Collapse All</button>
  <button class="btn" onclick="showFailed()">Show Non-Pass</button>
</div>

{case_sections}

<script>
function expandAll()   {{ document.querySelectorAll('details.case-section').forEach(d => d.open = true); }}
function collapseAll() {{ document.querySelectorAll('details.case-section').forEach(d => d.open = false); }}
function showFailed()  {{
  document.querySelectorAll('details.case-section').forEach(d => {{
    const s = d.dataset.status;
    d.open = s === 'FAIL' || s === 'ERROR';
  }});
}}
</script>
</body>
</html>
"""
