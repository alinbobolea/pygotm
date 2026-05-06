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


@dataclass
class CaseResult:
    case_name: str
    status: str            # "PASS" | "FAIL" | "ERROR"
    error: str | None
    py_nc_path: str
    ref_nc_path: str
    wall_time_s: float
    variables: list[VarResult] = field(default_factory=list)
    n_pass: int = 0
    n_fail: int = 0
    n_skip: int = 0


@dataclass
class Report:
    generated_at: str
    rtol: float
    atol: float
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
        vars_ = [VarResult(**v) for v in cd.pop("variables", [])]
        cases.append(CaseResult(**cd, variables=vars_))
    data["cases"] = cases
    data.setdefault("hardware", {})
    return Report(**data)


def _fmt(v: float | None, precision: int = 3) -> str:
    """Format a float in scientific notation with *precision* significant digits."""
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return "—"
    return f"{v:.{precision}e}"


def _fmt_full(v: float | None) -> str:
    """Format a float with full IEEE 754 double precision (all significant digits).

    Uses Python's repr(), which produces the shortest string guaranteed to
    round-trip back to the exact same float value (up to 17 significant digits).
    """
    if v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf"))):
        return "—"
    return repr(v)


def _rtol_latex(rtol: float) -> str:
    """Format rtol as a LaTeX string, e.g. '5 \\times 10^{-6}'."""
    exp = int(math.floor(math.log10(abs(rtol))))
    mantissa = rtol / 10.0 ** exp
    mantissa_str = f"{mantissa:.6g}".rstrip("0").rstrip(".")
    if mantissa_str == "1":
        return f"10^{{{exp}}}"
    return f"{mantissa_str} \\times 10^{{{exp}}}"


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m {sec:.0f}s"


def _status_cell(status: str) -> str:
    colours = {
        "PASS": "#2e7d32", "FAIL": "#c62828",
        "ERROR": "#c62828",
    }
    colour = colours.get(status, "#333")
    return f'<td style="color:{colour};font-weight:bold">{status}</td>'


def _hardware_section(hw: dict[str, str]) -> str:
    if not hw:
        return ""
    cpu_model = hw.get("cpu_model", "unknown")
    cpu_count = hw.get("cpu_count", "?")
    cores_per_socket = hw.get("cores_per_socket", "")
    sockets = hw.get("sockets", "")
    cpu_max_mhz = hw.get("cpu_max_mhz", "")
    ram = hw.get("ram_total", "unknown")
    numba = hw.get("numba_version", "unknown")
    python_ver = hw.get("python_version", "unknown")
    os_platform = hw.get("platform", "unknown")
    gpu_info = hw.get("gpu_info", "")
    execution_backend = hw.get("execution_backend", "cpu")

    core_detail = ""
    if cores_per_socket and sockets:
        try:
            total_phys = int(cores_per_socket) * int(sockets)
            core_detail = f" ({total_phys} physical cores)"
        except ValueError:
            pass
    freq_detail = ""
    if cpu_max_mhz:
        try:
            freq_detail = f" @ {float(cpu_max_mhz) / 1000:.2f} GHz"
        except ValueError:
            pass

    rows: list[tuple[str, str]] = [
        ("CPU", f"{cpu_model}{freq_detail}"),
        ("Logical CPUs", f"{cpu_count}{core_detail}"),
        ("RAM", ram),
        ("Execution backend", execution_backend),
        ("Numba", numba),
        ("Python", python_ver),
        ("OS", os_platform),
    ]
    if gpu_info:
        rows.insert(2, ("GPU", gpu_info))

    cells = "".join(
        f"<tr><th>{label}</th><td>{_html.escape(value)}</td></tr>"
        for label, value in rows
    )
    return f"""
<div class="hardware-box">
  <h3 style="margin:0 0 .5em">Hardware</h3>
  <table class="hw-table">
    <tbody>{cells}</tbody>
  </table>
</div>"""


def render_html(report: Report) -> str:
    """Render the validation report as a standalone HTML page."""
    rtol_latex = _rtol_latex(report.rtol)
    cases_passed = sum(1 for c in report.cases if c.status == "PASS")
    total_cases  = len(report.cases)
    verdict_colour = "#2e7d32" if report.verdict == "FULL PARITY" else (
        "#e65100" if report.verdict == "PARTIAL PARITY" else "#c62828"
    )
    total_wall = sum(c.wall_time_s for c in report.cases)

    rows_summary = ""
    for c in report.cases:
        total_checked = c.n_pass + c.n_fail
        time_str = _fmt_time(c.wall_time_s) if c.wall_time_s > 0 else "—"
        rows_summary += (
            f"<tr>"
            f"<td>{c.case_name}</td>"
            + _status_cell(c.status) +
            f"<td>{c.n_pass}/{total_checked}</td>"
            f"<td>{c.n_fail}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{time_str}</td>"
            f"</tr>\n"
        )

    total_pass = sum(c.n_pass for c in report.cases)
    total_fail = sum(c.n_fail for c in report.cases)
    rows_summary += (
        f"<tr style='font-weight:bold;border-top:2px solid #ccc'>"
        f"<td>TOTAL ({total_cases} cases)</td>"
        f"<td>—</td>"
        f"<td>{total_pass}/{total_pass + total_fail}</td>"
        f"<td>{total_fail}</td>"
        f"<td style='font-variant-numeric:tabular-nums'>{_fmt_time(total_wall)}</td>"
        f"</tr>\n"
    )

    case_sections = ""
    for c in report.cases:
        open_attr = " open" if c.status in ("FAIL", "ERROR") else ""
        time_str = _fmt_time(c.wall_time_s) if c.wall_time_s > 0 else "—"

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

        rows = ""
        for v in c.variables:
            # ref_at_worst and calc_at_worst: full precision via repr()
            # error metrics: 3-digit scientific notation for readability
            rows += (
                "<tr>"
                + _status_cell(v.status) +
                f"<td>{_html.escape(v.name)}</td>"
                f"<td><code>{_fmt_full(v.ref_at_worst)}</code></td>"
                f"<td><code>{_fmt_full(v.calc_at_worst)}</code></td>"
                f"<td>{_fmt(v.max_abs_err)}</td>"
                f"<td>{_fmt(v.max_rel_err)}</td>"
                f"<td>{_fmt(v.rmse)}</td>"
                f"<td>{_fmt(v.nrmse)}</td>"
                f"</tr>\n"
            )

        badge_cls = "badge-pass" if c.status == "PASS" else "badge-fail"
        case_sections += f"""
<details class="case-section" data-status="{c.status}"{open_attr}>
  <summary class="case-summary">
    <span class="case-title">{_html.escape(c.case_name)}</span>
    <span class="case-badge {badge_cls}">{c.status}</span>
    <span class="case-meta">
      {c.n_pass}/{c.n_pass+c.n_fail} vars pass &nbsp;·&nbsp;
      {c.n_fail} failed &nbsp;·&nbsp;
      wall time: {time_str}
    </span>
  </summary>
  <div class="case-body">
    <p style="font-size:0.85em;color:#555;margin:.4em 0">
      Python: <code>{_html.escape(c.py_nc_path)}</code><br>
      Reference: <code>{_html.escape(c.ref_nc_path)}</code>
    </p>
    <table>
      <thead>
        <tr>
          <th>Status</th><th>Variable</th>
          <th>Reference (full precision)</th>
          <th>Calculated (full precision)</th>
          <th>max_abs_err</th><th>max_rel_err ⚠</th>
          <th>RMSE</th><th>NRMSE</th>
        </tr>
      </thead>
      <tbody>
{rows}      </tbody>
    </table>
  </div>
</details>
"""

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
  body   {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 2rem auto; padding: 0 1rem; color: #222; }}
  h1     {{ border-bottom: 2px solid #ccc; padding-bottom: .4em; }}
  h2     {{ margin-top: 2.5em; border-bottom: 1px solid #ddd; padding-bottom: .3em; }}
  table  {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.85em; }}
  th, td {{ border: 1px solid #ddd; padding: .35em .6em; text-align: left; white-space: nowrap; }}
  th     {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover td {{ background: #fafafa; }}
  .verdict {{ font-size: 1.4em; font-weight: bold; padding: .5em 1em;
               border-left: 6px solid; margin: 1.2em 0; }}
  code   {{ background: #f5f5f5; padding: .1em .4em; border-radius: 3px; font-size: 0.9em; }}
  .hardware-box {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
                   padding: .8em 1.2em; margin: 1.2em 0; display: inline-block;
                   min-width: 420px; }}
  .hw-table {{ margin: 0; font-size: 0.88em; border: none; }}
  .hw-table th, .hw-table td {{ border: none; padding: .2em .7em .2em 0; background: transparent; }}
  .hw-table th {{ width: 140px; color: #555; font-weight: 600; }}
  .methodology {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
                  padding: 1em 1.4em; margin: 1.2em 0; }}
  .methodology h3 {{ margin: 0 0 .6em; font-size: 1em; color: #444; }}
  .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .5em 2em; }}
  .metric-item {{ padding: .4em 0; border-bottom: 1px solid #eee; }}
  .metric-name {{ font-weight: 600; font-size: 0.9em; }}
  .metric-desc {{ font-size: 0.85em; color: #555; margin-top: .15em; }}
  .pass-criterion {{ background: #e8f5e9; border-left: 4px solid #2e7d32;
                     padding: .6em 1em; margin-top: .8em; border-radius: 0 4px 4px 0; }}
  .pass-criterion strong {{ color: #1b5e20; }}
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
  .badge-pass  {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-fail  {{ background: #ffebee; color: #c62828; }}
  .badge-error {{ background: #ffebee; color: #c62828; }}
  .case-body  {{ padding: 0 1em 1em; }}
  .controls {{ display: flex; gap: .5em; margin: 1em 0; }}
  .btn {{ padding: .35em .9em; border: 1px solid #bbb; border-radius: 4px;
          background: #f5f5f5; cursor: pointer; font-size: 0.85em; }}
  .btn:hover {{ background: #e8e8e8; }}
</style>
</head>
<body>
<h1>pyGOTM Validation Report</h1>
<p style="color:#666;font-size:0.9em">Generated: {report.generated_at} &nbsp;·&nbsp; Total wall time: {_fmt_time(total_wall)}</p>

<div class="verdict" style="color:{verdict_colour};border-color:{verdict_colour}">
  {report.verdict} &nbsp; ({cases_passed}/{total_cases} cases passed)
</div>

{hw_html}

<h2>Methodology</h2>
<div class="methodology">
  <h3>Metrics</h3>
  <p style="font-size:0.85em;color:#555;margin:.2em 0 .8em">
    All metrics are computed over every time step and grid point in the flattened arrays
    \\(a\\) (pyGOTM) and \\(b\\) (Fortran reference).
    Reference and Calculated columns show all significant digits (IEEE 754 double precision).
  </p>
  <div class="metric-grid">
    <div class="metric-item">
      <div class="metric-name">Reference (full precision)</div>
      <div class="metric-desc">
        Fortran GOTM value \\(b_i\\) at \\(i^* = \\arg\\max |a - b|\\), shown with all significant digits.
      </div>
    </div>
    <div class="metric-item">
      <div class="metric-name">Calculated (full precision)</div>
      <div class="metric-desc">
        pyGOTM value \\(a_{{i^*}}\\) at that same index, shown with all significant digits.
      </div>
    </div>
    <div class="metric-item">
      <div class="metric-name">max_abs_err</div>
      <div class="metric-desc">\\(\\max_i |a_i - b_i|\\)</div>
    </div>
    <div class="metric-item">
      <div class="metric-name">max_rel_err ⚠</div>
      <div class="metric-desc">Unreliable near zero; use NRMSE as primary metric.</div>
    </div>
    <div class="metric-item">
      <div class="metric-name">RMSE</div>
      <div class="metric-desc">\\(\\sqrt{{\\frac{{1}}{{N}}\\sum_i (a_i - b_i)^2}}\\)</div>
    </div>
    <div class="metric-item">
      <div class="metric-name">NRMSE</div>
      <div class="metric-desc">\\(\\text{{RMSE}} / (\\max b - \\min b)\\) — primary metric.</div>
    </div>
  </div>
  <div class="pass-criterion">
    <strong>Pass criterion:</strong>
    \\[|a_i - b_i| \\leq \\max\\!\\left(10^{{-7}} \\cdot \\text{{range}}(b),\\; 10^{{-12}}\\right) + {rtol_latex} \\cdot |b_i|\\]
  </div>
</div>

<h2>Summary</h2>
<table>
  <thead><tr><th>Case</th><th>Status</th><th>Vars passed</th><th>Vars failed</th><th>Wall time</th></tr></thead>
  <tbody>{rows_summary}</tbody>
</table>

<div class="controls">
  <button class="btn" onclick="expandAll()">Expand All</button>
  <button class="btn" onclick="collapseAll()">Collapse All</button>
  <button class="btn" onclick="showFailed()">Show Failed</button>
</div>

{case_sections}

<script>
function expandAll()   {{ document.querySelectorAll('details.case-section').forEach(d => d.open = true); }}
function collapseAll() {{ document.querySelectorAll('details.case-section').forEach(d => d.open = false); }}
function showFailed()  {{
  document.querySelectorAll('details.case-section').forEach(d => {{
    d.open = d.dataset.status === 'FAIL' || d.dataset.status === 'ERROR';
  }});
}}
</script>
</body>
</html>
"""
