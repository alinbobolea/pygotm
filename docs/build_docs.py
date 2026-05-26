#!/usr/bin/env python3
"""Build pyGOTM Sphinx HTML documentation.

Usage
-----
    conda run -n pygotm python docs/build_docs.py
    conda run -n pygotm python docs/build_docs.py --clean
    conda run -n pygotm python docs/build_docs.py --strict
    conda run -n pygotm python docs/build_docs.py --open
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
BUILD_DIR = DOCS_DIR / "build"
INDEX_HTML = BUILD_DIR / "index.html"
FIGURES_SCRIPT = DOCS_DIR / "figures" / "build_figures.py"

#: The fixed set of GOTM 6.0.7 reference test cases.
GOTM_CASES: list[str] = [
    "asics_med",
    "blacksea",
    "channel",
    "couette",
    "entrainment",
    "estuary",
    "flex",
    "gotland",
    "lago_maggiore",
    "langmuir",
    "liverpool_bay",
    "medsea_east",
    "medsea_west",
    "nns_annual",
    "nns_seasonal",
    "ows_papa",
    "plume",
    "resolute",
    "reynolds",
    "rouse",
    "seagrass",
    "wave_breaking",
]

CASE_NOTES: dict[str, str] = {
    "couette": "Simple Couette flow.",
    "blacksea": "Black Sea seasonal cycle.",
    "channel": "Open-channel flow.",
    "entrainment": "Convective entrainment.",
    "estuary": "Estuarine circulation.",
    "flex": "FLEX experiment.",
    "gotland": "Baltic Sea Gotland Deep.",
    "lago_maggiore": "Alpine lake.",
    "langmuir": "Langmuir turbulence with Stokes drift.",
    "liverpool_bay": "Tidal mixing in Liverpool Bay.",
    "medsea_east": "Eastern Mediterranean.",
    "medsea_west": "Western Mediterranean.",
    "nns_annual": "North Sea annual cycle.",
    "nns_seasonal": "North Sea seasonal cycle.",
    "ows_papa": "Ocean Weather Station Papa.",
    "plume": "Freshwater plume.",
    "resolute": "Arctic mixing.",
    "reynolds": "Reynolds number scaling.",
    "rouse": "Rouse sediment profile.",
    "seagrass": "Seagrass canopy dynamics. See :ref:`fortran-parity-deviations`.",
    "wave_breaking": "Wave-breaking enhanced mixing.",
    "asics_med": "Mediterranean deep convection.",
}


def _case_status_counts(cases: list[dict[str, object]]) -> tuple[int, int, int]:
    pass_cases = sum(1 for case in cases if case.get("status") == "PASS")
    fail_cases = sum(1 for case in cases if case.get("status") == "FAIL")
    error_cases = sum(1 for case in cases if case.get("status") == "ERROR")
    return pass_cases, fail_cases, error_cases


def _case_count(case: dict[str, object], key: str) -> int:
    value = case.get(key, 0)
    return value if isinstance(value, int) else 0


def _case_name(case: dict[str, object]) -> str:
    value = case.get("case_name", "")
    return value if isinstance(value, str) else ""


def _case_status(case: dict[str, object]) -> str:
    value = case.get("status", "ERROR")
    return value if isinstance(value, str) else "ERROR"


def stage_validation_test_cases_summary(
    *,
    report_json: Path,
    output_path: Path,
) -> Path:
    """Generate the included validation case summary from ``report.json``.

    Plain ``sphinx-build`` must reflect the latest local validation run. This
    helper writes an include file from ``validation/report.json`` during Sphinx
    configuration import, avoiding stale checked-in timestamps and counts.

    If no report exists, the generated include explains how to create one
    instead of preserving old validation data.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not report_json.is_file():
        output_path.write_text(
            (
                "No generated validation report is available in this checkout.\n"
                "\n"
                "Run ``conda run -n pygotm python -m "
                "pygotm.validation.run_validation`` to create "
                "``validation/report.json``, then rebuild the documentation.\n"
                "\n"
            ),
            encoding="utf-8",
        )
        return output_path

    with report_json.open(encoding="utf-8") as f:
        report = json.load(f)

    cases = report.get("cases", [])
    if not isinstance(cases, list):
        msg = f"{report_json} does not contain a list-valued 'cases' field"
        raise ValueError(msg)

    case_dicts = [case for case in cases if isinstance(case, dict)]
    generated_at = str(report.get("generated_at", "unknown"))
    verdict = str(report.get("verdict", "UNKNOWN"))
    pass_cases, fail_cases, error_cases = _case_status_counts(case_dicts)
    total_pass = sum(_case_count(case, "n_pass") for case in case_dicts)
    total_marginal = sum(_case_count(case, "n_marginal") for case in case_dicts)
    total_discrepant = sum(_case_count(case, "n_discrepant") for case in case_dicts)
    total_broken = sum(_case_count(case, "n_broken") for case in case_dicts)

    lines = [
        "pyGOTM is validated against the 22 official GOTM 6.0.7 test cases.  "
        "The table below summarizes the latest generated "
        "``validation/report.html`` snapshot,",
        f"generated at ``{generated_at}``.",
        "",
        "Case status is aggregated from Frechet variable statuses:",
        "",
        "* ``PASS`` means every compared numeric variable has status ``PASS``.",
        "* ``FAIL`` means at least one compared variable is ``MARGINAL``,",
        "  ``DISCREPANT``, or ``BROKEN``.",
        "* ``ERROR`` means the case failed during setup, execution, or comparison",
        "  before a complete variable table could be produced.",
        "",
        f"The snapshot verdict is ``{verdict}``: {pass_cases} cases pass, "
        f"{fail_cases} cases fail, and {error_cases} cases error.",
        f"Across all cases, the variable totals are {total_pass} ``PASS``, "
        f"{total_marginal} ``MARGINAL``, {total_discrepant} ``DISCREPANT``, "
        f"and {total_broken} ``BROKEN``.",
        "",
        "Each case name in the table below links to its full per-case report",
        "(generated by the last local validation run). If a link 404s, regenerate",
        "the reports with ``conda run -n pygotm python -m "
        "pygotm.validation.run_validation`` and rebuild the documentation.",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 20 15 15 15 15 15 20",
        "",
        "   * - Case",
        "     - Case status",
        "     - PASS",
        "     - MARGINAL",
        "     - DISCREPANT",
        "     - BROKEN",
        "     - Notes",
    ]
    for case in case_dicts:
        case_name = _case_name(case)
        if not case_name:
            continue
        lines.extend(
            [
                f"   * - :doc:`{case_name} <cases/{case_name}>`",
                f"     - {_case_status(case)}",
                f"     - {_case_count(case, 'n_pass')}",
                f"     - {_case_count(case, 'n_marginal')}",
                f"     - {_case_count(case, 'n_discrepant')}",
                f"     - {_case_count(case, 'n_broken')}",
                f"     - {CASE_NOTES.get(case_name, '')}",
            ]
        )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def stage_validation_html(*, src: Path, staged_root: Path) -> list[Path]:
    """Copy ``<src>/*.html`` into ``<staged_root>/validation/``.

    Sphinx's ``html_extra_path`` copies an entire directory tree into the build
    output. The full ``validation/`` directory is too large (NetCDF outputs,
    reference data, per-case run directories) so we curate a small staging tree
    that mirrors the desired build-output layout::

        docs/_validation_html/validation/report.html
        docs/_validation_html/validation/<case>-gotm.html

    Sphinx then copies that tree to ``docs/build/validation/``.

    Existing staging content is wiped first so removed cases disappear from the
    build. Missing or empty ``src`` directories are not errors: the staged
    directory is created empty so ``html_extra_path`` always has a valid target.

    Args:
        src: The ``validation/`` directory at the repository root.
        staged_root: Directory that will hold ``validation/`` as a child. Must
            be the path registered in ``html_extra_path``.

    Returns:
        The list of files copied into the staging tree, in stable order.
    """
    staged_dir = staged_root / "validation"
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_dir.mkdir(parents=True)

    if not src.is_dir():
        return []

    copied: list[Path] = []
    for html in sorted(src.glob("*.html")):
        if not html.is_file():
            continue
        target = staged_dir / html.name
        shutil.copy2(html, target)
        copied.append(target)
    return copied


def stage_validation_rst_wrappers(*, cases_dir: Path) -> list[Path]:
    """Generate a Sphinx RST wrapper page for each GOTM case.

    Each page embeds the corresponding standalone case HTML (produced by the
    validation runner) inside a full-height ``<iframe>`` so Sphinx's theme
    navigation (sidebar, breadcrumbs, prev/next) is preserved.

    The iframe ``src`` is relative to the *rendered* case page location::

        docs/build/validation/cases/<case>.html
            → ../couette-gotm.html (in docs/build/validation/)

    The directory is wiped on every call so stale wrappers for removed cases
    never persist.

    Args:
        cases_dir: Destination directory for the RST files. Typically
            ``docs/validation/cases/`` in the Sphinx source tree (gitignored).

    Returns:
        Sorted list of generated RST paths.
    """
    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    cases_dir.mkdir(parents=True)

    generated: list[Path] = []
    for case in sorted(GOTM_CASES):
        title = case.replace("_", " ").title() + " Validation Report"
        underline = "=" * len(title)
        html_name = f"{case}-gotm.html"
        content = (
            f"{title}\n"
            f"{underline}\n"
            f"\n"
            f".. raw:: html\n"
            f"\n"
            f'   <iframe src="../{html_name}"\n'
            f'           style="width:100%;height:90vh;border:none;display:block;"\n'
            f'           title="{case} validation report">\n'
            f"   </iframe>\n"
        )
        rst_path = cases_dir / f"{case}.rst"
        rst_path.write_text(content, encoding="utf-8")
        generated.append(rst_path)
    return generated


def prepare_figure_cache(cache_root: Path | None = None) -> Path:
    """Set writable cache locations before importing Matplotlib figure code."""

    root = cache_root or BUILD_DIR / ".cache"
    matplotlib_dir = root / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    return root


def build_figures() -> None:
    spec = importlib.util.spec_from_file_location("build_figures", FIGURES_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--clean", action="store_true", help="Remove build directory before building"
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat Sphinx warnings as errors (-W --keep-going)",
    )
    p.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open index.html in the default browser after a successful build",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean and BUILD_DIR.exists():
        print(f"Removing {BUILD_DIR} …")
        shutil.rmtree(BUILD_DIR)

    prepare_figure_cache()
    print("Generating figures …")
    build_figures()

    cmd = ["sphinx-build", "-b", "html"]
    if args.strict:
        cmd += ["-W", "--keep-going"]
    cmd += [str(DOCS_DIR), str(BUILD_DIR)]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nSphinx build failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nDocs built: {INDEX_HTML}")

    if args.open_browser:
        webbrowser.open(INDEX_HTML.as_uri())


if __name__ == "__main__":
    main()
