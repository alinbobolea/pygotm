#!/usr/bin/env python3
"""Build pyGOTM Sphinx HTML documentation.

Usage
-----
    conda activate pygotm
    python docs/build_docs.py            # standard build
    python docs/build_docs.py --clean    # wipe build dir first
    python docs/build_docs.py --strict   # treat warnings as errors
    python docs/build_docs.py --open     # open browser after build
    python docs/build_docs.py --clean --strict --open
"""

from __future__ import annotations

import argparse
import importlib.util
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
