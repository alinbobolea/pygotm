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


def build_figures() -> None:
    spec = importlib.util.spec_from_file_location("build_figures", FIGURES_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean", action="store_true", help="Remove build directory before building")
    p.add_argument("--strict", action="store_true", help="Treat Sphinx warnings as errors (-W --keep-going)")
    p.add_argument("--open", action="store_true", dest="open_browser", help="Open index.html in the default browser after a successful build")
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
