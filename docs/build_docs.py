#!/usr/bin/env python3
"""Build pyGOTM Sphinx HTML documentation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent
BUILD_DIR = DOCS_DIR / "build" / "html"


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(DOCS_DIR), str(BUILD_DIR)],
        check=False,
    )
    if result.returncode != 0:
        print(f"Sphinx build failed with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"Docs built at: {BUILD_DIR}/index.html")


if __name__ == "__main__":
    main()
