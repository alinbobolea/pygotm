"""Stage validation HTML reports for inclusion in the Sphinx build.

Sphinx's ``html_extra_path`` copies an entire directory tree into the build
output. The full ``validation/`` directory is too large (NetCDF outputs,
reference data, per-case run directories) so we curate a small staging tree
that mirrors the desired build-output layout::

    docs/_validation_html/validation/report.html
    docs/_validation_html/validation/<case>-gotm.html

Sphinx then copies that tree to ``docs/build/validation/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = ["stage_validation_html"]


def stage_validation_html(*, src: Path, staged_root: Path) -> list[Path]:
    """Copy ``<src>/*.html`` into ``<staged_root>/validation/``.

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
