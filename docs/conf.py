"""Sphinx configuration for pyGOTM documentation."""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(DOCS_DIR))

from build_docs import (  # noqa: E402
    stage_validation_html,
    stage_validation_rst_wrappers,
    stage_validation_test_cases_summary,
)

project = "pyGOTM"
copyright = (
    "2026 pyGOTM contributors. "
    "Based on GOTM by Lars Umlauf, Hans Burchard, and Karsten Bolding."
)
author = "pyGOTM contributors"

try:
    release = metadata.version("pygotm")
except metadata.PackageNotFoundError:
    release = "unavailable"
version = ".".join(release.split(".")[:2]) if release != "unavailable" else release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

html_theme = "furo"
html_title = "pyGOTM Documentation"
_VALIDATION_HTML_STAGING = DOCS_DIR / "_validation_html"
stage_validation_html(
    src=REPO_ROOT / "validation",
    staged_root=_VALIDATION_HTML_STAGING,
)
html_extra_path = [str(_VALIDATION_HTML_STAGING.relative_to(DOCS_DIR))]

stage_validation_rst_wrappers(cases_dir=DOCS_DIR / "validation" / "cases")
stage_validation_test_cases_summary(
    report_json=REPO_ROOT / "validation" / "report.json",
    output_path=DOCS_DIR / "validation" / "_generated" / "test_cases_summary.inc",
)

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "footer_icons": [
        {
            "name": "GOTM",
            "url": "https://gotm.net",
            "html": ("Based on GOTM by Lars Umlauf, Hans Burchard & Karsten Bolding"),
            "class": "",
        },
    ],
}

# Use vendored intersphinx inventories under ``docs/_intersphinx`` first and
# fall back to the upstream URL only when present. Vendoring makes ``-W`` docs
# builds reproducible and resilient to restricted-network environments (CI,
# build sandboxes, offline mirrors). Refresh with
# ``conda run -n pygotm python docs/refresh_intersphinx.py``.
_INTERSPHINX_DIR = DOCS_DIR / "_intersphinx"
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3",
        (str(_INTERSPHINX_DIR / "python.inv"), None),
    ),
    "numpy": (
        "https://numpy.org/doc/stable",
        (str(_INTERSPHINX_DIR / "numpy.inv"), None),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy",
        (str(_INTERSPHINX_DIR / "scipy.inv"), None),
    ),
    "xarray": (
        "https://docs.xarray.dev/en/stable",
        (str(_INTERSPHINX_DIR / "xarray.inv"), None),
    ),
}
intersphinx_timeout = 5

# MathJax: define GOTM LaTeX macros so any surviving PROTEX LaTeX
# in docstrings renders correctly without manual conversion.
mathjax3_config = {
    "tex": {
        "macros": {
            "partder": [r"\frac{\partial #1}{\partial #2}", 2],
            "frstder": [r"\frac{\partial}{\partial #1}", 1],
            "mean": [r"\overline{#1}", 1],
            "comma": r",",
            "point": r".",
        }
    }
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autoclass_content = "both"
autodoc_typehints = "description"

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False

todo_include_todos = True

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

exclude_patterns = ["build", "Thumbs.db", ".DS_Store", "runtime-architecture.md"]
