"""Sphinx configuration for pyGOTM documentation."""

from __future__ import annotations

import sys
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(DOCS_DIR))

from build_docs import (  # noqa: E402
    stage_validation_html,
    stage_validation_rst_wrappers,
)

project = "pyGOTM"
copyright = (
    "2026 pyGOTM contributors. "
    "Based on GOTM by Lars Umlauf, Hans Burchard, and Karsten Bolding."
)
author = "pyGOTM contributors"
release = "0.1.0"
version = "0.1"

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

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

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
