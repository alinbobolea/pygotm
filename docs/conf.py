"""Sphinx configuration for pyGOTM documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

project = "pyGOTM"
copyright = "2026, pyGOTM contributors"
author = "pyGOTM contributors"
release = "0.1.0"
version = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]

html_theme = "furo"
html_title = "pyGOTM Documentation"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_ivar = True

myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]
