"""pyGOTM package."""

from __future__ import annotations

import warnings

from . import constants as constants

# Taichi 1.7.4 emits a Python 3.13+ deprecation warning from its own import-time
# locale helper (taichi._lib.utils.locale_encode -> locale.getdefaultlocale()).
# Scope this filter narrowly to the known upstream warning so normal project
# deprecations still surface.
warnings.filterwarnings(
    "ignore",
    message=(
        r"'locale\.getdefaultlocale' is deprecated and slated for removal in "
        r"Python 3\.15\..*"
    ),
    category=DeprecationWarning,
    module=r"taichi\._lib\.utils",
)

__all__ = ["constants"]
