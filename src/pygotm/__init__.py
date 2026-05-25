"""pyGOTM package."""

from __future__ import annotations

from importlib import metadata

from . import constants as constants

try:
    __version__ = metadata.version("pygotm")
except metadata.PackageNotFoundError:
    __version__ = "unavailable"

__all__ = ["__version__", "constants"]
