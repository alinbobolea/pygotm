"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: print_version() - prints GOTM and optional FABM versions
!
! !INTERFACE:
!   subroutine print_version()
!
! !DESCRIPTION:
!  Use git to obtain latest git hashes for GOTM and FABM.
!  Also print compiler information from GOTM.
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import platform
from importlib import metadata
from typing import TextIO

__all__ = ["collect_version_lines", "print_version"]


def _dist_version(package_name: str, fallback: str = "unavailable") -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return fallback


def collect_version_lines() -> tuple[str, ...]:
    """Collect user-facing version lines for the current runtime."""

    return (
        f"pyGOTM:  {_dist_version('pygotm', '0.1.0')}",
        f"Python:  {platform.python_version()}",
        f"NumPy:   {_dist_version('numpy')}",
        f"Numba:   {_dist_version('numba')}",
        f"xarray:  {_dist_version('xarray')}",
        f"NetCDF4: {_dist_version('netcdf4')}",
        f"GSW:     {_dist_version('gsw')}",
    )


def print_version(file: TextIO | None = None) -> str:
    """Return and optionally write the pyGOTM version summary."""

    output = "\n".join(collect_version_lines())
    if file is not None:
        file.write(output)
        file.write("\n")
    return output
