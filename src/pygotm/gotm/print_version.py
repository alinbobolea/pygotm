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
!  Original FORTRAN author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import TextIO

__all__ = [
    "collect_version_info",
    "collect_version_lines",
    "find_git_commit",
    "print_version",
]


def _dist_version(package_name: str, fallback: str = "unavailable") -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return fallback


def _candidate_git_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for start in (Path(__file__).resolve(), Path.cwd().resolve()):
        for path in (start, *start.parents):
            if (path / ".git").exists() and path not in candidates:
                candidates.append(path)
                break
    return tuple(candidates)


def find_git_commit() -> str:
    """Return the short Git commit for this checkout, or ``"unavailable"``."""

    for root in _candidate_git_roots():
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        if result.returncode == 0 and commit:
            return commit
    return "unavailable"


def collect_version_info() -> dict[str, str]:
    """Collect manifest-shaped version information for machine consumers."""

    return {
        "pygotm_version": _dist_version("pygotm"),
        "pygotm_git_commit": find_git_commit(),
        "python_version": platform.python_version(),
        "numpy_version": _dist_version("numpy"),
        "numba_version": _dist_version("numba"),
        "xarray_version": _dist_version("xarray"),
        "netcdf4_version": _dist_version("netcdf4"),
        "gsw_version": _dist_version("gsw"),
        "pyfabm_version": _dist_version("pyfabm"),
        "platform": f"{sys.platform}-{platform.machine().lower()}",
    }


def collect_version_lines() -> tuple[str, ...]:
    """Collect user-facing version lines for the current runtime."""

    info = collect_version_info()
    return (
        f"pyGOTM:  {info['pygotm_version']}",
        f"Git:     {info['pygotm_git_commit']}",
        f"Python:  {info['python_version']}",
        f"NumPy:   {info['numpy_version']}",
        f"Numba:   {info['numba_version']}",
        f"xarray:  {info['xarray_version']}",
        f"NetCDF4: {info['netcdf4_version']}",
        f"GSW:     {info['gsw_version']}",
        f"PyFABM:  {info['pyfabm_version']}",
        f"Platform:{info['platform']}",
    )


def print_version(file: TextIO | None = None) -> str:
    """Return and optionally write the pyGOTM version summary."""

    output = "\n".join(collect_version_lines())
    if file is not None:
        file.write(output)
        file.write("\n")
    return output
