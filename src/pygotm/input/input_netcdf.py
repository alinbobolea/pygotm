"""
NetCDF restart input — translation of ``input_netcdf.F90``.

Opens, validates, and reads GOTM restart (NetCDF) files.  Verifies that the
restart file's time metadata matches the requested simulation start time, then
reads scalar or 1-D profile variables into NumPy arrays.

Depends on :mod:`pygotm.util.time` for :func:`~pygotm.util.time.read_time_string`
and :func:`~pygotm.util.time.write_time_string` (Fortran:
``use time, only: read_time_string, write_time_string``).

Public interface: :func:`open_restart`, :func:`close_restart`,
:func:`check_restart_time`, :func:`read_restart_data`.

Original FORTRAN authors: Karsten Bolding, Jorn Bruggeman.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

from pygotm.util.time import read_time_string, write_time_string

__all__ = [
    "check_restart_time",
    "close_restart",
    "open_restart",
    "read_restart_data",
]

_restart_dataset: xr.Dataset | None = None


def open_restart(fn: str | Path) -> None:
    """Open a restart NetCDF file."""

    global _restart_dataset
    base = Path(fn)
    path = base if base.suffix == ".nc" else base.with_suffix(".nc")
    _restart_dataset = xr.open_dataset(path, decode_times=False, engine="scipy")


def close_restart() -> None:
    """Close the active restart NetCDF file."""

    global _restart_dataset
    if _restart_dataset is not None:
        _restart_dataset.close()
        _restart_dataset = None


def check_restart_time(
    var_name: str,
    *,
    expected_julian: int,
    expected_seconds: int,
) -> None:
    """Verify that the restart file time metadata matches the requested start."""

    if _restart_dataset is None:
        raise RuntimeError("restart file is not open")
    if var_name not in _restart_dataset.variables:
        msg = f"could not find time variable {var_name!r} in restart file"
        raise KeyError(msg)

    units = str(_restart_dataset[var_name].attrs.get("units", ""))
    match = re.search(r"since\s+(.+)$", units)
    if match is None:
        msg = f"time variable {var_name!r} has unsupported units {units!r}"
        raise ValueError(msg)

    julian, seconds = read_time_string(match.group(1).strip())
    if julian != expected_julian or seconds != expected_seconds:
        expected = write_time_string(expected_julian, expected_seconds)
        actual = write_time_string(julian, seconds)
        msg = (
            "restart time does not match requested start time: "
            f"expected {expected}, found {actual}"
        )
        raise ValueError(msg)


def read_restart_data(
    var_name: str,
    allow_missing_variable: bool,
    *,
    data_0d: np.ndarray | None = None,
    data_1d: np.ndarray | None = None,
) -> float | np.ndarray | None:
    """Read restart data into an optional destination or return a copy."""

    if _restart_dataset is None:
        raise RuntimeError("restart file is not open")
    if var_name not in _restart_dataset.variables:
        if allow_missing_variable:
            return None
        msg = f"variable {var_name!r} was not found in the restart file"
        raise KeyError(msg)

    values = np.asarray(_restart_dataset[var_name].values, dtype=np.float64)
    if data_0d is not None:
        scalar = float(np.ravel(values)[0])
        data_0d[...] = scalar
        return scalar
    if data_1d is not None:
        flat = np.ravel(values)
        if data_1d.shape[0] != flat.shape[0]:
            msg = (
                f"shape mismatch for {var_name!r}: "
                f"{flat.shape[0]} values in file, destination has {data_1d.shape[0]}"
            )
            raise ValueError(msg)
        data_1d[:] = flat
        return data_1d
    if values.ndim == 0:
        return float(values)
    return values.copy()
