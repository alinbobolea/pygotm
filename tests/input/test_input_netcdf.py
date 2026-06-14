"""Tests for pygotm.input.input_netcdf."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pygotm.input.input_netcdf import (
    check_restart_time,
    close_restart,
    open_restart,
    read_restart_data,
)
from pygotm.util.time import julian_day


def _write_restart_dataset(path: Path) -> None:
    dataset = xr.Dataset(
        data_vars={
            "time": xr.DataArray(
                [0.0],
                dims=("time",),
                attrs={"units": "seconds since 2000-01-01 00:00:00"},
            ),
            "eta": xr.DataArray(1.5),
            "temp": xr.DataArray(np.array([1.0, 2.0, 3.0]), dims=("z",)),
        }
    )
    dataset.to_netcdf(path, engine="scipy")


def test_restart_open_check_read_and_close(tmp_path: Path) -> None:
    path = tmp_path / "restart.nc"
    _write_restart_dataset(path)
    open_restart(path)
    try:
        check_restart_time(
            "time",
            expected_julian=julian_day(2000, 1, 1),
            expected_seconds=0,
        )
        assert read_restart_data("eta", False) == pytest.approx(1.5)
        values = np.zeros(3, dtype=np.float64)
        read_restart_data("temp", False, data_1d=values)
        assert np.allclose(values, np.array([1.0, 2.0, 3.0]))
    finally:
        close_restart()


def test_restart_time_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "restart.nc"
    _write_restart_dataset(path)
    open_restart(path)
    try:
        with pytest.raises(ValueError, match="does not match"):
            check_restart_time(
                "time",
                expected_julian=julian_day(2000, 1, 2),
                expected_seconds=0,
            )
    finally:
        close_restart()


def test_missing_restart_variable_can_be_ignored(tmp_path: Path) -> None:
    path = tmp_path / "restart.nc"
    _write_restart_dataset(path)
    open_restart(path)
    try:
        assert read_restart_data("missing", True) is None
        with pytest.raises(KeyError, match="missing"):
            read_restart_data("missing", False)
    finally:
        close_restart()


def test_check_restart_time_without_open_file_raises(tmp_path: Path) -> None:
    close_restart()  # ensure clean state
    with pytest.raises(RuntimeError, match="restart file is not open"):
        check_restart_time(
            "time", expected_julian=julian_day(2000, 1, 1), expected_seconds=0
        )


def test_read_restart_data_without_open_file_raises(tmp_path: Path) -> None:
    close_restart()  # ensure clean state
    with pytest.raises(RuntimeError, match="restart file is not open"):
        read_restart_data("temp", False)


def test_check_restart_time_missing_variable_raises(tmp_path: Path) -> None:
    path = tmp_path / "restart.nc"
    _write_restart_dataset(path)
    open_restart(path)
    try:
        with pytest.raises(KeyError, match="no_such_var"):
            check_restart_time(
                "no_such_var",
                expected_julian=julian_day(2000, 1, 1),
                expected_seconds=0,
            )
    finally:
        close_restart()


def test_check_restart_time_unsupported_units_raises(tmp_path: Path) -> None:
    import xarray as xr

    path = tmp_path / "bad_units.nc"
    dataset = xr.Dataset(
        data_vars={
            "time": xr.DataArray(
                [0.0],
                dims=("time",),
                attrs={"units": "dimensionless"},  # missing "since" keyword
            ),
        }
    )
    dataset.to_netcdf(path, engine="scipy")
    open_restart(path)
    try:
        with pytest.raises(ValueError, match="unsupported units"):
            check_restart_time(
                "time",
                expected_julian=julian_day(2000, 1, 1),
                expected_seconds=0,
            )
    finally:
        close_restart()


def test_read_restart_data_shape_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "restart.nc"
    _write_restart_dataset(path)
    open_restart(path)
    try:
        dest = np.zeros(10, dtype=np.float64)  # temp has 3 values, not 10
        with pytest.raises(ValueError, match="shape mismatch"):
            read_restart_data("temp", False, data_1d=dest)
    finally:
        close_restart()


def test_read_restart_data_scalar_into_0d_destination(tmp_path: Path) -> None:
    path = tmp_path / "restart.nc"
    _write_restart_dataset(path)
    open_restart(path)
    try:
        dest = np.zeros(1, dtype=np.float64)
        result = read_restart_data("eta", False, data_0d=dest)
        assert result == pytest.approx(1.5)
        assert dest[0] == pytest.approx(1.5)
    finally:
        close_restart()
