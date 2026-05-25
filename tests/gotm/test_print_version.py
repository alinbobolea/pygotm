"""Tests for pygotm.gotm.print_version."""

from __future__ import annotations

from pygotm.gotm.print_version import (
    collect_version_info,
    collect_version_lines,
    print_version,
)

_VERSION_KEYS = {
    "pygotm_version",
    "pygotm_git_commit",
    "python_version",
    "numpy_version",
    "numba_version",
    "xarray_version",
    "netcdf4_version",
    "gsw_version",
    "pyfabm_version",
    "platform",
}


def test_collect_version_info_uses_manifest_shaped_keys() -> None:
    info = collect_version_info()

    assert set(info) == _VERSION_KEYS
    assert all(isinstance(value, str) and value for value in info.values())


def test_collect_version_lines_contains_expected_labels() -> None:
    lines = collect_version_lines()
    assert any(line.startswith("pyGOTM:") for line in lines)
    assert any(line.startswith("Python:") for line in lines)
    assert any(line.startswith("NumPy:") for line in lines)
    assert any(line.startswith("PyFABM:") for line in lines)


def test_print_version_returns_multiline_string() -> None:
    output = print_version()
    assert "pyGOTM:" in output
    assert "Python:" in output
