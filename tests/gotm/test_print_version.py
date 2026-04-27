"""Tests for pygotm.gotm.print_version."""

from __future__ import annotations

from pygotm.gotm.print_version import collect_version_lines, print_version


def test_collect_version_lines_contains_expected_labels() -> None:
    lines = collect_version_lines()
    assert any(line.startswith("pyGOTM:") for line in lines)
    assert any(line.startswith("Python:") for line in lines)
    assert any(line.startswith("NumPy:") for line in lines)


def test_print_version_returns_multiline_string() -> None:
    output = print_version()
    assert "pyGOTM:" in output
    assert "Python:" in output
