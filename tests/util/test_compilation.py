"""Tests for util/compilation.py — build info metadata stub."""

import pygotm.util.compilation as compilation


def test_import() -> None:
    assert compilation is not None


def test_compiler_is_string() -> None:
    assert isinstance(compilation.compiler, str)


def test_compiler_id_is_string() -> None:
    assert isinstance(compilation.compiler_id, str)


def test_compiler_version_is_string() -> None:
    assert isinstance(compilation.compiler_version, str)


def test_all_exports() -> None:
    assert set(compilation.__all__) == {"compiler", "compiler_id", "compiler_version"}


def test_default_values_empty() -> None:
    assert compilation.compiler == ""
    assert compilation.compiler_id == ""
    assert compilation.compiler_version == ""
