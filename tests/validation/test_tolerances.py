"""Tests for validation/tolerances.py — per-variable tolerance configuration."""

from __future__ import annotations

import pytest

from pygotm.validation.tolerances import (
    DEFAULT_PYFABM_TOLERANCE,
    VARIABLE_TOLERANCES,
    VariableTolerance,
    classify_section,
    get_tolerance,
)


def test_variable_tolerance_is_frozen() -> None:
    tol = VariableTolerance(atol=1e-10, rtol=1e-8, scale_floor=1.0, section="pygotm")
    with pytest.raises((AttributeError, TypeError)):
        tol.atol = 1.0  # type: ignore[misc]


def test_known_gotm_variable_returns_pygotm_section() -> None:
    for name in ("temp", "salt", "u", "v", "tke", "eps", "num", "nuh"):
        assert get_tolerance(name).section == "pygotm", f"expected pygotm for {name!r}"


def test_unknown_variable_returns_pyfabm_default() -> None:
    tol = get_tolerance("oxygen_some_fabm_model")
    assert tol.section == "pyfabm"
    assert tol is DEFAULT_PYFABM_TOLERANCE


def test_classify_section_known_gotm_variable() -> None:
    assert classify_section("temp") == "pygotm"
    assert classify_section("salt") == "pygotm"
    assert classify_section("tke") == "pygotm"


def test_classify_section_unknown_variable_is_pyfabm() -> None:
    assert classify_section("some_fabm_tracer_xyz") == "pyfabm"


def test_all_registered_variables_have_positive_tolerances() -> None:
    for name, tol in VARIABLE_TOLERANCES.items():
        assert tol.atol > 0, f"{name}: atol must be positive"
        assert tol.rtol > 0, f"{name}: rtol must be positive"
        assert tol.scale_floor > 0, f"{name}: scale_floor must be positive"


def test_all_registered_variables_are_pygotm_section() -> None:
    for name, tol in VARIABLE_TOLERANCES.items():
        assert tol.section == "pygotm", f"{name}: should be pygotm section"


def test_default_pyfabm_tolerance_is_pyfabm_section() -> None:
    assert DEFAULT_PYFABM_TOLERANCE.section == "pyfabm"
    assert DEFAULT_PYFABM_TOLERANCE.atol > 0
    assert DEFAULT_PYFABM_TOLERANCE.rtol > 0
    assert DEFAULT_PYFABM_TOLERANCE.scale_floor > 0


def test_tke_atol_is_tighter_than_temperature() -> None:
    assert get_tolerance("tke").atol < get_tolerance("temp").atol


def test_get_tolerance_returns_same_object_as_registry() -> None:
    assert get_tolerance("temp") is VARIABLE_TOLERANCES["temp"]
