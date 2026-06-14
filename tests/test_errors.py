"""Tests for pygotm.errors — exit-code mapping and error utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from pygotm.errors import (
    EXIT_CONFIG_ERROR,
    EXIT_DEPENDENCY_UNAVAILABLE,
    EXIT_INTERNAL_ERROR,
    EXIT_IO_ERROR,
    EXIT_RUNTIME_FAILURE,
    EXIT_SUCCESS,
    EXIT_UNSUPPORTED_CONFIGURATION,
    IntegrationError,
    error_code_for_exception,
    run_with_exit_mapping,
)
from pygotm.gotm.runtime_builder import UnsupportedConfigurationError

yaml: Any = import_module("yaml")


# ---------------------------------------------------------------------------
# IntegrationError
# ---------------------------------------------------------------------------


def test_integration_error_str_returns_message() -> None:
    err = IntegrationError(code=EXIT_IO_ERROR, message="disk full")
    assert str(err) == "disk full"
    assert err.code == EXIT_IO_ERROR


def test_integration_error_is_frozen() -> None:
    err = IntegrationError(code=EXIT_IO_ERROR, message="disk full")
    with pytest.raises(AttributeError):
        err.code = EXIT_SUCCESS  # type: ignore[misc]


# ---------------------------------------------------------------------------
# error_code_for_exception
# ---------------------------------------------------------------------------


def test_error_code_integration_error_returns_its_own_code() -> None:
    err = IntegrationError(code=EXIT_IO_ERROR, message="test")
    assert error_code_for_exception(err) == EXIT_IO_ERROR


def test_error_code_validation_error_returns_config_error() -> None:
    from pydantic import BaseModel, ValidationError

    class M(BaseModel):
        x: int

    try:
        M.model_validate({"x": "not-an-int"})
    except ValidationError as exc:
        assert error_code_for_exception(exc) == EXIT_CONFIG_ERROR


def test_error_code_type_error_returns_config_error() -> None:
    err = TypeError("wrong type")
    assert error_code_for_exception(err) == EXIT_CONFIG_ERROR


def test_error_code_yaml_error_returns_config_error() -> None:
    try:
        yaml.safe_load(": : :")
    except yaml.YAMLError as exc:
        assert error_code_for_exception(exc) == EXIT_CONFIG_ERROR


def test_error_code_unsupported_configuration_error_returns_unsupported() -> None:
    err = UnsupportedConfigurationError("not supported")
    assert error_code_for_exception(err) == EXIT_UNSUPPORTED_CONFIGURATION


def test_error_code_not_implemented_error_returns_unsupported() -> None:
    err = NotImplementedError("not done yet")
    assert error_code_for_exception(err) == EXIT_UNSUPPORTED_CONFIGURATION


def test_error_code_file_not_found_returns_io_error() -> None:
    err = FileNotFoundError("missing.nc")
    assert error_code_for_exception(err) == EXIT_IO_ERROR


def test_error_code_permission_error_returns_io_error() -> None:
    err = PermissionError("read-only")
    assert error_code_for_exception(err) == EXIT_IO_ERROR


def test_error_code_os_error_returns_io_error() -> None:
    err = OSError("disk error")
    assert error_code_for_exception(err) == EXIT_IO_ERROR


def test_error_code_import_error_returns_dependency_unavailable() -> None:
    err = ImportError("no module")
    assert error_code_for_exception(err) == EXIT_DEPENDENCY_UNAVAILABLE


def test_error_code_module_not_found_returns_dependency_unavailable() -> None:
    err = ModuleNotFoundError("no module")
    assert error_code_for_exception(err) == EXIT_DEPENDENCY_UNAVAILABLE


def test_error_code_runtime_error_returns_runtime_failure() -> None:
    err = RuntimeError("something went wrong")
    assert error_code_for_exception(err) == EXIT_RUNTIME_FAILURE


def test_error_code_value_error_returns_runtime_failure() -> None:
    err = ValueError("bad value")
    assert error_code_for_exception(err) == EXIT_RUNTIME_FAILURE


def test_error_code_unknown_exception_returns_internal_error() -> None:
    err = ZeroDivisionError("div by zero")
    assert error_code_for_exception(err) == EXIT_INTERNAL_ERROR


def test_error_code_keyboard_interrupt_returns_internal_error() -> None:
    err = KeyboardInterrupt()
    assert error_code_for_exception(err) == EXIT_INTERNAL_ERROR


# ---------------------------------------------------------------------------
# run_with_exit_mapping
# ---------------------------------------------------------------------------


def test_run_with_exit_mapping_success_does_not_raise() -> None:
    run_with_exit_mapping(lambda: None)


def test_run_with_exit_mapping_maps_file_not_found_to_io_error() -> None:
    def _raise() -> None:
        raise FileNotFoundError("missing file")

    with pytest.raises(SystemExit) as exc_info:
        run_with_exit_mapping(_raise)
    assert exc_info.value.code == EXIT_IO_ERROR


def test_run_with_exit_mapping_maps_runtime_error_to_runtime_failure() -> None:
    def _raise() -> None:
        raise RuntimeError("something failed")

    with pytest.raises(SystemExit) as exc_info:
        run_with_exit_mapping(_raise)
    assert exc_info.value.code == EXIT_RUNTIME_FAILURE


def test_run_with_exit_mapping_maps_integration_error_to_its_own_code() -> None:
    def _raise() -> None:
        raise IntegrationError(code=EXIT_IO_ERROR, message="io error")

    with pytest.raises(SystemExit) as exc_info:
        run_with_exit_mapping(_raise)
    assert exc_info.value.code == EXIT_IO_ERROR


def test_run_with_exit_mapping_debug_reraises_original_exception() -> None:
    def _raise() -> None:
        raise FileNotFoundError("missing")

    with pytest.raises(FileNotFoundError, match="missing"):
        run_with_exit_mapping(_raise, debug=True)
