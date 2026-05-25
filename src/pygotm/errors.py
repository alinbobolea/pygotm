"""Shared integration exit-code mapping for public process boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

try:  # pragma: no cover - exercised when pyyaml is unavailable.
    yaml: Any = import_module("yaml")
except Exception:  # pragma: no cover
    yaml = None

from pydantic import ValidationError

from pygotm.gotm.runtime_builder import UnsupportedConfigurationError

EXIT_SUCCESS = 0
EXIT_VALIDATION_MISMATCH = 1
EXIT_CLICK_USAGE = 2
EXIT_CONFIG_ERROR = 10
EXIT_UNSUPPORTED_CONFIGURATION = 11
EXIT_RUNTIME_FAILURE = 12
EXIT_IO_ERROR = 13
EXIT_DEPENDENCY_UNAVAILABLE = 14
EXIT_INTERNAL_ERROR = 70

__all__ = [
    "EXIT_CLICK_USAGE",
    "EXIT_CONFIG_ERROR",
    "EXIT_DEPENDENCY_UNAVAILABLE",
    "EXIT_INTERNAL_ERROR",
    "EXIT_IO_ERROR",
    "EXIT_RUNTIME_FAILURE",
    "EXIT_SUCCESS",
    "EXIT_UNSUPPORTED_CONFIGURATION",
    "EXIT_VALIDATION_MISMATCH",
    "IntegrationError",
    "error_code_for_exception",
]


@dataclass(frozen=True, slots=True)
class IntegrationError(Exception):
    """Structured process-boundary error with a documented exit code."""

    code: int
    message: str

    def __str__(self) -> str:
        return self.message


def _is_yaml_error(exc: BaseException) -> bool:
    if yaml is None:
        return False
    yaml_error = getattr(yaml, "YAMLError", None)
    return bool(yaml_error is not None and isinstance(exc, yaml_error))


def error_code_for_exception(exc: BaseException) -> int:
    """Return the documented pyGOTM public exit code for *exc*."""

    if isinstance(exc, IntegrationError):
        return exc.code
    if isinstance(exc, ValidationError | TypeError) or _is_yaml_error(exc):
        return EXIT_CONFIG_ERROR
    if isinstance(exc, UnsupportedConfigurationError | NotImplementedError):
        return EXIT_UNSUPPORTED_CONFIGURATION
    if isinstance(exc, FileNotFoundError | PermissionError | OSError):
        return EXIT_IO_ERROR
    if isinstance(exc, ImportError | ModuleNotFoundError):
        return EXIT_DEPENDENCY_UNAVAILABLE
    if isinstance(exc, RuntimeError | ValueError):
        return EXIT_RUNTIME_FAILURE
    return EXIT_INTERNAL_ERROR


def run_with_exit_mapping(func: Callable[[], None], *, debug: bool = False) -> None:
    """Run *func*, mapping known exceptions to documented ``SystemExit`` codes."""

    try:
        func()
    except Exception as exc:
        if debug:
            raise
        raise SystemExit(error_code_for_exception(exc)) from exc
