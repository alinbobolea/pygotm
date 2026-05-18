"""Tests for the thin pyfabm engine wrapper."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pygotm.fabm.engine import FABMEngine


class FakeDependency:
    def __init__(self, name: str, value: object | None = 0.0) -> None:
        self.name = name
        self.value = value


class FakeDiagnostic:
    name = "oxygen"

    def __init__(self) -> None:
        self.value = np.array([1.0, 2.0], dtype=np.float64)


class ReadyModel:
    def __init__(self, path: str) -> None:
        self.path = path
        self.started = False
        self.state = np.array([1.0, 2.0], dtype=np.float64)
        self.dependencies = [FakeDependency("temperature", 10.0)]
        self.diagnostic_variables = [FakeDiagnostic()]

    def start(self) -> None:
        self.started = True

    def checkReady(self) -> bool:
        return True

    def findDependency(self, name: str) -> FakeDependency | None:
        for dependency in self.dependencies:
            if dependency.name == name:
                return dependency
        return None

    def getRates(self) -> np.ndarray:
        return self.state * 0.5


class TimeAwareModel(ReadyModel):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.rate_calls: list[tuple[float, bool, bool]] = []

    def getRates(
        self,
        t: float,
        *,
        surface: bool = True,
        bottom: bool = True,
    ) -> np.ndarray:
        self.rate_calls.append((t, surface, bottom))
        return self.state + t


class MissingDependencyModel(ReadyModel):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.dependencies = [FakeDependency("density", None)]

    def checkReady(self) -> bool:
        return False


def _fabm_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "fabm.yaml"
    path.write_text("instances: {}\n", encoding="utf-8")
    return path


def test_engine_initializes_sets_dependencies_and_gets_rates(tmp_path: Path) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=ReadyModel)

    engine.initialize()
    assert isinstance(engine.model, ReadyModel)
    assert engine.model.started
    assert engine.has_dependency("temperature")

    dependency = engine.model.findDependency("temperature")
    assert dependency is not None
    engine.set_dependency("temperature", np.array([3.0, 4.0], dtype=np.float64))
    assert isinstance(dependency.value, np.ndarray)
    np.testing.assert_allclose(dependency.value, [3.0, 4.0])

    engine.set_state(np.array([5.0, 7.0], dtype=np.float64))
    np.testing.assert_allclose(engine.get_rates(), [2.5, 3.5])
    np.testing.assert_allclose(engine.diagnostics()["oxygen"], [1.0, 2.0])


def test_engine_passes_time_and_boundary_flags_to_get_rates(tmp_path: Path) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=TimeAwareModel)
    engine.initialize()

    rates = engine.get_rates(surface=False, bottom=True, time=12.5)

    assert isinstance(engine.model, TimeAwareModel)
    assert engine.model.rate_calls == [(12.5, False, True)]
    np.testing.assert_allclose(rates, [13.5, 14.5])


def test_engine_reports_every_unresolved_dependency(tmp_path: Path) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=MissingDependencyModel)

    with pytest.raises(RuntimeError, match="density"):
        engine.initialize()


def test_engine_requires_pyfabm_when_no_factory_is_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "pyfabm":
            raise ImportError("blocked test import")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    engine = FABMEngine(_fabm_yaml(tmp_path))

    with pytest.raises(RuntimeError, match="pyfabm could not be imported"):
        engine.initialize()
