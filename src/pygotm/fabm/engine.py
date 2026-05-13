"""Thin Python wrapper around ``pyfabm.Model``."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pygotm.fabm.state import fabm_state_variable_names

FloatArray = NDArray[np.float64]

__all__ = ["FABMEngine"]


class FABMEngine:
    """Manage one pyfabm model while pyGOTM owns transport and timestepping."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self._model_factory = model_factory
        self.model: Any | None = None
        self._state = np.zeros(0, dtype=np.float64)
        self._rates = np.zeros(0, dtype=np.float64)
        self._dependency_buffers: dict[str, FloatArray] = {}

    def initialize(
        self,
        nlev: int | None = None,
        h_col: np.ndarray | None = None,
        *,
        skip_start: bool = False,
    ) -> None:
        """Construct and start the pyfabm model for a 1-D water column.

        *nlev* is the number of grid cells (pyGOTM ``nlev``). When provided,
        the model is created with ``shape=(nlev,)`` so pyfabm allocates 1-D
        arrays of the correct size. *h_col* (length *nlev*, bottom→surface) is
        the initial cell-thickness array required by pyfabm before ``start()``.
        """

        if not self.config_path.is_file():
            msg = f"FABM model YAML not found: {self.config_path}"
            raise RuntimeError(msg)

        factory = self._model_factory
        if factory is None:
            try:
                import pyfabm
            except ImportError as exc:
                msg = (
                    "FABM is enabled, but pyfabm could not be imported. "
                    "Install FABM's Python front-end and ensure its compiled "
                    "FABM libraries are discoverable before running this case."
                )
                raise RuntimeError(msg) from exc
            factory = pyfabm.Model

        try:
            if nlev is not None:
                self.model = factory(str(self.config_path), shape=(nlev,))  # type: ignore[call-arg]
            else:
                self.model = factory(str(self.config_path))
        except Exception as exc:
            msg = f"failed to initialize pyfabm.Model from {self.config_path}: {exc}"
            raise RuntimeError(msg) from exc

        # pyfabm 3.0: cell_thickness is a write-only property setter — use direct
        # assignment so pyfabm registers the array before start() / getRates().
        if nlev is not None and h_col is not None:
            try:
                arr = np.ascontiguousarray(h_col, dtype=np.float64)
                self.model.cell_thickness = arr
            except AttributeError:
                pass

        if not skip_start:
            self._start_model()
            self._state = self._read_model_state()
            self._rates = np.zeros_like(self._state)

    def has_dependency(self, name: str) -> bool:
        """Return whether the pyfabm model exposes a dependency by *name*."""

        return self._find_dependency(name) is not None

    def set_dependency(
        self,
        name: str,
        value: float | np.ndarray,
    ) -> None:
        """Set one FABM dependency from a scalar or contiguous float64 array."""

        dependency = self._require_dependency(name)
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value, dtype=np.float64)
            existing = self._dependency_buffers.get(name)
            if existing is None or existing.shape != array.shape:
                existing = np.empty_like(array)
                self._dependency_buffers[name] = existing
            np.copyto(existing, array)
            self._assign_value(dependency, existing)
            return

        self._assign_value(dependency, float(value))

    def set_state(self, state: np.ndarray) -> None:
        """Replace the pyGOTM-owned FABM state buffer."""

        array = np.ascontiguousarray(state, dtype=np.float64)
        if self._state.shape != array.shape:
            self._state = np.empty_like(array)
        np.copyto(self._state, array)
        if self.model is None:
            return
        if hasattr(self.model, "state"):
            self.model.state[...] = self._state
            return
        variables = self._state_variables()
        if variables:
            flat_state = self._state.reshape((self._state.shape[0], -1))
            for index, variable in enumerate(variables[: flat_state.shape[0]]):
                self._assign_value(variable, flat_state[index])

    @property
    def state(self) -> np.ndarray:
        """Return the current mutable FABM state buffer."""

        return self._state

    def get_rates(
        self,
        *,
        surface: bool = True,
        bottom: bool = True,
    ) -> np.ndarray:
        """Return FABM source rates for each layer.

        When *surface=True* (default for backwards compat) the returned rates
        include the air-sea surface exchange distributed over ALL layers by
        pyfabm — which is physically wrong for a 1D column.  Pass
        ``surface=False, bottom=False`` to get bulk-only reaction rates
        that can safely be applied to every layer, then handle boundary
        exchange explicitly on the top/bottom layers.
        """

        model = self._require_model()
        get_fn = getattr(model, "getRates", None) or getattr(model, "get_rates", None)
        if get_fn is None:
            msg = "pyfabm model does not expose getRates()"
            raise RuntimeError(msg)

        rates = self._call_get_rates_flags(get_fn, surface=surface, bottom=bottom)

        array = np.ascontiguousarray(rates, dtype=np.float64)
        if self._rates.shape != array.shape:
            self._rates = np.empty_like(array)
        np.copyto(self._rates, array)
        return self._rates

    def diagnostics(self) -> dict[str, np.ndarray | float]:
        """Return current FABM diagnostic values as NumPy arrays or floats."""

        model = self._require_model()
        diagnostics: dict[str, np.ndarray | float] = {}
        raw = getattr(model, "diagnostics", None)
        if isinstance(raw, dict):
            for name, value in raw.items():
                diagnostics[str(name)] = self._copy_diagnostic(value)

        for variable in self._diagnostic_variables():
            name = str(getattr(variable, "name", getattr(variable, "id", "")))
            if not name:
                continue
            diagnostics[name] = self._copy_diagnostic(getattr(variable, "value", 0.0))
        return diagnostics

    def unresolved_dependencies(self) -> tuple[str, ...]:
        """Return a best-effort list of pyfabm dependencies still unset."""

        missing: list[str] = []
        for dependency in self._dependencies():
            name = str(getattr(dependency, "name", getattr(dependency, "id", "")))
            if not name:
                name = repr(dependency)
            if self._dependency_is_missing(dependency):
                missing.append(name)
        return tuple(dict.fromkeys(missing))

    def start(self) -> None:
        """Call model.start() after dependencies are set; update state buffers.

        Use this when ``initialize()`` was called with ``skip_start=True`` to
        defer start until the caller has supplied initial dependency values.
        """

        self._start_model()
        self._state = self._read_model_state()
        self._rates = np.zeros_like(self._state)

    def state_variable_names(self) -> tuple[str, ...]:
        """Return state-variable names exposed by the wrapped pyfabm model."""

        return fabm_state_variable_names(self._require_model())

    def _require_model(self) -> Any:
        if self.model is None:
            msg = "FABMEngine.initialize() has not been called"
            raise RuntimeError(msg)
        return self.model

    def _start_model(self) -> None:
        model = self._require_model()
        if not self._check_ready():
            missing = self.unresolved_dependencies()
            deps = ", ".join(missing)
            msg = f"pyfabm model is not ready; unresolved dependencies: {deps}"
            raise RuntimeError(msg)
        start = getattr(model, "start", None)
        if callable(start):
            try:
                start()
            except Exception as exc:
                msg = f"pyfabm model.start() failed: {exc}"
                raise RuntimeError(msg) from exc

    def _check_ready(self) -> bool:
        model = self._require_model()
        check_ready = getattr(model, "checkReady", None)
        if not callable(check_ready):
            return True
        for args, kwargs in (((), {}), ((), {"stop": False}), ((False,), {})):
            try:
                return bool(check_ready(*args, **kwargs))
            except TypeError:
                continue
            except Exception as exc:
                msg = f"pyfabm model readiness check failed: {exc}"
                raise RuntimeError(msg) from exc
        return True

    def _read_model_state(self) -> FloatArray:
        model = self._require_model()
        if hasattr(model, "state"):
            return np.ascontiguousarray(model.state, dtype=np.float64)
        variables = self._state_variables()
        if not variables:
            return np.zeros(0, dtype=np.float64)
        values = [
            np.asarray(getattr(variable, "value", 0.0), dtype=np.float64)
            for variable in variables
        ]
        return np.ascontiguousarray(np.stack(values), dtype=np.float64)

    def _state_variables(self) -> list[Any]:
        model = self._require_model()
        variables = getattr(model, "state_variables", None)
        if variables is None:
            variables = getattr(model, "stateVariables", None)
        if variables is None:
            return []
        return list(variables)

    def _diagnostic_variables(self) -> list[Any]:
        model = self._require_model()
        for attr in ("diagnostic_variables", "diagnosticVariables"):
            variables = getattr(model, attr, None)
            if variables is not None:
                return list(variables)
        return []

    def _dependencies(self) -> Iterable[Any]:
        model = self._require_model()
        for attr in (
            "dependencies",
            "required_dependencies",
            "dependencies_unfulfilled",
        ):
            dependencies = getattr(model, attr, None)
            if dependencies is not None:
                return list(dependencies)
        return ()

    def _find_dependency(self, name: str) -> Any | None:
        model = self._require_model()
        finder = getattr(model, "findDependency", None)
        if callable(finder):
            try:
                dependency = finder(name)
            except Exception:
                dependency = None
            if dependency is not None:
                return dependency
        for dependency in self._dependencies():
            dependency_name = getattr(dependency, "name", getattr(dependency, "id", ""))
            if str(dependency_name) == name:
                return dependency
        return None

    def _require_dependency(self, name: str) -> Any:
        dependency = self._find_dependency(name)
        if dependency is None:
            msg = f"FABM dependency {name!r} is not exposed by this model"
            raise KeyError(msg)
        return dependency

    @staticmethod
    def _assign_value(target: Any, value: float | np.ndarray) -> None:
        if hasattr(target, "value"):
            current = target.value
            if isinstance(current, np.ndarray) and isinstance(value, np.ndarray):
                np.copyto(current, value)
            else:
                target.value = value
            return
        if callable(getattr(target, "set", None)):
            target.set(value)
            return
        msg = f"FABM target {target!r} cannot receive a value"
        raise RuntimeError(msg)

    @staticmethod
    def _copy_diagnostic(value: object) -> np.ndarray | float:
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value, dtype=np.float64).copy()
        if isinstance(value, (int, float, np.floating)):
            return float(value)
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            return float(array)
        return np.ascontiguousarray(array, dtype=np.float64).copy()

    @staticmethod
    def _dependency_is_missing(dependency: Any) -> bool:
        for attr in ("is_set", "is_fulfilled", "fulfilled", "ready"):
            value = getattr(dependency, attr, None)
            if value is not None:
                return not bool(value() if callable(value) else value)
        if bool(getattr(dependency, "missing", False)):
            return True
        return getattr(dependency, "value", None) is None

    def _call_get_rates_flags(
        self,
        method: Callable[..., Any],
        *,
        surface: bool,
        bottom: bool,
    ) -> Any:
        """Call getRates respecting pyfabm 3.0 surface/bottom kwargs."""
        for kwargs in (
            {"surface": surface, "bottom": bottom},
            {},
        ):
            try:
                return method(**kwargs) if kwargs else method()
            except TypeError:
                continue
        return method(self._state)
