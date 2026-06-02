"""Tests for the thin pyfabm engine wrapper."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pygotm.fabm.engine import FABMEngine
from pygotm.gotm.runtime_output import (
    FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES,
    allocate_runtime_output,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SELMA_MINIMAL_FABM = _PROJECT_ROOT / "tests/fixtures/fabm/selma_minimal.yaml"
_REFERENCE_FABM_CONFIGS = (
    _PROJECT_ROOT / "validation/reference/blacksea/fabm.yaml",
    _PROJECT_ROOT / "validation/reference/medsea_east/fabm.yaml",
    _PROJECT_ROOT / "validation/reference/medsea_west/fabm.yaml",
    _PROJECT_ROOT / "validation/reference/nns_annual/fabm.yaml",
    _PROJECT_ROOT / "validation/reference/rouse/fabm.yaml",
    _SELMA_MINIMAL_FABM,
)

_SELMA_MINIMAL_Z_OUTPUTS = {
    "selma_dd",
    "selma_aa",
    "selma_nn",
    "selma_po",
    "selma_o2",
    "selma_pw",
    "selma_Nit",
    "selma_Amm",
    "selma_Pho",
    "selma_DO_mg",
    "selma_DNP",
    "total_nitrogen",
    "total_phosphorus",
    "total_carbon",
    "attenuation_coefficient_of_photosynthetic_radiative_flux",
}

_SELMA_MINIMAL_SCALAR_OUTPUTS = {
    "selma_fl",
    "selma_pb",
    "selma_DNB",
    "selma_SBR",
    "selma_PBR",
    "selma_OFL",
    "total_nitrogen_at_interfaces",
    "total_nitrogen_at_bottom",
    "total_phosphorus_at_interfaces",
    "total_phosphorus_at_bottom",
    "total_carbon_at_interfaces",
    "total_carbon_at_bottom",
}


class FakeDependency:
    def __init__(self, name: str, value: object | None = 0.0) -> None:
        self.name = name
        self.value = value


class FakeDiagnostic:
    name = "oxygen"
    output_name = "oxygen"
    units = "mmol m-3"
    long_name = "oxygen"
    output = True

    def __init__(self) -> None:
        self.value = np.array([1.0, 2.0], dtype=np.float64)


class FakeVariable:
    def __init__(
        self,
        name: str,
        *,
        output_name: str | None = None,
        units: str = "",
        long_name: str = "",
        output: bool | None = None,
    ) -> None:
        self.name = name
        self.output_name = output_name
        self.units = units
        self.long_name = long_name
        if output is not None:
            self.output = output


class ReadyModel:
    def __init__(self, path: str) -> None:
        self.path = path
        self.started = False
        self.state = np.array([1.0, 2.0], dtype=np.float64)
        self.dependencies = [FakeDependency("temperature", 10.0)]
        self.diagnostic_variables = [FakeDiagnostic()]
        self.horizontal_diagnostic_variables = [
            FakeVariable(
                "surface_flux",
                output_name="surface_flux",
                units="mmol m-2 d-1",
                long_name="surface flux",
                output=True,
            )
        ]
        self.interior_state_variables = [
            FakeVariable(
                "npzd/phy",
                output_name="npzd_phy",
                units="mmol m-3",
                long_name="phytoplankton",
            )
        ]
        self.bottom_state_variables = [
            FakeVariable(
                "sed/fl",
                output_name="sed_fl",
                units="mmol m-2",
                long_name="fluff",
            )
        ]
        self.surface_state_variables = []
        self.find_calls: list[str] = []

    def start(self) -> None:
        self.started = True

    def checkReady(self) -> bool:
        return True

    def findDependency(self, name: str) -> FakeDependency | None:
        self.find_calls.append(name)
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

    oxygen = engine.diagnostic("oxygen", copy=False)
    assert isinstance(oxygen, np.ndarray)
    assert oxygen is engine.model.diagnostic_variables[0].value

    oxygen_copy = engine.diagnostic("oxygen")
    assert isinstance(oxygen_copy, np.ndarray)
    oxygen_copy[0] = 99.0
    np.testing.assert_allclose(engine.model.diagnostic_variables[0].value, [1.0, 2.0])


def test_engine_passes_time_and_boundary_flags_to_get_rates(tmp_path: Path) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=TimeAwareModel)
    engine.initialize()

    rates = engine.get_rates(surface=False, bottom=True, time=12.5)

    assert isinstance(engine.model, TimeAwareModel)
    assert engine.model.rate_calls == [(12.5, False, True)]
    np.testing.assert_allclose(rates, [13.5, 14.5])


def test_engine_reports_dynamic_output_specs(tmp_path: Path) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=ReadyModel)
    engine.initialize()

    specs = {spec.name: spec for spec in engine.output_variable_specs()}

    assert specs["npzd_phy"].kind == "z"
    assert specs["npzd_phy"].units == "mmol m-3"
    assert specs["sed_fl"].kind == "scalar"
    assert specs["oxygen"].kind == "z"
    assert specs["surface_flux"].kind == "scalar"


def test_engine_reports_minimal_selma_output_specs() -> None:
    pytest.importorskip("pyfabm")
    engine = FABMEngine(_SELMA_MINIMAL_FABM)
    engine.initialize(nlev=3, skip_start=True)

    specs = {spec.name: spec for spec in engine.output_variable_specs()}

    assert set(specs) == _SELMA_MINIMAL_Z_OUTPUTS | _SELMA_MINIMAL_SCALAR_OUTPUTS
    assert {name for name, spec in specs.items() if spec.kind == "z"} == (
        _SELMA_MINIMAL_Z_OUTPUTS
    )
    assert {name for name, spec in specs.items() if spec.kind == "scalar"} == (
        _SELMA_MINIMAL_SCALAR_OUTPUTS
    )
    assert specs["selma_nn"].units == "mmol N/m3"
    assert (
        specs["attenuation_coefficient_of_photosynthetic_radiative_flux"].units == "m-1"
    )


@pytest.mark.parametrize("fabm_config", _REFERENCE_FABM_CONFIGS)
def test_runtime_output_can_declare_every_known_fabm_output(
    fabm_config: Path,
) -> None:
    pytest.importorskip("pyfabm")
    if not fabm_config.is_file():
        pytest.skip(f"FABM config fixture is missing: {fabm_config}")
    engine = FABMEngine(fabm_config)
    engine.initialize(nlev=3, skip_start=True)
    output = allocate_runtime_output(nlev=3, nt=1)

    for spec in engine.output_variable_specs():
        output.declare_fabm_output(
            spec.name,
            3,
            z_profile=spec.kind == "z",
            attrs={"units": spec.units, "long_name": spec.long_name},
            replace_extra=spec.name in FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES,
        )
    output.validate(3)

    declared = set(output.fabm_z_profiles) | set(output.fabm_scalars)
    expected = {spec.name for spec in engine.output_variable_specs()}
    assert declared == expected


def test_engine_prefers_horizontal_diagnostic_when_output_name_is_shared(
    tmp_path: Path,
) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=ReadyModel)
    engine.initialize()
    assert isinstance(engine.model, ReadyModel)
    engine.model.diagnostic_variables.append(
        FakeVariable(
            "surface_flux",
            output_name="surface_flux",
            units="bad",
            long_name="bad duplicate",
            output=True,
        )
    )

    specs = {spec.name: spec for spec in engine.output_variable_specs()}

    assert specs["surface_flux"].kind == "scalar"
    assert specs["surface_flux"].units == "mmol m-2 d-1"


def test_engine_caches_optional_dependency_lookup(tmp_path: Path) -> None:
    engine = FABMEngine(_fabm_yaml(tmp_path), model_factory=ReadyModel)
    engine.initialize()
    assert isinstance(engine.model, ReadyModel)

    engine.model.find_calls.clear()
    first = np.array([3.0, 4.0], dtype=np.float64)
    second = np.array([5.0, 6.0], dtype=np.float64)

    assert engine.set_dependency_if_present("temperature", first)
    assert engine.set_dependency_if_present("temperature", second)
    assert not engine.set_dependency_if_present("missing", 1.0)
    assert not engine.set_dependency_if_present("missing", 2.0)

    assert engine.model.find_calls == ["temperature", "missing"]
    dependency = engine.model.findDependency("temperature")
    assert dependency is not None
    np.testing.assert_allclose(dependency.value, second)


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
