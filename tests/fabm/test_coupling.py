"""Tests for GOTM-to-FABM dependency coupling helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pygotm.fabm.coupling import apply_fabm_dependencies, copy_bioshade_feedback
from pygotm.fabm.fabm_loop import (
    _fabm_day_of_year,
    _par_with_bioext_from_attenuation,
    _record_fabm_output,
    _set_environment,
    _try_set,
    _update_light_from_diagnostics,
)


class RecordingEngine:
    def __init__(self) -> None:
        self.values: dict[str, float | np.ndarray] = {}

    def has_dependency(self, name: str) -> bool:
        return name != "density"

    def set_dependency(self, name: str, value: float | np.ndarray) -> None:
        self.values[name] = value

    def diagnostics(self) -> dict[str, np.ndarray | float]:
        return {
            "attenuation_coefficient_of_photosynthetic_radiative_flux": np.array(
                [1.0, 0.9, 0.8],
                dtype=np.float64,
            )
        }


def test_apply_fabm_dependencies_sets_available_dependencies() -> None:
    engine = RecordingEngine()
    profile = np.ones(3, dtype=np.float64)

    apply_fabm_dependencies(
        engine,
        temperature=profile,
        practical_salinity=profile * 2.0,
        density=profile * 1000.0,
        cell_thickness=profile * 0.5,
        downwelling_photosynthetic_radiative_flux=profile * 10.0,
    )

    assert "temperature" in engine.values
    assert "practical_salinity" in engine.values
    assert "density" not in engine.values
    assert "cell_thickness" in engine.values
    assert "downwelling_photosynthetic_radiative_flux" in engine.values


def test_copy_bioshade_feedback_uses_fabm_diagnostic() -> None:
    target = np.ones(3, dtype=np.float64)

    assert copy_bioshade_feedback(RecordingEngine(), target)
    np.testing.assert_allclose(target, [1.0, 0.9, 0.8])


def test_fabm_day_of_year_matches_gotm_fractional_calendar() -> None:
    assert _fabm_day_of_year(1.0, 0.0) == 0.0
    assert _fabm_day_of_year(1.0, 43200.0) == 0.5
    assert _fabm_day_of_year(2.0, 0.0) == 1.0
    assert _fabm_day_of_year(2.0, None) == 2.0


def test_fabm_environment_uses_center_depth_par() -> None:
    engine = RecordingEngine()
    nlev = 3
    profile = np.ones(nlev + 1, dtype=np.float64)
    h = np.array([0.0, 2.0, 2.0, 2.0], dtype=np.float64)
    rad = np.zeros(nlev + 1, dtype=np.float64)
    rad[nlev] = 100.0

    _set_environment(
        engine,
        object(),
        nlev,
        profile,
        profile * 2.0,
        profile * 1000.0,
        h,
        rad,
        light_A=0.4,
        light_g2=10.0,
    )

    expected_depth = np.array([5.0, 3.0, 1.0], dtype=np.float64)
    expected_par = 60.0 * np.exp(-expected_depth / 10.0)
    np.testing.assert_allclose(
        engine.values["downwelling_photosynthetic_radiative_flux"],
        expected_par,
    )
    assert engine.values["surface_downwelling_photosynthetic_radiative_flux"] == 60.0


def test_par_with_bioext_matches_gotm_light_formula() -> None:
    nlev = 3
    h = np.array([0.0, 2.0, 2.0, 2.0], dtype=np.float64)
    rad = np.zeros(nlev + 1, dtype=np.float64)
    rad[nlev] = 100.0
    light_g2 = 10.0
    local_ext = np.array([0.02, 0.03, 0.04], dtype=np.float64)

    par, surface_par = _par_with_bioext_from_attenuation(
        local_ext,
        h,
        rad,
        nlev,
        light_A=0.4,
        light_g2=light_g2,
    )

    expected = np.empty(nlev, dtype=np.float64)
    bioext = 0.0
    depths = np.array([5.0, 3.0, 1.0], dtype=np.float64)
    for idx in range(nlev - 1, -1, -1):
        bioext += local_ext[idx] * h[idx + 1] * 0.5
        expected[idx] = 60.0 * np.exp(-depths[idx] / light_g2 - bioext)
        bioext += local_ext[idx] * h[idx + 1] * 0.5
    np.testing.assert_allclose(par, expected)
    assert surface_par == 60.0


def test_update_light_uses_targeted_diagnostic_lookup() -> None:
    class TargetedDiagnosticEngine(RecordingEngine):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[tuple[str, bool]] = []

        def diagnostics(self) -> dict[str, np.ndarray | float]:
            raise AssertionError("full diagnostic scan should not be used")

        def diagnostic(
            self,
            name: str,
            *,
            copy: bool = True,
        ) -> np.ndarray | float | None:
            self.requests.append((name, copy))
            if name == "attenuation_coefficient_of_photosynthetic_radiative_flux":
                return np.array([0.02, 0.03, 0.04], dtype=np.float64)
            return None

    engine = TargetedDiagnosticEngine()
    nlev = 3
    h = np.array([0.0, 2.0, 2.0, 2.0], dtype=np.float64)
    rad = np.zeros(nlev + 1, dtype=np.float64)
    rad[nlev] = 100.0

    _update_light_from_diagnostics(
        engine,
        nlev,
        h,
        rad,
        light_A=0.4,
        light_g2=10.0,
    )

    assert engine.requests == [
        ("attenuation_coefficient_of_photosynthetic_radiative_flux", False)
    ]
    assert engine.values["surface_downwelling_photosynthetic_radiative_flux"] == 60.0
    assert "downwelling_photosynthetic_radiative_flux" in engine.values


def test_try_set_propagates_real_setter_failures() -> None:
    class FailingEngine:
        def set_dependency_if_present(
            self,
            name: str,
            value: np.ndarray,
        ) -> bool:
            raise RuntimeError(f"setter failed for {name}")

    with pytest.raises(RuntimeError, match="temperature"):
        _try_set(
            FailingEngine(),
            "temperature",
            np.ones(3, dtype=np.float64),
        )


def test_record_fabm_output_uses_cached_diagnostics_and_boundary_scalars() -> None:
    class EngineWithStaleDiagnostics:
        def diagnostics(self) -> dict[str, np.ndarray]:
            raise AssertionError("cached output diagnostics should be used")

    reference_z_profiles = {
        "jrc_med_ergom_Amm": np.zeros((1, 4), dtype=np.float64),
    }
    reference_scalars = {
        "jrc_med_ergom_OFL": np.zeros(1, dtype=np.float64),
        "jrc_med_ergom_DNB": np.zeros(1, dtype=np.float64),
        "jrc_med_ergom_fl": np.zeros(1, dtype=np.float64),
    }
    output = SimpleNamespace(
        nout=1,
        reference_z_profiles=reference_z_profiles,
        reference_scalars=reference_scalars,
    )
    cc = np.array([[10.0, 20.0, 30.0]], dtype=np.float64)
    cached_diagnostics = {
        "jrc/med/ergom/OFL": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "jrc/med/ergom/DNB": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        "jrc/med/ergom/Amm": np.array([7.0, 8.0, 9.0], dtype=np.float64),
    }

    _record_fabm_output(
        EngineWithStaleDiagnostics(),
        cc,
        [],
        [(0, reference_scalars["jrc_med_ergom_fl"], "jrc_med_ergom_fl")],
        output,
        0,
        3,
        diagnostics=cached_diagnostics,
    )

    assert reference_scalars["jrc_med_ergom_OFL"][0] == 3.0
    assert reference_scalars["jrc_med_ergom_DNB"][0] == 4.0
    assert reference_scalars["jrc_med_ergom_fl"][0] == 10.0
    np.testing.assert_allclose(
        reference_z_profiles["jrc_med_ergom_Amm"][0],
        [0.0, 7.0, 8.0, 9.0],
    )
