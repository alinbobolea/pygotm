"""Tests for GOTM-to-FABM dependency coupling helpers."""

from __future__ import annotations

import numpy as np

from pygotm.fabm.coupling import apply_fabm_dependencies, copy_bioshade_feedback
from pygotm.fabm.fabm_loop import (
    _fabm_day_of_year,
    _par_with_bioext_from_attenuation,
    _set_environment,
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
