"""Tests for GOTM-to-FABM dependency coupling helpers."""

from __future__ import annotations

import numpy as np

from pygotm.fabm.coupling import apply_fabm_dependencies, copy_bioshade_feedback


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
