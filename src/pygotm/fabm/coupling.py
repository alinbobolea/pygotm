"""Map pyGOTM-owned physical fields into FABM dependencies."""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = [
    "FABMDependencyEngine",
    "apply_fabm_dependencies",
    "copy_bioshade_feedback",
]


class FABMDependencyEngine(Protocol):
    """Subset of :class:`FABMEngine` used by coupling helpers."""

    def has_dependency(self, name: str) -> bool: ...

    def set_dependency(self, name: str, value: float | np.ndarray) -> None: ...

    def diagnostics(self) -> dict[str, np.ndarray | float]: ...


def _set_if_present(
    engine: FABMDependencyEngine,
    name: str,
    value: float | np.ndarray,
) -> None:
    if engine.has_dependency(name):
        engine.set_dependency(name, value)


def apply_fabm_dependencies(
    engine: FABMDependencyEngine,
    *,
    temperature: np.ndarray,
    practical_salinity: np.ndarray,
    density: np.ndarray,
    cell_thickness: np.ndarray,
    downwelling_photosynthetic_radiative_flux: np.ndarray,
) -> None:
    """Set the core GOTM-to-FABM dependencies in bulk."""

    _set_if_present(engine, "temperature", temperature)
    _set_if_present(engine, "practical_salinity", practical_salinity)
    _set_if_present(engine, "density", density)
    _set_if_present(engine, "cell_thickness", cell_thickness)
    _set_if_present(
        engine,
        "downwelling_photosynthetic_radiative_flux",
        downwelling_photosynthetic_radiative_flux,
    )


def copy_bioshade_feedback(
    engine: FABMDependencyEngine,
    target: np.ndarray,
) -> bool:
    """Copy FABM light-attenuation diagnostics into ``target`` when present."""

    diagnostics = engine.diagnostics()
    for name in (
        "attenuation_coefficient_of_photosynthetic_radiative_flux",
        "kc",
    ):
        value = diagnostics.get(name)
        if isinstance(value, np.ndarray) and value.shape == target.shape:
            np.copyto(target, value)
            return True
    return False
