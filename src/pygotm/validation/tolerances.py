"""Per-variable tolerance configuration and section classification for pyGOTM validation."""

# ruff: noqa: E501  -- tabular tolerance registry intentionally exceeds line limit

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "DEFAULT_PYFABM_TOLERANCE",
    "VARIABLE_TOLERANCES",
    "ValidationConfigurationError",
    "VariableTolerance",
    "classify_section",
    "get_tolerance",
]


class ValidationConfigurationError(Exception):
    """Raised when a variable has no tolerance configuration and no safe default."""


@dataclass(frozen=True)
class VariableTolerance:
    """Per-variable tolerance parameters for the three-indicator validation system.

    E_i = abs(calc_i - ref_i) / (atol + rtol * max(abs(ref_i), scale_floor))

    section: "pygotm" for core GOTM physics variables, "pyfabm" for FABM variables.
    """

    atol: float
    rtol: float
    scale_floor: float
    section: Literal["pygotm", "pyfabm"]


# Project-approved default tolerance for FABM variables not explicitly registered.
# FABM variable names are model-specific and cannot all be pre-registered.
DEFAULT_PYFABM_TOLERANCE = VariableTolerance(
    atol=1.0e-10,
    rtol=1.0e-6,
    scale_floor=1.0e-6,
    section="pyfabm",
)

# Registry of known GOTM physics output variables.
# Any variable NOT in this registry is classified as pyfabm using DEFAULT_PYFABM_TOLERANCE.
VARIABLE_TOLERANCES: dict[str, VariableTolerance] = {
    "temp": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0, section="pygotm"
    ),
    "salt": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0, section="pygotm"
    ),
    "u": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "v": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "h": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0e-4, section="pygotm"
    ),
    "rho": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0, section="pygotm"
    ),
    "buoy": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "NN": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-8, scale_floor=1.0e-10, section="pygotm"
    ),
    "SS": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-8, scale_floor=1.0e-10, section="pygotm"
    ),
    "ga": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "tke": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-10, section="pygotm"
    ),
    "eps": VariableTolerance(
        atol=1.0e-18, rtol=1.0e-7, scale_floor=1.0e-12, section="pygotm"
    ),
    "num": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-10, section="pygotm"
    ),
    "nuh": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-10, section="pygotm"
    ),
    "nus": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-10, section="pygotm"
    ),
    "nucl": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-10, section="pygotm"
    ),
    "L": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "P": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-8, scale_floor=1.0e-10, section="pygotm"
    ),
    "G": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-8, scale_floor=1.0e-10, section="pygotm"
    ),
    "Pb": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-8, scale_floor=1.0e-12, section="pygotm"
    ),
    "kb": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-12, section="pygotm"
    ),
    "epsb": VariableTolerance(
        atol=1.0e-18, rtol=1.0e-7, scale_floor=1.0e-14, section="pygotm"
    ),
    "an": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "cmue1": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "cmue2": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "as": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "at": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "avh": VariableTolerance(
        atol=1.0e-14, rtol=1.0e-7, scale_floor=1.0e-10, section="pygotm"
    ),
    "xP": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0e-3, section="pygotm"
    ),
    "fric": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "drag": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "taub": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "I_0": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0e-3, section="pygotm"
    ),
    "bioshade": VariableTolerance(
        atol=1.0e-10, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "PSTK": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-6, section="pygotm"
    ),
    "idpdy": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-8, section="pygotm"
    ),
    "idpdx": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-8, section="pygotm"
    ),
    "w": VariableTolerance(
        atol=1.0e-12, rtol=1.0e-8, scale_floor=1.0e-8, section="pygotm"
    ),
}


def get_tolerance(name: str) -> VariableTolerance:
    """Return per-variable tolerance parameters.

    Falls back to DEFAULT_PYFABM_TOLERANCE for variables not in the registry,
    treating them as FABM biogeochemical variables (names are model-specific).
    """
    return VARIABLE_TOLERANCES.get(name, DEFAULT_PYFABM_TOLERANCE)


def classify_section(name: str) -> Literal["pygotm", "pyfabm"]:
    """Return 'pygotm' for registered GOTM physics variables, 'pyfabm' otherwise."""
    return VARIABLE_TOLERANCES.get(name, DEFAULT_PYFABM_TOLERANCE).section
