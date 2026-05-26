"""Bundled GOTM reference-case fixtures for tests.

These cases are vendored at ``tests/fixtures/cases/`` so the test suite is
fully self-contained and runnable on a clean checkout. The seven bundled
cases cover the distinct physics regimes exercised by pyGOTM:

* ``couette`` — pure shear-driven turbulence
* ``channel`` — channel flow with buoyancy
* ``asics_med`` — ASICS Mediterranean (meteorological forcing)
* ``rouse`` — sediment-stratified turbulence (FABM)
* ``seagrass`` — vegetated canopy drag
* ``wave_breaking`` — surface wave-breaking turbulence
* ``entrainment`` — mixed-layer entrainment

Tests should resolve bundled cases through :func:`bundled_case` rather than
reaching into the (gitignored) top-level ``validation/reference/`` directory.
"""

from __future__ import annotations

from pathlib import Path

from pygotm.validation.reference import ValidationCase, resolve_reference_case

__all__ = [
    "BUNDLED_CASES_ROOT",
    "BUNDLED_CASE_NAMES",
    "bundled_case",
    "bundled_case_path",
]

BUNDLED_CASES_ROOT: Path = (Path(__file__).resolve().parent / "cases").resolve()

BUNDLED_CASE_NAMES: tuple[str, ...] = (
    "couette",
    "channel",
    "asics_med",
    "rouse",
    "seagrass",
    "wave_breaking",
    "entrainment",
)


def bundled_case(case_name: str) -> ValidationCase:
    """Resolve a bundled case to a :class:`ValidationCase` rooted under ``tests/fixtures/cases``."""

    return resolve_reference_case(case_name, cases_root=BUNDLED_CASES_ROOT)


def bundled_case_path(case_name: str, filename: str = "gotm.yaml") -> Path:
    """Return the absolute path of ``filename`` inside the bundled case directory."""

    return (BUNDLED_CASES_ROOT / case_name / filename).resolve()
