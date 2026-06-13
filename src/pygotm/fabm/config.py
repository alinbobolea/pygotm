"""Configuration helpers for the Python-level FABM bridge.

**FABM model YAML transformation (why the original** ``fabm.yaml`` **is rewritten).**
The GOTM-lake ``selmaprotbas`` phytoplankton model accepts two parameters that
exist **only** in the GOTM-lake fork's bundled FABM build:

* ``alpha`` — nutrient-uptake half-saturation, and
* ``beta`` — temperature growth-correction factor.

The conda ``pyfabm`` distribution (the stock upstream FABM library pyGOTM links
against) does not declare these parameters, so loading a reference
``fabm.yaml`` that contains them raises an "invalid configuration" error and the
case cannot start.

To run such cases at all, :func:`resolve_fabm_config_path` /
:func:`_normalized_fabm_config_path` materialize a pyfabm-compatible copy of the
model YAML with ``alpha``/``beta`` stripped from every
``selmaprotbas/phytoplankton`` instance and point the run at that copy. The
transformation is recorded in the NetCDF ``fabm_yaml_sha256`` attribute (it
hashes the materialized file), and the validation runner stages the same
materialized YAML into the case bundle, so the bundle stays self-consistent and
reproducible.

This stripping is a genuine **limitation**, not a cosmetic normalization: pyGOTM
then runs ``selmaprotbas`` with the upstream **defaults** for ``alpha``/``beta``
rather than the GOTM-lake tuned values, so the biogeochemistry cannot reach full
parity with the GOTM-lake reference until pyfabm exposes these parameters. See
``docs/validation/lake_erken_parity.md`` for the full lake_erken limitation
picture (this, plus the ``variable_bottom_index`` benthic-coupling limitation).
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "FABMConfig",
    "fabm_enabled",
    "load_fabm_config",
    "resolve_fabm_config_path",
]

# GOTM-lake-only selmaprotbas phytoplankton parameters that conda pyfabm cannot
# parse; stripped during materialization (see module docstring).
_LEGACY_SELMA_PHYTOPLANKTON_PARAMETERS = frozenset(("alpha", "beta"))


@dataclass(frozen=True, slots=True)
class FABMConfig:
    """Resolved FABM settings from a GOTM YAML document."""

    use: bool
    config_path: Path | None
    freshwater_impact: bool
    repair_state: bool
    shade_feedback: bool
    albedo_feedback: bool
    surface_drag_feedback: bool


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def fabm_enabled(document: dict[str, Any]) -> bool:
    """Return whether ``fabm.use`` is enabled in a GOTM YAML document."""

    return bool(_mapping(document.get("fabm")).get("use", False))


def resolve_fabm_config_path(
    gotm_yaml_path: str | Path,
    document: dict[str, Any],
) -> Path:
    """Resolve the FABM model YAML path for a FABM-active GOTM case."""

    raw_fabm = _mapping(document.get("fabm"))
    configured = (
        raw_fabm.get("config")
        or raw_fabm.get("config_file")
        or raw_fabm.get("yaml")
        or raw_fabm.get("file")
        or "fabm.yaml"
    )
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = Path(gotm_yaml_path).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        msg = f"FABM model YAML not found: {path}"
        raise RuntimeError(msg)
    return _normalized_fabm_config_path(path)


def _normalized_fabm_config_path(path: Path) -> Path:
    """Return a pyfabm-compatible FABM YAML path derived from *path*.

    Strips the GOTM-lake-only ``selmaprotbas`` phytoplankton ``alpha``/``beta``
    parameters (see module docstring) that conda ``pyfabm`` cannot parse. If the
    source contains none, the original path is returned unchanged (idempotent);
    otherwise a stripped copy is written under ``$TMPDIR/pygotm-fabm`` and that
    path is returned. The transformation is deterministic, so re-running a staged
    bundle reproduces the recorded ``fabm_yaml_sha256``.
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return path

    changed = False
    instances = _mapping(raw.get("instances"))
    for instance in instances.values():
        if not isinstance(instance, dict):
            continue
        if str(instance.get("model", "")) != "selmaprotbas/phytoplankton":
            continue
        parameters = _mapping(instance.get("parameters"))
        for key in _LEGACY_SELMA_PHYTOPLANKTON_PARAMETERS:
            if key in parameters:
                del parameters[key]
                changed = True

    if not changed:
        return path

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    target_dir = Path(tempfile.gettempdir()) / "pygotm-fabm"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{path.stem}-{digest}.yaml"
    target.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target


def load_fabm_config(
    document: dict[str, Any],
    gotm_yaml_path: str | Path,
) -> FABMConfig:
    """Load resolved FABM coupling settings from the pyGOTM document."""

    raw_fabm = _mapping(document.get("fabm"))
    feedbacks = _mapping(raw_fabm.get("feedbacks"))
    use = bool(raw_fabm.get("use", False))
    config_path = resolve_fabm_config_path(gotm_yaml_path, document) if use else None
    return FABMConfig(
        use=use,
        config_path=config_path,
        freshwater_impact=bool(raw_fabm.get("freshwater_impact", True)),
        repair_state=bool(raw_fabm.get("repair_state", False)),
        shade_feedback=bool(feedbacks.get("shade", False)),
        albedo_feedback=bool(feedbacks.get("albedo", False)),
        surface_drag_feedback=bool(feedbacks.get("surface_drag", False)),
    )
