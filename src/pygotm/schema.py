"""Machine-readable schema helpers for external integrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Any

from pygotm.config import GotmSettings, load_config
from pygotm.fabm.config import resolve_fabm_config_path
from pygotm.fabm.engine import FABMEngine
from pygotm.gotm.run_metadata import (
    PYGOTM_CONFIG_SCHEMA_VERSION,
    PYGOTM_OUTPUT_SCHEMA_VERSION,
    REQUIRED_NETCDF_ATTRS,
)
from pygotm.gotm.runtime_output import (
    EXTRA_SCALAR_OUTPUT_NAMES,
    EXTRA_Z_PROFILE_OUTPUT_NAMES,
    FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES,
)

__all__ = [
    "OutputVariable",
    "config_schema",
    "netcdf_attrs_schema",
    "output_schema",
]

from pygotm.schema_output_variables import (
    _CORE_OUTPUT_VARIABLES,
    _EXTRA_SCALAR_VARIABLES,
    _EXTRA_Z_PROFILE_VARIABLES,
    _FABM_SCHEMA,
    _FREE_FORM_SECTION_SCHEMAS,
    _TURBULENCE_SCHEMA,
    OutputVariable,
    _variable,
)


def config_schema() -> dict[str, Any]:
    """Return the curated GOTM config JSON Schema for editor integrations."""

    schema = GotmSettings.model_json_schema()
    schema["$id"] = "https://pygotm.org/schema/config/gotm-6.x-pygotm-1"
    schema["x-pygotm-schema-version"] = PYGOTM_CONFIG_SCHEMA_VERSION
    properties = schema.setdefault("properties", {})
    assert isinstance(properties, dict)
    for name, overlay in _FREE_FORM_SECTION_SCHEMAS.items():
        properties[name] = overlay
    properties["fabm"] = _FABM_SCHEMA
    properties["turbulence"] = _TURBULENCE_SCHEMA
    return schema


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _fabm_output_variables(fabm_path: Path) -> tuple[OutputVariable, ...]:
    engine = FABMEngine(fabm_path)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        engine.initialize(nlev=1, skip_start=True)
    variables: list[OutputVariable] = []
    for spec in engine.output_variable_specs():
        variables.append(
            _variable(
                spec.name,
                units=spec.units,
                long_name=spec.long_name,
                category="fabm",
                axis=spec.kind,
                state_dependent=True,
            )
        )
    return tuple(variables)


def _append_or_replace_fabm_output_variables(
    variables: list[OutputVariable],
    fabm_path: Path,
) -> None:
    core_names = {item.name for item in _CORE_OUTPUT_VARIABLES}
    extra_names = set(EXTRA_SCALAR_OUTPUT_NAMES) | set(EXTRA_Z_PROFILE_OUTPUT_NAMES)
    existing_names = {item.name for item in variables}
    promotable_names = set(FABM_PROMOTABLE_EXTRA_OUTPUT_NAMES)

    for variable in _fabm_output_variables(fabm_path):
        if variable.name in core_names:
            msg = f"FABM output {variable.name!r} collides with a core output variable"
            raise ValueError(msg)
        if variable.name in extra_names and variable.name not in promotable_names:
            msg = (
                f"FABM output {variable.name!r} collides with a core optional "
                "output variable"
            )
            raise ValueError(msg)
        if variable.name not in existing_names:
            variables.append(variable)
            existing_names.add(variable.name)
            continue
        if variable.name in promotable_names and variable.name in extra_names:
            for index, existing in enumerate(variables):
                if existing.name == variable.name:
                    variables[index] = variable
                    break
            existing_names.add(variable.name)
            continue
        msg = f"FABM output {variable.name!r} collides with an existing output variable"
        raise ValueError(msg)


def _state_dependent_variables(config_path: Path | None) -> tuple[OutputVariable, ...]:
    if config_path is None:
        return ()

    config = load_config(config_path)
    document = config.resolved_document()
    variables: list[OutputVariable] = []
    surface = _mapping(document.get("surface"))
    ice = _mapping(surface.get("ice"))
    ice_model = str(ice.get("model", "no_ice")).replace("-", "_").lower()
    if ice_model not in {"", "no_ice", "off", "none"}:
        ice_names = {"Hfrazil", "Hice", "Tf", "Tice_surface", "bottom_ice_energy"}
        ice_names.update(
            {
                "ocean_ice_flux",
                "ocean_ice_heat_flux",
                "ocean_ice_salt_flux",
                "surface_ice_energy",
            }
        )
        if ice_model == "winton":
            ice_names.update({"T1", "T2"})
        variables.extend(
            item for item in _EXTRA_SCALAR_VARIABLES if item.name in ice_names
        )

    fabm = _mapping(document.get("fabm"))
    if bool(fabm.get("use", False)):
        fabm_path = resolve_fabm_config_path(config_path, document)
        scalar_names = {"surface_albedo", "surface_drag_coefficient_in_air"}
        variables.extend(
            item for item in _EXTRA_SCALAR_VARIABLES if item.name in scalar_names
        )
        z_profile_names = {"attenuation_coefficient_of_photosynthetic_radiative_flux"}
        variables.extend(
            item for item in _EXTRA_Z_PROFILE_VARIABLES if item.name in z_profile_names
        )
        _append_or_replace_fabm_output_variables(variables, fabm_path)

    turbulence = _mapping(document.get("turbulence"))
    epsprof = _mapping(turbulence.get("epsprof"))
    if epsprof and str(epsprof.get("method", "off")).lower() not in {"off", "0"}:
        variables.extend(
            item for item in _EXTRA_Z_PROFILE_VARIABLES if item.name == "eps_obs"
        )
    return tuple(variables)


def output_schema(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return output-variable metadata, optionally augmented from *config_path*."""

    path = None if config_path is None else Path(config_path)
    variables = [asdict(item) for item in _CORE_OUTPUT_VARIABLES]
    variables.extend(asdict(item) for item in _state_dependent_variables(path))
    return {
        "schema_version": PYGOTM_OUTPUT_SCHEMA_VERSION,
        "variables": variables,
    }


def netcdf_attrs_schema() -> dict[str, Any]:
    """Return the stable NetCDF global-attribute contract."""

    return {
        "schema_version": PYGOTM_OUTPUT_SCHEMA_VERSION,
        "attributes": [
            {"name": name, "required": True} for name in REQUIRED_NETCDF_ATTRS
        ],
    }


def dumps_json(data: dict[str, Any]) -> str:
    """Return stable CLI JSON output for schema helpers."""

    return json.dumps(data, indent=2, sort_keys=True)
