"""Machine-readable schema helpers for external integrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pygotm.config import GotmSettings, load_config
from pygotm.gotm.run_metadata import (
    PYGOTM_CONFIG_SCHEMA_VERSION,
    PYGOTM_OUTPUT_SCHEMA_VERSION,
    REQUIRED_NETCDF_ATTRS,
    parse_fabm_models,
)
from pygotm.gotm.runtime_output import (
    REFERENCE_SCALAR_OUTPUT_NAMES,
    REFERENCE_Z_PROFILE_OUTPUT_NAMES,
)

__all__ = [
    "OutputVariable",
    "config_schema",
    "netcdf_attrs_schema",
    "output_schema",
]


@dataclass(frozen=True, slots=True)
class OutputVariable:
    """One machine-readable output-variable record."""

    name: str
    units: str
    long_name: str
    standard_name: str
    category: str
    dimensions: tuple[str, ...]
    state_dependent: bool


_INPUT_SETTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "method": {"type": "string"},
        "constant_value": {"type": "number"},
        "file": {"type": "string"},
        "column": {"type": "integer", "minimum": 1},
        "scale_factor": {"type": "number"},
        "offset": {"type": "number"},
    },
}

_FREE_FORM_SECTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "surface": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "fluxes": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["off", "kondo", "fairall"],
                    },
                    "heat": _INPUT_SETTING_SCHEMA,
                    "tx": _INPUT_SETTING_SCHEMA,
                    "ty": _INPUT_SETTING_SCHEMA,
                },
            },
            "u10": _INPUT_SETTING_SCHEMA,
            "v10": _INPUT_SETTING_SCHEMA,
            "airp": _INPUT_SETTING_SCHEMA,
            "airt": _INPUT_SETTING_SCHEMA,
            "hum": _INPUT_SETTING_SCHEMA,
            "cloud": _INPUT_SETTING_SCHEMA,
            "precip": _INPUT_SETTING_SCHEMA,
            "swr": _INPUT_SETTING_SCHEMA,
            "longwave_radiation": _INPUT_SETTING_SCHEMA,
            "albedo": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "method": {"type": "string"},
                    "constant_value": {"type": "number"},
                },
            },
            "ice": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "model": {"type": "string"},
                    "min_ice_thickness": {"type": "number"},
                    "winton": {"type": "object", "additionalProperties": True},
                    "mylake": {"type": "object", "additionalProperties": True},
                },
            },
        },
    },
    "bottom": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "u_vel": _INPUT_SETTING_SCHEMA,
            "v_vel": _INPUT_SETTING_SCHEMA,
            "z0": _INPUT_SETTING_SCHEMA,
            "drag": {"type": "object", "additionalProperties": True},
        },
    },
    "restart": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "load": {"type": "boolean"},
            "file": {"type": "string"},
        },
    },
    "output": {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "time_unit": {
                    "type": "string",
                    "enum": ["second", "hour", "day", "month", "year", "dt"],
                },
                "time_step": {"type": "integer", "minimum": 1},
                "time_method": {
                    "type": "string",
                    "enum": ["point", "mean", "integrated"],
                },
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {"source": {"type": "string"}},
                    },
                },
            },
        },
    },
    "equation_of_state": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "method": {
                "type": "string",
                "enum": ["full_teos-10", "linear_teos-10", "linear_custom"],
            },
            "rho0": {"type": "number"},
            "linear": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "T0": {"type": "number"},
                    "S0": {"type": "number"},
                    "p0": {"type": "number"},
                    "alpha": {"type": "number"},
                    "beta": {"type": "number"},
                    "cp": {"type": "number"},
                },
            },
        },
    },
}

_FABM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "use": {"type": "boolean"},
        "config": {"type": "string"},
        "config_file": {"type": "string"},
        "yaml": {"type": "string"},
        "file": {"type": "string"},
        "freshwater_impact": {"type": "boolean"},
        "repair_state": {"type": "boolean"},
        "feedbacks": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "shade": {"type": "boolean"},
                "albedo": {"type": "boolean"},
                "surface_drag": {"type": "boolean"},
            },
        },
    },
}

_TURBULENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "turb_method": {
            "type": "string",
            "enum": ["no_model", "first_order", "second_order", "cvmix"],
        },
        "tke_method": {
            "type": "string",
            "enum": ["local_eq", "tke", "mellor_yamada"],
        },
        "len_scale_method": {
            "type": "string",
            "enum": [
                "parabolic",
                "triangular",
                "xing_davies",
                "robert_ouellet",
                "blackadar",
                "bougeault_andre",
                "dissipation",
                "mellor_yamada",
                "gls",
                "omega",
            ],
        },
        "stab_method": {
            "type": "string",
            "enum": ["constant", "munk_anderson", "schumann_gerz"],
        },
        "epsprof": _INPUT_SETTING_SCHEMA,
        "bc": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "ubc_type": {
                    "type": "string",
                    "enum": ["logarithmic", "tke_injection"],
                },
                "lbc_type": {"type": "string"},
                "k_ubc": {"type": "number"},
                "k_lbc": {"type": "number"},
                "psi_ubc": {"type": "number"},
                "psi_lbc": {"type": "number"},
            },
        },
        "scnd": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "scnd_method": {"type": "string"},
                "scnd_coeff": {
                    "type": "string",
                    "enum": [
                        "custom",
                        "gibson_launder",
                        "mellor_yamada",
                        "kantha_clayson",
                        "luyten",
                        "canuto-a",
                        "canuto-b",
                        "cheng",
                    ],
                },
                "kb_method": {"type": "string"},
                "epsb_method": {"type": "string"},
            },
        },
        "iw": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "model": {"type": "string"},
                "alpha": {"type": "number"},
            },
        },
        "k_min": {"type": "number"},
        "eps_min": {"type": "number"},
        "kb_min": {"type": "number"},
        "epsb_min": {"type": "number"},
    },
}

_DIMENSIONS_BY_AXIS: dict[str, tuple[str, ...]] = {
    "time": ("time",),
    "scalar": ("time", "lat", "lon"),
    "z": ("time", "z", "lat", "lon"),
    "zi": ("time", "zi", "lat", "lon"),
}


def _display_name(name: str) -> str:
    return name.replace("_", " ")


def _variable(
    name: str,
    *,
    units: str = "",
    long_name: str | None = None,
    category: str = "diagnostic",
    axis: str = "scalar",
    standard_name: str = "",
    state_dependent: bool = False,
) -> OutputVariable:
    return OutputVariable(
        name=name,
        units=units,
        long_name=_display_name(name) if long_name is None else long_name,
        standard_name=standard_name,
        category=category,
        dimensions=_DIMENSIONS_BY_AXIS[axis],
        state_dependent=state_dependent,
    )


_CORE_OUTPUT_VARIABLES: tuple[OutputVariable, ...] = (
    _variable(
        "time",
        units="seconds",
        long_name="time",
        standard_name="time",
        category="coordinate",
        axis="time",
    ),
    _variable(
        "z", units="m", long_name="cell-centre depth", category="coordinate", axis="z"
    ),
    _variable(
        "zi", units="m", long_name="interface depth", category="coordinate", axis="zi"
    ),
    _variable(
        "rho_p",
        units="kg/m3",
        long_name="potential density",
        category="density",
        axis="z",
    ),
    _variable("zeta", units="m", long_name="surface elevation", category="meanflow"),
    _variable(
        "u_taus",
        units="m/s",
        long_name="surface friction velocity",
        category="meanflow",
    ),
    _variable("u10", units="m/s", long_name="10 m eastward wind", category="airsea"),
    _variable("v10", units="m/s", long_name="10 m northward wind", category="airsea"),
    _variable("airt", units="Celsius", long_name="air temperature", category="airsea"),
    _variable("airp", units="hPa", long_name="air pressure", category="airsea"),
    _variable("hum", units="%", long_name="humidity", category="airsea"),
    _variable(
        "es", units="hPa", long_name="saturation vapor pressure", category="airsea"
    ),
    _variable("ea", units="hPa", long_name="air vapor pressure", category="airsea"),
    _variable(
        "qs",
        units="kg/kg",
        long_name="surface saturation specific humidity",
        category="airsea",
    ),
    _variable(
        "qa", units="kg/kg", long_name="air specific humidity", category="airsea"
    ),
    _variable("rhoa", units="kg/m3", long_name="air density", category="airsea"),
    _variable("cloud", units="1", long_name="cloud cover fraction", category="airsea"),
    _variable("albedo", units="1", long_name="surface albedo", category="airsea"),
    _variable("precip", units="m/s", long_name="precipitation", category="airsea"),
    _variable("evap", units="m/s", long_name="evaporation", category="airsea"),
    _variable(
        "int_precip", units="m", long_name="integrated precipitation", category="airsea"
    ),
    _variable(
        "int_evap", units="m", long_name="integrated evaporation", category="airsea"
    ),
    _variable(
        "int_swr",
        units="J/m2",
        long_name="integrated shortwave radiation",
        category="airsea",
    ),
    _variable(
        "int_heat", units="J/m2", long_name="integrated heat flux", category="airsea"
    ),
    _variable(
        "int_total",
        units="J/m2",
        long_name="integrated total surface flux",
        category="airsea",
    ),
    _variable(
        "I_0", units="W/m2", long_name="surface shortwave radiation", category="airsea"
    ),
    _variable("qh", units="W/m2", long_name="sensible heat flux", category="airsea"),
    _variable("qe", units="W/m2", long_name="latent heat flux", category="airsea"),
    _variable(
        "ql", units="W/m2", long_name="longwave radiation flux", category="airsea"
    ),
    _variable("heat", units="W/m2", long_name="surface heat flux", category="airsea"),
    _variable(
        "tx",
        units="m2/s2",
        long_name="eastward surface stress divided by density",
        category="airsea",
    ),
    _variable(
        "ty",
        units="m2/s2",
        long_name="northward surface stress divided by density",
        category="airsea",
    ),
    _variable(
        "sst", units="Celsius", long_name="sea surface temperature", category="airsea"
    ),
    _variable(
        "sst_obs",
        units="Celsius",
        long_name="observed sea surface temperature",
        category="airsea",
    ),
    _variable("sss", units="g/kg", long_name="sea surface salinity", category="airsea"),
    _variable(
        "mld_surf",
        units="m",
        long_name="surface mixed-layer depth",
        category="diagnostics",
    ),
    _variable("u", units="m/s", long_name="x-velocity", category="meanflow", axis="z"),
    _variable("v", units="m/s", long_name="y-velocity", category="meanflow", axis="z"),
    _variable(
        "temp",
        units="Celsius",
        long_name="conservative temperature",
        category="meanflow",
        axis="z",
    ),
    _variable(
        "salt",
        units="g/kg",
        long_name="absolute salinity",
        category="meanflow",
        axis="z",
    ),
    _variable(
        "temp_obs",
        units="Celsius",
        long_name="observed temperature",
        category="observations",
        axis="z",
    ),
    _variable(
        "salt_obs",
        units="g/kg",
        long_name="observed salinity",
        category="observations",
        axis="z",
    ),
    _variable(
        "u_obs",
        units="m/s",
        long_name="observed x-velocity",
        category="observations",
        axis="z",
    ),
    _variable(
        "v_obs",
        units="m/s",
        long_name="observed y-velocity",
        category="observations",
        axis="z",
    ),
    _variable(
        "idpdx",
        units="m/s2",
        long_name="internal pressure gradient x",
        category="mimic_3d",
        axis="z",
    ),
    _variable(
        "idpdy",
        units="m/s2",
        long_name="internal pressure gradient y",
        category="mimic_3d",
        axis="z",
    ),
    _variable(
        "tke",
        units="m2/s2",
        long_name="turbulent kinetic energy",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "eps",
        units="m2/s3",
        long_name="dissipation rate",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "num",
        units="m2/s",
        long_name="eddy viscosity",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "nuh",
        units="m2/s",
        long_name="heat diffusivity",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "h", units="m", long_name="layer thickness", category="meanflow", axis="z"
    ),
    _variable(
        "xP",
        units="m2/s3",
        long_name="extra turbulence production",
        category="meanflow",
        axis="z",
    ),
    _variable(
        "fric",
        units="1",
        long_name="extra water-column friction coefficient",
        category="meanflow",
        axis="z",
    ),
    _variable(
        "drag",
        units="1",
        long_name="water-column drag coefficient",
        category="meanflow",
        axis="z",
    ),
    _variable(
        "avh", units="m2/s", long_name="eddy diffusivity", category="meanflow", axis="z"
    ),
    _variable(
        "bioshade",
        units="1",
        long_name="biological shading factor",
        category="meanflow",
        axis="z",
    ),
    _variable(
        "ga", units="1", long_name="coordinate scaling", category="meanflow", axis="z"
    ),
    _variable(
        "uu",
        units="m2/s2",
        long_name="velocity variance uu",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "vv",
        units="m2/s2",
        long_name="velocity variance vv",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "ww",
        units="m2/s2",
        long_name="velocity variance ww",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "NN",
        units="1/s2",
        long_name="buoyancy frequency squared",
        category="meanflow",
        axis="zi",
    ),
    _variable(
        "NNT",
        units="1/s2",
        long_name="temperature contribution to buoyancy frequency squared",
        category="meanflow",
        axis="zi",
    ),
    _variable(
        "NNS",
        units="1/s2",
        long_name="salinity contribution to buoyancy frequency squared",
        category="meanflow",
        axis="zi",
    ),
    _variable(
        "buoy", units="m/s2", long_name="buoyancy", category="meanflow", axis="z"
    ),
    _variable(
        "SS",
        units="1/s2",
        long_name="shear frequency squared",
        category="meanflow",
        axis="zi",
    ),
    _variable(
        "P",
        units="m2/s3",
        long_name="shear production",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "G",
        units="m2/s3",
        long_name="buoyancy production",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "Pb",
        units="m2/s3",
        long_name="variance buoyancy production",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "kb",
        units="m2/s2",
        long_name="buoyancy variance",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "epsb",
        units="m2/s3",
        long_name="buoyancy variance dissipation",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "L",
        units="m",
        long_name="turbulent length scale",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "PSTK",
        units="m2/s3",
        long_name="Stokes production",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "cmue1",
        units="1",
        long_name="stability function cmue1",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "cmue2",
        units="1",
        long_name="stability function cmue2",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "gamu",
        units="1",
        long_name="momentum stability function u",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "gamv",
        units="1",
        long_name="momentum stability function v",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "gamh",
        units="1",
        long_name="heat stability function",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "gams",
        units="1",
        long_name="salinity stability function",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "Rig",
        units="1",
        long_name="gradient Richardson number",
        category="diagnostics",
        axis="zi",
    ),
    _variable(
        "gamb",
        units="1",
        long_name="buoyancy stability function",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "gam",
        units="1",
        long_name="combined stability function",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "as",
        units="1",
        long_name="dimensionless shear",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "an",
        units="1",
        long_name="dimensionless stratification",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "at",
        units="1",
        long_name="dimensionless time scale",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "r", units="1", long_name="turbulence ratio", category="turbulence", axis="zi"
    ),
    _variable(
        "taux",
        units="m2/s2",
        long_name="turbulent momentum flux x",
        category="diagnostics",
        axis="zi",
    ),
    _variable(
        "tauy",
        units="m2/s2",
        long_name="turbulent momentum flux y",
        category="diagnostics",
        axis="zi",
    ),
    _variable(
        "u_taub", units="m/s", long_name="bottom friction velocity", category="meanflow"
    ),
    _variable("taub", units="Pa", long_name="bottom stress", category="meanflow"),
    _variable(
        "mld_bott",
        units="m",
        long_name="bottom mixed-layer depth",
        category="diagnostics",
    ),
    _variable(
        "rad",
        units="W/m2",
        long_name="shortwave radiation profile",
        category="meanflow",
        axis="zi",
    ),
    _variable(
        "us", units="m/s", long_name="Stokes drift x", category="stokes_drift", axis="z"
    ),
    _variable(
        "vs", units="m/s", long_name="Stokes drift y", category="stokes_drift", axis="z"
    ),
    _variable(
        "dusdz",
        units="1/s",
        long_name="vertical gradient of Stokes drift x",
        category="stokes_drift",
        axis="zi",
    ),
    _variable(
        "dvsdz",
        units="1/s",
        long_name="vertical gradient of Stokes drift y",
        category="stokes_drift",
        axis="zi",
    ),
    _variable(
        "us0", units="m/s", long_name="surface Stokes drift x", category="stokes_drift"
    ),
    _variable(
        "vs0", units="m/s", long_name="surface Stokes drift y", category="stokes_drift"
    ),
    _variable(
        "ds", units="m", long_name="Stokes drift decay scale", category="stokes_drift"
    ),
    _variable("Ekin", units="J/m2", long_name="kinetic energy", category="diagnostics"),
    _variable(
        "Epot", units="J/m2", long_name="potential energy", category="diagnostics"
    ),
    _variable(
        "Eturb", units="J/m2", long_name="turbulent energy", category="diagnostics"
    ),
    _variable(
        "nus",
        units="m2/s",
        long_name="salinity diffusivity",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "nucl",
        units="m2/s",
        long_name="Langmuir turbulent viscosity",
        category="turbulence",
        axis="zi",
    ),
    _variable(
        "rho",
        units="kg/m3",
        long_name="in-situ density",
        category="density",
        axis="z",
        state_dependent=True,
    ),
    _variable(
        "temp_p",
        units="Celsius",
        long_name="potential temperature",
        category="density",
        axis="z",
        state_dependent=True,
    ),
    _variable(
        "temp_i",
        units="Celsius",
        long_name="in-situ temperature",
        category="density",
        axis="z",
        state_dependent=True,
    ),
    _variable(
        "salt_p",
        units="psu",
        long_name="practical salinity",
        category="density",
        axis="z",
        state_dependent=True,
    ),
)

_REFERENCE_SCALAR_VARIABLES: tuple[OutputVariable, ...] = tuple(
    _variable(
        name,
        units="",
        long_name=_display_name(name),
        category="ice"
        if "ice" in name.lower()
        or name in {"Hfrazil", "Hice", "T1", "T2", "Tf", "Tice_surface"}
        else "fabm",
        axis="scalar",
        state_dependent=True,
    )
    for name in REFERENCE_SCALAR_OUTPUT_NAMES
)
_REFERENCE_Z_PROFILE_VARIABLES: tuple[OutputVariable, ...] = tuple(
    _variable(
        name,
        units="",
        long_name=_display_name(name),
        category="observations" if name == "eps_obs" else "fabm",
        axis="z",
        state_dependent=True,
    )
    for name in REFERENCE_Z_PROFILE_OUTPUT_NAMES
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
            item for item in _REFERENCE_SCALAR_VARIABLES if item.name in ice_names
        )

    fabm = _mapping(document.get("fabm"))
    if bool(fabm.get("use", False)):
        fabm_config = (
            fabm.get("config")
            or fabm.get("config_file")
            or fabm.get("yaml")
            or fabm.get("file")
            or "fabm.yaml"
        )
        fabm_path = Path(str(fabm_config))
        if not fabm_path.is_absolute():
            fabm_path = config_path.parent / fabm_path
        model_text = " ".join(parse_fabm_models(fabm_path)).lower()
        scalar_names = {"surface_albedo", "surface_drag_coefficient_in_air"}
        z_profile_names = {"attenuation_coefficient_of_photosynthetic_radiative_flux"}
        if "jrc_med_ergom" in model_text or "ergom" in model_text:
            scalar_names.update(
                name
                for name in REFERENCE_SCALAR_OUTPUT_NAMES
                if name.startswith("jrc_med_ergom")
            )
            z_profile_names.update(
                name
                for name in REFERENCE_Z_PROFILE_OUTPUT_NAMES
                if name.startswith("jrc_med_ergom")
            )
        if "bsem" in model_text:
            z_profile_names.update(
                name
                for name in REFERENCE_Z_PROFILE_OUTPUT_NAMES
                if name.startswith("bsem_")
            )
            z_profile_names.add("total_nitrogen")
        if "npzd" in model_text:
            z_profile_names.update(
                name
                for name in REFERENCE_Z_PROFILE_OUTPUT_NAMES
                if name.startswith("npzd_")
            )
            z_profile_names.add("total_nitrogen")
        if "bb/passive" in model_text:
            z_profile_names.add("sed_c")
        variables.extend(
            item for item in _REFERENCE_SCALAR_VARIABLES if item.name in scalar_names
        )
        variables.extend(
            item
            for item in _REFERENCE_Z_PROFILE_VARIABLES
            if item.name in z_profile_names
        )

    turbulence = _mapping(document.get("turbulence"))
    epsprof = _mapping(turbulence.get("epsprof"))
    if epsprof and str(epsprof.get("method", "off")).lower() not in {"off", "0"}:
        variables.extend(
            item for item in _REFERENCE_Z_PROFILE_VARIABLES if item.name == "eps_obs"
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
