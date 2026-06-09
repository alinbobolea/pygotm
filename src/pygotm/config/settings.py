"""
module settings

The original Fortran ``settings.F90`` extends GOTM's YAML settings system with
helpers that construct scalar/profile input descriptors. In the Python
translation, that role is fulfilled by Pydantic models plus small
normalisation helpers for GOTM-style YAML.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "WRITE_DETAIL_DEFAULT",
    "WRITE_DETAIL_FULL",
    "WRITE_DETAIL_MINIMAL",
    "ExtPressureSettings",
    "GotmSettings",
    "GradientCollectionSettings",
    "GridSettings",
    "InputSetting",
    "IntPressureSettings",
    "LightExtinctionSettings",
    "LocationSettings",
    "Mimic3DSettings",
    "ObservationTurbulenceSettings",
    "PlumeSettings",
    "ProfileRelaxationSettings",
    "SalinitySettings",
    "ScalarTidalSettings",
    "TimeSettings",
    "TidalConstituentSettings",
    "TemperatureSettings",
    "VelocitySettings",
    "VerticalVelocitySettings",
    "WaveSettings",
    "load_settings",
    "save_settings",
]

WRITE_DETAIL_MINIMAL = 0
WRITE_DETAIL_DEFAULT = 1
WRITE_DETAIL_FULL = 2

_INPUT_METHOD_CODES = {
    0: "constant",
    1: "constant",
    2: "file",
    3: "calculate",
}
_OPTIONAL_INPUT_METHOD_CODES = {
    0: "off",
    1: "constant",
    2: "file",
}
_TIDAL_INPUT_METHOD_CODES = {
    0: "constant",
    1: "tidal",
    2: "file",
}
_GRID_METHOD_CODES = {
    0: "analytical",
    1: "file_sigma",
    2: "file_h",
    3: "adaptive",
}
_PROFILE_METHOD_CODES = {
    0: "off",
    1: "constant",
    2: "file",
}
_ANALYTICAL_PROFILE_CODES = {
    1: "constant",
    2: "two_layer",
    3: "buoyancy",
}
_LIGHT_EXTINCTION_CODES = {
    1: "jerlov_i",
    2: "jerlov_1_50m",
    3: "jerlov_ia",
    4: "jerlov_ib",
    5: "jerlov_ii",
    6: "jerlov_iii",
    7: "custom",
}
_EXT_PRESSURE_CODES = {
    0: "elevation",
    1: "velocity",
    2: "average_velocity",
}
_INT_PRESSURE_CODES = {
    0: "none",
    1: "gradients",
    2: "plume",
}
_PLUME_CODES = {
    1: "surface",
    2: "bottom",
}
_W_ADV_DISCR_CODES = {
    1: "upstream",
    3: "p2",
    4: "superbee",
    5: "muscl",
    6: "p2_pdm",
    13: "splmax13",
}
_FLUX_METHOD_CODES = {
    0: "off",
    1: "kondo",
    2: "fairall",
}
_HUMIDITY_TYPE_CODES = {
    1: "relative",
    2: "wet_bulb",
    3: "dew_point",
    4: "specific",
}
_LONGWAVE_METHOD_CODES = {
    0: "file",
    1: "clark",
    2: "hastenrath_lamb",
    3: "bignami",
    4: "berliand_berliand",
    5: "josey1",
    6: "josey2",
}
_ALBEDO_METHOD_CODES = {
    0: "constant",
    1: "payne",
    2: "cogley",
}
_SSUV_METHOD_CODES = {
    0: "absolute",
    1: "relative",
}
_LAKE_ICE_MODEL_CODES = {
    0: "no_ice",
    1: "lebedev",
    2: "mylake",
    3: "winton",
}
_TURB_METHOD_CODES = {
    0: "no_model",
    2: "first_order",
    3: "second_order",
    100: "cvmix",
}
_TKE_METHOD_CODES = {
    1: "local_eq",
    2: "tke",
    3: "mellor_yamada",
}
_LEN_SCALE_METHOD_CODES = {
    1: "parabolic",
    2: "triangular",
    3: "xing_davies",
    4: "robert_ouellet",
    5: "blackadar",
    6: "bougeault_andre",
    8: "dissipation",
    9: "mellor_yamada",
    10: "gls",
}
_STAB_METHOD_CODES = {
    1: "constant",
    2: "munk_anderson",
    3: "schumann_gerz",
}
_SCND_METHOD_CODES = {
    1: "quasi_eq",
    2: "weak_eq_kb_eq",
}
_SCND_COEFF_CODES = {
    0: "custom",
    1: "gibson_launder",
    2: "mellor_yamada",
    3: "kantha_clayson",
    4: "luyten",
    5: "canuto_a",
    6: "canuto_b",
    7: "cheng",
}
_MY_LENGTH_CODES = {
    1: "parabolic",
    2: "triangular",
    3: "linear",
}
_EOS_METHOD_CODES = {
    1: "full_teos10",
    2: "full_teos10",
    3: "linear_teos10",
    4: "linear_custom",
}


def _format_timestamp(value: date | datetime) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return f"{value.isoformat()} 00:00:00"


def _normalize_settings_document(node: object, *, key: str | None = None) -> object:
    if isinstance(node, dict):
        normalized: dict[str, object] = {}
        for child_key, child_value in node.items():
            normalized[str(child_key)] = _normalize_settings_document(
                child_value,
                key=str(child_key),
            )
        return normalized

    if isinstance(node, list):
        return [_normalize_settings_document(item) for item in node]

    if isinstance(node, tuple):
        return tuple(_normalize_settings_document(item) for item in node)

    if isinstance(node, (date, datetime)):
        return _format_timestamp(node)

    if key == "method" and isinstance(node, bool):
        return "constant" if node else "off"

    if key == "file" and node is None:
        return ""

    return node


def _canonical_token(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return re.sub(r"[\s-]+", "_", text)


def _code_token(
    value: object,
    default: str,
    mapping: dict[int, str],
) -> str:
    if isinstance(value, bool):
        return _canonical_token(value, default)
    if isinstance(value, int):
        return mapping.get(value, str(value))
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            code = int(stripped)
            return mapping.get(code, stripped)
    return _canonical_token(value, default)


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}


def _copy_if_present(
    source: dict[str, object],
    target: dict[str, object],
    source_key: str,
    target_key: str,
) -> None:
    if source_key in source and target_key not in target:
        target[target_key] = source[source_key]


def _normalise_input_method(
    node: object,
    mapping: dict[int, str] = _INPUT_METHOD_CODES,
) -> None:
    if isinstance(node, dict) and "method" in node:
        node["method"] = _code_token(node["method"], "constant", mapping)


def _normalise_profile_section(
    node: dict[str, object],
    *,
    temp: bool,
) -> None:
    method = node.get("method")
    node["method"] = _code_token(method, "off", _PROFILE_METHOD_CODES)
    analytical = _as_mapping(node.get("analytical"))
    if not analytical:
        return

    analytical_method = _code_token(
        analytical.get("method"),
        "constant",
        _ANALYTICAL_PROFILE_CODES,
    )
    if method == 1 or node["method"] == "constant":
        node["method"] = analytical_method
    if temp:
        two_layer = _as_mapping(node.setdefault("two_layer", {}))
        _copy_if_present(analytical, two_layer, "z_t1", "z_s")
        _copy_if_present(analytical, two_layer, "t_1", "t_s")
        _copy_if_present(analytical, two_layer, "z_t2", "z_b")
        _copy_if_present(analytical, two_layer, "t_2", "t_b")
        if "t_1" in analytical and "constant_value" not in node:
            node["constant_value"] = analytical["t_1"]
    else:
        two_layer = _as_mapping(node.setdefault("two_layer", {}))
        _copy_if_present(analytical, two_layer, "z_s1", "z_s")
        _copy_if_present(analytical, two_layer, "s_1", "s_s")
        _copy_if_present(analytical, two_layer, "z_s2", "z_b")
        _copy_if_present(analytical, two_layer, "s_2", "s_b")
        if "s_1" in analytical and "constant_value" not in node:
            node["constant_value"] = analytical["s_1"]
    _copy_if_present(analytical, node, "obs_NN", "NN")


def _normalise_surface_section(document: dict[str, object]) -> None:
    surface = _as_mapping(document.get("surface"))
    if not surface:
        return

    meteo = _as_mapping(surface.get("meteo"))
    for key in (
        "u10",
        "v10",
        "airp",
        "airt",
        "hum",
        "cloud",
        "swr",
        "precip",
        "calc_evaporation",
        "ssuv_method",
    ):
        _copy_if_present(meteo, surface, key, key)

    fluxes = _as_mapping(surface.get("fluxes"))
    if "method" in fluxes:
        fluxes["method"] = _code_token(fluxes["method"], "off", _FLUX_METHOD_CODES)
    for key in ("heat", "tx", "ty"):
        _normalise_input_method(_as_mapping(fluxes.get(key)))

    for key in ("u10", "v10", "airp", "airt", "cloud", "swr", "precip"):
        _normalise_input_method(_as_mapping(surface.get(key)))
    hum = _as_mapping(surface.get("hum"))
    _normalise_input_method(hum)
    if "type" in hum:
        hum["type"] = _code_token(hum["type"], "relative", _HUMIDITY_TYPE_CODES)

    if "ssuv_method" in surface:
        surface["ssuv_method"] = _code_token(
            surface["ssuv_method"],
            "relative",
            _SSUV_METHOD_CODES,
        )

    longwave = _as_mapping(surface.get("longwave_radiation"))
    if "method" in longwave:
        longwave["method"] = _code_token(
            longwave["method"],
            "clark",
            _LONGWAVE_METHOD_CODES,
        )

    albedo = _as_mapping(surface.get("albedo"))
    if "method" in albedo:
        albedo["method"] = _code_token(
            albedo["method"],
            "payne",
            _ALBEDO_METHOD_CODES,
        )

    ice = _as_mapping(surface.get("ice"))
    if "model" in ice and isinstance(ice["model"], int):
        ice["model"] = _LAKE_ICE_MODEL_CODES.get(int(ice["model"]), str(ice["model"]))


def _normalise_mimic_3d_section(document: dict[str, object]) -> None:
    mimic = _as_mapping(document.get("mimic_3d"))
    if not mimic:
        return

    ext = _as_mapping(mimic.get("ext_pressure"))
    if "mode" in ext and "type" not in ext:
        ext["type"] = ext["mode"]
    if "type" in ext:
        ext["type"] = _code_token(ext["type"], "elevation", _EXT_PRESSURE_CODES)
    for key in ("dpdx", "dpdy"):
        _normalise_input_method(_as_mapping(ext.get(key)), _TIDAL_INPUT_METHOD_CODES)
    _normalise_input_method(_as_mapping(ext.get("h")))

    int_press = _as_mapping(mimic.get("int_pressure"))
    legacy_int_press = _as_mapping(mimic.get("int_press"))
    if legacy_int_press and not int_press:
        int_press = dict(legacy_int_press)
        mimic["int_pressure"] = int_press
    if "type" in int_press:
        int_press["type"] = _code_token(
            int_press["type"],
            "none",
            _INT_PRESSURE_CODES,
        )
    gradients = _as_mapping(int_press.setdefault("gradients", {}))
    for key in ("dsdx", "dsdy", "dtdx", "dtdy"):
        _copy_if_present(int_press, gradients, key, key)
        _normalise_input_method(
            _as_mapping(gradients.get(key)), _OPTIONAL_INPUT_METHOD_CODES
        )
    plume = _as_mapping(int_press.get("plume"))
    if "type" in plume:
        plume["type"] = _code_token(plume["type"], "bottom", _PLUME_CODES)

    zeta = _as_mapping(mimic.get("zeta"))
    _normalise_input_method(zeta, _TIDAL_INPUT_METHOD_CODES)

    w = _as_mapping(mimic.get("w"))
    _normalise_input_method(_as_mapping(w.get("max")), _OPTIONAL_INPUT_METHOD_CODES)
    _normalise_input_method(_as_mapping(w.get("height")), _INPUT_METHOD_CODES)
    if "adv_discr" in w:
        w["adv_discr"] = _code_token(w["adv_discr"], "p2_pdm", _W_ADV_DISCR_CODES)


def _normalise_turbulence_section(document: dict[str, object]) -> None:
    turbulence = _as_mapping(document.get("turbulence"))
    if not turbulence:
        return

    for key, mapping, default in (
        ("turb_method", _TURB_METHOD_CODES, "second_order"),
        ("tke_method", _TKE_METHOD_CODES, "tke"),
        ("len_scale_method", _LEN_SCALE_METHOD_CODES, "dissipation"),
        ("stab_method", _STAB_METHOD_CODES, "schumann_gerz"),
    ):
        if key in turbulence:
            turbulence[key] = _code_token(turbulence[key], default, mapping)

    scnd = _as_mapping(turbulence.get("scnd"))
    if "method" in scnd:
        scnd["method"] = _code_token(
            scnd["method"], "weak_eq_kb_eq", _SCND_METHOD_CODES
        )
    if "scnd_coeff" in scnd:
        scnd["scnd_coeff"] = _code_token(
            scnd["scnd_coeff"], "canuto_a", _SCND_COEFF_CODES
        )

    my = _as_mapping(turbulence.get("my"))
    if "length" in my:
        my["length"] = _code_token(my["length"], "parabolic", _MY_LENGTH_CODES)

    epsprof = _as_mapping(turbulence.get("epsprof"))
    _normalise_input_method(epsprof, _OPTIONAL_INPUT_METHOD_CODES)


def _normalise_eos_section(document: dict[str, object]) -> None:
    legacy = _as_mapping(document.get("eq_state"))
    if legacy and "equation_of_state" not in document:
        equation: dict[str, object] = {}
        if "method" in legacy:
            equation["method"] = _code_token(
                legacy["method"],
                "full_teos10",
                _EOS_METHOD_CODES,
            )
        linear: dict[str, object] = {}
        for source, target in (
            ("T0", "T0"),
            ("S0", "S0"),
            ("p0", "p0"),
            ("dtr0", "alpha"),
            ("dsr0", "beta"),
        ):
            _copy_if_present(legacy, linear, source, target)
        if linear:
            equation["linear"] = linear
        document["equation_of_state"] = equation

    equation = _as_mapping(document.get("equation_of_state"))
    if "method" in equation:
        equation["method"] = _code_token(
            equation["method"],
            "full_teos10",
            _EOS_METHOD_CODES,
        )


def _normalize_gotm_document(node: object) -> object:
    normalized = _normalize_settings_document(node)
    if not isinstance(normalized, dict):
        return normalized

    grid = _as_mapping(normalized.get("grid"))
    if "method" in grid:
        grid["method"] = _code_token(grid["method"], "analytical", _GRID_METHOD_CODES)

    temperature = _as_mapping(normalized.get("temperature"))
    if temperature:
        _normalise_profile_section(temperature, temp=True)

    salinity = _as_mapping(normalized.get("salinity"))
    if salinity:
        _normalise_profile_section(salinity, temp=False)

    light = _as_mapping(normalized.get("light_extinction"))
    if "method" in light:
        light["method"] = _code_token(
            light["method"],
            "jerlov_i",
            _LIGHT_EXTINCTION_CODES,
        )
    for key in ("A", "g1", "g2"):
        _normalise_input_method(_as_mapping(light.get(key)))

    velocities = _as_mapping(normalized.get("velocities"))
    for key in ("u", "v"):
        _normalise_input_method(
            _as_mapping(velocities.get(key)), _OPTIONAL_INPUT_METHOD_CODES
        )

    _normalise_surface_section(normalized)
    _normalise_mimic_3d_section(normalized)
    _normalise_turbulence_section(normalized)
    _normalise_eos_section(normalized)
    return normalized


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class InputSetting(_SettingsModel):
    """Configuration for a scalar or profile input in GOTM YAML."""

    method: str | int = "constant"
    constant_value: float = 0.0
    file: str = ""
    column: int = 1
    scale_factor: float = 1.0
    offset: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _coerce_scalar_value(cls, value: object) -> object:
        if value is None:
            return {}
        if isinstance(value, (int, float)):
            return {"method": "constant", "constant_value": float(value)}
        if isinstance(value, dict):
            return _normalize_settings_document(value)
        return value

    @model_validator(mode="after")
    def _normalise(self) -> InputSetting:
        self.method = _code_token(self.method, "constant", _INPUT_METHOD_CODES)
        return self

    @property
    def path(self) -> str:
        """Path to the external data file, if any."""

        return self.file

    @property
    def add_offset(self) -> float:
        """Offset applied to values read from file."""

        return self.offset


class ProfileRelaxationSettings(_SettingsModel):
    tau: float = 1.0e15
    h_s: float = 0.0
    tau_s: float = 1.0e15
    h_b: float = 0.0
    tau_b: float = 1.0e15


class TemperatureTwoLayerSettings(_SettingsModel):
    z_s: float = 0.0
    t_s: float = 0.0
    z_b: float = 0.0
    t_b: float = 0.0


class SalinityTwoLayerSettings(_SettingsModel):
    z_s: float = 0.0
    s_s: float = 0.0
    z_b: float = 0.0
    s_b: float = 0.0


class TemperatureSettings(InputSetting):
    method: str | int = "off"
    type: str = "in_situ"
    two_layer: TemperatureTwoLayerSettings = Field(
        default_factory=TemperatureTwoLayerSettings
    )
    NN: float = 0.0
    relax: ProfileRelaxationSettings = Field(default_factory=ProfileRelaxationSettings)

    @model_validator(mode="after")
    def _normalise_temperature(self) -> TemperatureSettings:
        self.method = _code_token(self.method, "off", _PROFILE_METHOD_CODES)
        self.type = _canonical_token(self.type, "in_situ")
        return self


class SalinitySettings(InputSetting):
    method: str | int = "off"
    type: str = "practical"
    two_layer: SalinityTwoLayerSettings = Field(
        default_factory=SalinityTwoLayerSettings
    )
    NN: float = 0.0
    relax: ProfileRelaxationSettings = Field(default_factory=ProfileRelaxationSettings)

    @model_validator(mode="after")
    def _normalise_salinity(self) -> SalinitySettings:
        self.method = _code_token(self.method, "off", _PROFILE_METHOD_CODES)
        self.type = _canonical_token(self.type, "practical")
        return self


class TidalConstituentSettings(_SettingsModel):
    period_1: float = 44714.0
    amp_1: float = 0.0
    phase_1: float = 0.0
    period_2: float = 43200.0
    amp_2: float = 0.0
    phase_2: float = 0.0


class ScalarTidalSettings(InputSetting):
    method: str | int = "constant"
    tidal: TidalConstituentSettings = Field(default_factory=TidalConstituentSettings)
    period_1: float = 44714.0
    period_2: float = 43200.0

    @model_validator(mode="before")
    @classmethod
    def _lift_nested_tidal_periods(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = _normalize_settings_document(value)
        if not isinstance(normalized, dict):
            return normalized

        tidal = normalized.get("tidal")
        if isinstance(tidal, dict):
            if "period_1" in tidal and "period_1" not in normalized:
                normalized["period_1"] = tidal["period_1"]
            if "period_2" in tidal and "period_2" not in normalized:
                normalized["period_2"] = tidal["period_2"]
        return normalized

    @model_validator(mode="after")
    def _normalise_tidal(self) -> ScalarTidalSettings:
        self.method = _code_token(self.method, "constant", _TIDAL_INPUT_METHOD_CODES)
        return self


class LightExtinctionSettings(_SettingsModel):
    method: str | int = "jerlov_i"
    A: InputSetting = Field(default_factory=lambda: InputSetting(constant_value=0.7))
    g1: InputSetting = Field(default_factory=lambda: InputSetting(constant_value=0.4))
    g2: InputSetting = Field(default_factory=lambda: InputSetting(constant_value=8.0))

    @model_validator(mode="after")
    def _normalise_method(self) -> LightExtinctionSettings:
        self.method = _code_token(
            self.method,
            "jerlov_i",
            _LIGHT_EXTINCTION_CODES,
        )
        return self


class GradientCollectionSettings(_SettingsModel):
    dtdx: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))
    dtdy: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))
    dsdx: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))
    dsdy: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))


class PlumeSettings(_SettingsModel):
    type: str = "bottom"
    x_slope: float = 0.0
    y_slope: float = 0.0

    @model_validator(mode="after")
    def _normalise_type(self) -> PlumeSettings:
        self.type = _canonical_token(self.type, "bottom")
        return self


class IntPressureSettings(_SettingsModel):
    type: str = "none"
    gradients: GradientCollectionSettings = Field(
        default_factory=GradientCollectionSettings
    )
    plume: PlumeSettings = Field(default_factory=PlumeSettings)
    t_adv: bool = False
    s_adv: bool = False

    @model_validator(mode="after")
    def _normalise_type(self) -> IntPressureSettings:
        self.type = _canonical_token(self.type, "none")
        return self


class ExtPressureSettings(_SettingsModel):
    type: str = "elevation"
    dpdx: ScalarTidalSettings = Field(default_factory=ScalarTidalSettings)
    dpdy: ScalarTidalSettings = Field(default_factory=ScalarTidalSettings)
    h: InputSetting = Field(default_factory=InputSetting)
    period_1: float = 44714.0
    period_2: float = 43200.0

    @model_validator(mode="after")
    def _normalise_type(self) -> ExtPressureSettings:
        self.type = _canonical_token(self.type, "elevation")
        return self


class VelocityRelaxationSettings(_SettingsModel):
    tau: float = 1.0e15
    ramp: float = 1.0e15


class VelocitySettings(_SettingsModel):
    u: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))
    v: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))
    relax: VelocityRelaxationSettings = Field(
        default_factory=VelocityRelaxationSettings
    )


class VerticalVelocitySettings(_SettingsModel):
    max: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))
    height: InputSetting = Field(default_factory=InputSetting)
    adv_discr: str = "p2_pdm"

    @model_validator(mode="after")
    def _normalise_adv_discr(self) -> VerticalVelocitySettings:
        self.adv_discr = _canonical_token(self.adv_discr, "p2_pdm")
        return self


class Mimic3DSettings(_SettingsModel):
    ext_pressure: ExtPressureSettings = Field(default_factory=ExtPressureSettings)
    int_pressure: IntPressureSettings = Field(default_factory=IntPressureSettings)
    zeta: ScalarTidalSettings = Field(default_factory=ScalarTidalSettings)
    w: VerticalVelocitySettings = Field(default_factory=VerticalVelocitySettings)


class WaveSettings(_SettingsModel):
    Hs: InputSetting = Field(default_factory=InputSetting)
    Tz: InputSetting = Field(default_factory=InputSetting)
    phiw: InputSetting = Field(default_factory=InputSetting)


class ObservationTurbulenceSettings(_SettingsModel):
    epsprof: InputSetting = Field(default_factory=lambda: InputSetting(method="off"))


class LocationSettings(_SettingsModel):
    name: str = "GOTM site"
    latitude: float = 0.0
    longitude: float = 0.0
    depth: float = 100.0
    hypsograph: str = ""


class TimeSettings(_SettingsModel):
    method: int = 2
    start: str = "2017-01-01 00:00:00"
    stop: str = "2018-01-01 00:00:00"
    dt: float = 3600.0
    cnpar: float = 1.0
    max_steps: int = Field(default=0, alias="MaxN")

    @model_validator(mode="before")
    @classmethod
    def _coerce_time_strings(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        for key in ("start", "stop"):
            time_value = normalized.get(key)
            if isinstance(time_value, (date, datetime)):
                normalized[key] = _format_timestamp(time_value)
        return normalized


class GridSettings(_SettingsModel):
    nlev: int = 100
    method: str | int = "analytical"
    ddu: float = 0.0
    ddl: float = 0.0
    file: str = ""

    @model_validator(mode="after")
    def _normalise_method(self) -> GridSettings:
        self.method = _code_token(self.method, "analytical", _GRID_METHOD_CODES)
        return self


class GotmSettings(_SettingsModel):
    """Top-level GOTM YAML settings used by Phase 5 and 6 modules."""

    version: int = 7
    title: str = "GOTM simulation"
    location: LocationSettings = Field(default_factory=LocationSettings)
    time: TimeSettings = Field(default_factory=TimeSettings)
    grid: GridSettings = Field(default_factory=GridSettings)
    temperature: TemperatureSettings = Field(default_factory=TemperatureSettings)
    salinity: SalinitySettings = Field(default_factory=SalinitySettings)
    light_extinction: LightExtinctionSettings = Field(
        default_factory=LightExtinctionSettings
    )
    mimic_3d: Mimic3DSettings = Field(default_factory=Mimic3DSettings)
    velocities: VelocitySettings = Field(default_factory=VelocitySettings)
    w: VerticalVelocitySettings = Field(default_factory=VerticalVelocitySettings)
    waves: WaveSettings = Field(default_factory=WaveSettings)
    turbulence: ObservationTurbulenceSettings = Field(
        default_factory=ObservationTurbulenceSettings
    )
    surface: dict[str, Any] = Field(default_factory=dict)
    bottom: dict[str, Any] = Field(default_factory=dict)
    restart: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    equation_of_state: dict[str, Any] = Field(default_factory=dict)


def load_settings(path: str | Path) -> GotmSettings:
    """Load GOTM YAML settings from *path*."""

    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        msg = f"top-level YAML document in {config_path} must be a mapping"
        raise TypeError(msg)
    return GotmSettings.model_validate(_normalize_gotm_document(raw))


def save_settings(
    settings: GotmSettings,
    path: str | Path,
    *,
    detail: int = WRITE_DETAIL_DEFAULT,
) -> None:
    """Write a normalised YAML representation of *settings* to *path*."""

    del detail  # Phase 6 keeps a single normalised serialisation path.
    config_path = Path(path)
    serialisable = settings.model_dump(by_alias=True, exclude_none=True)
    config_path.write_text(
        yaml.safe_dump(serialisable, sort_keys=False),
        encoding="utf-8",
    )
