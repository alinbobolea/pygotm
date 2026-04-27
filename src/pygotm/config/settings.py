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

import yaml  # type: ignore[import-untyped]
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


class _SettingsModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class InputSetting(_SettingsModel):
    """Configuration for a scalar or profile input in GOTM YAML."""

    method: str = "constant"
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
        self.method = _canonical_token(self.method, "constant")
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
    method: str = "off"
    type: str = "in_situ"
    two_layer: TemperatureTwoLayerSettings = Field(
        default_factory=TemperatureTwoLayerSettings
    )
    NN: float = 0.0
    relax: ProfileRelaxationSettings = Field(
        default_factory=ProfileRelaxationSettings
    )

    @model_validator(mode="after")
    def _normalise_temperature(self) -> TemperatureSettings:
        self.method = _canonical_token(self.method, "off")
        self.type = _canonical_token(self.type, "in_situ")
        return self


class SalinitySettings(InputSetting):
    method: str = "off"
    type: str = "practical"
    two_layer: SalinityTwoLayerSettings = Field(
        default_factory=SalinityTwoLayerSettings
    )
    NN: float = 0.0
    relax: ProfileRelaxationSettings = Field(
        default_factory=ProfileRelaxationSettings
    )

    @model_validator(mode="after")
    def _normalise_salinity(self) -> SalinitySettings:
        self.method = _canonical_token(self.method, "off")
        self.type = _canonical_token(self.type, "practical")
        return self


class TidalConstituentSettings(_SettingsModel):
    amp_1: float = 0.0
    phase_1: float = 0.0
    amp_2: float = 0.0
    phase_2: float = 0.0


class ScalarTidalSettings(InputSetting):
    method: str = "constant"
    tidal: TidalConstituentSettings = Field(default_factory=TidalConstituentSettings)
    period_1: float = 44714.0
    period_2: float = 43200.0

    @model_validator(mode="after")
    def _normalise_tidal(self) -> ScalarTidalSettings:
        self.method = _canonical_token(self.method, "constant")
        return self


class LightExtinctionSettings(_SettingsModel):
    method: str = "jerlov_i"
    A: InputSetting = Field(default_factory=lambda: InputSetting(constant_value=0.7))
    g1: InputSetting = Field(default_factory=lambda: InputSetting(constant_value=0.4))
    g2: InputSetting = Field(default_factory=lambda: InputSetting(constant_value=8.0))

    @model_validator(mode="after")
    def _normalise_method(self) -> LightExtinctionSettings:
        self.method = _canonical_token(self.method, "jerlov_i")
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


class Mimic3DSettings(_SettingsModel):
    ext_pressure: ExtPressureSettings = Field(default_factory=ExtPressureSettings)
    int_pressure: IntPressureSettings = Field(default_factory=IntPressureSettings)
    zeta: ScalarTidalSettings = Field(default_factory=ScalarTidalSettings)


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
    method: str = "analytical"
    ddu: float = 0.0
    ddl: float = 0.0
    file: str = ""

    @model_validator(mode="after")
    def _normalise_method(self) -> GridSettings:
        self.method = _canonical_token(self.method, "analytical")
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
    return GotmSettings.model_validate(_normalize_settings_document(raw))


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
