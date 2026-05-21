"""Tests for pygotm.config.settings."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from pygotm.config.settings import (
    GotmSettings,
    InputSetting,
    TemperatureSettings,
    load_settings,
    save_settings,
)

yaml: Any = import_module("yaml")


def test_input_setting_scalar_shorthand_creates_constant_input() -> None:
    setting = InputSetting.model_validate(3.5)
    assert setting.method == "constant"
    assert setting.constant_value == 3.5
    assert setting.column == 1


def test_temperature_setting_normalises_method_and_type() -> None:
    setting = TemperatureSettings.model_validate(
        {"method": "Two-Layer", "type": "In-Situ"}
    )
    assert setting.method == "two_layer"
    assert setting.type == "in_situ"


def test_load_settings_parses_minimal_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": 7,
                "location": {"latitude": 55.0, "longitude": 12.0, "depth": 40.0},
                "grid": {"nlev": 20, "method": "file-sigma"},
                "temperature": {
                    "method": "two-layer",
                    "type": "conservative",
                    "two_layer": {"z_s": 5.0, "t_s": 12.0, "z_b": 15.0, "t_b": 6.0},
                },
                "mimic_3d": {
                    "zeta": {
                        "method": "tidal",
                        "period_1": 100.0,
                        "tidal": {"amp_1": 0.2},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(config_path)
    assert isinstance(settings, GotmSettings)
    assert settings.location.latitude == 55.0
    assert settings.grid.method == "file_sigma"
    assert settings.temperature.method == "two_layer"
    assert settings.temperature.type == "conservative"
    assert settings.mimic_3d.zeta.method == "tidal"
    assert settings.mimic_3d.zeta.period_1 == 100.0


def test_load_settings_lifts_nested_tidal_periods_from_real_case() -> None:
    settings = load_settings(Path("validation/reference/seagrass/gotm.yaml"))

    assert settings.mimic_3d.zeta.method == "tidal"
    assert settings.mimic_3d.zeta.period_1 == 15.0
    assert settings.mimic_3d.zeta.period_2 == 43200.0


def test_load_settings_parses_mimic3d_vertical_velocity_from_real_cases() -> None:
    for case_path in (
        Path("validation/reference/nns_seasonal/gotm.yaml"),
        Path("validation/reference/reynolds/gotm.yaml"),
    ):
        settings = load_settings(case_path)

        assert settings.mimic_3d.w.max.method == "file"
        assert settings.mimic_3d.w.height.method == "file"


def test_save_settings_roundtrip(tmp_path: Path) -> None:
    settings = GotmSettings()
    out_path = tmp_path / "saved.yaml"
    save_settings(settings, out_path)
    reloaded = load_settings(out_path)
    assert reloaded.model_dump() == settings.model_dump()
