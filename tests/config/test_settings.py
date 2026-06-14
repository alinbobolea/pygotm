"""Tests for pygotm.config.settings."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from pygotm.config.settings import (
    GotmSettings,
    InputSetting,
    TemperatureSettings,
    load_settings,
    save_settings,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_LAKE_ERKEN_MINIMAL = _FIXTURES / "cases" / "lake_erken_minimal.yaml"

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
    from tests.fixtures import bundled_case_path

    settings = load_settings(bundled_case_path("seagrass"))

    assert settings.mimic_3d.zeta.method == "tidal"
    assert settings.mimic_3d.zeta.period_1 == 15.0
    assert settings.mimic_3d.zeta.period_2 == 43200.0


def test_load_settings_accepts_raw_lake_erken_numeric_codes() -> None:
    # Uses a synthetic fixture with the same raw GOTM numeric codes as the real
    # lake_erken case, without touching the gitignored validation/reference/ tree.
    settings = load_settings(_LAKE_ERKEN_MINIMAL)

    assert settings.location.hypsograph == "hypsograph.dat"
    assert settings.grid.method == "analytical"
    assert settings.temperature.method == "file"
    assert settings.salinity.method == "off"
    assert settings.light_extinction.method == "custom"
    assert settings.mimic_3d.ext_pressure.type == "elevation"
    assert settings.mimic_3d.w.max.method == "off"
    assert settings.mimic_3d.w.height.method == "constant"
    assert settings.mimic_3d.w.adv_discr == "p2_pdm"
    assert settings.velocities.u.method == "off"
    assert settings.velocities.v.method == "off"
    # Raw GOTM eq_state uses numeric mode (2=Jackett) + method (2=potential);
    # both must combine so the lake gets Jackett *potential* density (no
    # pressure term), not a silent fallback to full TEOS-10 in-situ density.
    assert settings.equation_of_state["method"] == "jackett_potential"


def test_eos_numeric_mode_method_combinations() -> None:
    from pygotm.config.settings import _eos_method_token

    assert _eos_method_token(2, 2) == "jackett_potential"
    assert _eos_method_token(2, 1) == "jackett_in_situ"
    assert _eos_method_token(1, 2) == "unesco_potential"
    assert _eos_method_token(1, 1) == "unesco_in_situ"
    # GOTM defaults (mode=2, method=2) when fields are absent.
    assert _eos_method_token(None, None) == "jackett_potential"
    # Linearised methods fall back to the method-only TEOS-10 tokens.
    assert _eos_method_token(2, 3) == "linear_teos10"
    assert _eos_method_token(2, 4) == "linear_custom"
    # Already-named tokens pass straight through.
    assert _eos_method_token(None, "full_teos10") == "full_teos10"


def test_save_settings_roundtrip(tmp_path: Path) -> None:
    settings = GotmSettings()
    out_path = tmp_path / "saved.yaml"
    save_settings(settings, out_path)
    reloaded = load_settings(out_path)
    assert reloaded.model_dump() == settings.model_dump()


def test_load_settings_raises_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_settings(missing)


def test_load_settings_raises_for_non_mapping_yaml(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_settings(path)


def test_load_settings_treats_empty_yaml_as_empty_settings(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    settings = load_settings(path)
    assert isinstance(settings, GotmSettings)
