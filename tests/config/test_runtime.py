"""Tests for the Phase 7 config runtime helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from pygotm.config import GotmConfig, GotmSettings, load_config

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_LAKE_ERKEN_MINIMAL = _FIXTURES / "cases" / "lake_erken_minimal.yaml"

yaml: Any = import_module("yaml")


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 7,
                "title": "runtime config",
                "location": {
                    "latitude": 55.0,
                    "longitude": 12.0,
                    "depth": 25.0,
                    "hypsograph": "hypsograph.dat",
                },
                "time": {
                    "start": "2000-01-01 00:00:00",
                    "stop": "2000-01-01 01:00:00",
                    "dt": 600.0,
                },
                "grid": {"nlev": 4, "file": "grid.dat"},
                "temperature": {
                    "method": "file",
                    "file": "t_prof.dat",
                    "type": "conservative",
                },
                "salinity": {
                    "method": "file",
                    "file": "s_prof.dat",
                    "type": "absolute",
                },
                "mimic_3d": {
                    "zeta": {
                        "method": "file",
                        "file": "zeta.dat",
                        "column": 1,
                    }
                },
                "surface": {
                    "u10": {
                        "method": "file",
                        "file": "meteo.dat",
                        "column": 1,
                    }
                },
                "turbulence": {
                    "turb_method": "second_order",
                    "epsprof": {
                        "method": "file",
                        "file": "eps.dat",
                        "column": 1,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_load_config_resolves_relative_file_paths_without_mutating_document(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_config(config_path)

    config = load_config(config_path)
    resolved = config.resolved_document()
    resolved_settings = config.resolved_settings()

    assert config.source_path == config_path.resolve()
    assert config.document["temperature"]["file"] == "t_prof.dat"
    assert config.document["location"]["hypsograph"] == "hypsograph.dat"
    assert config.document["surface"]["u10"]["file"] == "meteo.dat"
    assert config.document["turbulence"]["turb_method"] == "second_order"
    assert resolved["location"]["hypsograph"] == str(
        (tmp_path / "hypsograph.dat").resolve()
    )
    assert resolved["surface"]["u10"]["file"] == str((tmp_path / "meteo.dat").resolve())
    assert resolved_settings.location.hypsograph == str(
        (tmp_path / "hypsograph.dat").resolve()
    )
    assert resolved_settings.grid.file == str((tmp_path / "grid.dat").resolve())
    assert resolved_settings.temperature.file == str(
        (tmp_path / "t_prof.dat").resolve()
    )
    assert resolved_settings.salinity.file == str((tmp_path / "s_prof.dat").resolve())
    assert resolved_settings.mimic_3d.zeta.file == str(
        (tmp_path / "zeta.dat").resolve()
    )


def test_save_preserves_untyped_sections_and_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    out_path = tmp_path / "saved.yaml"
    _write_config(config_path)

    config = load_config(config_path)
    config.save(out_path)
    saved = yaml.safe_load(out_path.read_text(encoding="utf-8"))

    assert saved["temperature"]["file"] == "t_prof.dat"
    assert saved["surface"]["u10"]["file"] == "meteo.dat"
    assert saved["turbulence"]["turb_method"] == "second_order"
    assert saved["turbulence"]["epsprof"]["file"] == "eps.dat"


def test_from_settings_without_source_path_leaves_relative_paths_unmodified() -> None:
    settings = GotmSettings.model_validate(
        {
            "temperature": {"method": "file", "file": "t_prof.dat"},
            "mimic_3d": {"zeta": {"method": "file", "file": "zeta.dat"}},
        }
    )
    config = GotmConfig.from_settings(settings)
    resolved = config.resolved_settings()

    assert resolved.temperature.file == "t_prof.dat"
    assert resolved.mimic_3d.zeta.file == "zeta.dat"


def test_load_config_normalizes_lake_erken_legacy_document_aliases() -> None:
    # Uses a synthetic fixture with the same raw GOTM numeric codes and legacy
    # key aliases as the real lake_erken case, without touching the gitignored
    # validation/reference/ tree.
    config = load_config(_LAKE_ERKEN_MINIMAL)
    document = config.resolved_document()

    assert document["location"]["hypsograph"].endswith("hypsograph.dat")
    assert document["surface"]["u10"]["file"].endswith("Erken_MetFile_1960-2020.dat")
    assert document["surface"]["u10"]["method"] == "file"
    assert document["surface"]["fluxes"]["method"] == "fairall"
    assert document["surface"]["ssuv_method"] == "absolute"
    assert document["surface"]["ice"]["model"] == "mylake"
    assert document["mimic_3d"]["int_pressure"]["gradients"]["dtdx"]["method"] == "off"
    # Raw GOTM eq_state mode=2 (Jackett 2005) + method=2 (potential density)
    # must map to Jackett *potential* density, not a fallback to TEOS-10 in-situ
    # (which would add a spurious pressure term to the freshwater density).
    assert document["equation_of_state"]["method"] == "jackett_potential"


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    import pytest

    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)


def test_load_config_raises_for_non_mapping_yaml(tmp_path: Path) -> None:
    import pytest

    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a mapping"):
        load_config(path)


def test_gotm_config_from_settings_has_no_source_path() -> None:
    settings = GotmSettings.model_validate({"version": 7})
    config = GotmConfig.from_settings(settings)
    assert config.source_path is None


def test_gotm_config_from_settings_accepts_document_override() -> None:
    document = {"version": 7, "title": "explicit document"}
    settings = GotmSettings.model_validate(document)
    config = GotmConfig.from_settings(settings, document=document)
    assert config.document["title"] == "explicit document"
