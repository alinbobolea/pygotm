"""Tests for the Phase 7 single-column driver."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import xarray as xr

from pygotm.config import GotmConfig, GotmSettings
from pygotm.driver import GotmDriver
from pygotm.gotm.runtime_builder import UnsupportedConfigurationError

yaml: Any = import_module("yaml")

_COUETTE_CONFIG = Path("gotm-model/cases-runs/couette/gotm.yaml")


def _minimal_config_dict() -> dict[str, object]:
    return {
        "version": 7,
        "title": "driver test",
        "location": {"latitude": 55.0, "longitude": 12.0, "depth": 10.0},
        "time": {
            "start": "2000-01-01 00:00:00",
            "stop": "2000-01-01 00:20:00",
            "dt": 600.0,
        },
        "grid": {"nlev": 4},
        "temperature": {"method": "off"},
        "salinity": {"method": "off"},
        "mimic_3d": {
            "zeta": {
                "method": "tidal",
                "period_1": 1200.0,
                "tidal": {"amp_1": 0.5},
            }
        },
    }


def _write_config(path: Path, config: dict[str, object] | None = None) -> None:
    path.write_text(
        yaml.safe_dump(_minimal_config_dict() if config is None else config),
        encoding="utf-8",
    )


def _short_couette_config_text() -> str:
    config_text = _COUETTE_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "stop: 2005-01-02 00:00:00", "stop: 2005-01-01 00:00:20", 1
    )
    return config_text.replace("nlev: 100", "nlev: 8", 1)


def _write_short_couette_config(path: Path) -> None:
    path.write_text(_short_couette_config_text(), encoding="utf-8")


def _short_couette_config_dict() -> dict[str, object]:
    config = cast(
        dict[str, object],
        _strip_none(yaml.safe_load(_short_couette_config_text())),
    )
    surface = config.get("surface")
    if isinstance(surface, dict):
        for key in (
            "u10",
            "v10",
            "airp",
            "airt",
            "hum",
            "cloud",
            "longwave_radiation",
        ):
            surface.pop(key, None)
    return config


def _strip_none(value: object) -> object:
    if isinstance(value, dict):
        return {k: _strip_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_strip_none(item) for item in value]
    return value


def test_driver_run_returns_dataset_with_expected_axes_and_metadata(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run()

    assert dataset.sizes["time"] == 2
    assert dataset.sizes["z"] == 8
    assert dataset.sizes["zi"] == 9
    assert dataset["u"].dims == ("time", "z", "lat", "lon")
    assert dataset["u"].shape == (2, 8, 1, 1)
    assert dataset.attrs["runtime"] == "compiled"
    assert np.issubdtype(dataset["time"].dtype, np.floating)
    assert dataset["time"].attrs["units"] == "seconds since 2005-01-01 00:00:00"
    assert Path(str(dataset.attrs["source_yaml"])) == config_path.resolve()
    assert np.isfinite(dataset["u"].values).all()
    assert np.isfinite(dataset["tke"].values).all()


def test_driver_accepts_custom_yaml_filename(tmp_path: Path) -> None:
    config_path = tmp_path / "custom_input.yaml"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run(max_steps=1)

    assert Path(str(dataset.attrs["source_yaml"])) == config_path.resolve()
    assert dataset.attrs["runtime"] == "compiled"


def test_driver_run_writes_netcdf_output(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "output" / "result.nc"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run(output_path=output_path)

    assert output_path.is_file()
    with xr.open_dataset(output_path, engine="scipy") as reopened:
        assert reopened.sizes == dataset.sizes
        assert np.allclose(reopened["u"].values, dataset["u"].values)


def test_driver_accepts_in_memory_config() -> None:
    document = _short_couette_config_dict()
    settings = GotmSettings.model_validate(document)

    dataset = GotmDriver(GotmConfig.from_settings(settings, document=document)).run(
        max_steps=1
    )

    assert dataset.sizes["time"] == 2
    assert dataset.sizes["z"] == 8


def test_driver_max_steps_zero_returns_empty_time_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run(max_steps=0)

    assert dataset.sizes["time"] == 1
    assert dataset.sizes["z"] == 8
    assert dataset.sizes["zi"] == 9
    assert dataset.data_vars


def test_driver_output_false_runs_without_data_snapshots(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run(output=False)

    assert dataset.sizes["time"] == 0
    assert dataset.sizes["z"] == 8
    assert dataset.sizes["zi"] == 9
    assert not dataset.data_vars


def test_driver_resolves_relative_scalar_input_paths(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    config_path = case_dir / "gotm.yaml"
    zeta_path = case_dir / "zeta.dat"
    zeta_path.write_text(
        "\n".join(
            [
                "2000-01-01 00:00:00 1.0",
                "2000-01-01 01:00:00 3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = _minimal_config_dict()
    config["time"] = {
        "start": "2000-01-01 00:00:00",
        "stop": "2000-01-01 01:00:00",
        "dt": 1800.0,
    }
    config["mimic_3d"] = {"zeta": {"method": "file", "file": "zeta.dat", "column": 1}}
    _write_config(config_path, config)

    dataset = GotmDriver(config_path).run(output=False)

    assert dataset.sizes["time"] == 0
    assert not dataset.data_vars


def test_driver_raises_for_missing_yaml_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        GotmDriver(missing_path)


def _config_with_heat_flux(time_method: str, time_step: int = 1) -> dict[str, object]:
    """Minimal config with surface heat flux so that temperature changes each step."""
    return {
        "version": 7,
        "title": "time_method test",
        "location": {"latitude": 55.0, "longitude": 12.0, "depth": 10.0},
        "time": {
            "start": "2000-01-01 00:00:00",
            "stop": "2000-01-01 00:40:00",
            "dt": 600.0,
        },
        "grid": {"nlev": 4},
        "temperature": {"method": "constant"},
        "salinity": {"method": "off"},
        "surface": {
            "heat": {"method": "constant", "constant_value": -200.0},
            "swr": {"method": "constant", "constant_value": 0.0},
        },
        "output": {
            "result": {
                "time_unit": "dt",
                "time_step": time_step,
                "time_method": time_method,
            }
        },
    }


def test_output_time_method_raises_for_unknown_method() -> None:
    config = _config_with_heat_flux("bogus_method")
    settings = GotmSettings.model_validate(config)
    with pytest.raises(NotImplementedError, match="unsupported output time_method"):
        GotmDriver(settings).run()


def test_output_time_method_mean_averages_values_over_interval() -> None:
    """time_method: mean is rejected until compiled averaging is implemented."""
    with pytest.raises(UnsupportedConfigurationError, match="output.time_method"):
        GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("mean", 2))).run()


def test_output_time_method_integrated_sums_values_over_interval() -> None:
    """time_method: integrated is rejected until compiled accumulation exists."""
    with pytest.raises(UnsupportedConfigurationError, match="output.time_method"):
        GotmDriver(
            GotmSettings.model_validate(_config_with_heat_flux("integrated", 2))
        ).run()


def test_output_time_method_initial_snapshot_is_always_point() -> None:
    """Compiled point output still includes an initial state slot."""
    document = _short_couette_config_dict()
    settings = GotmSettings.model_validate(document)
    dataset = GotmDriver(GotmConfig.from_settings(settings, document=document)).run(
        max_steps=1
    )

    assert dataset.sizes["time"] == 2
    assert dataset["time"].values[0] == pytest.approx(0.0)
    assert dataset["time"].attrs["units"] == "seconds since 2005-01-01 00:00:00"
