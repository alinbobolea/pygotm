"""Tests for the Phase 7 single-column driver."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from pygotm.config import GotmSettings
from pygotm.driver import GotmDriver

yaml: Any = import_module("yaml")


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


def test_driver_run_returns_dataset_with_expected_axes_and_metadata(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_config(config_path)

    dataset = GotmDriver(config_path).run()

    assert dataset.sizes["time"] == 3
    assert dataset.sizes["z"] == 4
    assert dataset.sizes["zi"] == 5
    assert dataset["u"].shape == (3, 4)
    assert dataset["zeta_obs"].shape == (3,)
    assert np.issubdtype(dataset["time"].dtype, np.datetime64)
    assert Path(str(dataset.attrs["source_yaml"])) == config_path.resolve()
    assert np.isfinite(dataset["zeta_obs"].values).all()
    assert float(dataset["rho_p"].values[0, 0]) > 0.0


def test_driver_run_writes_netcdf_output(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "output" / "result.nc"
    _write_config(config_path)

    dataset = GotmDriver(config_path).run(output_path=output_path)

    assert output_path.is_file()
    with xr.open_dataset(output_path, engine="scipy") as reopened:
        assert reopened.sizes == dataset.sizes
        assert np.allclose(reopened["zeta_obs"].values, dataset["zeta_obs"].values)


def test_driver_accepts_in_memory_settings() -> None:
    settings = GotmSettings.model_validate(_minimal_config_dict())

    dataset = GotmDriver(settings).run(max_steps=1)

    assert dataset.sizes["time"] == 2
    assert dataset.sizes["z"] == 4


def test_driver_max_steps_zero_returns_empty_time_axis(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_config(config_path)

    dataset = GotmDriver(config_path).run(max_steps=0)

    assert dataset.sizes["time"] == 1
    assert dataset.sizes["z"] == 4
    assert dataset.sizes["zi"] == 5
    assert dataset.data_vars


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

    dataset = GotmDriver(config_path).run()

    assert np.allclose(dataset["zeta_obs"].values, np.array([1.0, 2.0, 3.0]))


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
    """time_method: mean outputs the time-average of each variable over the output interval."""
    point_ds = GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("point", 1))).run()
    mean_ds = GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("mean", 2))).run()

    # 4 steps / interval 2 = 2 output times, plus initial snapshot
    assert mean_ds.sizes["time"] == 3

    # mean_ds[1] must be the time-average of point_ds[1] and point_ds[2]
    expected_1 = (point_ds["temp"].values[1] + point_ds["temp"].values[2]) / 2.0
    np.testing.assert_allclose(mean_ds["temp"].values[1], expected_1, rtol=1e-10)

    # mean_ds[2] must be the time-average of point_ds[3] and point_ds[4]
    expected_2 = (point_ds["temp"].values[3] + point_ds["temp"].values[4]) / 2.0
    np.testing.assert_allclose(mean_ds["temp"].values[2], expected_2, rtol=1e-10)


def test_output_time_method_integrated_sums_values_over_interval() -> None:
    """time_method: integrated outputs the time-sum of each variable over the output interval."""
    point_ds = GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("point", 1))).run()
    integ_ds = GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("integrated", 2))).run()

    assert integ_ds.sizes["time"] == 3

    expected_1 = point_ds["temp"].values[1] + point_ds["temp"].values[2]
    np.testing.assert_allclose(integ_ds["temp"].values[1], expected_1, rtol=1e-10)

    expected_2 = point_ds["temp"].values[3] + point_ds["temp"].values[4]
    np.testing.assert_allclose(integ_ds["temp"].values[2], expected_2, rtol=1e-10)


def test_output_time_method_initial_snapshot_is_always_point() -> None:
    """The initial snapshot (before first integration step) is always a point snapshot."""
    point_ds = GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("point", 1))).run()
    mean_ds = GotmDriver(GotmSettings.model_validate(_config_with_heat_flux("mean", 2))).run()

    # The first snapshot (index 0) from both runs must be identical.
    np.testing.assert_allclose(mean_ds["temp"].values[0], point_ds["temp"].values[0], rtol=1e-12)
