"""Integration tests for NetCDF global provenance attributes."""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr

from pygotm.driver import GotmDriver
from pygotm.gotm.run_metadata import REQUIRED_NETCDF_ATTRS

_COUETTE_CONFIG = Path("validation/reference/couette/gotm.yaml")


def _write_short_couette_config(path: Path) -> None:
    config_text = _COUETTE_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "stop: 2005-01-02 00:00:00",
        "stop: 2005-01-01 00:00:20",
        1,
    )
    path.write_text(config_text.replace("nlev: 100", "nlev: 8", 1), encoding="utf-8")


def test_real_netcdf_output_has_required_studio_attrs(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    output_path = tmp_path / "result.nc"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run(output_path=output_path)
    dataset.close()

    with xr.open_dataset(output_path, engine="scipy") as reopened:
        for key in REQUIRED_NETCDF_ATTRS:
            assert key in reopened.attrs, f"missing NetCDF attr {key!r}"
        assert reopened.attrs["runtime"] == "compiled"
        assert reopened.attrs["turbulence_closure"] == "k-omega"
        assert reopened.attrs["fabm_active"] == "false"
        assert json.loads(reopened.attrs["fabm_models"]) == []


def test_empty_output_dataset_has_same_required_attrs(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)

    dataset = GotmDriver(config_path).run(output=False, max_steps=1)
    try:
        for key in REQUIRED_NETCDF_ATTRS:
            assert key in dataset.attrs, f"missing empty-output attr {key!r}"
        assert dataset.attrs["runtime"] == "compiled"
    finally:
        dataset.close()
