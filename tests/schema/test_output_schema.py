"""Tests for output schema metadata."""

from __future__ import annotations

from pathlib import Path

from pygotm.driver import GotmDriver
from pygotm.gotm.run_metadata import REQUIRED_NETCDF_ATTRS
from pygotm.schema import netcdf_attrs_schema, output_schema

_COUETTE_CONFIG = Path("validation/reference/couette/gotm.yaml")


def test_output_schema_has_core_variable_records() -> None:
    schema = output_schema()
    variables = {item["name"]: item for item in schema["variables"]}

    assert variables["temp"]["units"] == "Celsius"
    assert variables["temp"]["dimensions"] == ("time", "z", "lat", "lon")
    assert variables["tke"]["category"] == "turbulence"


def test_output_schema_covers_compiled_couette_output() -> None:
    dataset = GotmDriver(_COUETTE_CONFIG).run(max_steps=1)
    try:
        variables = {item["name"] for item in output_schema()["variables"]}
        assert set(dataset.data_vars).issubset(variables)
        assert {"time", "z", "zi"}.issubset(variables)
    finally:
        dataset.close()


def test_output_schema_reports_fabm_models_from_config(tmp_path: Path) -> None:
    (tmp_path / "fabm.yaml").write_text(
        "instances:\n  phy:\n    model: gotm/npzd\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        "version: 7\nfabm:\n  use: true\n  config_file: fabm.yaml\n",
        encoding="utf-8",
    )

    variables = {item["name"]: item for item in output_schema(config_path)["variables"]}

    assert "npzd_phy" in variables
    assert "npzd_nut" in variables
    assert "npzd_zoo" in variables
    assert variables["npzd_phy"]["state_dependent"] is True


def test_netcdf_attrs_schema_matches_required_attrs() -> None:
    schema = netcdf_attrs_schema()
    assert [item["name"] for item in schema["attributes"]] == list(
        REQUIRED_NETCDF_ATTRS
    )
