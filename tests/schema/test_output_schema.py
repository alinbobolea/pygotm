"""Tests for output schema metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from pygotm.driver import GotmDriver
from pygotm.fabm.engine import FABMOutputSpec
from pygotm.gotm.run_metadata import REQUIRED_NETCDF_ATTRS
from pygotm.schema import netcdf_attrs_schema, output_schema
from tests.fixtures import bundled_case_path

_COUETTE_CONFIG = bundled_case_path("couette")


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

    schema = output_schema(config_path)
    variables = {item["name"]: item for item in schema["variables"]}

    assert len(variables) == len(schema["variables"])
    assert "phy_phy" in variables
    assert "phy_nut" in variables
    assert "phy_zoo" in variables
    assert "phy_PAR" in variables
    assert "phy_phy_sms" not in variables
    assert variables["attenuation_coefficient_of_photosynthetic_radiative_flux"][
        "dimensions"
    ] == ("time", "z", "lat", "lon")
    assert (
        variables["attenuation_coefficient_of_photosynthetic_radiative_flux"][
            "category"
        ]
        == "fabm"
    )
    assert variables["phy_phy"]["units"] == "mmol m-3"
    assert variables["phy_phy"]["state_dependent"] is True


def test_output_schema_reports_previously_curated_fabm_model_generically(
    tmp_path: Path,
) -> None:
    (tmp_path / "fabm.yaml").write_text(
        "instances:\n"
        "  sed:\n"
        "    model: bb/passive\n"
        "    initialization:\n"
        "      c: 1.0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        "version: 7\nfabm:\n  use: true\n  config_file: fabm.yaml\n",
        encoding="utf-8",
    )

    variables = {item["name"]: item for item in output_schema(config_path)["variables"]}

    assert variables["sed_c"]["category"] == "fabm"
    assert variables["sed_c"]["dimensions"] == ("time", "z", "lat", "lon")


def test_output_schema_prefers_horizontal_fabm_diagnostic_for_shared_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HorizontalDiagnosticEngine:
        def __init__(self, path: str | Path) -> None:
            self.path = path

        def initialize(
            self,
            nlev: int | None = None,
            *,
            skip_start: bool = False,
        ) -> None:
            pass

        def output_variable_specs(self) -> tuple[FABMOutputSpec, ...]:
            return (
                FABMOutputSpec(
                    "jrc_med_ergom_DNB",
                    "scalar",
                    "mmol m-2 d-1",
                    "denitrification",
                ),
                FABMOutputSpec(
                    "jrc_med_ergom_OFL",
                    "scalar",
                    "mmol m-2 d-1",
                    "oxygen flux",
                ),
            )

    monkeypatch.setattr("pygotm.schema.FABMEngine", HorizontalDiagnosticEngine)
    (tmp_path / "fabm.yaml").write_text("instances: {}\n", encoding="utf-8")
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        "version: 7\nfabm:\n  use: true\n  config_file: fabm.yaml\n",
        encoding="utf-8",
    )

    schema = output_schema(config_path)
    variables = {item["name"]: item for item in schema["variables"]}

    assert len(variables) == len(schema["variables"])
    assert variables["jrc_med_ergom_DNB"]["dimensions"] == ("time", "lat", "lon")
    assert variables["jrc_med_ergom_OFL"]["dimensions"] == ("time", "lat", "lon")


def test_output_schema_rejects_fabm_collision_with_reserved_extra_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CollisionEngine:
        def __init__(self, path: str | Path) -> None:
            self.path = path

        def initialize(
            self,
            nlev: int | None = None,
            *,
            skip_start: bool = False,
        ) -> None:
            pass

        def output_variable_specs(self) -> tuple[FABMOutputSpec, ...]:
            return (FABMOutputSpec("Tf", "scalar", "", "freezing point"),)

    monkeypatch.setattr("pygotm.schema.FABMEngine", CollisionEngine)
    (tmp_path / "fabm.yaml").write_text("instances: {}\n", encoding="utf-8")
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        "version: 7\nfabm:\n  use: true\n  config_file: fabm.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="core optional output variable"):
        output_schema(config_path)


def test_output_schema_rejects_fabm_collision_with_core_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CollisionEngine:
        def __init__(self, path: str | Path) -> None:
            self.path = path

        def initialize(
            self,
            nlev: int | None = None,
            *,
            skip_start: bool = False,
        ) -> None:
            pass

        def output_variable_specs(self) -> tuple[FABMOutputSpec, ...]:
            return (FABMOutputSpec("temp", "z", "Celsius", "temperature"),)

    monkeypatch.setattr("pygotm.schema.FABMEngine", CollisionEngine)
    (tmp_path / "fabm.yaml").write_text("instances: {}\n", encoding="utf-8")
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(
        "version: 7\nfabm:\n  use: true\n  config_file: fabm.yaml\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="core output variable"):
        output_schema(config_path)


def test_netcdf_attrs_schema_matches_required_attrs() -> None:
    schema = netcdf_attrs_schema()
    assert [item["name"] for item in schema["attributes"]] == list(
        REQUIRED_NETCDF_ATTRS
    )
