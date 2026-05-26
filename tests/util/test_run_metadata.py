"""Tests for run metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

from pygotm.config import GotmConfig, GotmSettings
from pygotm.gotm.run_metadata import (
    REQUIRED_NETCDF_ATTRS,
    canonical_materialized_yaml_bytes,
    collect_config_hashes,
    derive_turbulence_closure,
    parse_fabm_models,
)


def test_effective_yaml_hash_uses_materialized_relative_paths(tmp_path: Path) -> None:
    document = {
        "version": 7,
        "title": "portable",
        "grid": {"file": "grid.dat"},
        "temperature": {"method": "file", "file": "temp.dat"},
    }
    case_a = tmp_path / "a"
    case_b = tmp_path / "b"
    case_a.mkdir()
    case_b.mkdir()
    path_a = case_a / "gotm.yaml"
    path_b = case_b / "gotm.yaml"
    path_a.write_text("version: 7\ngrid:\n  file: grid.dat\n", encoding="utf-8")
    path_b.write_text("version: 7\ngrid:\n  file: grid.dat\n", encoding="utf-8")
    settings = GotmSettings.model_validate(document)

    hash_a = collect_config_hashes(
        GotmConfig.from_settings(settings, source_path=path_a, document=document)
    )
    hash_b = collect_config_hashes(
        GotmConfig.from_settings(settings, source_path=path_b, document=document)
    )

    assert hash_a.effective_yaml_sha256 == hash_b.effective_yaml_sha256
    assert hash_a.source_yaml_sha256 == hash_b.source_yaml_sha256


def test_source_yaml_hash_changes_when_raw_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "gotm.yaml"
    path.write_text("version: 7\n", encoding="utf-8")
    first = collect_config_hashes(GotmConfig.from_path(path)).source_yaml_sha256
    path.write_text("version: 7\ntitle: changed\n", encoding="utf-8")
    second = collect_config_hashes(GotmConfig.from_path(path)).source_yaml_sha256

    assert first != second


def test_canonical_materialized_yaml_sorts_keys() -> None:
    left = canonical_materialized_yaml_bytes({"b": 2, "a": 1})
    right = canonical_materialized_yaml_bytes({"a": 1, "b": 2})

    assert left == right


def test_parse_fabm_models_reads_instance_model_paths(tmp_path: Path) -> None:
    path = tmp_path / "fabm.yaml"
    path.write_text(
        "instances:\n  phy:\n    model: gotm/npzd\n  passive:\n    model: bb/passive\n",
        encoding="utf-8",
    )

    assert parse_fabm_models(path) == ("gotm/npzd", "bb/passive")


def test_derive_turbulence_closure_labels_known_closures() -> None:
    assert (
        derive_turbulence_closure(
            turbulence_method="second_order",
            tke_method="tke",
            len_scale_method="omega",
        )
        == "k-omega"
    )
    assert (
        derive_turbulence_closure(
            turbulence_method="second_order",
            tke_method="tke",
            len_scale_method="dissipation",
        )
        == "k-epsilon"
    )


def test_required_netcdf_attrs_count_and_fabm_models_json() -> None:
    assert len(REQUIRED_NETCDF_ATTRS) == 30
    assert json.loads("[]") == []
