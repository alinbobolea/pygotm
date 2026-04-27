"""Tests for pygotm.gotm.gotm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from pygotm.gotm.gotm import finalize_gotm, initialize_gotm, integrate_gotm


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 7,
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
        ),
        encoding="utf-8",
    )


def test_initialize_and_finalize_gotm(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_config(config_path)
    run = initialize_gotm(config_path)
    assert run.initialized
    assert run.registry.list()
    finalize_gotm(run)
    assert not run.initialized


def test_integrate_gotm_collects_snapshots(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_config(config_path)
    run = initialize_gotm(config_path)
    try:
        integrate_gotm(run)
        assert len(run.snapshots) == run.time.MaxN + 1
        assert run.snapshot_times == [
            "2000-01-01 00:00:00",
            "2000-01-01 00:10:00",
            "2000-01-01 00:20:00",
        ]
        zeta_values = [snapshot["zeta_obs"] for snapshot in run.snapshots]
        assert max(zeta_values) > min(zeta_values)
        assert np.isfinite(run.snapshots[0]["rho_p"]).all()
    finally:
        finalize_gotm(run)


@pytest.mark.parametrize(
    "eqstate_method",
    ["full_teos-10", "full_teos_10", "linear_teos-10", "linear_teos_10"],
)
def test_initialize_gotm_accepts_teos10_equation_of_state_variants(
    tmp_path: Path, eqstate_method: str
) -> None:
    """full_teos-10 / full_teos_10 (reference YAML form) must not raise NotImplementedError."""
    config = {
        "version": 7,
        "location": {"latitude": 55.0, "longitude": 12.0, "depth": 10.0},
        "time": {
            "start": "2000-01-01 00:00:00",
            "stop": "2000-01-01 00:00:00",
            "dt": 600.0,
        },
        "grid": {"nlev": 4},
        "temperature": {"method": "off"},
        "salinity": {"method": "off"},
        "equation_of_state": {"method": eqstate_method},
    }
    config_path = tmp_path / "gotm.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run = initialize_gotm(config_path)
    finalize_gotm(run)


def test_initialize_gotm_can_write_default_yaml_and_schema_without_input_file(
    tmp_path: Path,
) -> None:
    yaml_out = tmp_path / "default.yaml"
    schema_out = tmp_path / "schema.json"
    run = initialize_gotm(
        tmp_path / "missing.yaml",
        write_yaml_path=str(yaml_out),
        write_schema_path=str(schema_out),
    )
    assert yaml_out.is_file()
    assert schema_out.is_file()
    finalize_gotm(run)
