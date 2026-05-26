"""Tests for pygotm.gotm.gotm."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped,unused-ignore]

from pygotm.airsea.airsea import AirSeaDriverState
from pygotm.gotm.gotm import (
    _configure_airsea_from_document,
    _configure_output_schedule,
    finalize_gotm,
    initialize_gotm,
    integrate_gotm,
)
from pygotm.icethm import IceModelEnum

_COUETTE_CONFIG = Path("validation/reference/couette/gotm.yaml")


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


def _write_short_couette_config(path: Path) -> None:
    config_text = _COUETTE_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "stop: 2005-01-02 00:00:00", "stop: 2005-01-01 00:00:20", 1
    )
    config_text = config_text.replace("nlev: 100", "nlev: 8", 1)
    path.write_text(config_text, encoding="utf-8")


def _write_vertical_advection_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 7,
                "location": {"latitude": 55.0, "longitude": 12.0, "depth": 10.0},
                "time": {
                    "start": "2000-01-01 00:00:00",
                    "stop": "2000-01-01 00:01:00",
                    "dt": 60.0,
                },
                "grid": {"nlev": 8},
                "temperature": {"method": "off"},
                "salinity": {"method": "off"},
                "mimic_3d": {
                    "w": {
                        "max": {"method": "constant", "constant_value": 1.0e-4},
                        "height": {"method": "constant", "constant_value": -5.0},
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


def test_integrate_gotm_uses_compiled_runtime_without_snapshots(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)
    run = initialize_gotm(config_path)
    try:
        integrate_gotm(run)
        assert run.snapshots == []
        assert run.snapshot_times == []
        assert run.time.timestr == "2005-01-01 00:00:20"
        assert run.meanflow.u is not None
        assert np.isfinite(run.meanflow.u).all()
    finally:
        finalize_gotm(run)


def test_integrate_gotm_can_skip_snapshots(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)
    run = initialize_gotm(config_path)
    try:
        integrate_gotm(run, output=False)

        assert run.snapshots == []
        assert run.snapshot_times == []
        assert run.time.timestr == "2005-01-01 00:00:20"
    finally:
        finalize_gotm(run)


def test_integrate_gotm_rejects_fabm_active_case_before_zero_output(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_short_couette_config(config_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "use: false                           # enable FABM",
        "use: true                            # enable FABM",
    )
    config_path.write_text(text, encoding="utf-8")
    (tmp_path / "fabm.yaml").write_text("instances: {}\n", encoding="utf-8")

    run = initialize_gotm(config_path)
    try:
        with pytest.raises(RuntimeError, match="FABM|pyfabm"):
            integrate_gotm(run, max_steps=1, output=True)
    finally:
        finalize_gotm(run)


@pytest.mark.slow
def test_integrate_gotm_accepts_active_vertical_advection(tmp_path: Path) -> None:
    config_path = tmp_path / "gotm.yaml"
    _write_vertical_advection_config(config_path)
    run = initialize_gotm(config_path)
    try:
        integrate_gotm(run, max_steps=1, output=False)

        assert run.observations.w_adv_input.method != 0
        assert run.meanflow.w is not None
        assert np.any(run.meanflow.w[1 : run.nlev] != 0.0)
    finally:
        finalize_gotm(run)


@pytest.mark.parametrize(
    "eqstate_method",
    ["full_teos-10", "full_teos_10", "linear_teos-10", "linear_teos_10"],
)
def test_initialize_gotm_accepts_teos10_equation_of_state_variants(
    tmp_path: Path, eqstate_method: str
) -> None:
    """TEOS-10 spelling variants should initialize cleanly."""
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


@pytest.mark.parametrize("ice_model", ["basal_melt", "winton"])
def test_airsea_configuration_accepts_validation_ice_models(ice_model: str) -> None:
    state = AirSeaDriverState()

    _inputs, ice_params = _configure_airsea_from_document(
        state,
        {"surface": {"ice": {"model": ice_model}}},
    )

    expected = {
        "basal_melt": IceModelEnum.BASAL_MELT,
        "winton": IceModelEnum.WINTON,
    }
    assert ice_params.model == expected[ice_model]


def test_output_schedule_tracks_active_vertical_slice() -> None:
    schedule = _configure_output_schedule(
        {
            "output": {
                "plume": {
                    "time_unit": "dt",
                    "time_step": 400,
                    "k_start": 300,
                    "k1_start": 300,
                },
                "ice": {
                    "is_active": False,
                    "time_unit": "hour",
                    "time_step": 1,
                    "k_start": 1,
                    "k1_start": 1,
                },
            }
        },
        dt=8.64,
    )

    assert schedule.interval_steps == 400
    assert schedule.k_start == 300
    assert schedule.k1_start == 300


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
