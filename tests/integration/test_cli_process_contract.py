"""Subprocess tests for the public pyGOTM process contract."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pygotm.errors import EXIT_RUNTIME_FAILURE
from tests.fixtures import bundled_case_path

_COUETTE_CONFIG = bundled_case_path("couette")
_SELMA_MINIMAL_FABM = (
    Path(__file__).resolve().parents[1] / "fixtures/fabm/selma_minimal.yaml"
)


def _pygotm_script() -> str:
    script = shutil.which("pygotm")
    if script is None:
        pytest.skip("pygotm console script is not installed")
    return script


def _run_pygotm(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_pygotm_script(), *args],
        cwd=cwd,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert result.returncode == 0, detail
    assert "cannot cache function" not in result.stdout
    assert "cannot cache function" not in result.stderr


def _write_short_couette_config(path: Path) -> None:
    config_text = _COUETTE_CONFIG.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "stop: 2005-01-02 00:00:00",
        "stop: 2005-01-01 00:00:20",
        1,
    )
    path.write_text(config_text.replace("nlev: 100", "nlev: 8", 1), encoding="utf-8")


def _write_minimal_selma_config(path: Path, *, fabm_file: str = "fabm.yaml") -> None:
    path.write_text(
        f"""
version: 7
title: Minimal SELMA public CLI smoke test
location:
  latitude: 55.0
  longitude: 12.0
  depth: 10.0
time:
  start: 2000-01-01 00:00:00
  stop: 2000-01-01 00:01:00
  dt: 60.0
grid:
  nlev: 3
temperature:
  method: constant
  constant_value: 10.0
salinity:
  method: constant
  constant_value: 35.0
surface:
  fluxes:
    method: off
    heat:
      method: constant
      constant_value: 0.0
    tx:
      method: constant
      constant_value: 0.0
    ty:
      method: constant
      constant_value: 0.0
  u10:
    method: constant
    constant_value: 0.0
  v10:
    method: constant
    constant_value: 0.0
  swr:
    method: constant
    constant_value: 100.0
  longwave_radiation:
    method: constant
    constant_value: 0.0
  precip:
    method: constant
    constant_value: 0.0
fabm:
  use: true
  config_file: {fabm_file}
output:
  selma:
    time_unit: dt
    time_step: 1
    time_method: point
    variables:
    - source: /*
""".lstrip(),
        encoding="utf-8",
    )


def _write_fabm_case(
    tmp_path: Path,
    *,
    fabm_file: str = "fabm.yaml",
    write_sidecar: bool = True,
) -> tuple[Path, Path]:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    if write_sidecar:
        fabm_path = case_dir / fabm_file
        fabm_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_SELMA_MINIMAL_FABM, fabm_path)
    config_path = case_dir / "gotm.yaml"
    _write_minimal_selma_config(config_path, fabm_file=fabm_file)
    return case_dir, config_path


def test_run_cli_from_foreign_cwd_writes_truncated_netcdf(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    cwd = tmp_path / "cwd"
    case_dir.mkdir()
    cwd.mkdir()
    config_path = case_dir / "gotm.yaml"
    output_path = tmp_path / "out.nc"
    _write_short_couette_config(config_path)

    result = _run_pygotm(
        [
            "run",
            str(config_path),
            "--output",
            str(output_path),
            "--max-steps",
            "2",
        ],
        cwd=cwd,
    )

    _assert_success(result)
    assert output_path.is_file()
    with xr.open_dataset(output_path, engine="scipy") as dataset:
        assert dataset.sizes["time"] >= 1
        assert np.isfinite(dataset["temp"].values).all()

        schema_result = _run_pygotm(["schema", "netcdf-attrs", "--json"], cwd=cwd)
        _assert_success(schema_result)
        required = {
            item["name"] for item in json.loads(schema_result.stdout)["attributes"]
        }
        assert required.issubset(dataset.attrs)


def test_run_cli_progress_json_is_parseable_from_subprocess(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    cwd = tmp_path / "cwd"
    case_dir.mkdir()
    cwd.mkdir()
    config_path = case_dir / "gotm.yaml"
    output_path = tmp_path / "out-progress.nc"
    _write_short_couette_config(config_path)

    result = _run_pygotm(
        [
            "run",
            str(config_path),
            "--output",
            str(output_path),
            "--max-steps",
            "2",
            "--progress",
            "json",
        ],
        cwd=cwd,
    )

    _assert_success(result)
    events = [json.loads(line) for line in result.stderr.splitlines() if line.strip()]
    assert events[0]["event"] == "started"
    assert any(event.get("event") == "phase" for event in events)
    assert events[-1]["event"] == "finished"
    assert events[-1]["exit_code"] == 0
    assert output_path.is_file()


def test_fabm_config_file_runs_through_public_cli_from_foreign_cwd(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyfabm")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _, config_path = _write_fabm_case(tmp_path)
    output_path = tmp_path / "fabm-out.nc"

    result = _run_pygotm(
        [
            "run",
            str(config_path),
            "--output",
            str(output_path),
            "--max-steps",
            "2",
        ],
        cwd=cwd,
    )

    _assert_success(result)
    assert output_path.is_file()
    with xr.open_dataset(output_path, engine="scipy") as dataset:
        assert str(dataset.attrs["fabm_active"]).lower() == "true"
        assert "selma/selma" in json.loads(str(dataset.attrs["fabm_models"]))


def test_fabm_missing_config_file_fails_with_resolved_path(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    case_dir, config_path = _write_fabm_case(
        tmp_path,
        fabm_file="missing.yaml",
        write_sidecar=False,
    )
    output_path = tmp_path / "missing-out.nc"
    missing_path = (case_dir / "missing.yaml").resolve()

    result = _run_pygotm(
        [
            "run",
            str(config_path),
            "--output",
            str(output_path),
            "--max-steps",
            "1",
        ],
        cwd=cwd,
    )

    assert result.returncode == EXIT_RUNTIME_FAILURE
    assert "FABM model YAML not found" in result.stderr
    assert str(missing_path) in result.stderr
    assert not output_path.exists()


def test_schema_output_resolves_fabm_config_file_from_foreign_cwd(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyfabm")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _, config_path = _write_fabm_case(tmp_path, fabm_file="sidecars/fabm.yaml")

    result = _run_pygotm(
        ["schema", "output", "--config", str(config_path), "--json"],
        cwd=cwd,
    )

    _assert_success(result)
    variables = {item["name"] for item in json.loads(result.stdout)["variables"]}
    assert "selma_dd" in variables
    assert "attenuation_coefficient_of_photosynthetic_radiative_flux" in variables
