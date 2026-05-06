"""Tests for validation/runner.py — case execution and validation logic."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from pygotm.validation.runner import validate_case


def test_validate_case_skip_run_missing_nc_returns_error(tmp_path: Path) -> None:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "pygotm.validation.runner._ref_path",
                return_value=tmp_path / "ref.nc",
            )
        )
        stack.enter_context(
            patch(
                "pygotm.validation.runner._yaml_path",
                return_value=tmp_path / "gotm.yaml",
            )
        )
        result = validate_case("couette", tmp_path / "runs", skip_run=True)
    assert result.status == "ERROR"
    assert "not found" in (result.error or "").lower()


def test_validate_case_skip_run_existing_nc(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    case_dir = runs_dir / "couette"
    case_dir.mkdir(parents=True)
    nc_path = case_dir / "couette.nc"

    arr = np.linspace(0.0, 1.0, 10)
    ds = xr.Dataset({"u": (["t"], arr)})
    ds.to_netcdf(nc_path, engine="scipy")

    ref_nc = tmp_path / "ref.nc"
    ds.to_netcdf(ref_nc, engine="scipy")

    with ExitStack() as stack:
        stack.enter_context(
            patch("pygotm.validation.runner._ref_path", return_value=ref_nc)
        )
        stack.enter_context(
            patch(
                "pygotm.validation.runner._yaml_path",
                return_value=tmp_path / "gotm.yaml",
            )
        )
        result = validate_case("couette", runs_dir, skip_run=True)

    assert result.status == "PASS"
    assert result.wall_time_s == 0.0
    assert result.n_pass >= 1
    assert result.n_fail == 0


def test_validate_case_counts_vars_correctly(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    case_dir = runs_dir / "couette"
    case_dir.mkdir(parents=True)

    arr = np.linspace(0.0, 1.0, 10)
    # u: present in both, identical → PASS
    # v: present in both, large error → FAIL
    # w: only in ref → SKIP
    py_ds  = xr.Dataset({"u": (["t"], arr), "v": (["t"], arr + 1000.0)})
    ref_ds = xr.Dataset({"u": (["t"], arr), "v": (["t"], arr), "w": (["t"], arr)})

    py_ds.to_netcdf(runs_dir / "couette" / "couette.nc", engine="scipy")
    ref_nc = tmp_path / "ref.nc"
    ref_ds.to_netcdf(ref_nc, engine="scipy")

    with ExitStack() as stack:
        stack.enter_context(
            patch("pygotm.validation.runner._ref_path", return_value=ref_nc)
        )
        stack.enter_context(
            patch(
                "pygotm.validation.runner._yaml_path",
                return_value=tmp_path / "gotm.yaml",
            )
        )
        result = validate_case("couette", runs_dir, skip_run=True)

    assert result.n_pass == 1
    assert result.n_fail == 1
    assert result.n_skip == 1
    assert result.status == "FAIL"


def test_validate_case_threads_on_step_to_run_case(tmp_path: Path) -> None:
    steps_received: list[tuple[int, int]] = []

    def fake_run_case(
        case_name: str,
        runs_dir: Path,
        on_step: object = None,
    ) -> tuple[Path, float]:
        nc = tmp_path / f"{case_name}.nc"
        arr = np.zeros(5)
        xr.Dataset({"u": (["t"], arr)}).to_netcdf(nc, engine="scipy")
        if callable(on_step):
            on_step(1, 10)
            on_step(10, 10)
        return nc, 0.5

    ref_nc = tmp_path / "ref.nc"
    xr.Dataset({"u": (["t"], np.zeros(5))}).to_netcdf(ref_nc, engine="scipy")

    def capture(current: int, total: int) -> None:
        steps_received.append((current, total))

    with ExitStack() as stack:
        stack.enter_context(
            patch("pygotm.validation.runner.run_case", side_effect=fake_run_case)
        )
        stack.enter_context(
            patch("pygotm.validation.runner._ref_path", return_value=ref_nc)
        )
        validate_case("couette", tmp_path, skip_run=False, on_step=capture)

    assert (1, 10) in steps_received
    assert (10, 10) in steps_received


def test_validate_case_run_exception_returns_error(tmp_path: Path) -> None:
    ref_nc = tmp_path / "ref.nc"
    xr.Dataset({"u": (["t"], np.zeros(5))}).to_netcdf(ref_nc, engine="scipy")

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "pygotm.validation.runner.run_case",
                side_effect=RuntimeError("boom"),
            )
        )
        stack.enter_context(
            patch("pygotm.validation.runner._ref_path", return_value=ref_nc)
        )
        result = validate_case("couette", tmp_path, skip_run=False)

    assert result.status == "ERROR"
    assert "RuntimeError" in (result.error or "")
    assert "boom" in (result.error or "")
