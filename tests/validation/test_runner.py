"""Tests for validation/runner.py — case execution and validation logic."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from pygotm.validate import ValidationCase
from pygotm.validation.compare import VarResult
from pygotm.validation.report import CaseResult
from pygotm.validation.runner import validate_case, validate_case_to_html


def _fake_case(
    tmp_path: Path,
    *,
    name: str = "couette",
    ref_path: Path | None = None,
    yaml_path: Path | None = None,
) -> ValidationCase:
    return ValidationCase(
        name=name,
        directory=tmp_path / name,
        yaml_path=yaml_path or tmp_path / "gotm.yaml",
        reference_path=ref_path or tmp_path / "ref.nc",
    )


def test_validate_case_skip_run_missing_nc_returns_error(tmp_path: Path) -> None:
    with patch(
        "pygotm.validate.resolve_reference_case",
        return_value=_fake_case(tmp_path),
    ):
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

    with patch(
        "pygotm.validate.resolve_reference_case",
        return_value=_fake_case(tmp_path, ref_path=ref_nc),
    ):
        result = validate_case("couette", runs_dir, skip_run=True)

    assert result.status == "PASS"
    assert result.wall_time_s == 0.0
    assert result.n_pass >= 1
    assert result.n_broken == 0 and result.n_marginal == 0 and result.n_discrepant == 0


def test_validate_case_to_html_writes_case_report_before_stripping_summary(
    tmp_path: Path,
) -> None:
    variable = VarResult(
        name="temp",
        section="pygotm",
        status="MARGINAL",
        color="yellow",
        reference_at_worst=1.0,
        calculated_at_worst=1.1,
        d_raw=0.1,
        d_norm=0.02,
        plot_html="<div id='plot'>plot</div>",
    )
    case_result = CaseResult(
        case_name="couette",
        status="FAIL",
        error=None,
        py_nc_path="validation/runs/couette/couette.nc",
        ref_nc_path="validation/reference/couette/couette.nc",
        wall_time_s=0.0,
        task_name="couette-gotm",
        variables=[variable],
        n_marginal=1,
    )

    with patch("pygotm.validation.runner.validate_case", return_value=case_result):
        summary = validate_case_to_html(
            "couette",
            tmp_path / "runs",
            tmp_path,
            generated_at="2026-05-22T00:00:00Z",
            hardware={"execution_backend": "cpu"},
            skip_run=True,
        )

    assert summary.variables == []
    html = (tmp_path / "couette-gotm.html").read_text(encoding="utf-8")
    assert "temp" in html
    assert "plot" in html


def test_validate_case_counts_vars_correctly(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    case_dir = runs_dir / "couette"
    case_dir.mkdir(parents=True)

    arr = np.linspace(0.0, 1.0, 10)
    # u: present in both, identical → PASS
    # v: present in both, large error -> BROKEN (d_norm >= 0.20)
    # w: only in ref → BROKEN; pyGOTM must emit every Fortran reference variable.
    py_ds = xr.Dataset({"u": (["t"], arr), "v": (["t"], arr + 1000.0)})
    ref_ds = xr.Dataset({"u": (["t"], arr), "v": (["t"], arr), "w": (["t"], arr)})

    py_ds.to_netcdf(runs_dir / "couette" / "couette.nc", engine="scipy")
    ref_nc = tmp_path / "ref.nc"
    ref_ds.to_netcdf(ref_nc, engine="scipy")

    with patch(
        "pygotm.validate.resolve_reference_case",
        return_value=_fake_case(tmp_path, ref_path=ref_nc),
    ):
        result = validate_case("couette", runs_dir, skip_run=True)

    assert result.n_pass == 1
    assert result.n_broken == 2
    assert result.status == "FAIL"


def test_validate_case_writes_turbulence_debug_dump(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    case_dir = runs_dir / "couette"
    case_dir.mkdir(parents=True)

    ref_tke = np.asarray([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    py_tke = ref_tke.copy()
    py_tke[1, 2] += 0.25

    xr.Dataset(
        {"tke": (["time", "zi"], py_tke)},
    ).to_netcdf(case_dir / "couette.nc", engine="scipy")
    ref_nc = tmp_path / "ref.nc"
    xr.Dataset(
        {"tke": (["time", "zi"], ref_tke)},
    ).to_netcdf(ref_nc, engine="scipy")

    with patch(
        "pygotm.validate.resolve_reference_case",
        return_value=_fake_case(tmp_path, ref_path=ref_nc),
    ):
        result = validate_case(
            "couette",
            runs_dir,
            skip_run=True,
            debug_turbulence=True,
        )

    debug_path = case_dir / "turbulence_debug.json"
    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    worst_tke = next(
        row
        for row in payload["per_time"]
        if row["variable"] == "tke" and row["time_index"] == 1
    )
    assert result.status == "FAIL"
    assert worst_tke["max_abs_err"] == 0.25
    assert worst_tke["spatial_index"] == {"zi": 2}


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
            patch(
                "pygotm.validate.resolve_reference_case",
                return_value=_fake_case(tmp_path, ref_path=ref_nc),
            )
        )
        result = validate_case("couette", tmp_path, skip_run=False)

    assert result.status == "ERROR"
    assert "RuntimeError" in (result.error or "")
    assert "boom" in (result.error or "")
