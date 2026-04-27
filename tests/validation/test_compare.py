"""Tests for validation/compare.py — NetCDF comparison logic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pygotm.validation.compare import ATOL, RTOL, VarResult, compare_nc


def _write_nc(path: Path, arrays: dict[str, np.ndarray]) -> None:
    ds = xr.Dataset({k: (["t"], v) for k, v in arrays.items()})
    ds.to_netcdf(path, engine="scipy")


def test_var_result_fields() -> None:
    v = VarResult(
        name="u", status="PASS",
        ref_at_worst=1.0, calc_at_worst=1.0,
        max_abs_err=0.0, max_rel_err=0.0, rmse=0.0, nrmse=0.0,
    )
    assert v.name == "u"
    assert v.status == "PASS"


def test_identical_arrays_all_pass(tmp_path: Path) -> None:
    arr = np.linspace(0.1, 1.0, 50)
    _write_nc(tmp_path / "py.nc", {"u": arr})
    _write_nc(tmp_path / "ref.nc", {"u": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert len(results) == 1
    assert results[0].status == "PASS"
    assert results[0].max_abs_err == pytest.approx(0.0, abs=1e-15)
    assert results[0].rmse == pytest.approx(0.0, abs=1e-15)


def test_identical_zero_array_passes(tmp_path: Path) -> None:
    arr = np.zeros(50)
    _write_nc(tmp_path / "py.nc", {"u": arr})
    _write_nc(tmp_path / "ref.nc", {"u": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert results[0].status == "PASS"


def test_large_error_fails(tmp_path: Path) -> None:
    ref = np.ones(10)
    py  = ref + 1.0
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert results[0].status == "FAIL"


def test_shape_mismatch_fails(tmp_path: Path) -> None:
    _write_nc(tmp_path / "py.nc", {"u": np.ones(10)})
    _write_nc(tmp_path / "ref.nc", {"u": np.ones(20)})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert results[0].status == "FAIL"
    assert math.isinf(results[0].max_abs_err)


def test_missing_variable_is_skipped(tmp_path: Path) -> None:
    _write_nc(tmp_path / "py.nc", {"u": np.ones(10)})
    _write_nc(tmp_path / "ref.nc", {"u": np.ones(10), "v": np.ones(10)})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    by_name = {r.name: r for r in results}
    assert by_name["u"].status == "PASS"
    assert by_name["v"].status == "SKIP"
    assert math.isnan(by_name["v"].max_abs_err)


def test_rmse_computed_correctly(tmp_path: Path) -> None:
    ref = np.zeros(4)
    py  = np.array([1.0, 2.0, 3.0, 4.0])
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    expected_rmse = float(np.sqrt(np.mean(py ** 2)))
    assert results[0].rmse == pytest.approx(expected_rmse, rel=1e-10)


def test_nrmse_nan_for_zero_range_reference(tmp_path: Path) -> None:
    ref = np.ones(10) * 5.0
    py  = ref + 1e-5
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert math.isnan(results[0].nrmse)


def test_worst_index_points_to_max_abs_error(tmp_path: Path) -> None:
    ref = np.zeros(5)
    py  = np.array([0.1, 0.5, 2.0, 0.3, 0.1])
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert results[0].calc_at_worst == pytest.approx(2.0, rel=1e-15)
    assert results[0].ref_at_worst  == pytest.approx(0.0, abs=1e-15)


def test_value_at_tolerance_boundary_passes(tmp_path: Path) -> None:
    ref = np.linspace(0.0, 1.0, 100)
    ref_rng = float(np.max(ref) - np.min(ref))
    atol_var = max(2e-6 * ref_rng, ATOL)
    py = ref + atol_var * 0.999
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert results[0].status == "PASS"


def test_value_just_outside_tolerance_fails(tmp_path: Path) -> None:
    ref = np.linspace(0.0, 1.0, 100)
    ref_rng = float(np.max(ref) - np.min(ref))
    atol_var = max(2e-6 * ref_rng, ATOL)
    py = ref + atol_var * 1.001
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    assert results[0].status == "FAIL"


def test_custom_rtol_affects_pass_fail(tmp_path: Path) -> None:
    ref = np.ones(10) * 1000.0
    py  = ref * (1 + 1e-4)
    _write_nc(tmp_path / "py.nc", {"u": py})
    _write_nc(tmp_path / "ref.nc", {"u": ref})
    assert compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", rtol=1e-6)[0].status == "FAIL"
    assert compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", rtol=1e-3)[0].status == "PASS"


def test_multiple_variables_reported_independently(tmp_path: Path) -> None:
    ref_u = np.linspace(0.0, 1.0, 20)
    ref_v = np.linspace(0.0, 1.0, 20)
    py_u  = ref_u
    py_v  = ref_v + 10.0
    _write_nc(tmp_path / "py.nc", {"u": py_u, "v": py_v})
    _write_nc(tmp_path / "ref.nc", {"u": ref_u, "v": ref_v})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc")
    by_name = {r.name: r for r in results}
    assert by_name["u"].status == "PASS"
    assert by_name["v"].status == "FAIL"
