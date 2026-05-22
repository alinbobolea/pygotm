"""Tests for validation/compare.py Frechet validation logic."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import pygotm.validation.compare as compare_mod
from pygotm.validation.compare import VarResult, compare_nc


def _write_nc(
    path: Path,
    arrays: dict[str, np.ndarray],
    *,
    dim: str = "t",
    times: np.ndarray | None = None,
) -> None:
    coords = {dim: times} if times is not None else None
    ds = xr.Dataset({k: ([dim], v) for k, v in arrays.items()}, coords=coords)
    ds.to_netcdf(path, engine="scipy")


def _write_2d_nc(path: Path, name: str, values: np.ndarray) -> None:
    ds = xr.Dataset({name: (["time", "z"], values)})
    ds.to_netcdf(path, engine="scipy")


def _result_for_delta(tmp_path: Path, delta: float) -> VarResult:
    ref = np.sin(np.linspace(0.0, 20.0, 200)) + 2.0
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    return compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]


def test_var_result_has_required_fields() -> None:
    v = VarResult(
        name="temp",
        section="pygotm",
        status="PASS",
        color="green",
        reference_at_worst=10.0,
        calculated_at_worst=10.0,
        d_raw=0.0,
        d_norm=0.0,
        plot_html=None,
    )
    assert v.name == "temp"
    assert v.section == "pygotm"
    assert v.status == "PASS"
    assert v.color == "green"
    assert v.d_raw == pytest.approx(0.0)
    assert v.d_norm == pytest.approx(0.0)
    assert v.metric_mode == "d_norm"
    assert v.score is None
    assert v.peak_d_norm is None


def test_var_result_has_no_old_metric_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(VarResult)}
    old_metrics = {"primary_score", "birge_ratio", "normalized_signed_bias"}
    assert not field_names & old_metrics


def test_identical_series_is_pass(tmp_path: Path) -> None:
    arr = np.linspace(1.0, 10.0, 50)
    _write_nc(tmp_path / "py.nc", {"temp": arr})
    _write_nc(tmp_path / "ref.nc", {"temp": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["temp"].status == "PASS"
    assert by_name["temp"].color == "green"
    assert by_name["temp"].d_raw == pytest.approx(0.0)
    assert by_name["temp"].d_norm == pytest.approx(0.0)
    assert by_name["temp"].metric_mode == "d_norm"


def test_raw_below_abs_tolerance_forces_zero_distances(tmp_path: Path) -> None:
    ref = np.linspace(1.0, 2.0, 50)
    py = ref + 1.0e-15
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.status == "PASS"
    assert result.d_raw == pytest.approx(0.0)
    assert result.d_norm == pytest.approx(0.0)


def test_dnorm_just_over_pass_threshold_is_marginal(tmp_path: Path) -> None:
    result = _result_for_delta(tmp_path, 0.04)
    assert result.status == "MARGINAL"
    assert result.color == "yellow"
    assert 0.01 <= result.d_norm < 0.05


def test_dnorm_just_over_marginal_threshold_is_discrepant(tmp_path: Path) -> None:
    result = _result_for_delta(tmp_path, 0.12)
    assert result.status == "DISCREPANT"
    assert result.color == "orange"
    assert 0.05 <= result.d_norm < 0.20


def test_dnorm_over_broken_threshold_is_broken(tmp_path: Path) -> None:
    result = _result_for_delta(tmp_path, 0.5)
    assert result.status == "BROKEN"
    assert result.color == "red"
    assert result.d_norm >= 0.20


def test_sparse_localized_excursion_uses_normalized_frechet_score(
    tmp_path: Path,
) -> None:
    ref = np.sin(np.linspace(0.0, 20.0, 200)) + 2.0
    py = ref.copy()
    py[-1] += 0.5
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})

    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]

    assert result.metric_mode == "d_norm"
    assert (result.score if result.score is not None else result.d_norm) == (
        pytest.approx(result.d_norm)
    )
    assert result.peak_d_norm is not None


def test_compare_uses_hybrid_normalization_and_peak_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, float, float, int]] = []

    def fake_frechet_raw_and_normalized(
        *_args: object,
        robust: bool,
        q_low: float,
        q_high: float,
        frechet_k: int,
        **_kwargs: object,
    ) -> dict[str, float | str]:
        calls.append((robust, q_low, q_high, frechet_k))
        return {
            "d_raw": 0.0,
            "d_norm": 0.0,
            "normalization_mode": "test",
        }

    monkeypatch.setattr(
        compare_mod,
        "frechet_raw_and_normalized",
        fake_frechet_raw_and_normalized,
    )
    arr = np.linspace(1.0, 2.0, 20)
    _write_nc(tmp_path / "py.nc", {"temp": arr, "oxygen": arr})
    _write_nc(tmp_path / "ref.nc", {"temp": arr, "oxygen": arr})

    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")

    assert [result.name for result in results] == ["temp", "oxygen"]
    assert calls == [
        (False, 0.1, 99.9, 200),
        (False, 0.1, 99.9, 400),
        (True, 0.1, 99.9, 200),
        (False, 0.1, 99.9, 400),
    ]


def test_near_zero_nn_small_relative_deviation_is_pass(tmp_path: Path) -> None:
    ref = np.full(50, 1.3e-6)
    py = ref * (1.0 + 3.0e-3)
    _write_nc(tmp_path / "py.nc", {"NN": py})
    _write_nc(tmp_path / "ref.nc", {"NN": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.metric_mode == "d_rel"
    assert result.score == pytest.approx(result.d_raw / np.max(np.abs(py)))
    assert result.score is not None and result.score < 0.01
    assert result.status == "PASS"


def test_near_zero_nn_meaningful_relative_deviation_is_discrepant(
    tmp_path: Path,
) -> None:
    ref = np.full(50, 5.0e-7)
    py = ref * 0.86
    _write_nc(tmp_path / "py.nc", {"NN": py})
    _write_nc(tmp_path / "ref.nc", {"NN": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.metric_mode == "d_rel"
    assert result.score == pytest.approx(0.14)
    assert result.status == "DISCREPANT"


def test_nn_above_magnitude_floor_uses_dnorm(tmp_path: Path) -> None:
    ref = np.full(50, 5.0e-3)
    py = ref.copy()
    _write_nc(tmp_path / "py.nc", {"NN": py})
    _write_nc(tmp_path / "ref.nc", {"NN": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.metric_mode == "d_norm"
    assert result.score == pytest.approx(result.d_norm)
    assert result.status == "PASS"


def test_reference_and_calculated_at_worst_use_max_abs_difference(
    tmp_path: Path,
) -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    py = np.array([1.0, 2.1, 2.4, 4.0])
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.reference_at_worst == pytest.approx(3.0)
    assert result.calculated_at_worst == pytest.approx(2.4)


def test_union_time_grid_interpolation_can_produce_pass(tmp_path: Path) -> None:
    _write_nc(
        tmp_path / "py.nc",
        {"temp": np.array([1.0, 2.0, 3.0])},
        dim="time",
        times=np.array([0.0, 1.0, 2.0]),
    )
    _write_nc(
        tmp_path / "ref.nc",
        {"temp": np.array([1.0, 3.0])},
        dim="time",
        times=np.array([0.0, 2.0]),
    )
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.status == "PASS"
    assert result.d_norm == pytest.approx(0.0)


def test_marginal_and_discrepant_variables_get_plots(tmp_path: Path) -> None:
    marginal = _result_for_delta(tmp_path, 0.04)
    assert marginal.status == "MARGINAL"
    assert marginal.plot_html is not None
    assert "x unified" in marginal.plot_html


def test_pass_and_broken_variables_have_no_plot(tmp_path: Path) -> None:
    passing = _result_for_delta(tmp_path, 0.0)
    broken = _result_for_delta(tmp_path, 0.5)
    assert passing.status == "PASS"
    assert passing.plot_html is None
    assert broken.status == "BROKEN"
    assert broken.plot_html is None


def test_plot_title_contains_required_fields(tmp_path: Path) -> None:
    result = _result_for_delta(tmp_path, 0.12)
    assert result.plot_html is not None
    assert "test" in result.plot_html
    assert "temp" in result.plot_html
    assert "DISCREPANT" in result.plot_html
    assert "Normalized Frechet" in result.plot_html


def test_shape_mismatch_gives_broken_status(tmp_path: Path) -> None:
    _write_nc(tmp_path / "py.nc", {"temp": np.ones(10)}, dim="z")
    _write_nc(tmp_path / "ref.nc", {"temp": np.ones(20)}, dim="z")
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].status == "BROKEN"


def test_non_time_dimension_shape_mismatch_gives_broken_status(tmp_path: Path) -> None:
    _write_2d_nc(tmp_path / "py.nc", "temp", np.ones((4, 2)))
    _write_2d_nc(tmp_path / "ref.nc", "temp", np.ones((4, 3)))
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.status == "BROKEN"


def test_missing_reference_variable_gives_broken_status(tmp_path: Path) -> None:
    _write_nc(tmp_path / "py.nc", {"temp": np.ones(10)})
    _write_nc(tmp_path / "ref.nc", {"temp": np.ones(10), "salt": np.ones(10)})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["salt"].status == "BROKEN"


def test_extra_pygotm_variable_gives_broken_status(tmp_path: Path) -> None:
    _write_nc(tmp_path / "py.nc", {"temp": np.ones(10), "salt": np.ones(10)})
    _write_nc(tmp_path / "ref.nc", {"temp": np.ones(10)})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["salt"].status == "BROKEN"


def test_nan_values_are_masked_not_propagated(tmp_path: Path) -> None:
    ref = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
    py = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert not math.isnan(result.d_norm)
    assert result.status == "PASS"


def test_all_nan_overlap_gives_broken_status(tmp_path: Path) -> None:
    ref = np.array([float("nan"), float("nan")])
    py = np.array([float("nan"), float("nan")])
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    result = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")[0]
    assert result.status == "BROKEN"
