"""Tests for validation/compare.py — three-indicator validation logic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pygotm.validation.compare import (
    VarResult,
    compare_nc,
)


def _write_nc(path: Path, arrays: dict[str, np.ndarray]) -> None:
    ds = xr.Dataset({k: (["t"], v) for k, v in arrays.items()})
    ds.to_netcdf(path, engine="scipy")


# ---------------------------------------------------------------------------
# VarResult structure
# ---------------------------------------------------------------------------


def test_var_result_has_required_fields() -> None:
    v = VarResult(
        name="temp",
        section="pygotm",
        status="PASS",
        color="green",
        reference_at_worst=10.0,
        calculated_at_worst=10.0,
        primary_score=0.1,
        birge_ratio=0.05,
        normalized_signed_bias=0.0,
        plot_html=None,
    )
    assert v.name == "temp"
    assert v.section == "pygotm"
    assert v.status == "PASS"
    assert v.color == "green"
    assert v.primary_score == pytest.approx(0.1)


def test_var_result_has_no_deprecated_fields() -> None:
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(VarResult)}
    deprecated = {"max_abs_err", "max_rel_err", "rmse", "nrmse", "mean_abs_err", "mean_rel_err", "r2", "correlation"}
    assert not field_names & deprecated, f"deprecated fields present: {field_names & deprecated}"


# ---------------------------------------------------------------------------
# Status classification from primary_score
# ---------------------------------------------------------------------------


def test_primary_score_zero_is_pass(tmp_path: Path) -> None:
    arr = np.linspace(1.0, 10.0, 50)
    _write_nc(tmp_path / "py.nc", {"temp": arr})
    _write_nc(tmp_path / "ref.nc", {"temp": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["temp"].status == "PASS"
    assert by_name["temp"].color == "green"


def test_primary_score_just_over_1_is_marginal(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 1.5 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["temp"].status == "MARGINAL"
    assert by_name["temp"].color == "yellow"


def test_primary_score_just_over_3_is_discrepant(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 5.0 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["temp"].status == "DISCREPANT"
    assert by_name["temp"].color == "orange"


def test_primary_score_over_10_is_broken(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 50.0 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    by_name = {r.name: r for r in results}
    assert by_name["temp"].status == "BROKEN"
    assert by_name["temp"].color == "red"


# ---------------------------------------------------------------------------
# Three indicators — not old metrics
# ---------------------------------------------------------------------------


def test_primary_score_is_p99_of_E_i(tmp_path: Path) -> None:
    ref = np.ones(500) * 1.0
    atol, rtol, sf = 1.0e-10, 1.0e-8, 1.0
    delta = 2.0 * (atol + rtol * np.maximum(np.abs(ref), sf))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.primary_score == pytest.approx(2.0, rel=1e-4)


def test_birge_ratio_is_rms_of_E_i(tmp_path: Path) -> None:
    ref = np.ones(500) * 1.0
    atol, rtol, sf = 1.0e-10, 1.0e-8, 1.0
    delta = 3.0 * (atol + rtol * np.maximum(np.abs(ref), sf))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.birge_ratio == pytest.approx(3.0, rel=1e-4)


def test_normalized_signed_bias_positive_offset(tmp_path: Path) -> None:
    ref = np.ones(200) * 5.0
    py = ref + 1.0e-7
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].normalized_signed_bias > 0.0


def test_normalized_signed_bias_negative_offset(tmp_path: Path) -> None:
    ref = np.ones(200) * 5.0
    py = ref - 1.0e-7
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].normalized_signed_bias < 0.0


def test_no_deprecated_metrics_in_var_result(tmp_path: Path) -> None:
    arr = np.linspace(0.1, 1.0, 50)
    _write_nc(tmp_path / "py.nc", {"temp": arr})
    _write_nc(tmp_path / "ref.nc", {"temp": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert not hasattr(r, "max_abs_err")
    assert not hasattr(r, "max_rel_err")
    assert not hasattr(r, "rmse")
    assert not hasattr(r, "nrmse")


# ---------------------------------------------------------------------------
# worst_index is argmax(E_i)
# ---------------------------------------------------------------------------


def test_reference_at_worst_is_at_argmax_E_i(tmp_path: Path) -> None:
    ref = np.ones(10) * 1.0
    py = ref.copy()
    atol, rtol, sf = 1.0e-10, 1.0e-8, 1.0
    denominator = atol + rtol * max(abs(ref[3]), sf)
    py[3] = ref[3] + 100.0 * denominator
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.reference_at_worst == pytest.approx(1.0, rel=1e-12)
    assert r.calculated_at_worst == pytest.approx(py[3], rel=1e-12)


# ---------------------------------------------------------------------------
# Section classification
# ---------------------------------------------------------------------------


def test_known_gotm_variable_is_pygotm_section(tmp_path: Path) -> None:
    arr = np.ones(10) * 5.0
    _write_nc(tmp_path / "py.nc", {"temp": arr})
    _write_nc(tmp_path / "ref.nc", {"temp": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].section == "pygotm"


def test_unknown_variable_is_pyfabm_section(tmp_path: Path) -> None:
    arr = np.ones(10) * 5.0
    _write_nc(tmp_path / "py.nc", {"fabm_algae_carbon": arr})
    _write_nc(tmp_path / "ref.nc", {"fabm_algae_carbon": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].section == "pyfabm"


# ---------------------------------------------------------------------------
# Plot creation rule
# ---------------------------------------------------------------------------


def test_pass_variable_has_no_plot(tmp_path: Path) -> None:
    arr = np.ones(50) * 5.0
    _write_nc(tmp_path / "py.nc", {"temp": arr})
    _write_nc(tmp_path / "ref.nc", {"temp": arr})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].status == "PASS"
    assert results[0].plot_html is None


def test_marginal_variable_has_plot(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 1.5 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.status == "MARGINAL"
    assert r.plot_html is not None
    assert len(r.plot_html) > 100


def test_discrepant_variable_has_plot(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 5.0 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.status == "DISCREPANT"
    assert r.plot_html is not None


def test_broken_variable_has_no_plot(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 50.0 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.status == "BROKEN"
    assert r.plot_html is None


# ---------------------------------------------------------------------------
# Plot title content
# ---------------------------------------------------------------------------


def test_plot_title_contains_required_fields(tmp_path: Path) -> None:
    ref = np.ones(200) * 1.0
    delta = 5.0 * (1.0e-10 + 1.0e-8 * np.maximum(np.abs(ref), 1.0))
    py = ref + delta
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="seagrass")
    r = results[0]
    assert r.plot_html is not None
    assert "seagrass" in r.plot_html
    assert "temp" in r.plot_html
    assert "DISCREPANT" in r.plot_html


# ---------------------------------------------------------------------------
# Structure errors → BROKEN
# ---------------------------------------------------------------------------


def test_shape_mismatch_gives_broken_status(tmp_path: Path) -> None:
    _write_nc(tmp_path / "py.nc", {"temp": np.ones(10)})
    _write_nc(tmp_path / "ref.nc", {"temp": np.ones(20)})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    assert results[0].status == "BROKEN"


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


# ---------------------------------------------------------------------------
# Birge vs primary_score: status follows primary_score, not Birge
# ---------------------------------------------------------------------------


def test_status_follows_primary_score_not_birge(tmp_path: Path) -> None:
    """98 values at E_i=0.5 (Birge≈0.86) + 2 values at E_i=5.0 (p99=5.0) → DISCREPANT.

    With 100 values sorted as [0.5×98, 5.0×2], numpy p99 lands at index 98 = 5.0.
    Birge = sqrt((98×0.25 + 2×25)/100) ≈ 0.86 — well below the PASS threshold of 1.
    If status were Birge-driven it would be PASS; primary_score-driven gives DISCREPANT.
    """
    ref = np.ones(100) * 1.0
    atol, rtol, sf = 1.0e-10, 1.0e-8, 1.0
    denom = atol + rtol * max(1.0, sf)
    py = ref.copy()
    py[:98] = ref[:98] + 0.5 * denom
    py[98:] = ref[98:] + 5.0 * denom
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert r.birge_ratio < 1.0, f"birge should be < 1, got {r.birge_ratio}"
    assert r.primary_score > 3.0, f"primary_score should be > 3, got {r.primary_score}"
    assert r.status == "DISCREPANT"


# ---------------------------------------------------------------------------
# NaN / invalid masking
# ---------------------------------------------------------------------------


def test_nan_values_are_masked_not_propagated(tmp_path: Path) -> None:
    ref = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
    py = np.array([1.0, 2.0, float("nan"), 4.0, 5.0])
    _write_nc(tmp_path / "py.nc", {"temp": py})
    _write_nc(tmp_path / "ref.nc", {"temp": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    assert not math.isnan(r.primary_score)
    assert r.status == "PASS"


def test_near_zero_values_handled_by_atol_and_scale_floor(tmp_path: Path) -> None:
    # tke: atol=1e-14, rtol=1e-7, scale_floor=1e-10
    ref = np.full(50, 1.0e-12)
    py = ref + 0.5e-14
    _write_nc(tmp_path / "py.nc", {"tke": py})
    _write_nc(tmp_path / "ref.nc", {"tke": ref})
    results = compare_nc(tmp_path / "py.nc", tmp_path / "ref.nc", case_name="test")
    r = results[0]
    # E_i = 0.5e-14 / (1e-14 + 1e-7 * max(1e-12, 1e-10)) = 0.5e-14 / ~1e-14 ≈ 0.5
    assert r.status == "PASS"
