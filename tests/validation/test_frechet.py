"""Unit tests for the discrete Fréchet kernel and dynamic normalization."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.validation.frechet import (
    discrete_frechet_iter,
    discrete_frechet_iter_numba,
    dynamic_log_range_normalize_pair,
    frechet_raw_and_normalized,
)


def test_identical_series_zero_raw_frechet() -> None:
    a = np.linspace(1.0, 10.0, 100)
    result = frechet_raw_and_normalized(a, a)
    assert result["d_raw"] == pytest.approx(0.0, abs=1e-10)
    assert result["d_norm"] == pytest.approx(0.0, abs=1e-10)


def test_raw_below_abs_tol_gives_zero_dnorm() -> None:
    a = np.linspace(1.0, 10.0, 100)
    b = a + 1e-15  # below default frechet_abs_tol=1e-12
    result = frechet_raw_and_normalized(a, b, abs_tolerance=1e-12)
    assert result["d_norm"] == pytest.approx(0.0)


def test_numba_matches_python_reference() -> None:
    rng = np.random.default_rng(42)
    a = rng.random(80)
    b = a + 0.05 * rng.random(80)
    P = np.ascontiguousarray(a.reshape(-1, 1), dtype=np.float64)
    Q = np.ascontiguousarray(b.reshape(-1, 1), dtype=np.float64)
    d_py = discrete_frechet_iter(P, Q)
    d_nb = float(discrete_frechet_iter_numba(P, Q))
    assert d_py == pytest.approx(d_nb, rel=1e-10)


def test_frechet_is_symmetric() -> None:
    a = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
    b = np.array([1.1, 1.9, 1.6, 2.8, 2.6])
    r1 = frechet_raw_and_normalized(a, b)
    r2 = frechet_raw_and_normalized(b, a)
    assert r1["d_raw"] == pytest.approx(r2["d_raw"], rel=1e-10)


def test_all_zeros_returns_zero() -> None:
    a = np.zeros(50)
    b = np.zeros(50)
    result = frechet_raw_and_normalized(a, b)
    assert result["d_raw"] == pytest.approx(0.0, abs=1e-10)
    assert result["d_norm"] == pytest.approx(0.0, abs=1e-10)


def test_nan_filtered_before_frechet() -> None:
    a = np.array([1.0, float("nan"), 3.0, 4.0, 5.0])
    b = np.array([1.0, float("nan"), 3.0, 4.0, 5.0])
    result = frechet_raw_and_normalized(a, b)
    assert result["d_raw"] == pytest.approx(0.0, abs=1e-10)


def test_linear_normalization_for_small_span() -> None:
    a = np.ones(50) * 5.0
    b = np.ones(50) * 5.1
    _, _, meta = dynamic_log_range_normalize_pair(a, b, switch_oom=2.0)
    assert meta["mode"] == "linear"


def test_log_normalization_for_large_span() -> None:
    a = np.logspace(-6, 0, 100)  # 6 decades
    b = a * 1.01
    _, _, meta = dynamic_log_range_normalize_pair(a, b, switch_oom=2.0)
    assert meta["mode"] == "log"


def test_normalized_output_clipped_to_0_1() -> None:
    a = np.linspace(1.0, 100.0, 200)
    b = a + 0.5
    a_n, b_n, _ = dynamic_log_range_normalize_pair(a, b)
    assert np.all(a_n >= 0.0) and np.all(a_n <= 1.0)
    assert np.all(b_n >= 0.0) and np.all(b_n <= 1.0)


def test_dnorm_increases_with_divergence() -> None:
    ref = np.linspace(1.0, 100.0, 200)
    small_delta = frechet_raw_and_normalized(ref, ref + 1.0)["d_norm"]
    large_delta = frechet_raw_and_normalized(ref, ref + 20.0)["d_norm"]
    assert large_delta > small_delta


def test_frechet_empty_input_returns_zero() -> None:
    a = np.array([], dtype=float)
    b = np.array([], dtype=float)
    result = frechet_raw_and_normalized(a, b)
    assert result["d_raw"] == pytest.approx(0.0)
    assert result["d_norm"] == pytest.approx(0.0)


def test_rel_tolerance_short_circuits_to_zero_dnorm() -> None:
    # d_raw ≈ 5e-4, signal_scale=1000, rel_tol=1e-6 → 1e-3 > 5e-4 → triggers
    a = np.ones(50) * 1000.0
    b = a + 5.0e-4
    result = frechet_raw_and_normalized(
        a, b, abs_tolerance=1.0e-12, rel_tolerance=1.0e-6
    )
    assert result["d_norm"] == pytest.approx(0.0)
    assert result["normalization_mode"] == "rel_tolerance"


def test_rel_tolerance_preserves_nonzero_d_raw() -> None:
    # d_raw must be reported accurately even when rel-tol fires
    a = np.ones(50) * 1000.0
    b = a + 5.0e-4
    result = frechet_raw_and_normalized(
        a, b, abs_tolerance=1.0e-12, rel_tolerance=1.0e-6
    )
    assert result["d_raw"] > 0.0


def test_rel_tolerance_not_triggered_when_signal_small() -> None:
    # signal_scale=1, rel_tol=1e-6 → threshold=1e-6; d_raw ~0.5 >> 1e-6
    a = np.ones(50) * 1.0
    b = a + 0.5
    result = frechet_raw_and_normalized(
        a, b, abs_tolerance=1.0e-12, rel_tolerance=1.0e-6
    )
    assert result["normalization_mode"] != "rel_tolerance"
    assert result["d_norm"] > 0.0
