"""Tests for exponential Stokes drift profiles."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pygotm.stokes_drift.stokes_drift_exp import (
    stokes_drift_exp,
    stokes_drift_exp_batch,
)


def _grid(nlev: int) -> tuple[np.ndarray, np.ndarray]:
    zi = np.linspace(-10.0, 0.0, nlev + 1)
    z = np.zeros(nlev + 1, dtype=np.float64)
    z[1:] = 0.5 * (zi[:-1] + zi[1:])
    return z, zi


def test_exponential_profile_matches_fortran_cell_average_formula() -> None:
    nlev = 5
    z, zi = _grid(nlev)
    us = np.zeros(nlev + 1)
    vs = np.zeros(nlev + 1)

    stokes_drift_exp(nlev, z, zi, 0.2, -0.1, 3.0, us, vs)

    for k in range(1, nlev + 1):
        dz = zi[k] - zi[k - 1]
        kdz = 0.5 * dz / 3.0
        tmp = math.sinh(kdz) / kdz * math.exp((z[k] - zi[nlev]) / 3.0)
        assert us[k] == pytest.approx(tmp * 0.2)
        assert vs[k] == pytest.approx(tmp * -0.1)
    assert us[0] == 0.0
    assert vs[0] == 0.0
    assert np.isfinite(us).all()
    assert np.isfinite(vs).all()


def test_large_kdz_uses_fortran_overflow_guard_branch() -> None:
    nlev = 2
    z, zi = _grid(nlev)
    us = np.zeros(nlev + 1)
    vs = np.zeros(nlev + 1)

    stokes_drift_exp(nlev, z, zi, 1.0, 2.0, 0.001, us, vs)

    expected = math.exp((z[1] - zi[nlev]) / 0.001)
    assert us[1] == pytest.approx(expected)
    assert vs[1] == pytest.approx(2.0 * expected)


def test_nonpositive_decay_depth_fails_loudly() -> None:
    z, zi = _grid(3)
    with pytest.raises(ValueError):
        stokes_drift_exp(3, z, zi, 1.0, 0.0, 0.0, np.zeros(4), np.zeros(4))


def test_batch_matches_single_column_bitwise_for_identical_columns() -> None:
    nlev = 4
    z, zi = _grid(nlev)
    us_single = np.zeros(nlev + 1)
    vs_single = np.zeros(nlev + 1)
    stokes_drift_exp(nlev, z, zi, 0.12, 0.03, 2.5, us_single, vs_single)

    us_batch = np.zeros((2, nlev + 1), dtype=np.float64)
    vs_batch = np.zeros((2, nlev + 1), dtype=np.float64)
    stokes_drift_exp_batch(
        2,
        nlev,
        np.stack([z, z]),
        np.stack([zi, zi]),
        np.array([0.12, 0.12]),
        np.array([0.03, 0.03]),
        np.array([2.5, 2.5]),
        us_batch,
        vs_batch,
    )

    np.testing.assert_array_equal(us_batch[0], us_single)
    np.testing.assert_array_equal(us_batch[1], us_single)
    np.testing.assert_array_equal(vs_batch[0], vs_single)
    np.testing.assert_array_equal(vs_batch[1], vs_single)
