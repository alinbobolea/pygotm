"""Tests for empirical theory-wave Stokes drift."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.stokes_drift.stokes_drift_theory import (
    US0_TO_U10,
    stokes_drift_theory,
    stokes_drift_theory_batch,
    stokes_drift_theory_srf,
)


def _grid(nlev: int) -> tuple[np.ndarray, np.ndarray]:
    zi = np.linspace(-30.0, 0.0, nlev + 1)
    z = np.zeros(nlev + 1, dtype=np.float64)
    z[1:] = 0.5 * (zi[:-1] + zi[1:])
    return z, zi


def test_surface_layer_average_zero_for_zero_wind_or_tiny_depth() -> None:
    assert stokes_drift_theory_srf(0.0, 5.0, 9.81) == 0.0
    assert stokes_drift_theory_srf(10.0, 1.0e-5, 9.81) == 0.0


def test_surface_layer_average_positive_for_valid_wind_and_depth() -> None:
    value = stokes_drift_theory_srf(10.0, 5.0, 9.81)
    assert value > 0.0
    assert np.isfinite(value)


def test_theory_profile_projects_onto_wind_direction() -> None:
    nlev = 6
    z, zi = _grid(nlev)
    stokes_srf = np.zeros(nlev + 1)
    us = np.zeros(nlev + 1)
    vs = np.zeros(nlev + 1)

    us0, vs0, ds = stokes_drift_theory(
        nlev,
        z,
        zi,
        3.0,
        4.0,
        9.81,
        stokes_srf,
        us,
        vs,
    )

    assert us0 == pytest.approx(US0_TO_U10 * 3.0)
    assert vs0 == pytest.approx(US0_TO_U10 * 4.0)
    assert ds > 0.0
    np.testing.assert_allclose(vs[1:] / us[1:], np.full(nlev, 4.0 / 3.0))
    assert np.all(us[1:] >= 0.0)
    assert np.all(np.isfinite(us))
    assert np.all(np.isfinite(vs))


def test_theory_profile_zero_wind_clears_outputs() -> None:
    nlev = 3
    z, zi = _grid(nlev)
    stokes_srf = np.full(nlev + 1, 7.0)
    us = np.full(nlev + 1, 8.0)
    vs = np.full(nlev + 1, 9.0)

    us0, vs0, ds = stokes_drift_theory(nlev, z, zi, 0.0, 0.0, 9.81, stokes_srf, us, vs)

    assert (us0, vs0, ds) == (0.0, 0.0, 0.0)
    np.testing.assert_array_equal(us, np.zeros(nlev + 1))
    np.testing.assert_array_equal(vs, np.zeros(nlev + 1))
    np.testing.assert_array_equal(stokes_srf, np.zeros(nlev + 1))


def test_batch_matches_single_column() -> None:
    nlev = 4
    z, zi = _grid(nlev)
    srf_single = np.zeros(nlev + 1)
    us_single = np.zeros(nlev + 1)
    vs_single = np.zeros(nlev + 1)
    scalars_single = stokes_drift_theory(
        nlev, z, zi, 6.0, 2.0, 9.81, srf_single, us_single, vs_single
    )

    srf_batch = np.zeros((2, nlev + 1), dtype=np.float64)
    us_batch = np.zeros((2, nlev + 1), dtype=np.float64)
    vs_batch = np.zeros((2, nlev + 1), dtype=np.float64)
    scalars = np.zeros((2, 3), dtype=np.float64)
    stokes_drift_theory_batch(
        2,
        nlev,
        np.stack([z, z]),
        np.stack([zi, zi]),
        np.array([6.0, 6.0]),
        np.array([2.0, 2.0]),
        9.81,
        srf_batch,
        us_batch,
        vs_batch,
        scalars,
    )

    np.testing.assert_allclose(us_batch[0], us_single)
    np.testing.assert_allclose(vs_batch[1], vs_single)
    np.testing.assert_allclose(scalars[0], np.array(scalars_single))
    np.testing.assert_allclose(scalars[1], np.array(scalars_single))
