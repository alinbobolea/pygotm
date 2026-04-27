"""Tests for pygotm.observations.const_nnt."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.observations.const_nnt import const_NNT
from pygotm.util.density import METHOD_LINEAR_USER, DensityState, init_density


def _density_state() -> DensityState:
    state = DensityState()
    state.density_method = METHOD_LINEAR_USER
    state.rho0 = 1027.0
    state.T0 = 10.0
    state.S0 = 35.0
    state.alpha0 = 2.0e-4
    state.beta0 = 7.5e-4
    init_density(state, 4)
    return state


def test_const_nnt_matches_linear_eos_recurrence() -> None:
    z = np.array([0.0, -8.0, -6.0, -3.0, -1.0], dtype=np.float64)
    zi = np.array([0.0, -9.0, -7.0, -4.0, -2.0], dtype=np.float64)
    profile = const_NNT(_density_state(), 4, z, zi, 12.0, 35.0, 1.0e-4, 9.81)
    delta = 1.0e-4 / (9.81 * 2.0e-4)
    assert profile[4] == pytest.approx(12.0)
    assert profile[3] == pytest.approx(12.0 - delta * (z[4] - z[3]))
    assert profile[2] == pytest.approx(profile[3] - delta * (z[3] - z[2]))


def test_const_nnt_zero_buoyancy_frequency_is_constant() -> None:
    z = np.array([0.0, -2.0, -1.0], dtype=np.float64)
    zi = np.array([0.0, -2.5, -1.5], dtype=np.float64)
    profile = const_NNT(_density_state(), 2, z, zi, 6.5, 35.0, 0.0, 9.81)
    assert np.allclose(profile[1:], 6.5)
