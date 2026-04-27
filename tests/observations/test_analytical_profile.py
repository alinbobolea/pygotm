"""Tests for pygotm.observations.analytical_profile."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.observations.analytical_profile import analytical_profile


def test_analytical_profile_builds_two_layers_with_linear_transition() -> None:
    z = np.array([0.0, -4.0, -3.0, -2.0, -1.0, -0.5], dtype=np.float64)
    profile = analytical_profile(5, z, 1.0, 10.0, 3.0, 20.0)
    assert profile[5] == pytest.approx(10.0)
    assert profile[4] == pytest.approx(10.0)
    assert profile[3] == pytest.approx(12.5)
    assert profile[2] == pytest.approx(17.5)
    assert profile[1] == pytest.approx(20.0)


def test_analytical_profile_rejects_inverted_interface_depths() -> None:
    with pytest.raises(ValueError, match="z2 should be larger than z1"):
        analytical_profile(2, np.array([0.0, -1.0, -0.5]), 2.0, 1.0, 1.0, 2.0)


def test_analytical_profile_contains_no_nan_or_inf() -> None:
    z = np.array([0.0, -3.0, -2.0, -1.0], dtype=np.float64)
    profile = analytical_profile(3, z, 0.5, 4.0, 2.5, 8.0)
    assert not np.any(np.isnan(profile))
    assert not np.any(np.isinf(profile))
