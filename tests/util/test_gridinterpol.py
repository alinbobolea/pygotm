"""Tests for util/gridinterpol.py — Step 1.7 of GOTM translation plan."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.util.gridinterpol import gridinterpol

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NLEV = 10


def _make_obs(N: int, cols: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Return obs_z and obs_prof with padding index 0 (GOTM convention)."""
    # obs_z shape (N+1,): index 0 is padding, indices 1..N are actual levels
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 1.0, N)  # depths 0..1 at indices 1..N
    obs_prof = np.zeros((N + 1, cols))
    obs_prof[1:, :] = obs_z[1:, np.newaxis]  # value == depth (linear profile)
    return obs_z, obs_prof


def _model_z(nlev: int, zmin: float = 0.0, zmax: float = 1.0) -> np.ndarray:
    """Uniform model grid with nlev+1 levels."""
    return np.linspace(zmin, zmax, nlev + 1)


# ---------------------------------------------------------------------------
# Import / smoke
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.util.gridinterpol import gridinterpol  # noqa: F401


def test_smoke_single_column() -> None:
    obs_z, obs_prof = _make_obs(N=5, cols=1)
    model_z = _model_z(NLEV)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)
    assert result.shape == (NLEV + 1, 1)


def test_smoke_multi_column() -> None:
    obs_z, obs_prof = _make_obs(N=5, cols=3)
    model_z = _model_z(NLEV)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)
    assert result.shape == (NLEV + 1, 3)


# ---------------------------------------------------------------------------
# Analytic verification: linear profile interpolates exactly
# ---------------------------------------------------------------------------


def test_linear_profile_exact_interpolation() -> None:
    """A linear obs profile must reproduce exact values on the model grid."""
    N = 20
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 1.0, N)
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[1:, 0] = obs_z[1:]  # value = depth → linear

    model_z = _model_z(NLEV, zmin=obs_z[1], zmax=obs_z[N])
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    # model_prof[i, 0] should equal model_z[i] for i in 1..nlev
    for i in range(1, NLEV + 1):
        assert result[i, 0] == pytest.approx(model_z[i], rel=1e-12)


def test_constant_profile_exact() -> None:
    """A constant obs profile must reproduce the same constant everywhere."""
    N = 5
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 1.0, N)
    obs_prof = np.full((N + 1, 1), 42.0)

    model_z = _model_z(NLEV, zmin=obs_z[1], zmax=obs_z[N])
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    for i in range(1, NLEV + 1):
        assert result[i, 0] == pytest.approx(42.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Boundary conditions: k=0 and k=nlev
# ---------------------------------------------------------------------------


def test_index_zero_not_touched() -> None:
    """model_prof[0] is never set by gridinterpol (Fortran loop starts at 1)."""
    obs_z, obs_prof = _make_obs(N=5, cols=1)
    model_z = _model_z(NLEV)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)
    assert result[0, 0] == 0.0  # left at initialised zero


def test_top_level_within_range() -> None:
    """model_prof[nlev] at the top of the obs range must equal obs_prof[N]."""
    N = 5
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 1.0, N)
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[1:, 0] = np.arange(1, N + 1, dtype=float)

    # Place top model level exactly at obs_z[N]
    model_z = _model_z(NLEV, zmin=obs_z[1], zmax=obs_z[N])
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)
    assert result[NLEV, 0] == pytest.approx(obs_prof[N, 0], rel=1e-12)


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


def test_surface_extrapolation() -> None:
    """Model levels above obs_z[N] must receive the topmost obs value."""
    N = 5
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 0.5, N)  # obs only goes up to 0.5
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[N, 0] = 99.0  # topmost obs value

    # Model grid goes from 0 to 1.0 — top half is above obs range
    model_z = _model_z(NLEV, zmin=0.0, zmax=1.0)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    # All model levels above obs_z[N]=0.5 must equal obs_prof[N]
    for i in range(1, NLEV + 1):
        if model_z[i] >= obs_z[N]:
            assert result[i, 0] == pytest.approx(99.0, rel=1e-12)


def test_bottom_extrapolation() -> None:
    """Model levels below obs_z[1] must receive the lowest obs value."""
    N = 5
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.5, 1.0, N)  # obs starts at 0.5
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[1, 0] = 77.0  # lowest obs value

    # Model grid goes from 0 to 1.0 — bottom half is below obs range
    model_z = _model_z(NLEV, zmin=0.0, zmax=1.0)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    for i in range(1, NLEV + 1):
        if model_z[i] <= obs_z[1]:
            assert result[i, 0] == pytest.approx(77.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Multiple columns
# ---------------------------------------------------------------------------


def test_multi_column_independent() -> None:
    """Each column must be interpolated independently."""
    N = 10
    cols = 4
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 1.0, N)
    obs_prof = np.zeros((N + 1, cols))
    for j in range(cols):
        obs_prof[1:, j] = (j + 1) * obs_z[1:]  # col j: value = (j+1)*depth

    model_z = _model_z(NLEV, zmin=obs_z[1], zmax=obs_z[N])
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    for j in range(cols):
        for i in range(1, NLEV + 1):
            expected = (j + 1) * model_z[i]
            assert result[i, j] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# NaN / Inf guard
# ---------------------------------------------------------------------------


def test_no_nan_inf_standard_inputs() -> None:
    obs_z, obs_prof = _make_obs(N=10, cols=2)
    model_z = _model_z(NLEV)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_no_nan_inf_with_extrapolation() -> None:
    """NaN/Inf must not appear even when model grid extends beyond obs range."""
    N = 5
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.2, 0.8, N)
    obs_prof = np.random.default_rng(42).random((N + 1, 2))

    model_z = _model_z(NLEV, zmin=0.0, zmax=1.0)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)
    assert not np.any(np.isnan(result[1:]))
    assert not np.any(np.isinf(result[1:]))


# ---------------------------------------------------------------------------
# Physical bounds
# ---------------------------------------------------------------------------


def test_interpolated_values_within_obs_range() -> None:
    """Interior interpolated values must stay within the obs profile range."""
    N = 8
    obs_z = np.zeros(N + 1)
    obs_z[1:] = np.linspace(0.0, 1.0, N)
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[1:, 0] = np.sort(np.random.default_rng(7).random(N))

    obs_min = obs_prof[1:, 0].min()
    obs_max = obs_prof[1:, 0].max()

    model_z = _model_z(NLEV, zmin=obs_z[1], zmax=obs_z[N])
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    for i in range(1, NLEV + 1):
        assert obs_min - 1e-14 <= result[i, 0] <= obs_max + 1e-14


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_coarse_obs_fine_model() -> None:
    """Fine model grid interpolated from coarser obs grid must be self-consistent."""
    N = 3  # very coarse obs: 3 levels
    obs_z = np.zeros(N + 1)
    obs_z[1:] = [0.0, 0.5, 1.0]
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[1:, 0] = [0.0, 5.0, 10.0]  # linear from 0 to 10

    model_z = _model_z(20, zmin=0.0, zmax=1.0)  # fine 20-level model
    result = gridinterpol(obs_z, obs_prof, model_z, nlev=20)

    # Value at any interior model level should equal 10 * model_z[i]
    for i in range(1, 21):
        if 0.0 < model_z[i] < 1.0:
            assert result[i, 0] == pytest.approx(10.0 * model_z[i], rel=1e-12)


def test_single_obs_level() -> None:
    """With N=1 obs level, all model levels should receive that constant value."""
    N = 1
    obs_z = np.zeros(N + 1)
    obs_z[1] = 0.5  # single obs level at depth 0.5
    obs_prof = np.zeros((N + 1, 1))
    obs_prof[1, 0] = 3.14

    model_z = _model_z(NLEV, zmin=0.0, zmax=1.0)
    result = gridinterpol(obs_z, obs_prof, model_z, NLEV)

    # All model levels: model_z >= obs_z[N]=0.5 or model_z <= obs_z[1]=0.5
    # Either way extrapolation gives obs_prof[1]=3.14
    for i in range(1, NLEV + 1):
        assert result[i, 0] == pytest.approx(3.14, rel=1e-12)
