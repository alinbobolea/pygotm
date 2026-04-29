"""Tests for pygotm.meanflow.stratification — buoyancy frequency N²."""

from __future__ import annotations

import numpy as np

from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.stratification import stratification
from pygotm.meanflow.updategrid import updategrid
from pygotm.util.density import METHOD_LINEAR_USER, DensityState, init_density

_NLEV = 10
_DEPTH = 25.0
_DT = 3600.0

_ALPHA = 2.0e-4
_BETA = 7.5e-4
_GRAVITY = 9.81
_RHO0 = 1025.0


def _make_state(nlev: int = _NLEV, depth: float = _DEPTH) -> MeanflowState:
    state = MeanflowState()
    init_meanflow(state, gravity=_GRAVITY)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return state


def _make_density(nlev: int = _NLEV) -> DensityState:
    ds = DensityState()
    ds.density_method = METHOD_LINEAR_USER
    ds.rho0 = _RHO0
    ds.alpha0 = _ALPHA
    ds.beta0 = _BETA
    init_density(ds, nlev)
    return ds


def _call(state: MeanflowState, density_state: DensityState, nlev: int = _NLEV) -> None:
    stratification(state, density_state, nlev)


def test_import() -> None:
    from pygotm.meanflow.stratification import stratification as _s  # noqa: F401
    assert callable(_s)


def test_smoke() -> None:
    state = _make_state()
    ds = _make_density()
    _call(state, ds)


def test_neutral_stratification_zero_N2() -> None:
    """Uniform temperature and salinity profile must give N²=0 everywhere."""
    state = _make_state()
    ds = _make_density()
    assert state.T is not None
    assert state.S is not None
    state.T[:] = 15.0
    state.S[:] = 35.0
    _call(state, ds)
    assert np.all(state.NN == 0.0)
    assert np.all(state.NNT == 0.0)
    assert np.all(state.NNS == 0.0)


def test_linear_T_constant_NNT() -> None:
    r"""Linear temperature profile must yield constant N²_T = g*α*ΔT/H."""
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    delta_T = 5.0
    T_bottom = 10.0
    for k in range(nlev + 1):
        state.T[k] = T_bottom + delta_T * k / nlev
    state.S[:] = 35.0

    _call(state, ds)

    expected_NNT = _ALPHA * _GRAVITY * delta_T / depth

    assert state.NNS is not None
    assert state.NNT is not None
    assert state.NN is not None

    np.testing.assert_allclose(state.NNT[1:nlev], expected_NNT, rtol=1e-10)
    np.testing.assert_allclose(state.NNS[1:nlev], 0.0, atol=1e-15)
    np.testing.assert_allclose(state.NN[1:nlev], expected_NNT, rtol=1e-10)


def test_linear_S_constant_NNS() -> None:
    r"""Linear salinity profile must yield constant N²_S = -g*β*ΔS/H."""
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    delta_S = 2.0
    S_bottom = 34.0
    for k in range(nlev + 1):
        state.S[k] = S_bottom + delta_S * k / nlev
    state.T[:] = 15.0

    _call(state, ds)

    expected_NNS = -_BETA * _GRAVITY * delta_S / depth

    assert state.NNS is not None
    assert state.NNT is not None
    assert state.NN is not None

    np.testing.assert_allclose(state.NNS[1:nlev], expected_NNS, rtol=1e-10)
    np.testing.assert_allclose(state.NNT[1:nlev], 0.0, atol=1e-15)
    np.testing.assert_allclose(state.NN[1:nlev], expected_NNS, rtol=1e-10)


def test_NN_equals_NNT_plus_NNS() -> None:
    """N² must equal NNT + NNS at every interface."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    rng = np.random.default_rng(42)
    state.T[:] = 10.0 + rng.uniform(-2.0, 2.0, nlev + 1)
    state.S[:] = 35.0 + rng.uniform(-1.0, 1.0, nlev + 1)

    _call(state, ds)

    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None

    np.testing.assert_allclose(state.NN, state.NNT + state.NNS, atol=1e-15)


def test_boundary_values_zero() -> None:
    """Indices 0 and nlev of NN, NNT, NNS must always be zero."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(5.0, 20.0, nlev + 1)
    state.S[:] = np.linspace(30.0, 38.0, nlev + 1)

    _call(state, ds)

    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None

    for arr, name in [(state.NN, "NN"), (state.NNT, "NNT"), (state.NNS, "NNS")]:
        assert arr[0] == 0.0, f"{name}[0] must be zero"
        assert arr[nlev] == 0.0, f"{name}[nlev] must be zero"


def test_stable_stratification_positive_NN() -> None:
    """Warm water over cold (stable) must give NN > 0 at interior interfaces."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(5.0, 20.0, nlev + 1)
    state.S[:] = 35.0

    _call(state, ds)

    assert state.NN is not None
    assert np.all(state.NN[1:nlev] > 0.0)


def test_unstable_stratification_negative_NN() -> None:
    """Cold water over warm (unstable) must give NN < 0 at interior interfaces."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(20.0, 5.0, nlev + 1)
    state.S[:] = 35.0

    _call(state, ds)

    assert state.NN is not None
    assert np.all(state.NN[1:nlev] < 0.0)


def test_no_nan_inf() -> None:
    """No NaN or Inf in outputs for a realistic T/S profile."""
    nlev = 50
    depth = 100.0
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(4.0, 22.0, nlev + 1)
    state.S[:] = np.linspace(33.0, 37.0, nlev + 1)

    _call(state, ds, nlev=nlev)

    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None

    assert np.all(np.isfinite(state.NN))
    assert np.all(np.isfinite(state.NNT))
    assert np.all(np.isfinite(state.NNS))


def test_single_layer() -> None:
    """With nlev=1 there are no interior interfaces; only boundaries (= 0)."""
    nlev = 1
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = [5.0, 20.0]
    state.S[:] = [35.0, 35.0]

    _call(state, ds, nlev=nlev)

    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None

    np.testing.assert_array_equal(state.NN, 0.0)
    np.testing.assert_array_equal(state.NNT, 0.0)
    np.testing.assert_array_equal(state.NNS, 0.0)


def test_large_nlev() -> None:
    """N² computation runs without error for nlev=200."""
    nlev = 200
    state = _make_state(nlev=nlev, depth=500.0)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(2.0, 18.0, nlev + 1)
    state.S[:] = np.linspace(33.0, 37.0, nlev + 1)

    _call(state, ds, nlev=nlev)

    assert state.NN is not None
    assert np.all(np.isfinite(state.NN))
    assert state.NN[0] == 0.0
    assert state.NN[nlev] == 0.0


def test_idempotent() -> None:
    """Calling stratification twice with unchanged inputs gives identical output."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(5.0, 20.0, nlev + 1)
    state.S[:] = np.linspace(33.0, 37.0, nlev + 1)

    _call(state, ds)
    assert state.NN is not None
    nn_first = state.NN.copy()

    _call(state, ds)
    np.testing.assert_array_equal(state.NN, nn_first)


def test_matches_analytic_decomposition() -> None:
    """NN, NNT, NNS all match the analytic formula for a combined T/S gradient."""
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    delta_T = 5.0
    delta_S = 2.0
    for k in range(nlev + 1):
        state.T[k] = 10.0 + delta_T * k / nlev
        state.S[k] = 34.0 + delta_S * k / nlev

    _call(state, ds)

    expected_NNT = _ALPHA * _GRAVITY * delta_T / depth
    expected_NNS = -_BETA * _GRAVITY * delta_S / depth

    assert state.NNT is not None
    assert state.NNS is not None
    assert state.NN is not None

    np.testing.assert_allclose(state.NNT[1:nlev], expected_NNT, rtol=1e-10)
    np.testing.assert_allclose(state.NNS[1:nlev], expected_NNS, rtol=1e-10)
    np.testing.assert_allclose(
        state.NN[1:nlev], expected_NNT + expected_NNS, rtol=1e-10,
    )
