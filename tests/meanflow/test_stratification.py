"""Tests for pygotm.meanflow.stratification — buoyancy frequency N²."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.stratification import (
    StratificationWorkspace,
    step_stratification,
    stratification,
)
from pygotm.meanflow.updategrid import updategrid
from pygotm.util.density import METHOD_LINEAR_USER, DensityState, init_density

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NLEV = 10
_DEPTH = 25.0
_DT = 3600.0

# Physically realistic constants for linear EOS tests
_ALPHA = 2.0e-4   # thermal expansion coefficient [1/K]
_BETA = 7.5e-4    # haline contraction coefficient [kg/g]
_GRAVITY = 9.81   # [m/s²]
_RHO0 = 1025.0


def _make_state(nlev: int = _NLEV, depth: float = _DEPTH) -> MeanflowState:
    state = MeanflowState()
    init_meanflow(state, gravity=_GRAVITY)
    state.depth = depth
    state.grid_method = 0  # equidistant — no file I/O
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
    # alpha and beta are set to alpha0/beta0 by init_density for METHOD_LINEAR_USER
    return ds


def _call(
    state: MeanflowState,
    density_state: DensityState,
    nlev: int = _NLEV,
) -> None:
    stratification(state, density_state, nlev)


def _run_step_stratification(
    state: MeanflowState,
    density_state: DensityState,
    nlev: int,
    *,
    n_cols: int,
) -> StratificationWorkspace:
    assert state.h is not None
    assert state.T is not None
    assert state.S is not None
    assert density_state.alpha is not None
    assert density_state.beta is not None

    ws = StratificationWorkspace(nlev=nlev, n_cols=n_cols)
    for col in range(n_cols):
        fill_field_from_array(ws.h, state.h, col=col)
        fill_field_from_array(ws.T, state.T, col=col)
        fill_field_from_array(ws.S, state.S, col=col)
        fill_field_from_array(ws.alpha, density_state.alpha, col=col)
        fill_field_from_array(ws.beta, density_state.beta, col=col)

    step_stratification(
        n_cols,
        nlev,
        state.gravity,
        ws.h,
        ws.T,
        ws.S,
        ws.alpha,
        ws.beta,
        ws.NN,
        ws.NNT,
        ws.NNS,
    )
    return ws


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.stratification import stratification as _s  # noqa: F401

    assert callable(_s)


# ---------------------------------------------------------------------------
# 2. Smoke test
# ---------------------------------------------------------------------------


def test_smoke() -> None:
    state = _make_state()
    ds = _make_density()
    _call(state, ds)


# ---------------------------------------------------------------------------
# 3. Neutral stratification — uniform T and S → N² = 0
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 4. Analytic: linear T profile → constant NNT
# ---------------------------------------------------------------------------


def test_linear_T_constant_NNT() -> None:
    r"""Linear temperature profile must yield constant N²_T = g*α*ΔT/H.

    With equal layer thicknesses h = H/nlev and T[k] increasing linearly
    from bottom (k=0) to surface (k=nlev):
        idz = 2/(h+h) = nlev/H
        dT  = delta_T / nlev
        NNT = α * g * (delta_T/nlev) * (nlev/H) = α * g * delta_T / H
    """
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    delta_T = 5.0  # warm surface, cool bottom
    T_bottom = 10.0
    # T[k] at cell centres; layers 1..nlev, so use indices 1..nlev
    # Simple: fill all including ghost cells with linear profile
    for k in range(nlev + 1):
        state.T[k] = T_bottom + delta_T * k / nlev
    state.S[:] = 35.0  # uniform salinity — no NNS contribution

    _call(state, ds)

    # Expected analytic NNT at interior interfaces
    expected_NNT = _ALPHA * _GRAVITY * delta_T / depth

    assert state.NNS is not None
    assert state.NNT is not None
    assert state.NN is not None

    # Interior interfaces 1..nlev-1
    np.testing.assert_allclose(
        state.NNT[1:nlev], expected_NNT, rtol=1e-10,
        err_msg="NNT deviates from analytic value for linear T profile",
    )
    np.testing.assert_allclose(
        state.NNS[1:nlev], 0.0, atol=1e-15,
        err_msg="NNS should be zero for uniform salinity",
    )
    np.testing.assert_allclose(
        state.NN[1:nlev], expected_NNT, rtol=1e-10,
        err_msg="NN should equal NNT when salinity is uniform",
    )


# ---------------------------------------------------------------------------
# 5. Analytic: linear S profile → constant NNS
# ---------------------------------------------------------------------------


def test_linear_S_constant_NNS() -> None:
    r"""Linear salinity profile must yield constant N²_S = -g*β*ΔS/H.

    With freshwater increasing toward surface (haline destabilisation):
        NNS = -β * g * (delta_S/nlev) * (nlev/H) = -β * g * delta_S / H
    """
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    delta_S = 2.0  # saltier at surface (stabilising)
    S_bottom = 34.0
    for k in range(nlev + 1):
        state.S[k] = S_bottom + delta_S * k / nlev
    state.T[:] = 15.0  # uniform temperature — no NNT contribution

    _call(state, ds)

    expected_NNS = -_BETA * _GRAVITY * delta_S / depth

    assert state.NNS is not None
    assert state.NNT is not None
    assert state.NN is not None

    np.testing.assert_allclose(
        state.NNS[1:nlev], expected_NNS, rtol=1e-10,
        err_msg="NNS deviates from analytic value for linear S profile",
    )
    np.testing.assert_allclose(
        state.NNT[1:nlev], 0.0, atol=1e-15,
        err_msg="NNT should be zero for uniform temperature",
    )
    np.testing.assert_allclose(
        state.NN[1:nlev], expected_NNS, rtol=1e-10,
        err_msg="NN should equal NNS when temperature is uniform",
    )


# ---------------------------------------------------------------------------
# 6. N² = NNT + NNS decomposition
# ---------------------------------------------------------------------------


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

    np.testing.assert_allclose(
        state.NN, state.NNT + state.NNS, atol=1e-15,
        err_msg="NN must equal NNT + NNS at every level",
    )


# ---------------------------------------------------------------------------
# 7. Boundary conditions: indices 0 and nlev are zero
# ---------------------------------------------------------------------------


def test_boundary_values_zero() -> None:
    """Indices 0 and nlev of NN, NNT, NNS must always be zero."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    # Non-trivial profiles to ensure boundary zeroing is explicit
    state.T[:] = np.linspace(5.0, 20.0, nlev + 1)
    state.S[:] = np.linspace(30.0, 38.0, nlev + 1)

    _call(state, ds)

    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None

    for arr, name in [(state.NN, "NN"), (state.NNT, "NNT"), (state.NNS, "NNS")]:
        assert arr[0] == 0.0, f"{name}[0] must be zero"
        assert arr[nlev] == 0.0, f"{name}[nlev] must be zero"


# ---------------------------------------------------------------------------
# 8. Stable stratification: warm-over-cold → NN > 0
# ---------------------------------------------------------------------------


def test_stable_stratification_positive_NN() -> None:
    """Warm water over cold (stable) must give NN > 0 at interior interfaces."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    # Temperature increasing from bottom (cold) to surface (warm): stable
    state.T[:] = np.linspace(5.0, 20.0, nlev + 1)
    state.S[:] = 35.0

    _call(state, ds)

    assert state.NN is not None

    assert np.all(state.NN[1:nlev] > 0.0), "Stable stratification must give NN > 0"


# ---------------------------------------------------------------------------
# 9. Unstable stratification: cold-over-warm → NN < 0
# ---------------------------------------------------------------------------


def test_unstable_stratification_negative_NN() -> None:
    """Cold water over warm (unstable) must give NN < 0 at interior interfaces."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    # Temperature decreasing from bottom (warm) to surface (cold): unstable
    state.T[:] = np.linspace(20.0, 5.0, nlev + 1)
    state.S[:] = 35.0

    _call(state, ds)

    assert state.NN is not None

    assert np.all(state.NN[1:nlev] < 0.0), "Unstable stratification must give NN < 0"


# ---------------------------------------------------------------------------
# 10. No NaN or Inf for valid inputs
# ---------------------------------------------------------------------------


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

    assert np.all(np.isfinite(state.NN)), "NN must be finite"
    assert np.all(np.isfinite(state.NNT)), "NNT must be finite"
    assert np.all(np.isfinite(state.NNS)), "NNS must be finite"


# ---------------------------------------------------------------------------
# 11. Single layer (nlev=1)
# ---------------------------------------------------------------------------


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

    # nlev=1 → i = range(1,1) is empty; only boundary fills apply
    np.testing.assert_array_equal(state.NN, 0.0)
    np.testing.assert_array_equal(state.NNT, 0.0)
    np.testing.assert_array_equal(state.NNS, 0.0)


# ---------------------------------------------------------------------------
# 12. Large nlev stress test
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 13. Idempotency: calling twice with same inputs gives same result
# ---------------------------------------------------------------------------


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


def test_step_stratification_matches_reference_and_multicolumn_parity() -> None:
    """Kernel output must match the NumPy reference for one and two columns."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    ds = _make_density(nlev=nlev)

    assert state.T is not None
    assert state.S is not None

    state.T[:] = np.linspace(4.0, 18.0, nlev + 1)
    state.S[:] = 35.0 + 0.4 * np.sin(np.linspace(0.0, np.pi, nlev + 1))

    _call(state, ds, nlev=nlev)

    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None

    nn_ref = state.NN.copy()
    nnt_ref = state.NNT.copy()
    nns_ref = state.NNS.copy()

    ws_single = _run_step_stratification(state, ds, nlev, n_cols=1)
    np.testing.assert_allclose(read_field_array(ws_single.NN), nn_ref, rtol=1e-12)
    np.testing.assert_allclose(read_field_array(ws_single.NNT), nnt_ref, rtol=1e-12)
    np.testing.assert_allclose(read_field_array(ws_single.NNS), nns_ref, rtol=1e-12)

    ws_multi = _run_step_stratification(state, ds, nlev, n_cols=2)
    for col in range(2):
        np.testing.assert_allclose(
            read_field_array(ws_multi.NN, col=col),
            nn_ref,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            read_field_array(ws_multi.NNT, col=col),
            nnt_ref,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            read_field_array(ws_multi.NNS, col=col),
            nns_ref,
            rtol=1e-12,
        )
