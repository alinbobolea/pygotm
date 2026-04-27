"""Tests for pygotm.meanflow.shear — shear-frequency squared (M²)."""

from __future__ import annotations

import numpy as np
from type_helpers import ReadyMeanflowState, require_meanflow_state

from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.shear import shear
from pygotm.meanflow.updategrid import updategrid

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_NLEV = 10
_DEPTH = 25.0
_DT = 3600.0
_CNPAR = 0.5


def _make_state(nlev: int = _NLEV, depth: float = _DEPTH) -> ReadyMeanflowState:
    state = MeanflowState()
    init_meanflow(state)
    state.depth = depth
    state.grid_method = 0  # equidistant — no file I/O
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return require_meanflow_state(state)


def _call(
    state: ReadyMeanflowState,
    nlev: int = _NLEV,
    cnpar: float = _CNPAR,
    dusdz: np.ndarray | None = None,
    dvsdz: np.ndarray | None = None,
) -> None:
    shear(state, nlev, cnpar, dusdz=dusdz, dvsdz=dvsdz)


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.shear import shear as _s  # noqa: F401

    assert callable(_s)


# ---------------------------------------------------------------------------
# 2. Smoke test
# ---------------------------------------------------------------------------


def test_smoke() -> None:
    state = _make_state()
    _call(state)


# ---------------------------------------------------------------------------
# 3. Zero velocity → zero shear
# ---------------------------------------------------------------------------


def test_zero_velocity_gives_zero_shear() -> None:
    """All velocity arrays zero → SS, SSU, SSV all zero."""
    state = _make_state()
    # u, v, uo, vo are already zero from post_init_meanflow
    _call(state)
    assert np.all(state.SS == 0.0)
    assert np.all(state.SSU == 0.0)
    assert np.all(state.SSV == 0.0)


def test_zero_velocity_gives_zero_stokes_terms() -> None:
    state = _make_state()
    _call(state)
    assert np.all(state.SSCSTK == 0.0)
    assert np.all(state.SSSTK == 0.0)


# ---------------------------------------------------------------------------
# 4. Analytic verification: steady-state linear velocity profile
#    u[k] = U0 * k/nlev, uo == u → SSU = (U0/depth)²  for all interior i
# ---------------------------------------------------------------------------


def test_linear_u_steady_state_gives_uniform_SSU() -> None:
    """Linear U with u == uo (no time tendency) → SSU = (dU/dz)² = (U0/H)²."""
    nlev = 20
    depth = 100.0
    U0 = 1.0
    state = _make_state(nlev=nlev, depth=depth)

    # Uniform layer thickness after updategrid with equidistant grid
    h_val = depth / nlev
    assert np.allclose(state.h[1 : nlev + 1], h_val, rtol=1e-12)

    # Linear profile: u[k] = U0 * k / nlev
    k = np.arange(nlev + 1, dtype=float)
    state.u[:] = U0 * k / nlev
    state.uo[:] = state.u.copy()  # steady state: old == new
    # v = 0
    state.v[:] = 0.0
    state.vo[:] = 0.0

    _call(state, nlev=nlev)

    expected_ssu = (U0 / depth) ** 2
    # Interior interfaces i = 1 .. nlev-1
    assert np.allclose(state.SSU[1:nlev], expected_ssu, rtol=1e-10)
    assert np.allclose(state.SSV[1:nlev], 0.0, atol=1e-30)
    assert np.allclose(state.SS[1:nlev], expected_ssu, rtol=1e-10)


def test_linear_v_steady_state_gives_uniform_SSV() -> None:
    """Linear V with v == vo → SSV = (V0/H)² for all interior interfaces."""
    nlev = 20
    depth = 50.0
    V0 = 0.5
    state = _make_state(nlev=nlev, depth=depth)

    k = np.arange(nlev + 1, dtype=float)
    state.v[:] = V0 * k / nlev
    state.vo[:] = state.v.copy()
    state.u[:] = 0.0
    state.uo[:] = 0.0

    _call(state, nlev=nlev)

    expected_ssv = (V0 / depth) ** 2
    assert np.allclose(state.SSV[1:nlev], expected_ssv, rtol=1e-10)
    assert np.allclose(state.SSU[1:nlev], 0.0, atol=1e-30)


def test_combined_uv_steady_state() -> None:
    """Simultaneous linear U and V: SS = SSU + SSV."""
    nlev = 10
    depth = 30.0
    U0 = 1.0
    V0 = 0.5
    state = _make_state(nlev=nlev, depth=depth)

    k = np.arange(nlev + 1, dtype=float)
    state.u[:] = U0 * k / nlev
    state.uo[:] = state.u.copy()
    state.v[:] = V0 * k / nlev
    state.vo[:] = state.v.copy()

    _call(state, nlev=nlev)

    expected_ss = (U0 / depth) ** 2 + (V0 / depth) ** 2
    assert np.allclose(state.SS[1:nlev], expected_ss, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Boundary fill
# ---------------------------------------------------------------------------


def test_boundary_fill_index_zero() -> None:
    """Index-0 values must equal index-1 (seabed ghost cell fill)."""
    state = _make_state()
    k = np.arange(_NLEV + 1, dtype=float)
    state.u[:] = k / _NLEV
    state.uo[:] = state.u.copy()
    _call(state)

    assert state.SSU[0] == state.SSU[1]
    assert state.SSV[0] == state.SSV[1]
    assert state.SS[0] == state.SS[1]
    assert state.SSCSTK[0] == state.SSCSTK[1]
    assert state.SSSTK[0] == state.SSSTK[1]


def test_boundary_fill_index_nlev() -> None:
    """Index-nlev values must equal index-(nlev-1) (surface ghost cell fill)."""
    nlev = _NLEV
    state = _make_state()
    k = np.arange(nlev + 1, dtype=float)
    state.u[:] = k / nlev
    state.uo[:] = state.u.copy()
    _call(state, nlev=nlev)

    assert state.SSU[nlev] == state.SSU[nlev - 1]
    assert state.SSV[nlev] == state.SSV[nlev - 1]
    assert state.SS[nlev] == state.SS[nlev - 1]
    assert state.SSCSTK[nlev] == state.SSCSTK[nlev - 1]
    assert state.SSSTK[nlev] == state.SSSTK[nlev - 1]


# ---------------------------------------------------------------------------
# 6. Stokes drift terms
# ---------------------------------------------------------------------------


def test_stokes_cross_shear_analytic() -> None:
    """SSCSTK = dusdz * dU/dz for pure U shear and Stokes in x only."""
    nlev = 10
    depth = 20.0
    U0 = 1.0
    state = _make_state(nlev=nlev, depth=depth)

    k = np.arange(nlev + 1, dtype=float)
    state.u[:] = U0 * k / nlev
    state.uo[:] = state.u.copy()
    state.v[:] = 0.0
    state.vo[:] = 0.0

    # Constant Stokes drift shear in x-direction
    us_shear = 0.1  # s⁻¹
    dusdz = np.full(nlev + 1, us_shear)
    dvsdz = np.zeros(nlev + 1)

    _call(state, nlev=nlev, dusdz=dusdz, dvsdz=dvsdz)

    # SSCSTK[i] = dusdz[i] * (u[i+1]-u[i]) / h_mid
    #            = us_shear * (U0/nlev) / (depth/nlev)
    #            = us_shear * U0/depth
    expected = us_shear * U0 / depth
    assert np.allclose(state.SSCSTK[1:nlev], expected, rtol=1e-10)


def test_stokes_shear_squared_analytic() -> None:
    """SSSTK = dusdz² + dvsdz² at each interface."""
    nlev = 8
    state = _make_state(nlev=nlev)

    us = 0.2
    vs = 0.3
    dusdz = np.full(nlev + 1, us)
    dvsdz = np.full(nlev + 1, vs)

    _call(state, nlev=nlev, dusdz=dusdz, dvsdz=dvsdz)

    expected = us**2 + vs**2
    assert np.allclose(state.SSSTK[1:nlev], expected, rtol=1e-10)


def test_stokes_default_none_equals_zero_arrays() -> None:
    """Passing dusdz=None must be identical to passing zero arrays."""
    nlev = 10
    state1 = _make_state(nlev=nlev)
    state2 = _make_state(nlev=nlev)

    k = np.arange(nlev + 1, dtype=float)
    for s in (state1, state2):
        s.u[:] = k / nlev
        s.uo[:] = s.u.copy()

    shear(state1, nlev, _CNPAR, dusdz=None, dvsdz=None)
    shear(state2, nlev, _CNPAR, dusdz=np.zeros(nlev + 1), dvsdz=np.zeros(nlev + 1))

    assert np.array_equal(state1.SSCSTK, state2.SSCSTK)
    assert np.array_equal(state1.SSSTK, state2.SSSTK)


# ---------------------------------------------------------------------------
# 7. cnpar sensitivity
# ---------------------------------------------------------------------------


def test_cnpar_fully_explicit() -> None:
    """cnpar=0 (fully explicit) must not crash and give finite results."""
    state = _make_state()
    k = np.arange(_NLEV + 1, dtype=float)
    state.u[:] = k / _NLEV
    state.uo[:] = state.u.copy()
    _call(state, cnpar=0.0)
    assert np.all(np.isfinite(state.SS))


def test_cnpar_fully_implicit() -> None:
    """cnpar=1 (fully implicit) must not crash and give finite results."""
    state = _make_state()
    k = np.arange(_NLEV + 1, dtype=float)
    state.u[:] = k / _NLEV
    state.uo[:] = state.u.copy()
    _call(state, cnpar=1.0)
    assert np.all(np.isfinite(state.SS))


def test_cnpar_steady_state_invariant() -> None:
    """For u == uo, SSU must be independent of cnpar."""
    nlev = 20
    depth = 40.0
    U0 = 1.0
    state_05 = _make_state(nlev=nlev, depth=depth)
    state_10 = _make_state(nlev=nlev, depth=depth)

    k = np.arange(nlev + 1, dtype=float)
    for s in (state_05, state_10):
        s.u[:] = U0 * k / nlev
        s.uo[:] = s.u.copy()

    shear(state_05, nlev, 0.5)
    shear(state_10, nlev, 1.0)

    assert np.allclose(state_05.SSU, state_10.SSU, rtol=1e-12)


# ---------------------------------------------------------------------------
# 8. Physical bounds — steady-state shear is non-negative
# ---------------------------------------------------------------------------


def test_SS_nonnegative_steady_state() -> None:
    """For u == uo and v == vo, SS[i] >= 0 at all interfaces."""
    nlev = 15
    state = _make_state(nlev=nlev)
    rng = np.random.default_rng(42)
    state.u[:] = rng.uniform(-1.0, 1.0, nlev + 1)
    state.uo[:] = state.u.copy()
    state.v[:] = rng.uniform(-1.0, 1.0, nlev + 1)
    state.vo[:] = state.v.copy()

    _call(state, nlev=nlev)

    # For u == uo, SSU = (u[i+1]-u[i])^2 / h_mid^2 * scaling >= 0
    assert np.all(state.SS >= 0.0)
    assert np.all(state.SSU >= 0.0)
    assert np.all(state.SSV >= 0.0)


def test_SSSTK_nonnegative() -> None:
    """SSSTK = dusdz² + dvsdz² is always non-negative."""
    nlev = 10
    state = _make_state(nlev=nlev)
    rng = np.random.default_rng(7)
    dusdz = rng.uniform(-0.5, 0.5, nlev + 1)
    dvsdz = rng.uniform(-0.5, 0.5, nlev + 1)
    _call(state, nlev=nlev, dusdz=dusdz, dvsdz=dvsdz)
    assert np.all(state.SSSTK >= 0.0)


# ---------------------------------------------------------------------------
# 9. NaN / Inf guard
# ---------------------------------------------------------------------------


def test_no_nan_inf_random_inputs() -> None:
    """No NaN or Inf for a set of random but physically valid velocity inputs."""
    nlev = 30
    state = _make_state(nlev=nlev, depth=50.0)
    rng = np.random.default_rng(99)
    state.u[:] = rng.uniform(-2.0, 2.0, nlev + 1)
    state.uo[:] = rng.uniform(-2.0, 2.0, nlev + 1)
    state.v[:] = rng.uniform(-2.0, 2.0, nlev + 1)
    state.vo[:] = rng.uniform(-2.0, 2.0, nlev + 1)
    dusdz = rng.uniform(-0.1, 0.1, nlev + 1)
    dvsdz = rng.uniform(-0.1, 0.1, nlev + 1)

    _call(state, nlev=nlev, dusdz=dusdz, dvsdz=dvsdz)

    assert np.all(np.isfinite(state.SS))
    assert np.all(np.isfinite(state.SSU))
    assert np.all(np.isfinite(state.SSV))
    assert np.all(np.isfinite(state.SSCSTK))
    assert np.all(np.isfinite(state.SSSTK))


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


def test_single_layer() -> None:
    """nlev=1: interior loop is empty; boundary fill must not error."""
    nlev = 1
    depth = 10.0
    state = _make_state(nlev=nlev, depth=depth)
    state.u[0] = 0.0
    state.u[1] = 1.0
    state.uo[:] = state.u.copy()
    _call(state, nlev=nlev)
    # With nlev=1 the interior loop (range(1,1)) is empty;
    # boundary fill copies SSU[1] (untouched=0) to SSU[0] and SSU[1]
    assert np.all(np.isfinite(state.SS))


def test_uniform_velocity_gives_zero_shear() -> None:
    """Depth-uniform velocity (no gradient) → zero M²."""
    nlev = 10
    state = _make_state(nlev=nlev)
    state.u[:] = 1.5  # constant in depth
    state.uo[:] = 1.5
    state.v[:] = 0.3
    state.vo[:] = 0.3
    _call(state)
    assert np.allclose(state.SS, 0.0, atol=1e-30)


def test_large_nlev() -> None:
    """nlev=100 (typical GOTM resolution) must complete without error."""
    nlev = 100
    state = _make_state(nlev=nlev, depth=100.0)
    rng = np.random.default_rng(1)
    state.u[:] = rng.uniform(-1, 1, nlev + 1)
    state.uo[:] = rng.uniform(-1, 1, nlev + 1)
    _call(state, nlev=nlev)
    assert np.all(np.isfinite(state.SS))
