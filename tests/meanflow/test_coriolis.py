"""Tests for pygotm.meanflow.coriolis — Coriolis rotation.

Analytic validation targets:
  - Speed invariance: ||(u, v)|| is preserved under rotation.
  - Zero Coriolis (cori=0): no change in velocities.
  - Quarter-period rotation (omega = pi/2): (1, 0) -> (0, -1).
  - Half-period rotation (omega = pi): (1, 0) -> (-1, 0).
  - Full-period rotation (omega = 2*pi): (1, 0) -> (1, 0) (identity).
  - Stokes drift removed after rotation (Eulerian output only).
  - Layer 0 (k=0) is never modified.

Tests verify all 8 AGENTS.md requirements:
  1. Import without error
  2. Smoke test (valid inputs)
  3. Physical bounds (speed preserved)
  4. Analytic verification (rotation matrix, rtol<=1e-10)
  5. Boundary condition k=0 untouched
  6. Edge cases (zero velocity, zero Coriolis, Stokes drift)
  7. Multi-column parity (step_coriolis n_cols=1 == coriolis)
  8. No NaN/Inf for valid inputs
"""

from __future__ import annotations

import math

import numpy as np

from pygotm.meanflow.coriolis import coriolis, step_coriolis_batch
from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.updategrid import updategrid

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NLEV = 10
_DEPTH = 20.0
_DT = 3600.0


def _make_state(
    nlev: int = _NLEV,
    depth: float = _DEPTH,
    cori: float = 0.0,
) -> MeanflowState:
    state = MeanflowState()
    init_meanflow(state)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    # Override cori directly (latitude=0 gives cori=0 from post_init)
    state.cori = cori
    updategrid(state, nlev, _DT, zeta=0.0)
    return state


def _set_uniform_velocity(state: MeanflowState, u: float, v: float) -> None:
    """Set uniform (u, v) across all layers 1..nlev."""
    assert state.u is not None
    assert state.v is not None
    nlev = len(state.u) - 1
    state.u[1 : nlev + 1] = u
    state.v[1 : nlev + 1] = v


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.coriolis import coriolis as _c  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Smoke test
# ---------------------------------------------------------------------------


def test_smoke_zero_coriolis() -> None:
    state = _make_state(cori=0.0)
    _set_uniform_velocity(state, u=0.5, v=0.3)
    coriolis(state, _NLEV, _DT)


def test_smoke_nonzero_coriolis() -> None:
    state = _make_state(cori=1e-4)
    _set_uniform_velocity(state, u=0.5, v=0.3)
    coriolis(state, _NLEV, _DT)


# ---------------------------------------------------------------------------
# 3. Physical bounds — speed preserved under rotation
# ---------------------------------------------------------------------------


def test_speed_preserved_uniform() -> None:
    """||u||^2 + ||v||^2 is conserved under rotation (rotation is isometric)."""
    nlev = _NLEV
    state = _make_state(cori=1e-4)
    assert state.u is not None
    assert state.v is not None
    rng = np.random.default_rng(42)
    state.u[1 : nlev + 1] = rng.uniform(-1.0, 1.0, nlev)
    state.v[1 : nlev + 1] = rng.uniform(-1.0, 1.0, nlev)

    speed_before = np.sqrt(state.u[1 : nlev + 1] ** 2 + state.v[1 : nlev + 1] ** 2)
    coriolis(state, nlev, _DT)
    speed_after = np.sqrt(state.u[1 : nlev + 1] ** 2 + state.v[1 : nlev + 1] ** 2)

    np.testing.assert_allclose(speed_after, speed_before, rtol=1e-14)


def test_speed_preserved_multiple_steps() -> None:
    """Speed is preserved over many time steps."""
    nlev = _NLEV
    state = _make_state(cori=7.3e-5)
    _set_uniform_velocity(state, u=1.0, v=0.0)

    speed_0 = math.sqrt(1.0)
    for _ in range(100):
        coriolis(state, nlev, _DT)

    assert state.u is not None
    assert state.v is not None
    speed_final = np.sqrt(state.u[1 : nlev + 1] ** 2 + state.v[1 : nlev + 1] ** 2)
    np.testing.assert_allclose(speed_final, speed_0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Analytic verification — rotation matrix
# ---------------------------------------------------------------------------


def test_zero_coriolis_no_change() -> None:
    """cori=0 => omega=0 => identity rotation => velocities unchanged."""
    nlev = _NLEV
    state = _make_state(cori=0.0)
    u0, v0 = 0.7, -0.3
    _set_uniform_velocity(state, u=u0, v=v0)
    coriolis(state, nlev, _DT)

    assert state.u is not None
    assert state.v is not None
    np.testing.assert_allclose(state.u[1 : nlev + 1], u0, rtol=1e-15)
    np.testing.assert_allclose(state.v[1 : nlev + 1], v0, rtol=1e-15)


def test_quarter_period_rotation() -> None:
    """omega = pi/2 rotates (1, 0) to (0, -1).

    Rotation matrix:
        [cos  sin] [1]   [0]
        [-sin cos] [0] = [-1]
    """
    nlev = _NLEV
    # Set cori*dt = pi/2 exactly
    dt = 1.0
    cori = math.pi / 2.0
    state = _make_state(cori=cori)
    _set_uniform_velocity(state, u=1.0, v=0.0)
    coriolis(state, nlev, dt)

    assert state.u is not None
    assert state.v is not None
    np.testing.assert_allclose(state.u[1 : nlev + 1], 0.0, atol=1e-15)
    np.testing.assert_allclose(state.v[1 : nlev + 1], -1.0, rtol=1e-15)


def test_half_period_rotation() -> None:
    """omega = pi rotates (1, 0) to (-1, 0)."""
    nlev = _NLEV
    dt = 1.0
    cori = math.pi
    state = _make_state(cori=cori)
    _set_uniform_velocity(state, u=1.0, v=0.0)
    coriolis(state, nlev, dt)

    assert state.u is not None
    assert state.v is not None
    np.testing.assert_allclose(state.u[1 : nlev + 1], -1.0, rtol=1e-15)
    np.testing.assert_allclose(state.v[1 : nlev + 1], 0.0, atol=1e-15)


def test_full_period_rotation() -> None:
    """omega = 2*pi rotates back to the original vector."""
    nlev = _NLEV
    dt = 1.0
    cori = 2.0 * math.pi
    state = _make_state(cori=cori)
    u0, v0 = 0.8, -0.6
    _set_uniform_velocity(state, u=u0, v=v0)
    coriolis(state, nlev, dt)

    assert state.u is not None
    assert state.v is not None
    np.testing.assert_allclose(state.u[1 : nlev + 1], u0, rtol=1e-14)
    np.testing.assert_allclose(state.v[1 : nlev + 1], v0, rtol=1e-14)


def test_rotation_general_angle() -> None:
    """Check rotation against direct numpy computation for an arbitrary angle."""
    nlev = _NLEV
    dt = 1.0
    cori = 1.23456
    omega = cori * dt
    cos_w = math.cos(omega)
    sin_w = math.sin(omega)
    u0, v0 = 0.4, -0.9

    state = _make_state(cori=cori)
    _set_uniform_velocity(state, u=u0, v=v0)
    coriolis(state, nlev, dt)

    expected_u = u0 * cos_w + v0 * sin_w
    expected_v = -u0 * sin_w + v0 * cos_w

    assert state.u is not None
    assert state.v is not None
    np.testing.assert_allclose(state.u[1 : nlev + 1], expected_u, rtol=1e-15)
    np.testing.assert_allclose(state.v[1 : nlev + 1], expected_v, rtol=1e-15)


def test_inertial_oscillation_period() -> None:
    """After n steps of dt each with omega = 2*pi/n, velocity returns to origin.

    This verifies the inertial oscillation frequency: T_inertial = 2*pi / |f|.
    """
    nlev = 5
    n_steps = 360  # 360 steps of 1 degree each
    dt = 1.0
    cori = 2.0 * math.pi / n_steps  # one full rotation in n_steps

    state = _make_state(nlev=nlev, cori=cori)
    u0, v0 = 1.0, 0.0
    _set_uniform_velocity(state, u=u0, v=v0)

    for _ in range(n_steps):
        coriolis(state, nlev, dt)

    assert state.u is not None
    assert state.v is not None
    # Accumulated floating-point error over 360 steps is small but not 1e-15
    np.testing.assert_allclose(state.u[1 : nlev + 1], u0, atol=1e-10)
    np.testing.assert_allclose(state.v[1 : nlev + 1], v0, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. Boundary conditions — k=0 never modified
# ---------------------------------------------------------------------------


def test_k0_not_modified() -> None:
    """Layer k=0 (seabed interface) must remain zero (Coriolis only touches 1..nlev)."""
    state = _make_state(cori=1e-4)
    _set_uniform_velocity(state, u=1.0, v=1.0)
    assert state.u is not None
    assert state.v is not None
    state.u[0] = 99.0  # sentinel
    state.v[0] = 99.0
    coriolis(state, _NLEV, _DT)
    assert state.u[0] == 99.0
    assert state.v[0] == 99.0


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


def test_zero_velocity() -> None:
    """(0, 0) remains (0, 0) after rotation."""
    state = _make_state(cori=1e-4)
    _set_uniform_velocity(state, u=0.0, v=0.0)
    coriolis(state, _NLEV, _DT)

    assert state.u is not None
    assert state.v is not None
    np.testing.assert_allclose(state.u, 0.0, atol=1e-30)
    np.testing.assert_allclose(state.v, 0.0, atol=1e-30)


def test_stokes_drift_removed_from_output() -> None:
    """Stokes drift is added before rotation and removed after; the Eulerian
    velocity change mirrors the pure-Eulerian case when usprof=vsprof=0."""
    nlev = _NLEV
    dt = 1.0
    cori = math.pi / 4.0  # 45-degree rotation

    # Case 1: no Stokes drift
    state1 = _make_state(cori=cori)
    _set_uniform_velocity(state1, u=1.0, v=0.0)
    coriolis(state1, nlev, dt)

    # Case 2: uniform Stokes drift that is exactly equal to (u, v)
    # The rotation acts on Lagrangian velocity = (2, 0), giving (sqrt(2), -sqrt(2))
    # Then subtract Stokes (1, 0): Eulerian = (sqrt(2)-1, -sqrt(2))
    state2 = _make_state(cori=cori)
    _set_uniform_velocity(state2, u=1.0, v=0.0)
    usprof = np.ones(nlev + 1)  # Stokes drift = 1 m/s in x
    vsprof = np.zeros(nlev + 1)
    coriolis(state2, nlev, dt, usprof=usprof, vsprof=vsprof)

    cos_w = math.cos(math.pi / 4.0)
    sin_w = math.sin(math.pi / 4.0)

    assert state2.u is not None
    assert state2.v is not None
    # Lagrangian (2, 0) -> (2*cos, -2*sin) -> Eulerian = (2*cos-1, -2*sin)
    expected_u = 2.0 * cos_w - 1.0
    expected_v = -2.0 * sin_w
    np.testing.assert_allclose(state2.u[1 : nlev + 1], expected_u, rtol=1e-14)
    np.testing.assert_allclose(state2.v[1 : nlev + 1], expected_v, rtol=1e-14)


def test_stokes_drift_zero_equals_no_stokes() -> None:
    """Passing zero Stokes drift gives identical result to omitting it."""
    nlev = _NLEV
    dt = 1.0
    cori = 0.5

    state1 = _make_state(cori=cori)
    _set_uniform_velocity(state1, u=0.3, v=-0.7)
    coriolis(state1, nlev, dt)

    state2 = _make_state(cori=cori)
    _set_uniform_velocity(state2, u=0.3, v=-0.7)
    zeros = np.zeros(nlev + 1)
    coriolis(state2, nlev, dt, usprof=zeros, vsprof=zeros)

    assert state1.u is not None
    assert state1.v is not None
    assert state2.u is not None
    assert state2.v is not None
    np.testing.assert_array_equal(state1.u, state2.u)
    np.testing.assert_array_equal(state1.v, state2.v)


def test_varying_velocity_profile() -> None:
    """Non-uniform velocity profile: each layer rotates independently."""
    nlev = 5
    dt = 1.0
    cori = math.pi / 3.0  # 60-degree rotation
    cos_w = math.cos(cori * dt)
    sin_w = math.sin(cori * dt)

    state = _make_state(nlev=nlev, cori=cori)
    assert state.u is not None
    assert state.v is not None
    u_init = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    v_init = np.array([0.0, 0.0, 1.0, -1.0, 0.5, -0.5])
    state.u[:] = u_init
    state.v[:] = v_init

    coriolis(state, nlev, dt)

    # k=0 unchanged
    assert state.u[0] == u_init[0]
    assert state.v[0] == v_init[0]

    # Layers 1..nlev rotate exactly
    for i in range(1, nlev + 1):
        expected_u = u_init[i] * cos_w + v_init[i] * sin_w
        expected_v = -u_init[i] * sin_w + v_init[i] * cos_w
        assert abs(state.u[i] - expected_u) < 1e-14, f"layer {i} u mismatch"
        assert abs(state.v[i] - expected_v) < 1e-14, f"layer {i} v mismatch"


# ---------------------------------------------------------------------------
# 8. NaN/Inf guard
# ---------------------------------------------------------------------------


def test_no_nan_inf_typical() -> None:
    """No NaN or Inf for typical mid-latitude inputs."""
    nlev = 20
    state = _make_state(nlev=nlev, cori=7.3e-5)
    rng = np.random.default_rng(7)
    assert state.u is not None
    assert state.v is not None
    state.u[1 : nlev + 1] = rng.uniform(-2.0, 2.0, nlev)
    state.v[1 : nlev + 1] = rng.uniform(-2.0, 2.0, nlev)
    coriolis(state, nlev, _DT)
    assert np.all(np.isfinite(state.u[1 : nlev + 1]))
    assert np.all(np.isfinite(state.v[1 : nlev + 1]))


def test_no_nan_inf_large_coriolis() -> None:
    """No NaN or Inf even for artificially large Coriolis parameter."""
    state = _make_state(cori=1.0)  # unphysically large but numerically safe
    _set_uniform_velocity(state, u=1.0, v=0.0)
    coriolis(state, _NLEV, _DT)
    assert state.u is not None
    assert state.v is not None
    assert np.all(np.isfinite(state.u))
    assert np.all(np.isfinite(state.v))


# ---------------------------------------------------------------------------
# 7. Batch parity — step_coriolis_batch == coriolis (single column)
# ---------------------------------------------------------------------------


def test_batch_single_col_matches_numpy() -> None:
    """step_coriolis_batch with batch_size=1 matches coriolis()."""
    nlev = _NLEV
    dt = _DT
    cori_val = 7.3e-5
    n_cols = 1

    state_np = _make_state(cori=cori_val)
    rng = np.random.default_rng(99)
    u_init = np.zeros(nlev + 1)
    v_init = np.zeros(nlev + 1)
    u_init[1 : nlev + 1] = rng.uniform(-1.0, 1.0, nlev)
    v_init[1 : nlev + 1] = rng.uniform(-1.0, 1.0, nlev)
    assert state_np.u is not None
    assert state_np.v is not None
    state_np.u[:] = u_init
    state_np.v[:] = v_init
    coriolis(state_np, nlev, dt)

    omega = cori_val * dt
    u_b = np.tile(u_init, (n_cols, 1))
    v_b = np.tile(v_init, (n_cols, 1))
    usprof_b = np.zeros((n_cols, nlev + 1))
    vsprof_b = np.zeros((n_cols, nlev + 1))

    step_coriolis_batch(
        n_cols,
        nlev,
        math.cos(omega),
        math.sin(omega),
        u_b,
        v_b,
        usprof_b,
        vsprof_b,
    )

    assert state_np.u is not None
    assert state_np.v is not None
    np.testing.assert_allclose(
        u_b[0, 1 : nlev + 1], state_np.u[1 : nlev + 1], rtol=1e-14
    )
    np.testing.assert_allclose(
        v_b[0, 1 : nlev + 1], state_np.v[1 : nlev + 1], rtol=1e-14
    )


def test_batch_multi_col_uniform() -> None:
    """step_coriolis_batch with batch_size=3 identical columns gives identical results."""
    nlev = _NLEV
    dt = _DT
    cori_val = 1e-4
    n_cols = 3

    rng = np.random.default_rng(17)
    u_init = np.zeros(nlev + 1)
    v_init = np.zeros(nlev + 1)
    u_init[1 : nlev + 1] = rng.uniform(-1.0, 1.0, nlev)
    v_init[1 : nlev + 1] = rng.uniform(-1.0, 1.0, nlev)

    omega = cori_val * dt
    u_b = np.tile(u_init, (n_cols, 1))
    v_b = np.tile(v_init, (n_cols, 1))
    usprof_b = np.zeros((n_cols, nlev + 1))
    vsprof_b = np.zeros((n_cols, nlev + 1))

    step_coriolis_batch(
        n_cols,
        nlev,
        math.cos(omega),
        math.sin(omega),
        u_b,
        v_b,
        usprof_b,
        vsprof_b,
    )

    for col in range(1, n_cols):
        np.testing.assert_array_equal(u_b[col], u_b[0])
        np.testing.assert_array_equal(v_b[col], v_b[0])
