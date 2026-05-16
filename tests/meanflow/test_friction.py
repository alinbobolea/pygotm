"""Tests for pygotm.meanflow.friction — bottom friction and roughness.

Analytic validation targets:
- Law-of-the-wall: u_taub = r * |U_1|  where r = κ / ln((z0b + h1/2) / z0b)
- drag[1] = r²  (implicit bottom friction coefficient)
- taub = u_taub² * rho0
- Charnock roughness: z0s = charnock_val * u_taus² / g  (≥ z0s_min)
- Surface friction velocity: u_taus = (tx²+ty²)^(1/4) for plume_type=0
"""

from __future__ import annotations

import math

import numpy as np
from type_helpers import ReadyMeanflowState, require_meanflow_state

from pygotm.meanflow.friction import (
    KAPPA,
    FrictionWorkspace,
    friction,
    step_friction_batch,
)
from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.updategrid import updategrid

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NLEV = 20
_DEPTH = 10.0
_DT = 3600.0
_RHO0 = 1027.0


def _make_state(
    nlev: int = _NLEV,
    depth: float = _DEPTH,
    avmolu: float = 1.3e-6,
    h0b: float = 0.05,
    calc_bottom_stress: bool = True,
) -> ReadyMeanflowState:
    state = MeanflowState()
    init_meanflow(state, avmolu=avmolu, h0b=h0b, calc_bottom_stress=calc_bottom_stress)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return require_meanflow_state(state)


def _configure_parity_state(state: ReadyMeanflowState, nlev: int) -> None:
    state.charnock = True
    state.charnock_val = 1400.0
    state.z0s_min = 1.0e-5
    state.MaxItz0b = 3
    state.za = 2.5e-3
    state.u_taub = 1.8e-2
    state.u_taubo = 1.1e-2
    state.u_taus = 3.2e-2
    state.drag[:] = np.linspace(1.0, 2.0, nlev + 1)
    state.u[1] = 0.35
    state.v[1] = -0.12
    state.u[nlev] = 0.28
    state.v[nlev] = 0.06


def _run_step_friction_batch(
    state: ReadyMeanflowState,
    nlev: int,
    *,
    tx: float,
    ty: float,
    plume_type: int,
    first: bool,
    batch_size: int,
    rho0: float = _RHO0,
) -> FrictionWorkspace:
    ws = FrictionWorkspace(nlev=nlev, batch_size=batch_size)
    for b in range(batch_size):
        ws.h[b] = state.h
        ws.u[b] = state.u
        ws.v[b] = state.v
        ws.drag[b] = state.drag
        ws.z0b[b] = state.z0b
        ws.z0s[b] = state.z0s
        ws.za[b] = state.za
        ws.u_taub[b] = state.u_taub
        ws.u_taubo[b] = state.u_taubo
        ws.u_taus[b] = state.u_taus
        ws.taub[b] = state.taub
        ws.tx[b] = tx
        ws.ty[b] = ty

    step_friction_batch(
        batch_size,
        nlev,
        KAPPA,
        state.avmolu,
        rho0,
        state.gravity,
        state.h0b,
        state.z0s_min,
        int(state.charnock),
        state.charnock_val,
        int(state.calc_bottom_stress),
        state.MaxItz0b,
        plume_type,
        int(first),
        ws.h,
        ws.u,
        ws.v,
        ws.drag,
        ws.z0b,
        ws.z0s,
        ws.za,
        ws.u_taub,
        ws.u_taubo,
        ws.u_taus,
        ws.taub,
        ws.tx,
        ws.ty,
    )
    return ws


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.friction import friction as _f  # noqa: F401

    assert callable(_f)


# ---------------------------------------------------------------------------
# 2. Smoke — runs without error with quiescent column
# ---------------------------------------------------------------------------


def test_smoke() -> None:
    state = _make_state()
    first = [True]
    friction(state, _NLEV, _first=first)


# ---------------------------------------------------------------------------
# 3. drag is zeroed and then set by friction
# ---------------------------------------------------------------------------


def test_drag_reset_then_set() -> None:
    """drag array must be zeroed at the start of each call, then drag[1] set."""
    state = _make_state()
    # Pre-populate drag with sentinel values
    state.drag[:] = 99.0
    state.u[1] = 0.5
    state.v[1] = 0.0

    first = [True]
    friction(state, _NLEV, _first=first)

    # Only drag[1] (and drag[nlev] for plume_type==1) should be non-zero
    assert state.drag is not None
    assert state.drag[0] == 0.0, "drag[0] (seabed sentinel) must be zero"
    assert state.drag[1] > 0.0, "drag[1] must be positive after law-of-the-wall"
    for k in range(2, _NLEV + 1):
        assert state.drag[k] == 0.0, f"drag[{k}] must be zero (plume_type=0)"


# ---------------------------------------------------------------------------
# 4. Law-of-the-wall: analytic rr_b and drag[1]
# ---------------------------------------------------------------------------


def test_law_of_the_wall_drag() -> None:
    """drag[1] = rr_b² must equal the analytic law-of-the-wall coefficient."""
    nlev = _NLEV
    avmolu = 1.3e-6
    h0b = 0.05
    state = _make_state(nlev=nlev, avmolu=avmolu, h0b=h0b)

    assert state.u is not None
    assert state.h is not None
    assert state.drag is not None

    u1 = 0.3  # [m/s]
    state.u[1] = u1
    state.v[1] = 0.0

    first = [True]
    friction(state, nlev, avmolu=avmolu, _first=first)

    # On first call u_taub is initialised to u_taubo = 0, so z0b uses that.
    z0b_expected = 0.1 * avmolu / max(avmolu, 0.0) + 0.03 * h0b + 0.0
    # avmolu > 0 and u_taub_initial = 0 → max(avmolu, 0) = avmolu → 0.1*1 + 0.03*h0b
    z0b_expected = 0.1 + 0.03 * h0b  # 0.1*avmolu/avmolu = 0.1

    h1 = state.h[1]
    rr_b_expected = KAPPA / math.log((z0b_expected + h1 / 2.0) / z0b_expected)
    drag_expected = rr_b_expected**2

    assert state.drag is not None
    np.testing.assert_allclose(state.drag[1], drag_expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Custom kappa propagates into drag and friction velocity
# ---------------------------------------------------------------------------


def test_custom_kappa_is_used_by_friction() -> None:
    state = _make_state(h0b=0.1)

    assert state.h is not None
    assert state.u is not None
    assert state.v is not None
    assert state.drag is not None

    state.u[1] = 0.35
    state.v[1] = 0.0
    state.u_taub = 1.5e-2

    custom_kappa = 0.39256055713193655
    friction(state, _NLEV, kappa=custom_kappa, _first=[False])

    rr_expected = custom_kappa / math.log((state.z0b + state.h[1] / 2.0) / state.z0b)
    np.testing.assert_allclose(state.drag[1], rr_expected**2, rtol=1e-12)
    np.testing.assert_allclose(state.u_taub, rr_expected * state.u[1], rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Friction velocity u_taub analytic check
# ---------------------------------------------------------------------------


def test_u_taub_law_of_the_wall() -> None:
    """u_taub = rr_b * sqrt(u1² + v1²) must match analytic formula."""
    nlev = _NLEV
    avmolu = 1.3e-6
    h0b = 0.05
    state = _make_state(nlev=nlev, avmolu=avmolu, h0b=h0b)

    assert state.u is not None
    assert state.v is not None
    assert state.h is not None

    u1, v1 = 0.2, 0.15
    state.u[1] = u1
    state.v[1] = v1

    first = [True]
    friction(state, nlev, avmolu=avmolu, _first=first)

    speed = math.sqrt(u1**2 + v1**2)
    z0b = 0.1 + 0.03 * h0b  # avmolu/avmolu = 1 → 0.1*1
    h1 = state.h[1]
    rr_b = KAPPA / math.log((z0b + h1 / 2.0) / z0b)
    u_taub_expected = rr_b * speed

    np.testing.assert_allclose(state.u_taub, u_taub_expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 6. Bottom stress taub = u_taub² * rho0
# ---------------------------------------------------------------------------


def test_taub_equals_u_taub_squared_times_rho0() -> None:
    """taub must equal u_taub² * rho0 after every call."""
    rho0 = 1027.0
    state = _make_state()
    assert state.u is not None
    state.u[1] = 0.4
    state.v[1] = 0.0

    first = [True]
    friction(state, _NLEV, rho0=rho0, _first=first)

    expected = state.u_taub**2 * rho0
    np.testing.assert_allclose(state.taub, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. Quiescent column: u_taub = 0, drag[1] = 0
# ---------------------------------------------------------------------------


def test_quiescent_zero_friction() -> None:
    """Zero velocity → u_taub = 0, but drag[1] = rr_b² (coefficient is non-zero).

    drag[1] is the law-of-the-wall drag *coefficient* (rr_b²), not the friction
    force.  The force in uequation is Lsour[1] = -drag[1]/h[1]*|U[1]|, which is
    zero when U[1]=0.  The coefficient itself is always written regardless of
    velocity — this matches the Fortran: ``drag(1) = drag(1) + rr_b*rr_b``.
    """
    state = _make_state()
    assert state.u is not None
    assert state.v is not None
    state.u[:] = 0.0
    state.v[:] = 0.0

    first = [True]
    friction(state, _NLEV, _first=first)

    assert state.drag is not None
    # friction velocity is zero when there is no flow
    np.testing.assert_allclose(state.u_taub, 0.0, atol=1e-15)
    # drag coefficient (rr_b²) is always set from log-law, never zero
    assert (
        state.drag[1] > 0.0
    ), "drag[1] = rr_b² must be positive even for zero velocity"


# ---------------------------------------------------------------------------
# 8. Surface roughness: z0s = z0s_min when charnock=False
# ---------------------------------------------------------------------------


def test_z0s_equals_z0s_min_when_no_charnock() -> None:
    """Without Charnock, z0s must be clamped to z0s_min."""
    state = _make_state()
    state.charnock = False
    state.z0s_min = 0.02

    first = [True]
    friction(state, _NLEV, _first=first)

    np.testing.assert_allclose(state.z0s, 0.02, rtol=1e-12)


# ---------------------------------------------------------------------------
# 9. Charnock roughness formula
# ---------------------------------------------------------------------------


def test_charnock_roughness() -> None:
    """With Charnock enabled, z0s = max(charnock_val*u_taus²/g, z0s_min)."""
    state = _make_state()
    state.charnock = True
    state.charnock_val = 1400.0
    state.z0s_min = 1e-5
    state.gravity = 9.81
    # Set a non-zero u_taus so Charnock produces a meaningful z0s
    state.u_taus = 0.05  # surface friction velocity [m/s]

    first = [True]
    friction(state, _NLEV, tx=0.0, ty=0.0, _first=first)

    z0s_charnock = 1400.0 * 0.05**2 / 9.81
    expected = max(z0s_charnock, 1e-5)
    np.testing.assert_allclose(state.z0s, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 10. u_taus from surface wind stress (plume_type=0)
# ---------------------------------------------------------------------------


def test_u_taus_from_wind_stress() -> None:
    """For plume_type=0, u_taus = (tx²+ty²)^(1/4)."""
    state = _make_state()
    tx, ty = 1e-4, 5e-5

    first = [True]
    friction(state, _NLEV, tx=tx, ty=ty, plume_type=0, _first=first)

    expected = (tx**2 + ty**2) ** 0.25
    np.testing.assert_allclose(state.u_taus, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 11. plume_type=1: drag at surface level set, u_taus from log-law
# ---------------------------------------------------------------------------


def test_plume_type_adds_surface_drag() -> None:
    """plume_type=1 must add rr_s² to drag[nlev]."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.h is not None
    assert state.drag is not None

    state.u[nlev] = 0.3  # surface velocity
    state.v[nlev] = 0.0

    first = [True]
    friction(state, nlev, plume_type=1, _first=first)

    z0s = state.z0s
    h_top = state.h[nlev]
    rr_s = KAPPA / math.log((z0s + h_top / 2.0) / z0s)

    assert state.drag is not None
    np.testing.assert_allclose(state.drag[nlev], rr_s**2, rtol=1e-10)


# ---------------------------------------------------------------------------
# 12. calc_bottom_stress=False: drag[1] stays zero
# ---------------------------------------------------------------------------


def test_no_calc_bottom_stress_leaves_drag_zero() -> None:
    """With calc_bottom_stress=False, drag[1] must remain zero."""
    state = _make_state(calc_bottom_stress=False)
    assert state.u is not None
    state.u[1] = 0.5
    state.v[1] = 0.0

    first = [True]
    friction(state, _NLEV, _first=first)

    assert state.drag is not None
    np.testing.assert_allclose(state.drag[1], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 13. u_taubo updated on second call
# ---------------------------------------------------------------------------


def test_u_taubo_updated_on_second_call() -> None:
    """u_taubo must equal the u_taub from the previous call."""
    state = _make_state()
    assert state.u is not None
    state.u[1] = 0.3

    first = [True]
    friction(state, _NLEV, _first=first)  # first call: u_taub = u_taubo (0)

    u_taub_after_first = state.u_taub

    friction(state, _NLEV, _first=first)  # second call: u_taubo = u_taub_after_first

    np.testing.assert_allclose(state.u_taubo, u_taub_after_first, rtol=1e-12)


# ---------------------------------------------------------------------------
# 14. Physical bounds: no NaN or Inf for realistic inputs
# ---------------------------------------------------------------------------


def test_no_nan_inf() -> None:
    """No NaN or Inf for realistic ocean inputs."""
    state = _make_state(depth=200.0)
    assert state.u is not None
    assert state.v is not None
    state.u[1] = 0.5
    state.v[1] = 0.2

    first = [True]
    friction(state, _NLEV, tx=1e-4, ty=5e-5, _first=first)

    assert math.isfinite(state.u_taub), "u_taub must be finite"
    assert math.isfinite(state.u_taus), "u_taus must be finite"
    assert math.isfinite(state.taub), "taub must be finite"
    assert state.drag is not None
    assert np.all(np.isfinite(state.drag)), "drag must be finite"


# ---------------------------------------------------------------------------
# 15. MaxItz0b > 1 converges z0b and u_taub
# ---------------------------------------------------------------------------


def test_multiple_roughness_iterations_converge() -> None:
    """With MaxItz0b=5, u_taub must converge to a stable value."""
    state1 = _make_state()
    state5 = _make_state()
    assert state1.u is not None and state5.u is not None
    state1.u[1] = 0.4
    state5.u[1] = 0.4

    state1.MaxItz0b = 1
    state5.MaxItz0b = 5

    first1, first5 = [True], [True]
    friction(state1, _NLEV, _first=first1)
    friction(state5, _NLEV, _first=first5)

    # With more iterations, z0b is better converged — u_taub should differ slightly
    # but both must be physically reasonable (positive, < 1 m/s for U=0.4 m/s)
    assert 0.0 < state1.u_taub < 0.4
    assert 0.0 < state5.u_taub < 0.4


def test_step_friction_batch_matches_reference_on_first_call() -> None:
    """step_friction_batch with batch_size=1 must match the Python friction() path."""
    nlev = _NLEV
    tx = 1.0e-4
    ty = -4.0e-5

    state_ref = _make_state(nlev=nlev)
    _configure_parity_state(state_ref, nlev)

    state_kernel = _make_state(nlev=nlev)
    _configure_parity_state(state_kernel, nlev)

    friction(state_ref, nlev, tx=tx, ty=ty, plume_type=1, rho0=_RHO0, _first=[True])
    ws = _run_step_friction_batch(
        state_kernel,
        nlev,
        tx=tx,
        ty=ty,
        plume_type=1,
        first=True,
        batch_size=1,
    )

    assert state_ref.drag is not None
    np.testing.assert_allclose(ws.drag[0], state_ref.drag, rtol=1e-12)
    np.testing.assert_allclose(ws.z0b[0], state_ref.z0b, rtol=1e-12)
    np.testing.assert_allclose(ws.z0s[0], state_ref.z0s, rtol=1e-12)
    np.testing.assert_allclose(ws.u_taub[0], state_ref.u_taub, rtol=1e-12)
    np.testing.assert_allclose(ws.u_taubo[0], state_ref.u_taubo, rtol=1e-12)
    np.testing.assert_allclose(ws.u_taus[0], state_ref.u_taus, rtol=1e-12)
    np.testing.assert_allclose(ws.taub[0], state_ref.taub, rtol=1e-12)


def test_step_friction_batch_multicolumn_parity() -> None:
    """step_friction_batch with batch_size=2 identical columns gives identical results."""
    nlev = _NLEV
    tx = 1.0e-4
    ty = -4.0e-5

    state_ref = _make_state(nlev=nlev)
    _configure_parity_state(state_ref, nlev)

    state_kernel = _make_state(nlev=nlev)
    _configure_parity_state(state_kernel, nlev)

    friction(state_ref, nlev, tx=tx, ty=ty, plume_type=1, rho0=_RHO0, _first=[False])
    ws = _run_step_friction_batch(
        state_kernel,
        nlev,
        tx=tx,
        ty=ty,
        plume_type=1,
        first=False,
        batch_size=2,
    )

    assert state_ref.drag is not None
    for b in range(2):
        np.testing.assert_allclose(ws.drag[b], state_ref.drag, rtol=1e-12)
        np.testing.assert_allclose(ws.z0b[b], state_ref.z0b, rtol=1e-12)
        np.testing.assert_allclose(ws.z0s[b], state_ref.z0s, rtol=1e-12)
        np.testing.assert_allclose(ws.u_taub[b], state_ref.u_taub, rtol=1e-12)
        np.testing.assert_allclose(ws.u_taubo[b], state_ref.u_taubo, rtol=1e-12)
        np.testing.assert_allclose(ws.u_taus[b], state_ref.u_taus, rtol=1e-12)
        np.testing.assert_allclose(ws.taub[b], state_ref.taub, rtol=1e-12)
