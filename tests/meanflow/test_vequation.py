"""Tests for pygotm.meanflow.vequation — V-momentum equation.

Analytic validation target: Couette flow in the y-direction.
With no pressure gradient, no Coriolis, and a steady wind stress ty applied
at the surface over a quiescent water column, the Crank-Nicolson implicit
diffusion converges toward a linear velocity profile in the steady state:

    V(z) = ty / (avmolu + num) * (z + depth)

Tests verify:
  - Correct Neumann upper BC (wind stress applied as surface flux)
  - Correct Neumann lower BC (bottom friction treated as implicit source)
  - Convergence to analytic Couette profile
  - Mass conservation (column-mean momentum change from surface forcing)
  - No NaN/Inf for valid inputs
  - Surface plume friction source term at k=nlev
"""

from __future__ import annotations

import numpy as np
from type_helpers import ReadyMeanflowState, require_meanflow_state

from pygotm.meanflow.friction import friction
from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.updategrid import updategrid
from pygotm.meanflow.vequation import VEquationWorkspace, step_vequation, vequation

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_NLEV = 20
_DEPTH = 10.0
_DT = 3600.0
_CNPAR = 0.6


def _make_state(
    nlev: int = _NLEV,
    depth: float = _DEPTH,
    avmolu: float = 1.3e-6,
) -> ReadyMeanflowState:
    state = MeanflowState()
    init_meanflow(state, avmolu=avmolu)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return require_meanflow_state(state)


def _zeros(nlev: int = _NLEV) -> np.ndarray:
    return np.zeros(nlev + 1)


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------


def test_import() -> None:
    from pygotm.meanflow.vequation import vequation as _v  # noqa: F401

    assert callable(_v)


# ---------------------------------------------------------------------------
# 2. Smoke — runs without error with all-zero inputs
# ---------------------------------------------------------------------------


def test_smoke() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    num = _zeros(nlev)
    nucl = _zeros(nlev)
    gamv = _zeros(nlev)
    vequation(state, nlev, _DT, _CNPAR, ty=0.0, num=num, nucl=nucl, gamv=gamv)


# ---------------------------------------------------------------------------
# 3. vo is saved before v is updated
# ---------------------------------------------------------------------------


def test_vo_saves_old_v() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None
    assert state.vo is not None
    state.v[:] = np.linspace(0.0, 0.5, nlev + 1)
    v_before = state.v.copy()
    vequation(
        state,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )
    np.testing.assert_array_equal(state.vo, v_before)


# ---------------------------------------------------------------------------
# 4. avh = num + avmolu written into state
# ---------------------------------------------------------------------------


def test_avh_equals_num_plus_avmolu() -> None:
    nlev = _NLEV
    avmolu = 1.3e-6
    state = _make_state(nlev=nlev, avmolu=avmolu)
    rng = np.random.default_rng(0)
    num = np.abs(rng.uniform(1e-5, 1e-3, nlev + 1))
    vequation(
        state, nlev, _DT, _CNPAR, ty=0.0, num=num, nucl=_zeros(nlev), gamv=_zeros(nlev)
    )
    assert state.avh is not None
    np.testing.assert_allclose(state.avh, num + avmolu, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. No wind, no friction, quiescent — V remains zero
# ---------------------------------------------------------------------------


def test_quiescent_no_forcing() -> None:
    """Zero wind, no drag, zero initial V → V stays zero."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None
    assert state.drag is not None
    state.v[:] = 0.0
    state.drag[:] = 0.0
    vequation(
        state,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )
    np.testing.assert_allclose(state.v[1 : nlev + 1], 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# 6. Physical bounds — surface stress drives positive V
# ---------------------------------------------------------------------------


def test_surface_stress_accelerates_v() -> None:
    """Positive wind stress must increase column-mean V."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None
    assert state.drag is not None
    state.v[:] = 0.0
    state.drag[:] = 0.0
    ty = 1e-4
    vequation(
        state,
        nlev,
        _DT,
        _CNPAR,
        ty=ty,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )
    assert state.v is not None
    assert np.mean(state.v[1 : nlev + 1]) > 0.0


# ---------------------------------------------------------------------------
# 7. Momentum injection from surface flux
# ---------------------------------------------------------------------------


def test_momentum_budget_surface_only() -> None:
    """Without drag, ΔV_mean ≈ ty * dt / depth."""
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    assert state.v is not None
    assert state.drag is not None
    assert state.h is not None
    state.v[:] = 0.0
    state.drag[:] = 0.0
    ty = 1e-4

    vequation(
        state,
        nlev,
        _DT,
        _CNPAR,
        ty=ty,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )

    h = state.h
    v_mean = np.sum(state.v[1 : nlev + 1] * h[1 : nlev + 1]) / depth
    expected = ty * _DT / depth
    np.testing.assert_allclose(v_mean, expected, rtol=0.02)


# ---------------------------------------------------------------------------
# 8. Couette flow convergence (analytic steady-state)
# ---------------------------------------------------------------------------


def test_couette_gradient_convergence() -> None:
    r"""Couette flow: velocity gradient converges to ty/ν."""
    nlev = 20
    depth = 2.0
    num_val = 1e-2
    avmolu = 1e-6
    ty = 1e-4
    dt = 10.0
    n_steps = 600

    state = _make_state(nlev=nlev, depth=depth, avmolu=avmolu)
    init_meanflow(state, avmolu=avmolu, h0b=0.5, calc_bottom_stress=True)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, dt, zeta=0.0)

    assert state.v is not None
    assert state.drag is not None
    assert state.h is not None

    state.u[:] = 0.0
    state.v[:] = 0.0

    num = np.full(nlev + 1, num_val)
    nucl = np.zeros(nlev + 1)
    gamv = np.zeros(nlev + 1)

    first = [True]
    for _ in range(n_steps):
        friction(state, nlev, avmolu=avmolu, tx=0.0, ty=ty, _first=first)
        vequation(state, nlev, dt, 0.6, ty=ty, num=num, nucl=nucl, gamv=gamv)

    h = state.h
    nu_eff = num_val + avmolu
    expected_grad = ty / nu_eff

    grad = np.array(
        [
            (state.v[k + 1] - state.v[k]) / (0.5 * (h[k] + h[k + 1]))
            for k in range(2, nlev)
        ]
    )

    np.testing.assert_allclose(
        grad,
        expected_grad,
        rtol=5e-2,
        err_msg="Velocity gradient must converge to ty/ν (Couette gradient)",
    )


def test_workspace_reused_across_wrapper_calls() -> None:
    """Single-column wrapper must reuse its Taichi workspace across time steps."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    num = np.full(nlev + 1, 1e-3)
    nucl = np.zeros(nlev + 1)
    gamv = np.zeros(nlev + 1)

    vequation(state, nlev, _DT, _CNPAR, ty=1e-4, num=num, nucl=nucl, gamv=gamv)
    first_workspace = state._kernel_workspaces["vequation"]

    for _ in range(3):
        vequation(state, nlev, _DT, _CNPAR, ty=1e-4, num=num, nucl=nucl, gamv=gamv)

    assert state._kernel_workspaces["vequation"] is first_workspace


# ---------------------------------------------------------------------------
# 9. External pressure gradient drives V when ext_method=0
# ---------------------------------------------------------------------------


def test_external_pressure_gradient() -> None:
    """Negative dpdy (adverse pressure) must produce negative V tendency."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None
    assert state.drag is not None
    state.v[:] = 0.0
    state.drag[:] = 0.0

    dpdy = 1e-5
    vequation(
        state,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        ext_method=0,
        dpdy=dpdy,
    )

    assert state.v is not None
    assert np.all(
        state.v[1 : nlev + 1] < 0.0
    ), "Adverse pressure gradient must drive V negative"


def test_ext_method_nonzero_ignores_dpdy() -> None:
    """When ext_method != 0 the external pressure gradient is zeroed."""
    nlev = _NLEV
    state_0 = _make_state(nlev=nlev)
    state_1 = _make_state(nlev=nlev)
    assert state_0.v is not None and state_1.v is not None
    assert state_0.drag is not None and state_1.drag is not None
    state_0.v[:] = 0.0
    state_1.v[:] = 0.0
    state_0.drag[:] = 0.0
    state_1.drag[:] = 0.0

    vequation(
        state_0,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        ext_method=1,
        dpdy=1e-5,
    )
    vequation(
        state_1,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        ext_method=1,
        dpdy=0.0,
    )

    assert state_0.v is not None and state_1.v is not None
    np.testing.assert_array_equal(state_0.v, state_1.v)


# ---------------------------------------------------------------------------
# 10. Stokes drift gradient adds momentum
# ---------------------------------------------------------------------------


def test_stokes_gradient_effect() -> None:
    """Non-zero nucl*dvsdz must change V compared to zero Stokes drift."""
    nlev = _NLEV
    state_stokes = _make_state(nlev=nlev)
    state_none = _make_state(nlev=nlev)
    assert state_stokes.v is not None and state_none.v is not None
    assert state_stokes.drag is not None and state_none.drag is not None
    state_stokes.v[:] = 0.0
    state_none.v[:] = 0.0
    state_stokes.drag[:] = 0.0
    state_none.drag[:] = 0.0

    nucl = np.full(nlev + 1, 1e-3)
    dvsdz = np.linspace(0.0, 1e-2, nlev + 1)

    vequation(
        state_stokes,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=nucl,
        gamv=_zeros(nlev),
        dvsdz=dvsdz,
    )
    vequation(
        state_none,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )

    assert state_stokes.v is not None and state_none.v is not None
    assert not np.allclose(
        state_stokes.v, state_none.v
    ), "Stokes drift gradient must change V"


# ---------------------------------------------------------------------------
# 11. Relaxation: large tau_r (no relaxation) == baseline
# ---------------------------------------------------------------------------


def test_large_relax_tau_equals_no_relax() -> None:
    """Relaxation time >> dt must not affect the solution."""
    nlev = _NLEV
    state_a = _make_state(nlev=nlev)
    state_b = _make_state(nlev=nlev)
    assert state_a.v is not None and state_b.v is not None
    state_a.v[:] = 0.5
    state_b.v[:] = 0.5

    num = np.full(nlev + 1, 1e-3)
    tau_r = np.full(nlev + 1, 1e15)
    vprof = np.zeros(nlev + 1)

    vequation(
        state_a,
        nlev,
        _DT,
        _CNPAR,
        ty=1e-4,
        num=num,
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        vel_relax_tau=tau_r,
        vprof=vprof,
    )
    vequation(
        state_b,
        nlev,
        _DT,
        _CNPAR,
        ty=1e-4,
        num=num,
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )

    assert state_a.v is not None and state_b.v is not None
    np.testing.assert_allclose(state_a.v, state_b.v, rtol=1e-12)


# ---------------------------------------------------------------------------
# 12. Relaxation: short tau_r pulls V toward vprof
# ---------------------------------------------------------------------------


def test_relax_pulls_toward_vprof() -> None:
    """Short relaxation time scale must nudge V toward the target profile."""
    nlev = _NLEV
    state_relax = _make_state(nlev=nlev)
    state_free = _make_state(nlev=nlev)
    assert state_relax.v is not None and state_free.v is not None
    state_relax.v[1:] = 0.0
    state_free.v[1:] = 0.0

    num = np.full(nlev + 1, 1e-3)
    tau_r = np.full(nlev + 1, _DT)
    vprof = np.ones(nlev + 1)

    vequation(
        state_relax,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=num,
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        vel_relax_tau=tau_r,
        vprof=vprof,
    )
    vequation(
        state_free,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=num,
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )

    assert state_relax.v is not None and state_free.v is not None
    assert np.mean(state_relax.v[1:]) > np.mean(state_free.v[1:])


# ---------------------------------------------------------------------------
# 13. Boundary values: k=0 sentinel unchanged
# ---------------------------------------------------------------------------


def test_sentinel_level_unchanged() -> None:
    """Index 0 (seabed sentinel) must not be modified by the solver."""
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.v is not None
    state.v[0] = 99.0
    vequation(
        state,
        nlev,
        _DT,
        _CNPAR,
        ty=1e-4,
        num=np.full(nlev + 1, 1e-3),
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
    )
    assert state.v is not None
    assert state.v[0] == 99.0, "Sentinel level k=0 must not be modified"


# ---------------------------------------------------------------------------
# 14. No NaN or Inf for realistic inputs
# ---------------------------------------------------------------------------


def test_no_nan_inf() -> None:
    """No NaN or Inf in V after one step with realistic ocean inputs."""
    nlev = 50
    state = _make_state(nlev=nlev, depth=200.0)
    assert state.v is not None
    assert state.drag is not None
    state.v[:] = np.linspace(0.0, 0.3, nlev + 1)
    state.drag[1] = 2e-3

    num = np.linspace(1e-4, 1e-2, nlev + 1)
    vequation(
        state,
        nlev,
        _DT,
        0.6,
        ty=5e-5,
        num=num,
        nucl=np.zeros(nlev + 1),
        gamv=np.zeros(nlev + 1),
        ext_method=0,
        dpdy=1e-6,
    )

    assert state.v is not None
    assert np.all(np.isfinite(state.v)), "V must be finite after one step"


# ---------------------------------------------------------------------------
# 15. Surface plume friction modifies V at k=nlev
# ---------------------------------------------------------------------------


def test_plume_active_modifies_surface_layer() -> None:
    """plume_active=True adds extra friction at k=nlev."""
    nlev = _NLEV
    state_plume = _make_state(nlev=nlev)
    state_nopl = _make_state(nlev=nlev)
    assert state_plume.v is not None and state_nopl.v is not None
    assert state_plume.drag is not None and state_nopl.drag is not None

    # Set non-zero initial V and drag at surface so plume term is non-trivial
    v_init = np.linspace(0.1, 0.5, nlev + 1)
    state_plume.v[:] = v_init
    state_nopl.v[:] = v_init
    state_plume.drag[:] = 1e-3
    state_nopl.drag[:] = 1e-3
    # u contributes to speed; set non-zero to ensure non-zero plume term
    state_plume.u[:] = 0.1
    state_nopl.u[:] = 0.1

    num = np.full(nlev + 1, 1e-3)

    vequation(
        state_plume,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=num,
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        plume_active=True,
    )
    vequation(
        state_nopl,
        nlev,
        _DT,
        _CNPAR,
        ty=0.0,
        num=num,
        nucl=_zeros(nlev),
        gamv=_zeros(nlev),
        plume_active=False,
    )

    assert state_plume.v is not None and state_nopl.v is not None
    # Surface layer must differ; plume friction reduces V at k=nlev
    assert (
        state_plume.v[nlev] != state_nopl.v[nlev]
    ), "plume_active must change surface-layer V"
    assert (
        state_plume.v[nlev] < state_nopl.v[nlev]
    ), "Plume friction must reduce surface V (extra drag)"


# ---------------------------------------------------------------------------
# 16. step_vequation kernel: multi-column consistency
# ---------------------------------------------------------------------------


def test_multi_column_parity() -> None:
    """step_vequation with n_cols=2, identical columns → identical results."""
    nlev = _NLEV
    n_cols = 2
    avmolu = 1.3e-6
    gravity = 9.81

    ws = VEquationWorkspace(nlev=nlev, n_cols=n_cols)

    rng = np.random.default_rng(7)
    v_init = rng.uniform(-0.2, 0.2, nlev + 1)
    u_init = rng.uniform(-0.1, 0.1, nlev + 1)
    h_val = np.full(nlev + 1, _DEPTH / nlev)
    h_val[0] = 0.0
    num_val = np.linspace(1e-4, 1e-2, nlev + 1)

    for col in range(n_cols):
        for k in range(nlev + 1):
            ws.v[col, k] = float(v_init[k])
            ws.vo[col, k] = float(v_init[k])
            ws.u[col, k] = float(u_init[k])
            ws.h[col, k] = float(h_val[k])
            ws.w[col, k] = 0.0
            ws.drag[col, k] = 0.0
            ws.num[col, k] = float(num_val[k])
            ws.nucl[col, k] = 0.0
            ws.dvsdz[col, k] = 0.0
            ws.idpdy[col, k] = 0.0
            ws.vprof[col, k] = 0.0
            ws.tau_r[col, k] = 1e15
        ws.ty[col, 0] = 1e-4
        ws.dzetady[col, 0] = 0.0

    step_vequation(
        n_cols,
        nlev,
        _DT,
        _CNPAR,
        avmolu,
        gravity,
        0,
        0,
        4,
        0,
        ws.v,
        ws.vo,
        ws.u,
        ws.h,
        ws.w,
        ws.drag,
        ws.num,
        ws.nucl,
        ws.dvsdz,
        ws.idpdy,
        ws.vprof,
        ws.tau_r,
        ws.ty,
        ws.dzetady,
        ws.avh,
        ws.q_sour,
        ws.l_sour,
        ws.av,
        ws.bv,
        ws.cv,
        ws.dv,
        ws.rv,
        ws.qv,
        ws.adv_cv,
    )

    v0 = np.array([ws.v[0, k] for k in range(nlev + 1)])
    v1 = np.array([ws.v[1, k] for k in range(nlev + 1)])
    np.testing.assert_allclose(
        v0, v1, rtol=1e-12, err_msg="Identical columns must give identical results"
    )
