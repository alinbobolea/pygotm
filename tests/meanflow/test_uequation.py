"""Tests for pygotm.meanflow.uequation — U-momentum equation."""

import numpy as np

from pygotm.meanflow.friction import friction
from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.uequation import step_uequation, uequation
from pygotm.meanflow.updategrid import updategrid

_NLEV = 20
_DEPTH = 10.0
_DT = 3600.0
_CNPAR = 0.6
_LONG = 1.0e15


def _make_state(nlev=_NLEV, depth=_DEPTH, avmolu=1.3e-6):
    state = MeanflowState()
    init_meanflow(state, avmolu=avmolu)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    return state


def _zeros(nlev=_NLEV):
    return np.zeros(nlev + 1, dtype=np.float64)


def _run_step(
    state,
    nlev,
    dt,
    cnpar,
    *,
    num=None,
    nucl=None,
    tx=0.0,
    dpdx=0.0,
    ext_method=0,
    w_adv_active=False,
    w_adv_discr=4,
    seagrass_active=False,
    plume_active=False,
    idpdx=None,
    dusdz=None,
    tau_r=None,
    uprof=None,
):
    uequation(
        state,
        nlev,
        dt,
        cnpar,
        tx,
        num if num is not None else _zeros(nlev),
        nucl if nucl is not None else _zeros(nlev),
        _zeros(nlev),
        ext_method=ext_method,
        dpdx=dpdx,
        idpdx=idpdx,
        dusdz=dusdz,
        w_adv_active=w_adv_active,
        w_adv_discr=w_adv_discr,
        vel_relax_tau=tau_r,
        uprof=uprof,
        seagrass_active=seagrass_active,
        plume_active=plume_active,
    )


def test_import():
    from pygotm.meanflow.uequation import step_uequation as _u  # noqa: F401

    assert callable(_u)


def test_smoke_step_uequation():
    state = _make_state()
    _run_step(state, _NLEV, _DT, _CNPAR, num=_zeros(_NLEV), nucl=_zeros(_NLEV))


def test_uo_saves_old_u():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    state.u[:] = np.linspace(0.0, 0.5, nlev + 1)
    u_before = state.u.copy()
    _run_step(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))
    assert state.uo is not None
    np.testing.assert_array_equal(state.uo, u_before)


def test_avh_equals_num_plus_avmolu():
    nlev = _NLEV
    avmolu = 1.3e-6
    state = _make_state(nlev=nlev, avmolu=avmolu)
    rng = np.random.default_rng(0)
    num = np.abs(rng.uniform(1.0e-5, 1.0e-3, nlev + 1))
    _run_step(state, nlev, _DT, _CNPAR, num=num, nucl=_zeros(nlev))
    assert state.avh is not None
    np.testing.assert_allclose(state.avh, num + avmolu, rtol=1.0e-12)


def test_quiescent_no_forcing():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0
    _run_step(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))
    np.testing.assert_allclose(state.u[1 : nlev + 1], 0.0, atol=1.0e-15)


def test_surface_stress_accelerates_u():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0
    _run_step(state, nlev, _DT, _CNPAR, tx=1.0e-4, num=_zeros(nlev), nucl=_zeros(nlev))
    assert np.mean(state.u[1 : nlev + 1]) > 0.0


def test_momentum_budget_surface_only():
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    assert state.u is not None
    assert state.drag is not None
    assert state.h is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0
    tx = 1.0e-4
    _run_step(state, nlev, _DT, _CNPAR, tx=tx, num=_zeros(nlev), nucl=_zeros(nlev))
    u_mean = np.sum(state.u[1 : nlev + 1] * state.h[1 : nlev + 1]) / depth
    expected = tx * _DT / depth
    np.testing.assert_allclose(u_mean, expected, rtol=0.02)


def test_bottom_friction_decelerates_u():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[1:] = 0.2
    state.v[1:] = 0.0
    state.drag[:] = 0.0
    state.drag[1] = 5.0e-3
    before = state.u.copy()
    _run_step(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))
    assert np.mean(state.u[1:]) < np.mean(before[1:])


def test_seagrass_inner_friction_decelerates_interior_layers():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[1:] = 0.2
    state.v[1:] = 0.0
    state.drag[:] = 0.0
    state.drag[2:] = 5.0e-3

    with_seagrass = _make_state(nlev=nlev)
    assert with_seagrass.u is not None
    assert with_seagrass.drag is not None
    with_seagrass.u[:] = state.u
    with_seagrass.v[:] = state.v
    with_seagrass.drag[:] = state.drag

    _run_step(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev), seagrass_active=False)
    _run_step(with_seagrass, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev), seagrass_active=True)

    assert np.mean(with_seagrass.u[2:]) < np.mean(state.u[2:])


def test_plume_active_modifies_surface_layer():
    nlev = _NLEV
    plume_state = _make_state(nlev=nlev)
    base_state = _make_state(nlev=nlev)
    assert plume_state.u is not None and base_state.u is not None
    assert plume_state.v is not None and base_state.v is not None
    assert plume_state.drag is not None and base_state.drag is not None

    u_init = np.linspace(0.1, 0.5, nlev + 1)
    plume_state.u[:] = u_init
    base_state.u[:] = u_init
    plume_state.v[:] = 0.1
    base_state.v[:] = 0.1
    plume_state.drag[:] = 1.0e-3
    base_state.drag[:] = 1.0e-3

    num = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    _run_step(plume_state, nlev, _DT, _CNPAR, num=num, nucl=_zeros(nlev), plume_active=True)
    _run_step(base_state, nlev, _DT, _CNPAR, num=num, nucl=_zeros(nlev), plume_active=False)

    assert plume_state.u[nlev] < base_state.u[nlev]


def test_couette_gradient_convergence():
    nlev = 20
    depth = 2.0
    num_val = 1.0e-2
    avmolu = 1.0e-6
    tx = 1.0e-4
    dt = 10.0
    n_steps = 600

    state = _make_state(nlev=nlev, depth=depth, avmolu=avmolu)
    init_meanflow(state, avmolu=avmolu, h0b=0.5, calc_bottom_stress=True)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, dt, zeta=0.0)

    assert state.u is not None
    assert state.v is not None
    assert state.h is not None
    state.u[:] = 0.0
    state.v[:] = 0.0

    num = np.full(nlev + 1, num_val, dtype=np.float64)
    first = [True]
    for _ in range(n_steps):
        friction(state, nlev, avmolu=avmolu, tx=tx, ty=0.0, _first=first)
        uequation(state, nlev, dt, 0.6, tx, num, _zeros(nlev), _zeros(nlev))

    nu_eff = num_val + avmolu
    expected_grad = tx / nu_eff
    h = state.h
    grad = np.array(
        [
            (state.u[k + 1] - state.u[k]) / (0.5 * (h[k] + h[k + 1]))
            for k in range(2, nlev)
        ]
    )
    np.testing.assert_allclose(grad, expected_grad, rtol=5.0e-2)


def test_external_pressure_gradient():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0
    _run_step(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev), ext_method=0, dpdx=1.0e-5)
    assert np.all(state.u[1 : nlev + 1] < 0.0)


def test_ext_method_nonzero_ignores_dzetadx():
    nlev = _NLEV
    state_a = _make_state(nlev=nlev)
    state_b = _make_state(nlev=nlev)
    assert state_a.u is not None and state_b.u is not None
    assert state_a.drag is not None and state_b.drag is not None
    state_a.u[:] = 0.0
    state_b.u[:] = 0.0
    state_a.drag[:] = 0.0
    state_b.drag[:] = 0.0

    _run_step(state_a, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev), ext_method=1, dpdx=1.0e-5)
    _run_step(state_b, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev), ext_method=1, dpdx=0.0)
    np.testing.assert_array_equal(state_a.u, state_b.u)


def test_stokes_gradient_effect():
    nlev = _NLEV
    state_stokes = _make_state(nlev=nlev)
    state_none = _make_state(nlev=nlev)
    assert state_stokes.u is not None and state_none.u is not None
    assert state_stokes.drag is not None and state_none.drag is not None
    state_stokes.u[:] = 0.0
    state_none.u[:] = 0.0
    state_stokes.drag[:] = 0.0
    state_none.drag[:] = 0.0

    nucl = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    dusdz = np.linspace(0.0, 1.0e-2, nlev + 1)

    _run_step(state_stokes, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=nucl, dusdz=dusdz)
    _run_step(state_none, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))
    assert not np.allclose(state_stokes.u, state_none.u)


def test_large_relax_tau_equals_no_relax():
    nlev = _NLEV
    state_a = _make_state(nlev=nlev)
    state_b = _make_state(nlev=nlev)
    assert state_a.u is not None and state_b.u is not None
    state_a.u[:] = 0.5
    state_b.u[:] = 0.5

    num = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    tau_r = np.full(nlev + 1, _LONG, dtype=np.float64)
    uprof = np.zeros(nlev + 1, dtype=np.float64)

    _run_step(state_a, nlev, _DT, _CNPAR, tx=1.0e-4, num=num, nucl=_zeros(nlev), tau_r=tau_r, uprof=uprof)
    _run_step(state_b, nlev, _DT, _CNPAR, tx=1.0e-4, num=num, nucl=_zeros(nlev))
    np.testing.assert_allclose(state_a.u, state_b.u, rtol=1.0e-12)


def test_relax_pulls_toward_uprof():
    nlev = _NLEV
    state_relax = _make_state(nlev=nlev)
    state_free = _make_state(nlev=nlev)
    assert state_relax.u is not None and state_free.u is not None
    state_relax.u[1:] = 0.0
    state_free.u[1:] = 0.0

    num = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    tau_r = np.full(nlev + 1, _DT, dtype=np.float64)
    uprof = np.ones(nlev + 1, dtype=np.float64)

    _run_step(state_relax, nlev, _DT, _CNPAR, num=num, nucl=_zeros(nlev), tau_r=tau_r, uprof=uprof)
    _run_step(state_free, nlev, _DT, _CNPAR, num=num, nucl=_zeros(nlev))
    assert np.mean(state_relax.u[1:]) > np.mean(state_free.u[1:])


def test_sentinel_level_unchanged():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    state.u[0] = 99.0
    _run_step(state, nlev, _DT, _CNPAR, tx=1.0e-4, num=np.full(nlev + 1, 1.0e-3, dtype=np.float64), nucl=_zeros(nlev))
    assert state.u[0] == 99.0


def test_no_nan_inf():
    nlev = 50
    state = _make_state(nlev=nlev, depth=200.0)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = np.linspace(0.0, 0.3, nlev + 1)
    state.drag[:] = 0.0
    state.drag[1] = 2.0e-3
    _run_step(
        state, nlev, _DT, 0.6,
        tx=5.0e-5,
        num=np.linspace(1.0e-4, 1.0e-2, nlev + 1),
        nucl=_zeros(nlev),
        ext_method=0,
        dpdx=1.0e-6,
    )
    assert np.all(np.isfinite(state.u))


def test_multi_column_parity():
    nlev = _NLEV
    batch_size = 2
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None
    state.u[:] = np.linspace(-0.1, 0.2, nlev + 1)
    state.v[:] = np.linspace(0.05, -0.05, nlev + 1)

    num = np.linspace(1.0e-4, 1.0e-2, nlev + 1)
    nucl = np.full(nlev + 1, 2.0e-4, dtype=np.float64)
    dusdz = np.linspace(0.0, 1.0e-3, nlev + 1)
    tau_r = np.full(nlev + 1, _LONG, dtype=np.float64)
    uprof = np.zeros(nlev + 1, dtype=np.float64)
    tx_val = 1.0e-4

    state_ref = _make_state(nlev=nlev)
    state_ref.u[:] = state.u.copy()
    state_ref.v[:] = state.v.copy()
    _run_step(state_ref, nlev, _DT, _CNPAR, num=num, nucl=nucl, dusdz=dusdz, tau_r=tau_r, uprof=uprof, tx=tx_val)
    u_single = state_ref.u.copy()

    u_b = np.tile(state.u, (batch_size, 1)).astype(np.float64)
    uo_b = np.tile(state.uo, (batch_size, 1)).astype(np.float64)
    v_b = np.tile(state.v, (batch_size, 1)).astype(np.float64)
    h_b = np.tile(state.h, (batch_size, 1)).astype(np.float64)
    w_b = np.tile(state.w, (batch_size, 1)).astype(np.float64)
    drag_b = np.tile(state.drag, (batch_size, 1)).astype(np.float64)
    num_b = np.tile(num, (batch_size, 1)).astype(np.float64)
    nucl_b = np.tile(nucl, (batch_size, 1)).astype(np.float64)
    dusdz_b = np.tile(dusdz, (batch_size, 1)).astype(np.float64)
    idpdx_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    uprof_b = np.tile(uprof, (batch_size, 1)).astype(np.float64)
    tau_r_b = np.tile(tau_r, (batch_size, 1)).astype(np.float64)
    tx_b = np.full(batch_size, tx_val, dtype=np.float64)
    dzetadx_b = np.zeros(batch_size, dtype=np.float64)
    avh_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    q_sour_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    l_sour_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    au_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    bu_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    cu_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    du_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    ru_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    qu_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    adv_cu_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)

    step_uequation(
        batch_size, nlev, _DT, _CNPAR, state.avmolu, state.gravity,
        0, 0, 4, 0, 0,
        tx_b, dzetadx_b,
        u_b, uo_b, v_b, h_b, w_b, drag_b,
        num_b, nucl_b, dusdz_b, idpdx_b, uprof_b, tau_r_b,
        avh_b, q_sour_b, l_sour_b,
        au_b, bu_b, cu_b, du_b, ru_b, qu_b, adv_cu_b,
    )

    np.testing.assert_allclose(u_b[0], u_single)
    np.testing.assert_allclose(u_b[1], u_single)


def test_kernel_reproducibility():
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None
    state.u[:] = np.linspace(-0.05, 0.15, nlev + 1)
    state.v[:] = np.linspace(0.02, -0.01, nlev + 1)

    num = np.linspace(1.0e-4, 1.0e-3, nlev + 1)
    tx_val = 2.0e-5

    state_a = _make_state(nlev=nlev)
    state_b = _make_state(nlev=nlev)
    for s in (state_a, state_b):
        s.u[:] = state.u.copy()
        s.v[:] = state.v.copy()

    _run_step(state_a, nlev, _DT, _CNPAR, tx=tx_val, num=num, nucl=_zeros(nlev))
    _run_step(state_b, nlev, _DT, _CNPAR, tx=tx_val, num=num, nucl=_zeros(nlev))

    np.testing.assert_array_equal(state_a.u, state_b.u)
    np.testing.assert_array_equal(state_a.uo, state_b.uo)
