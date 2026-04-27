"""Tests for pygotm.meanflow.uequation — U-momentum equation.

All execution goes through the production Taichi kernel ``step_uequation``.
Single-column checks use ``n_cols=1`` and populate the workspace directly.
"""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, fill_field_scalar, read_field_array
from type_helpers import ReadyMeanflowState, require_meanflow_state

from pygotm.meanflow.friction import friction
from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.uequation import UEquationWorkspace, step_uequation
from pygotm.meanflow.updategrid import updategrid

_NLEV = 20
_DEPTH = 10.0
_DT = 3600.0
_CNPAR = 0.6
_LONG = 1.0e15


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
    return np.zeros(nlev + 1, dtype=np.float64)


def _build_workspace(
    state: ReadyMeanflowState,
    nlev: int,
    *,
    workspace: UEquationWorkspace | None = None,
    num: np.ndarray | None = None,
    nucl: np.ndarray | None = None,
    dusdz: np.ndarray | None = None,
    idpdx: np.ndarray | None = None,
    uprof: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    tx: float = 0.0,
    dzetadx: float = 0.0,
) -> UEquationWorkspace:
    ws = workspace if workspace is not None else UEquationWorkspace(nlev=nlev, n_cols=1)
    fill_field_from_array(ws.u, state.u)
    fill_field_from_array(ws.uo, state.uo)
    fill_field_from_array(ws.v, state.v)
    fill_field_from_array(ws.h, state.h)
    fill_field_from_array(ws.w, state.w)
    fill_field_from_array(ws.drag, state.drag)
    fill_field_from_array(ws.num, num if num is not None else _zeros(nlev))
    fill_field_from_array(ws.nucl, nucl if nucl is not None else _zeros(nlev))
    fill_field_from_array(ws.dusdz, dusdz if dusdz is not None else _zeros(nlev))
    fill_field_from_array(ws.idpdx, idpdx if idpdx is not None else _zeros(nlev))
    fill_field_from_array(ws.uprof, uprof if uprof is not None else _zeros(nlev))
    fill_field_from_array(
        ws.tau_r,
        tau_r if tau_r is not None else np.full(nlev + 1, _LONG, dtype=np.float64),
    )

    ws.tx.fill(0.0)
    ws.dzetadx.fill(0.0)
    ws.avh.fill(0.0)
    ws.q_sour.fill(0.0)
    ws.l_sour.fill(0.0)
    ws.au.fill(0.0)
    ws.bu.fill(0.0)
    ws.cu.fill(0.0)
    ws.du.fill(0.0)
    ws.ru.fill(0.0)
    ws.qu.fill(0.0)
    ws.adv_cu.fill(0.0)

    fill_field_scalar(ws.tx, tx)
    fill_field_scalar(ws.dzetadx, dzetadx)
    return ws


def _run_single_column(
    state: ReadyMeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    *,
    workspace: UEquationWorkspace | None = None,
    num: np.ndarray | None = None,
    nucl: np.ndarray | None = None,
    dusdz: np.ndarray | None = None,
    idpdx: np.ndarray | None = None,
    uprof: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    tx: float = 0.0,
    dzetadx: float = 0.0,
    ext_method: int = 0,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
    seagrass_active: bool = False,
    plume_active: bool = False,
) -> UEquationWorkspace:
    ws = _build_workspace(
        state,
        nlev,
        workspace=workspace,
        num=num,
        nucl=nucl,
        dusdz=dusdz,
        idpdx=idpdx,
        uprof=uprof,
        tau_r=tau_r,
        tx=tx,
        dzetadx=dzetadx,
    )

    step_uequation(
        1,
        nlev,
        dt,
        cnpar,
        state.avmolu,
        state.gravity,
        ext_method,
        int(w_adv_active),
        w_adv_discr,
        int(seagrass_active),
        int(plume_active),
        ws.u,
        ws.uo,
        ws.v,
        ws.h,
        ws.w,
        ws.drag,
        ws.num,
        ws.nucl,
        ws.dusdz,
        ws.idpdx,
        ws.uprof,
        ws.tau_r,
        ws.tx,
        ws.dzetadx,
        ws.avh,
        ws.q_sour,
        ws.l_sour,
        ws.au,
        ws.bu,
        ws.cu,
        ws.du,
        ws.ru,
        ws.qu,
        ws.adv_cu,
    )

    assert state.u is not None
    assert state.uo is not None
    assert state.avh is not None
    state.u[:] = read_field_array(ws.u)
    state.uo[:] = read_field_array(ws.uo)
    state.avh[:] = read_field_array(ws.avh)
    return ws


def test_import() -> None:
    from pygotm.meanflow.uequation import step_uequation as _u  # noqa: F401

    assert callable(_u)


def test_smoke_step_uequation() -> None:
    state = _make_state()
    _run_single_column(
        state,
        _NLEV,
        _DT,
        _CNPAR,
        num=_zeros(_NLEV),
        nucl=_zeros(_NLEV),
    )


def test_uo_saves_old_u() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    state.u[:] = np.linspace(0.0, 0.5, nlev + 1)
    u_before = state.u.copy()

    _run_single_column(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))

    assert state.uo is not None
    np.testing.assert_array_equal(state.uo, u_before)


def test_avh_equals_num_plus_avmolu() -> None:
    nlev = _NLEV
    avmolu = 1.3e-6
    state = _make_state(nlev=nlev, avmolu=avmolu)
    rng = np.random.default_rng(0)
    num = np.abs(rng.uniform(1.0e-5, 1.0e-3, nlev + 1))

    _run_single_column(state, nlev, _DT, _CNPAR, num=num, nucl=_zeros(nlev))

    assert state.avh is not None
    np.testing.assert_allclose(state.avh, num + avmolu, rtol=1.0e-12)


def test_quiescent_no_forcing() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0

    _run_single_column(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))

    np.testing.assert_allclose(state.u[1 : nlev + 1], 0.0, atol=1.0e-15)


def test_surface_stress_accelerates_u() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        tx=1.0e-4,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
    )

    assert np.mean(state.u[1 : nlev + 1]) > 0.0


def test_momentum_budget_surface_only() -> None:
    nlev = _NLEV
    depth = _DEPTH
    state = _make_state(nlev=nlev, depth=depth)
    assert state.u is not None
    assert state.drag is not None
    assert state.h is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0
    tx = 1.0e-4

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        tx=tx,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
    )

    u_mean = np.sum(state.u[1 : nlev + 1] * state.h[1 : nlev + 1]) / depth
    expected = tx * _DT / depth
    np.testing.assert_allclose(u_mean, expected, rtol=0.02)


def test_bottom_friction_decelerates_u() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[1:] = 0.2
    state.v[1:] = 0.0
    state.drag[:] = 0.0
    state.drag[1] = 5.0e-3
    before = state.u.copy()

    _run_single_column(state, nlev, _DT, _CNPAR, num=_zeros(nlev), nucl=_zeros(nlev))

    assert np.mean(state.u[1:]) < np.mean(before[1:])


def test_seagrass_inner_friction_decelerates_interior_layers() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None
    assert state.drag is not None
    state.u[1:] = 0.2
    state.v[1:] = 0.0
    state.drag[:] = 0.0
    state.drag[2:] = 5.0e-3

    with_seagrass = _make_state(nlev=nlev)
    assert with_seagrass.u is not None
    assert with_seagrass.v is not None
    assert with_seagrass.drag is not None
    with_seagrass.u[:] = state.u
    with_seagrass.v[:] = state.v
    with_seagrass.drag[:] = state.drag

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        seagrass_active=False,
    )
    _run_single_column(
        with_seagrass,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        seagrass_active=True,
    )

    assert np.mean(with_seagrass.u[2:]) < np.mean(state.u[2:])


def test_plume_active_modifies_surface_layer() -> None:
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

    _run_single_column(
        plume_state,
        nlev,
        _DT,
        _CNPAR,
        num=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        nucl=_zeros(nlev),
        plume_active=True,
    )
    _run_single_column(
        base_state,
        nlev,
        _DT,
        _CNPAR,
        num=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        nucl=_zeros(nlev),
        plume_active=False,
    )

    assert plume_state.u[nlev] < base_state.u[nlev]


def test_couette_gradient_convergence() -> None:
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
    workspace = UEquationWorkspace(nlev=nlev, n_cols=1)
    first = [True]
    for _ in range(n_steps):
        friction(state, nlev, avmolu=avmolu, tx=tx, ty=0.0, _first=first)
        _run_single_column(
            state,
            nlev,
            dt,
            0.6,
            workspace=workspace,
            tx=tx,
            num=num,
            nucl=_zeros(nlev),
        )

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


def test_external_pressure_gradient() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = 0.0
    state.drag[:] = 0.0

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        ext_method=0,
        dzetadx=1.0e-5,
    )

    assert np.all(state.u[1 : nlev + 1] < 0.0)


def test_ext_method_nonzero_ignores_dzetadx() -> None:
    nlev = _NLEV
    state_a = _make_state(nlev=nlev)
    state_b = _make_state(nlev=nlev)
    assert state_a.u is not None and state_b.u is not None
    assert state_a.drag is not None and state_b.drag is not None
    state_a.u[:] = 0.0
    state_b.u[:] = 0.0
    state_a.drag[:] = 0.0
    state_b.drag[:] = 0.0

    _run_single_column(
        state_a,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        ext_method=1,
        dzetadx=1.0e-5,
    )
    _run_single_column(
        state_b,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
        ext_method=1,
        dzetadx=0.0,
    )

    np.testing.assert_array_equal(state_a.u, state_b.u)


def test_stokes_gradient_effect() -> None:
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

    _run_single_column(
        state_stokes,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=nucl,
        dusdz=dusdz,
    )
    _run_single_column(
        state_none,
        nlev,
        _DT,
        _CNPAR,
        num=_zeros(nlev),
        nucl=_zeros(nlev),
    )

    assert not np.allclose(state_stokes.u, state_none.u)


def test_large_relax_tau_equals_no_relax() -> None:
    nlev = _NLEV
    state_a = _make_state(nlev=nlev)
    state_b = _make_state(nlev=nlev)
    assert state_a.u is not None and state_b.u is not None
    state_a.u[:] = 0.5
    state_b.u[:] = 0.5

    num = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    tau_r = np.full(nlev + 1, _LONG, dtype=np.float64)
    uprof = np.zeros(nlev + 1, dtype=np.float64)

    _run_single_column(
        state_a,
        nlev,
        _DT,
        _CNPAR,
        tx=1.0e-4,
        num=num,
        nucl=_zeros(nlev),
        tau_r=tau_r,
        uprof=uprof,
    )
    _run_single_column(
        state_b,
        nlev,
        _DT,
        _CNPAR,
        tx=1.0e-4,
        num=num,
        nucl=_zeros(nlev),
    )

    np.testing.assert_allclose(state_a.u, state_b.u, rtol=1.0e-12)


def test_relax_pulls_toward_uprof() -> None:
    nlev = _NLEV
    state_relax = _make_state(nlev=nlev)
    state_free = _make_state(nlev=nlev)
    assert state_relax.u is not None and state_free.u is not None
    state_relax.u[1:] = 0.0
    state_free.u[1:] = 0.0

    num = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    tau_r = np.full(nlev + 1, _DT, dtype=np.float64)
    uprof = np.ones(nlev + 1, dtype=np.float64)

    _run_single_column(
        state_relax,
        nlev,
        _DT,
        _CNPAR,
        num=num,
        nucl=_zeros(nlev),
        tau_r=tau_r,
        uprof=uprof,
    )
    _run_single_column(
        state_free,
        nlev,
        _DT,
        _CNPAR,
        num=num,
        nucl=_zeros(nlev),
    )

    assert np.mean(state_relax.u[1:]) > np.mean(state_free.u[1:])


def test_sentinel_level_unchanged() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    state.u[0] = 99.0

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        tx=1.0e-4,
        num=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        nucl=_zeros(nlev),
    )

    assert state.u[0] == 99.0


def test_no_nan_inf() -> None:
    nlev = 50
    state = _make_state(nlev=nlev, depth=200.0)
    assert state.u is not None
    assert state.drag is not None
    state.u[:] = np.linspace(0.0, 0.3, nlev + 1)
    state.drag[:] = 0.0
    state.drag[1] = 2.0e-3

    _run_single_column(
        state,
        nlev,
        _DT,
        0.6,
        tx=5.0e-5,
        num=np.linspace(1.0e-4, 1.0e-2, nlev + 1),
        nucl=_zeros(nlev),
        ext_method=0,
        dzetadx=1.0e-6,
    )

    assert np.all(np.isfinite(state.u))


def test_multi_column_parity() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None
    assert state.h is not None
    state.u[:] = np.linspace(-0.1, 0.2, nlev + 1)
    state.v[:] = np.linspace(0.05, -0.05, nlev + 1)

    num = np.linspace(1.0e-4, 1.0e-2, nlev + 1)
    nucl = np.full(nlev + 1, 2.0e-4, dtype=np.float64)
    dusdz = np.linspace(0.0, 1.0e-3, nlev + 1)
    tau_r = np.full(nlev + 1, _LONG, dtype=np.float64)
    uprof = np.zeros(nlev + 1, dtype=np.float64)

    single = _build_workspace(
        state,
        nlev,
        num=num,
        nucl=nucl,
        dusdz=dusdz,
        tau_r=tau_r,
        uprof=uprof,
        tx=1.0e-4,
    )
    step_uequation(
        1,
        nlev,
        _DT,
        _CNPAR,
        state.avmolu,
        state.gravity,
        0,
        0,
        4,
        0,
        0,
        single.u,
        single.uo,
        single.v,
        single.h,
        single.w,
        single.drag,
        single.num,
        single.nucl,
        single.dusdz,
        single.idpdx,
        single.uprof,
        single.tau_r,
        single.tx,
        single.dzetadx,
        single.avh,
        single.q_sour,
        single.l_sour,
        single.au,
        single.bu,
        single.cu,
        single.du,
        single.ru,
        single.qu,
        single.adv_cu,
    )
    single_result = read_field_array(single.u)

    multi = UEquationWorkspace(nlev=nlev, n_cols=2)
    for col in range(2):
        fill_field_from_array(multi.u, state.u, col=col)
        fill_field_from_array(multi.uo, state.uo, col=col)
        fill_field_from_array(multi.v, state.v, col=col)
        fill_field_from_array(multi.h, state.h, col=col)
        fill_field_from_array(multi.w, state.w, col=col)
        fill_field_from_array(multi.drag, state.drag, col=col)
        fill_field_from_array(multi.num, num, col=col)
        fill_field_from_array(multi.nucl, nucl, col=col)
        fill_field_from_array(multi.dusdz, dusdz, col=col)
        fill_field_from_array(multi.idpdx, _zeros(nlev), col=col)
        fill_field_from_array(multi.uprof, uprof, col=col)
        fill_field_from_array(multi.tau_r, tau_r, col=col)
        fill_field_scalar(multi.tx, 1.0e-4, col=col)
        fill_field_scalar(multi.dzetadx, 0.0, col=col)

    multi.avh.fill(0.0)
    multi.q_sour.fill(0.0)
    multi.l_sour.fill(0.0)
    multi.au.fill(0.0)
    multi.bu.fill(0.0)
    multi.cu.fill(0.0)
    multi.du.fill(0.0)
    multi.ru.fill(0.0)
    multi.qu.fill(0.0)
    multi.adv_cu.fill(0.0)

    step_uequation(
        2,
        nlev,
        _DT,
        _CNPAR,
        state.avmolu,
        state.gravity,
        0,
        0,
        4,
        0,
        0,
        multi.u,
        multi.uo,
        multi.v,
        multi.h,
        multi.w,
        multi.drag,
        multi.num,
        multi.nucl,
        multi.dusdz,
        multi.idpdx,
        multi.uprof,
        multi.tau_r,
        multi.tx,
        multi.dzetadx,
        multi.avh,
        multi.q_sour,
        multi.l_sour,
        multi.au,
        multi.bu,
        multi.cu,
        multi.du,
        multi.ru,
        multi.qu,
        multi.adv_cu,
    )

    np.testing.assert_allclose(read_field_array(multi.u, col=0), single_result)
    np.testing.assert_allclose(read_field_array(multi.u, col=1), single_result)


def test_kernel_reproducibility() -> None:
    nlev = _NLEV
    state = _make_state(nlev=nlev)
    assert state.u is not None
    assert state.v is not None
    state.u[:] = np.linspace(-0.05, 0.15, nlev + 1)
    state.v[:] = np.linspace(0.02, -0.01, nlev + 1)

    num = np.linspace(1.0e-4, 1.0e-3, nlev + 1)
    tau_r = np.full(nlev + 1, _LONG, dtype=np.float64)

    ws_a = _build_workspace(state, nlev, num=num, tau_r=tau_r, tx=2.0e-5)
    ws_b = _build_workspace(state, nlev, num=num, tau_r=tau_r, tx=2.0e-5)

    for ws in (ws_a, ws_b):
        step_uequation(
            1,
            nlev,
            _DT,
            _CNPAR,
            state.avmolu,
            state.gravity,
            0,
            0,
            4,
            0,
            0,
            ws.u,
            ws.uo,
            ws.v,
            ws.h,
            ws.w,
            ws.drag,
            ws.num,
            ws.nucl,
            ws.dusdz,
            ws.idpdx,
            ws.uprof,
            ws.tau_r,
            ws.tx,
            ws.dzetadx,
            ws.avh,
            ws.q_sour,
            ws.l_sour,
            ws.au,
            ws.bu,
            ws.cu,
            ws.du,
            ws.ru,
            ws.qu,
            ws.adv_cu,
        )

    assert np.array_equal(read_field_array(ws_a.u), read_field_array(ws_b.u))
    assert np.array_equal(read_field_array(ws_a.uo), read_field_array(ws_b.uo))
