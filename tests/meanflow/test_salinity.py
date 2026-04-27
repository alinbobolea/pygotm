"""Tests for pygotm.meanflow.salinity — salinity diffusion equation.

All execution goes through the production Taichi kernel ``step_salinity``.
Single-column checks use ``n_cols=1`` and populate the workspace directly.
"""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, fill_field_scalar, read_field_array
from type_helpers import ReadyMeanflowState, require_meanflow_state

from pygotm.meanflow.meanflow import (
    MeanflowState,
    init_meanflow,
    post_init_meanflow,
)
from pygotm.meanflow.salinity import SalinityWorkspace, step_salinity
from pygotm.meanflow.updategrid import updategrid

_NLEV = 20
_DEPTH = 50.0
_DT = 3600.0
_CNPAR = 0.6
_LONG = 1.0e15


def _make_state(
    nlev: int = _NLEV,
    depth: float = _DEPTH,
    avmolS: float = 1.1e-9,
    S_init: float = 35.0,
) -> ReadyMeanflowState:
    state = MeanflowState()
    init_meanflow(state, avmolS=avmolS)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    ready_state = require_meanflow_state(state)
    ready_state.S[:] = S_init
    return ready_state


def _zeros(nlev: int = _NLEV) -> np.ndarray:
    return np.zeros(nlev + 1, dtype=np.float64)


def _prepare_workspace(
    state: ReadyMeanflowState,
    nlev: int,
    *,
    workspace: SalinityWorkspace | None = None,
    wflux: float = 0.0,
    sflux: float = 0.0,
    nus: np.ndarray | None = None,
    gams: np.ndarray | None = None,
    dsdx: np.ndarray | None = None,
    dsdy: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    Sobs: np.ndarray | None = None,
) -> SalinityWorkspace:
    ws = workspace if workspace is not None else SalinityWorkspace(nlev=nlev, n_cols=1)
    fill_field_from_array(ws.S, state.S)
    fill_field_from_array(ws.h, state.h)
    fill_field_from_array(ws.w, state.w)
    fill_field_from_array(ws.u, state.u)
    fill_field_from_array(ws.v, state.v)
    fill_field_from_array(ws.nus, nus if nus is not None else _zeros(nlev))
    fill_field_from_array(ws.gams, gams if gams is not None else _zeros(nlev))
    fill_field_from_array(ws.Sobs, Sobs if Sobs is not None else _zeros(nlev))
    fill_field_from_array(
        ws.tau_r,
        tau_r if tau_r is not None else np.full(nlev + 1, _LONG, dtype=np.float64),
    )
    fill_field_from_array(ws.dsdx, dsdx if dsdx is not None else _zeros(nlev))
    fill_field_from_array(ws.dsdy, dsdy if dsdy is not None else _zeros(nlev))

    ws.wflux.fill(0.0)
    ws.sflux.fill(0.0)
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

    fill_field_scalar(ws.wflux, wflux)
    fill_field_scalar(ws.sflux, sflux)
    return ws


def _step_workspace(
    ws: SalinityWorkspace,
    state: ReadyMeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    *,
    s_adv: bool = False,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
) -> None:
    step_salinity(
        1,
        nlev,
        dt,
        cnpar,
        state.avmolS,
        int(w_adv_active),
        w_adv_discr,
        int(s_adv),
        ws.S,
        ws.h,
        ws.w,
        ws.u,
        ws.v,
        ws.nus,
        ws.gams,
        ws.Sobs,
        ws.tau_r,
        ws.wflux,
        ws.sflux,
        ws.dsdx,
        ws.dsdy,
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


def _sync_workspace_to_state(ws: SalinityWorkspace, state: ReadyMeanflowState) -> None:
    state.S[:] = read_field_array(ws.S)
    state.avh[:] = read_field_array(ws.avh)


def _run_single_column(
    state: ReadyMeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    *,
    workspace: SalinityWorkspace | None = None,
    wflux: float = 0.0,
    sflux: float = 0.0,
    nus: np.ndarray | None = None,
    gams: np.ndarray | None = None,
    dsdx: np.ndarray | None = None,
    dsdy: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    Sobs: np.ndarray | None = None,
    s_adv: bool = False,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
) -> SalinityWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        wflux=wflux,
        sflux=sflux,
        nus=nus,
        gams=gams,
        dsdx=dsdx,
        dsdy=dsdy,
        tau_r=tau_r,
        Sobs=Sobs,
    )
    _step_workspace(
        ws,
        state,
        nlev,
        dt,
        cnpar,
        s_adv=s_adv,
        w_adv_active=w_adv_active,
        w_adv_discr=w_adv_discr,
    )
    _sync_workspace_to_state(ws, state)
    return ws


def test_import() -> None:
    from pygotm.meanflow.salinity import step_salinity as _s  # noqa: F401

    assert callable(_s)


def test_smoke_workspace_creation() -> None:
    ws = SalinityWorkspace(nlev=_NLEV, n_cols=1)
    assert ws is not None


def test_smoke_step_salinity_kernel() -> None:
    state = _make_state()
    _run_single_column(
        state,
        _NLEV,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=np.full(_NLEV + 1, 1.0e-4),
        gams=_zeros(_NLEV),
    )


def test_wflux_is_carried_through_api_without_affecting_gotm_solution() -> None:
    nlev = _NLEV
    wet = _make_state(nlev=nlev)
    dry = _make_state(nlev=nlev)
    nus = np.full(nlev + 1, 1.0e-4, dtype=np.float64)

    _run_single_column(
        wet,
        nlev,
        _DT,
        _CNPAR,
        wflux=1.0e-6,
        sflux=0.0,
        nus=nus,
        gams=_zeros(nlev),
    )
    _run_single_column(
        dry,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=nus,
        gams=_zeros(nlev),
    )

    np.testing.assert_allclose(wet.S, dry.S, rtol=1.0e-12, atol=1.0e-12)


def test_physical_bounds_salinity() -> None:
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    ws = _prepare_workspace(
        state,
        nlev,
        wflux=0.0,
        sflux=0.0,
        nus=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        gams=_zeros(nlev),
    )

    for _ in range(100):
        _step_workspace(ws, state, nlev, _DT, _CNPAR)

    _sync_workspace_to_state(ws, state)
    assert float(np.min(state.S[1:])) >= 0.0
    assert float(np.max(state.S[1:])) < 100.0


def test_sinusoidal_decay_analytic() -> None:
    nlev = 20
    depth = 10.0
    nu = 1.0e-2
    dt = 10.0
    cnpar = 1.0
    n_steps = 500
    t_total = n_steps * dt
    decay_rate = nu * (np.pi / depth) ** 2

    dz = depth / nlev
    z_k = np.array([(k - 0.5) * dz for k in range(nlev + 1)], dtype=np.float64)

    amplitude = 2.0
    background = 35.0

    state = _make_state(nlev=nlev, depth=depth, S_init=background)
    assert state.S is not None
    state.S[:] = background + amplitude * np.cos(np.pi * z_k / depth)
    state.S[0] = background

    ws = _prepare_workspace(
        state,
        nlev,
        wflux=0.0,
        sflux=0.0,
        nus=np.full(nlev + 1, nu, dtype=np.float64),
        gams=_zeros(nlev),
    )

    for _ in range(n_steps):
        _step_workspace(ws, state, nlev, dt, cnpar)

    _sync_workspace_to_state(ws, state)
    expected = background + amplitude * np.cos(np.pi * z_k / depth) * np.exp(
        -decay_rate * t_total
    )

    np.testing.assert_allclose(state.S[1:], expected[1:], rtol=0.02)


def test_upper_neumann_bc_salt_input() -> None:
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    top_before = float(state.S[nlev])

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=-1.0,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
    )

    assert float(state.S[nlev]) > top_before


def test_lower_neumann_bc_no_bottom_flux() -> None:
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    bottom_before = float(state.S[1])

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
    )

    assert abs(float(state.S[1]) - bottom_before) < 1.0e-14


def test_zero_index_unchanged() -> None:
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    state.S[0] = -999.0

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=np.full(nlev + 1, 1.0e-4),
        gams=_zeros(nlev),
    )

    assert float(state.S[0]) == -999.0


def test_zero_all_forcing_unchanged() -> None:
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    S_before = state.S.copy()

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
    )

    np.testing.assert_allclose(state.S[1:], S_before[1:], atol=1.0e-14)


def test_patankar_no_negative_salinity() -> None:
    """S=0 exactly would divide by zero, matching the Fortran behaviour."""

    state = _make_state(S_init=1.0e-2)
    nlev = _NLEV
    ws = _prepare_workspace(
        state,
        nlev,
        wflux=0.0,
        sflux=1.0,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
    )

    for _ in range(25):
        _step_workspace(ws, state, nlev, _DT, _CNPAR)

    _sync_workspace_to_state(ws, state)
    assert float(np.min(state.S[1:])) >= 0.0


def test_relaxation_towards_observed() -> None:
    nlev = _NLEV
    state = _make_state(S_init=35.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
        tau_r=np.full(nlev + 1, _DT, dtype=np.float64),
        Sobs=np.full(nlev + 1, 38.0, dtype=np.float64),
    )

    for k in range(1, nlev + 1):
        assert float(state.S[k]) > 35.0
        assert float(state.S[k]) < 38.0


def test_multicol_parity() -> None:
    nlev = _NLEV
    state = _make_state(S_init=35.0)
    assert state.S is not None
    assert state.h is not None
    nus = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    gams = _zeros(nlev)

    single = _prepare_workspace(
        state,
        nlev,
        wflux=0.0,
        sflux=0.0,
        nus=nus,
        gams=gams,
    )
    _step_workspace(single, state, nlev, _DT, _CNPAR)
    single_result = read_field_array(single.S)

    multi = SalinityWorkspace(nlev=nlev, n_cols=2)
    for col in range(2):
        fill_field_from_array(multi.S, state.S, col=col)
        fill_field_from_array(multi.h, state.h, col=col)
        fill_field_from_array(multi.w, state.w, col=col)
        fill_field_from_array(multi.u, state.u, col=col)
        fill_field_from_array(multi.v, state.v, col=col)
        fill_field_from_array(multi.nus, nus, col=col)
        fill_field_from_array(multi.gams, gams, col=col)
        fill_field_from_array(multi.Sobs, _zeros(nlev), col=col)
        fill_field_from_array(
            multi.tau_r,
            np.full(nlev + 1, _LONG, dtype=np.float64),
            col=col,
        )
        fill_field_from_array(multi.dsdx, _zeros(nlev), col=col)
        fill_field_from_array(multi.dsdy, _zeros(nlev), col=col)
        fill_field_scalar(multi.wflux, 0.0, col=col)
        fill_field_scalar(multi.sflux, 0.0, col=col)

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

    step_salinity(
        2,
        nlev,
        _DT,
        _CNPAR,
        state.avmolS,
        0,
        4,
        0,
        multi.S,
        multi.h,
        multi.w,
        multi.u,
        multi.v,
        multi.nus,
        multi.gams,
        multi.Sobs,
        multi.tau_r,
        multi.wflux,
        multi.sflux,
        multi.dsdx,
        multi.dsdy,
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

    np.testing.assert_allclose(read_field_array(multi.S, col=0), single_result)
    np.testing.assert_allclose(read_field_array(multi.S, col=1), single_result)


def test_no_nan_inf_typical_forcing() -> None:
    nlev = _NLEV
    state = _make_state(S_init=35.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.1,
        nus=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        gams=_zeros(nlev),
    )

    assert np.all(np.isfinite(state.S[1:]))


def test_no_nan_inf_zero_diffusivity() -> None:
    nlev = _NLEV
    state = _make_state(S_init=35.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.0,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
    )

    assert np.all(np.isfinite(state.S[1:]))


def test_no_nan_inf_patankar_path() -> None:
    nlev = _NLEV
    state = _make_state(S_init=35.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        wflux=0.0,
        sflux=0.5,
        nus=np.full(nlev + 1, 1.0e-4, dtype=np.float64),
        gams=_zeros(nlev),
    )

    assert np.all(np.isfinite(state.S[1:]))
