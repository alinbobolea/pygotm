"""Tests for pygotm.meanflow.temperature — temperature diffusion equation.

All execution goes through the production Taichi kernel ``step_temperature``.
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
from pygotm.meanflow.temperature import TemperatureWorkspace, step_temperature
from pygotm.meanflow.updategrid import updategrid

_NLEV = 20
_DEPTH = 50.0
_DT = 3600.0
_CNPAR = 0.6
_RHO0 = 1027.0
_CP = 3991.86795711963
_A_DEFAULT = 0.58
_G1_DEFAULT = 0.35
_G2_DEFAULT = 23.0
_LONG = 1.0e15


def _make_state(
    nlev: int = _NLEV,
    depth: float = _DEPTH,
    avmolT: float = 1.4e-7,
    T_init: float = 10.0,
    S_init: float = 35.0,
) -> ReadyMeanflowState:
    state = MeanflowState()
    init_meanflow(state, avmolT=avmolT)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    ready_state = require_meanflow_state(state)
    ready_state.T[:] = T_init
    ready_state.S[:] = S_init
    return ready_state


def _zeros(nlev: int = _NLEV) -> np.ndarray:
    return np.zeros(nlev + 1, dtype=np.float64)


def _prepare_workspace(
    state: ReadyMeanflowState,
    nlev: int,
    *,
    workspace: TemperatureWorkspace | None = None,
    I_0: float = 0.0,
    wflux: float = 0.0,
    hflux: float = 0.0,
    nuh: np.ndarray | None = None,
    gamh: np.ndarray | None = None,
    dtdx: np.ndarray | None = None,
    dtdy: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    Tobs: np.ndarray | None = None,
) -> TemperatureWorkspace:
    ws = (
        workspace
        if workspace is not None
        else TemperatureWorkspace(nlev=nlev, n_cols=1)
    )
    fill_field_from_array(ws.T, state.T)
    fill_field_from_array(ws.S, state.S)
    fill_field_from_array(ws.h, state.h)
    fill_field_from_array(ws.w, state.w)
    fill_field_from_array(ws.u, state.u)
    fill_field_from_array(ws.v, state.v)
    fill_field_from_array(ws.nuh, nuh if nuh is not None else _zeros(nlev))
    fill_field_from_array(ws.gamh, gamh if gamh is not None else _zeros(nlev))
    fill_field_from_array(ws.bioshade, state.bioshade)
    fill_field_from_array(ws.Tobs, Tobs if Tobs is not None else _zeros(nlev))
    fill_field_from_array(
        ws.tau_r,
        tau_r if tau_r is not None else np.full(nlev + 1, _LONG, dtype=np.float64),
    )
    fill_field_from_array(ws.dtdx, dtdx if dtdx is not None else _zeros(nlev))
    fill_field_from_array(ws.dtdy, dtdy if dtdy is not None else _zeros(nlev))

    ws.I_0.fill(0.0)
    ws.wflux.fill(0.0)
    ws.hflux.fill(0.0)
    ws.rad.fill(0.0)
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

    fill_field_scalar(ws.I_0, I_0)
    fill_field_scalar(ws.wflux, wflux)
    fill_field_scalar(ws.hflux, hflux)
    return ws


def _step_workspace(
    ws: TemperatureWorkspace,
    state: ReadyMeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    *,
    t_adv: bool = False,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
) -> None:
    step_temperature(
        1,
        nlev,
        dt,
        cnpar,
        state.avmolT,
        _RHO0,
        _CP,
        _A_DEFAULT,
        _G1_DEFAULT,
        _G2_DEFAULT,
        int(w_adv_active),
        w_adv_discr,
        int(t_adv),
        ws.T,
        ws.S,
        ws.h,
        ws.w,
        ws.u,
        ws.v,
        ws.nuh,
        ws.gamh,
        ws.bioshade,
        ws.rad,
        ws.Tobs,
        ws.tau_r,
        ws.I_0,
        ws.wflux,
        ws.hflux,
        ws.dtdx,
        ws.dtdy,
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


def _sync_workspace_to_state(
    ws: TemperatureWorkspace, state: ReadyMeanflowState
) -> None:
    state.T[:] = read_field_array(ws.T)
    state.avh[:] = read_field_array(ws.avh)
    state.rad[:] = read_field_array(ws.rad)


def _run_single_column(
    state: ReadyMeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    *,
    workspace: TemperatureWorkspace | None = None,
    I_0: float = 0.0,
    wflux: float = 0.0,
    hflux: float = 0.0,
    nuh: np.ndarray | None = None,
    gamh: np.ndarray | None = None,
    dtdx: np.ndarray | None = None,
    dtdy: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    Tobs: np.ndarray | None = None,
    t_adv: bool = False,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
) -> TemperatureWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        I_0=I_0,
        wflux=wflux,
        hflux=hflux,
        nuh=nuh,
        gamh=gamh,
        dtdx=dtdx,
        dtdy=dtdy,
        tau_r=tau_r,
        Tobs=Tobs,
    )
    _step_workspace(
        ws,
        state,
        nlev,
        dt,
        cnpar,
        t_adv=t_adv,
        w_adv_active=w_adv_active,
        w_adv_discr=w_adv_discr,
    )
    _sync_workspace_to_state(ws, state)
    return ws


def test_import() -> None:
    from pygotm.meanflow.temperature import step_temperature as _t  # noqa: F401

    assert callable(_t)


def test_smoke_workspace_creation() -> None:
    ws = TemperatureWorkspace(nlev=_NLEV, n_cols=1)
    assert ws is not None


def test_smoke_step_temperature_kernel() -> None:
    state = _make_state()
    _run_single_column(
        state,
        _NLEV,
        _DT,
        _CNPAR,
        I_0=100.0,
        wflux=0.0,
        hflux=-50.0,
        nuh=np.full(_NLEV + 1, 1.0e-4),
        gamh=_zeros(_NLEV),
    )


def test_physical_bounds_temperature() -> None:
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    nuh = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    gamh = _zeros(nlev)
    ws = _prepare_workspace(
        state,
        nlev,
        I_0=200.0,
        wflux=0.0,
        hflux=-100.0,
        nuh=nuh,
        gamh=gamh,
    )

    for _ in range(100):
        _step_workspace(ws, state, nlev, _DT, _CNPAR)

    _sync_workspace_to_state(ws, state)
    assert float(np.min(state.T[1:])) > -5.0
    assert float(np.max(state.T[1:])) < 50.0


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
    background = 15.0

    state = _make_state(nlev=nlev, depth=depth)
    assert state.T is not None
    state.T[:] = background + amplitude * np.cos(np.pi * z_k / depth)
    state.T[0] = background

    nuh = np.full(nlev + 1, nu, dtype=np.float64)
    ws = _prepare_workspace(
        state,
        nlev,
        I_0=0.0,
        wflux=0.0,
        hflux=0.0,
        nuh=nuh,
        gamh=_zeros(nlev),
    )

    for _ in range(n_steps):
        _step_workspace(ws, state, nlev, dt, cnpar)

    _sync_workspace_to_state(ws, state)
    expected = background + amplitude * np.cos(np.pi * z_k / depth) * np.exp(
        -decay_rate * t_total
    )

    np.testing.assert_allclose(state.T[1:], expected[1:], rtol=0.02)


def test_radiation_energy_conservation() -> None:
    nlev = 40
    depth = 200.0
    dt = 1.0
    cnpar = 1.0
    I_0 = 500.0

    state = _make_state(nlev=nlev, depth=depth, T_init=15.0)
    assert state.T is not None
    assert state.h is not None
    T_before = state.T.copy()

    _run_single_column(
        state,
        nlev,
        dt,
        cnpar,
        I_0=I_0,
        wflux=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )

    dT = state.T[1:] - T_before[1:]
    heat_added = float(np.sum(dT * state.h[1:]))
    expected = I_0 * dt / (_RHO0 * _CP)
    assert abs(heat_added - expected) < 0.01 * expected + 1.0e-12


def test_upper_neumann_bc_surface_heating() -> None:
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    top_before = float(state.T[nlev])

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        wflux=0.0,
        hflux=-500.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )

    assert float(state.T[nlev]) > top_before


def test_lower_neumann_bc_no_bottom_flux() -> None:
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    bottom_before = float(state.T[1])

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        wflux=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )

    assert abs(float(state.T[1]) - bottom_before) < 1.0e-14


def test_zero_index_unchanged() -> None:
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    state.T[0] = -999.0

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=100.0,
        wflux=0.0,
        hflux=-50.0,
        nuh=np.full(nlev + 1, 1.0e-4),
        gamh=_zeros(nlev),
    )

    assert float(state.T[0]) == -999.0


def test_zero_surface_flux_zero_radiation() -> None:
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    T_before = state.T.copy()

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        wflux=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )

    np.testing.assert_allclose(state.T[1:], T_before[1:], atol=1.0e-14)


def test_ice_correction_suppresses_warming_flux() -> None:
    state = _make_state(T_init=-3.0, S_init=35.0)
    nlev = _NLEV
    assert state.T is not None
    T_before = state.T.copy()

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        wflux=0.0,
        hflux=-200.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )

    np.testing.assert_allclose(state.T[1:], T_before[1:], atol=1.0e-14)


def test_relaxation_towards_observed() -> None:
    nlev = _NLEV
    state = _make_state(T_init=10.0)
    Tobs = np.full(nlev + 1, 20.0, dtype=np.float64)
    tau_r = np.full(nlev + 1, _DT, dtype=np.float64)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        wflux=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
        tau_r=tau_r,
        Tobs=Tobs,
    )

    for k in range(1, nlev + 1):
        assert float(state.T[k]) > 10.0
        assert float(state.T[k]) < 20.0


def test_multicol_parity() -> None:
    nlev = _NLEV
    state = _make_state(T_init=15.0)
    assert state.T is not None
    assert state.S is not None
    assert state.h is not None
    nuh = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    gamh = _zeros(nlev)
    I_0 = 150.0
    hflux = -100.0

    single = _prepare_workspace(
        state,
        nlev,
        I_0=I_0,
        wflux=0.0,
        hflux=hflux,
        nuh=nuh,
        gamh=gamh,
    )
    _step_workspace(single, state, nlev, _DT, _CNPAR)
    single_result = read_field_array(single.T)

    multi = TemperatureWorkspace(nlev=nlev, n_cols=2)
    for col in range(2):
        fill_field_from_array(multi.T, state.T, col=col)
        fill_field_from_array(multi.S, state.S, col=col)
        fill_field_from_array(multi.h, state.h, col=col)
        fill_field_from_array(multi.w, state.w, col=col)
        fill_field_from_array(multi.u, state.u, col=col)
        fill_field_from_array(multi.v, state.v, col=col)
        fill_field_from_array(multi.nuh, nuh, col=col)
        fill_field_from_array(multi.gamh, gamh, col=col)
        fill_field_from_array(multi.bioshade, state.bioshade, col=col)
        fill_field_from_array(multi.Tobs, _zeros(nlev), col=col)
        fill_field_from_array(
            multi.tau_r,
            np.full(nlev + 1, _LONG, dtype=np.float64),
            col=col,
        )
        fill_field_from_array(multi.dtdx, _zeros(nlev), col=col)
        fill_field_from_array(multi.dtdy, _zeros(nlev), col=col)
        fill_field_scalar(multi.I_0, I_0, col=col)
        fill_field_scalar(multi.wflux, 0.0, col=col)
        fill_field_scalar(multi.hflux, hflux, col=col)

    multi.rad.fill(0.0)
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

    step_temperature(
        2,
        nlev,
        _DT,
        _CNPAR,
        state.avmolT,
        _RHO0,
        _CP,
        _A_DEFAULT,
        _G1_DEFAULT,
        _G2_DEFAULT,
        0,
        4,
        0,
        multi.T,
        multi.S,
        multi.h,
        multi.w,
        multi.u,
        multi.v,
        multi.nuh,
        multi.gamh,
        multi.bioshade,
        multi.rad,
        multi.Tobs,
        multi.tau_r,
        multi.I_0,
        multi.wflux,
        multi.hflux,
        multi.dtdx,
        multi.dtdy,
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

    np.testing.assert_allclose(read_field_array(multi.T, col=0), single_result)
    np.testing.assert_allclose(read_field_array(multi.T, col=1), single_result)


def test_no_nan_inf_typical_forcing() -> None:
    nlev = _NLEV
    state = _make_state(T_init=12.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=300.0,
        wflux=0.0,
        hflux=-200.0,
        nuh=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        gamh=_zeros(nlev),
    )

    assert np.all(np.isfinite(state.T[1:]))
    assert np.all(np.isfinite(state.rad))


def test_no_nan_inf_extreme_radiation() -> None:
    nlev = 5
    state = _make_state(nlev=nlev, depth=5.0, T_init=5.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=1000.0,
        wflux=0.0,
        hflux=0.0,
        nuh=np.full(nlev + 1, 1.0e-4, dtype=np.float64),
        gamh=_zeros(nlev),
    )

    assert np.all(np.isfinite(state.T[1:]))


def test_no_nan_inf_zero_diffusivity() -> None:
    nlev = _NLEV
    state = _make_state(T_init=15.0)

    _run_single_column(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=200.0,
        wflux=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )

    assert np.all(np.isfinite(state.T[1:]))
