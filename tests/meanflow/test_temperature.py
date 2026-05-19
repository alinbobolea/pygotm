"""Tests for pygotm.meanflow.temperature — temperature diffusion equation."""

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.temperature import (
    _A_DEFAULT,
    _G1_DEFAULT,
    _G2_DEFAULT,
    step_temperature,
    temperature,
)
from pygotm.meanflow.updategrid import updategrid

_NLEV = 20
_DEPTH = 50.0
_DT = 3600.0
_CNPAR = 0.6
_RHO0 = 1027.0
_CP = 3991.86795711963
_LONG = 1.0e15


def _make_state(nlev=_NLEV, depth=_DEPTH, avmolT=1.4e-7, T_init=10.0, S_init=35.0):
    state = MeanflowState()
    init_meanflow(state, avmolT=avmolT)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    assert state.T is not None
    assert state.S is not None
    state.T[:] = T_init
    state.S[:] = S_init
    return state


def _zeros(nlev=_NLEV):
    return np.zeros(nlev + 1, dtype=np.float64)


def _run_step(
    state,
    nlev,
    dt,
    cnpar,
    *,
    I_0=0.0,
    wflux=0.0,
    hflux=0.0,
    nuh=None,
    gamh=None,
    Tobs=None,
    tau_r=None,
    dtdx=None,
    dtdy=None,
    t_adv=False,
    w_adv_active=False,
    w_adv_discr=4,
    apply_simple_ice_correction=False,
):
    temperature(
        state,
        nlev,
        dt,
        cnpar,
        I_0,
        wflux,
        hflux,
        nuh if nuh is not None else _zeros(nlev),
        gamh if gamh is not None else _zeros(nlev),
        rho0=_RHO0,
        cp=_CP,
        Tobs=Tobs,
        tau_r=tau_r,
        dtdx=dtdx,
        dtdy=dtdy,
        w_adv_active=w_adv_active,
        w_adv_discr=w_adv_discr,
        t_adv=t_adv,
        apply_simple_ice_correction=apply_simple_ice_correction,
    )


def test_import():
    from pygotm.meanflow.temperature import step_temperature as _t  # noqa: F401

    assert callable(_t)


def test_smoke():
    state = _make_state()
    _run_step(
        state,
        _NLEV,
        _DT,
        _CNPAR,
        I_0=100.0,
        hflux=-50.0,
        nuh=np.full(_NLEV + 1, 1.0e-4),
        gamh=_zeros(_NLEV),
    )


def test_physical_bounds_temperature():
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    nuh = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    for _ in range(100):
        _run_step(
            state,
            nlev,
            _DT,
            _CNPAR,
            I_0=200.0,
            hflux=-100.0,
            nuh=nuh,
            gamh=_zeros(nlev),
        )
    assert float(np.min(state.T[1:])) > -5.0
    assert float(np.max(state.T[1:])) < 50.0


def test_sinusoidal_decay_analytic():
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
    for _ in range(n_steps):
        _run_step(
            state, nlev, dt, cnpar, I_0=0.0, hflux=0.0, nuh=nuh, gamh=_zeros(nlev)
        )

    expected = background + amplitude * np.cos(np.pi * z_k / depth) * np.exp(
        -decay_rate * t_total
    )
    np.testing.assert_allclose(state.T[1:], expected[1:], rtol=0.02)


def test_radiation_energy_conservation():
    nlev = 40
    depth = 200.0
    dt = 1.0
    cnpar = 1.0
    I_0 = 500.0

    state = _make_state(nlev=nlev, depth=depth, T_init=15.0)
    assert state.T is not None
    assert state.h is not None
    T_before = state.T.copy()

    _run_step(
        state, nlev, dt, cnpar, I_0=I_0, hflux=0.0, nuh=_zeros(nlev), gamh=_zeros(nlev)
    )

    dT = state.T[1:] - T_before[1:]
    heat_added = float(np.sum(dT * state.h[1:]))
    expected = I_0 * dt / (_RHO0 * _CP)
    assert abs(heat_added - expected) < 0.01 * expected + 1.0e-12


def test_upper_neumann_bc_surface_heating():
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    top_before = float(state.T[nlev])
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        hflux=-500.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
        apply_simple_ice_correction=True,
    )
    assert float(state.T[nlev]) > top_before


def test_lower_neumann_bc_no_bottom_flux():
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    bottom_before = float(state.T[1])
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )
    assert abs(float(state.T[1]) - bottom_before) < 1.0e-14


def test_zero_index_unchanged():
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    state.T[0] = -999.0
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=100.0,
        hflux=-50.0,
        nuh=np.full(nlev + 1, 1.0e-4),
        gamh=_zeros(nlev),
    )
    assert float(state.T[0]) == -999.0


def test_zero_surface_flux_zero_radiation():
    state = _make_state(T_init=10.0)
    nlev = _NLEV
    assert state.T is not None
    T_before = state.T.copy()
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )
    np.testing.assert_allclose(state.T[1:], T_before[1:], atol=1.0e-14)


def test_ice_correction_suppresses_warming_flux():
    state = _make_state(T_init=-3.0, S_init=35.0)
    nlev = _NLEV
    assert state.T is not None
    T_before = state.T.copy()
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        hflux=-200.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
        apply_simple_ice_correction=True,
    )
    np.testing.assert_allclose(state.T[1:], T_before[1:], atol=1.0e-14)


def test_relaxation_towards_observed():
    nlev = _NLEV
    state = _make_state(T_init=10.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=0.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
        tau_r=np.full(nlev + 1, _DT, dtype=np.float64),
        Tobs=np.full(nlev + 1, 20.0, dtype=np.float64),
    )
    assert state.T is not None
    for k in range(1, nlev + 1):
        assert float(state.T[k]) > 10.0
        assert float(state.T[k]) < 20.0


def test_multicol_parity():
    nlev = _NLEV
    batch_size = 2
    I_0 = 150.0
    hflux = -100.0
    nuh = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    gamh = _zeros(nlev)

    state = _make_state(T_init=15.0)
    assert state.T is not None
    assert state.S is not None

    state_ref = _make_state(T_init=15.0)
    _run_step(state_ref, nlev, _DT, _CNPAR, I_0=I_0, hflux=hflux, nuh=nuh, gamh=gamh)
    T_single = state_ref.T.copy()

    diff_t_up = -hflux / (_RHO0 * _CP)

    T_b = np.tile(state.T, (batch_size, 1)).astype(np.float64)
    S_b = np.tile(state.S, (batch_size, 1)).astype(np.float64)
    h_b = np.tile(state.h, (batch_size, 1)).astype(np.float64)
    w_b = np.tile(state.w, (batch_size, 1)).astype(np.float64)
    u_b = np.tile(state.u, (batch_size, 1)).astype(np.float64)
    v_b = np.tile(state.v, (batch_size, 1)).astype(np.float64)
    nuh_b = np.tile(nuh, (batch_size, 1)).astype(np.float64)
    gamh_b = np.tile(gamh, (batch_size, 1)).astype(np.float64)
    assert state.bioshade is not None
    bioshade_b = np.tile(state.bioshade, (batch_size, 1)).astype(np.float64)
    rad_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    Tobs_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    tau_r_b = np.full((batch_size, nlev + 1), _LONG, dtype=np.float64)
    i_0_b = np.full(batch_size, I_0, dtype=np.float64)
    diff_t_up_b = np.full(batch_size, diff_t_up, dtype=np.float64)
    dtdx_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    dtdy_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
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

    step_temperature(
        batch_size,
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
        T_b,
        S_b,
        h_b,
        w_b,
        u_b,
        v_b,
        nuh_b,
        gamh_b,
        bioshade_b,
        rad_b,
        Tobs_b,
        tau_r_b,
        i_0_b,
        diff_t_up_b,
        0,
        dtdx_b,
        dtdy_b,
        avh_b,
        q_sour_b,
        l_sour_b,
        au_b,
        bu_b,
        cu_b,
        du_b,
        ru_b,
        qu_b,
        adv_cu_b,
    )

    np.testing.assert_allclose(T_b[0], T_single)
    np.testing.assert_allclose(T_b[1], T_single)


def test_no_nan_inf_typical_forcing():
    nlev = _NLEV
    state = _make_state(T_init=12.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=300.0,
        hflux=-200.0,
        nuh=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        gamh=_zeros(nlev),
    )
    assert state.T is not None
    assert state.rad is not None
    assert np.all(np.isfinite(state.T[1:]))
    assert np.all(np.isfinite(state.rad))


def test_no_nan_inf_extreme_radiation():
    nlev = 5
    state = _make_state(nlev=nlev, depth=5.0, T_init=5.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=1000.0,
        hflux=0.0,
        nuh=np.full(nlev + 1, 1.0e-4, dtype=np.float64),
        gamh=_zeros(nlev),
    )
    assert state.T is not None
    assert np.all(np.isfinite(state.T[1:]))


def test_no_nan_inf_zero_diffusivity():
    nlev = _NLEV
    state = _make_state(T_init=15.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        I_0=200.0,
        hflux=0.0,
        nuh=_zeros(nlev),
        gamh=_zeros(nlev),
    )
    assert state.T is not None
    assert np.all(np.isfinite(state.T[1:]))
