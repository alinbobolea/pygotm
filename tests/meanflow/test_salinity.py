"""Tests for pygotm.meanflow.salinity — salinity diffusion equation."""

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState, init_meanflow, post_init_meanflow
from pygotm.meanflow.salinity import salinity, step_salinity
from pygotm.meanflow.updategrid import updategrid

_NLEV = 20
_DEPTH = 50.0
_DT = 3600.0
_CNPAR = 0.6
_LONG = 1.0e15


def _make_state(nlev=_NLEV, depth=_DEPTH, avmolS=1.1e-9, S_init=35.0):
    state = MeanflowState()
    init_meanflow(state, avmolS=avmolS)
    state.depth = depth
    state.grid_method = 0
    post_init_meanflow(state, nlev, latitude=0.0)
    updategrid(state, nlev, _DT, zeta=0.0)
    assert state.S is not None
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
    wflux=0.0,
    sflux=0.0,
    nus=None,
    gams=None,
    Sobs=None,
    tau_r=None,
    dsdx=None,
    dsdy=None,
    s_adv=False,
    w_adv_active=False,
    w_adv_discr=4,
):
    salinity(
        state,
        nlev,
        dt,
        cnpar,
        wflux,
        sflux,
        nus if nus is not None else _zeros(nlev),
        gams if gams is not None else _zeros(nlev),
        Sobs=Sobs,
        tau_r=tau_r,
        dsdx=dsdx,
        dsdy=dsdy,
        w_adv_active=w_adv_active,
        w_adv_discr=w_adv_discr,
        s_adv=s_adv,
    )


def test_import():
    from pygotm.meanflow.salinity import step_salinity as _s  # noqa: F401

    assert callable(_s)


def test_smoke():
    state = _make_state()
    _run_step(
        state, _NLEV, _DT, _CNPAR, nus=np.full(_NLEV + 1, 1.0e-4), gams=_zeros(_NLEV)
    )


def test_wflux_is_carried_through_api_without_affecting_gotm_solution():
    nlev = _NLEV
    nus = np.full(nlev + 1, 1.0e-4, dtype=np.float64)
    wet = _make_state(nlev=nlev)
    dry = _make_state(nlev=nlev)
    _run_step(
        wet, nlev, _DT, _CNPAR, wflux=1.0e-6, sflux=0.0, nus=nus, gams=_zeros(nlev)
    )
    _run_step(dry, nlev, _DT, _CNPAR, wflux=0.0, sflux=0.0, nus=nus, gams=_zeros(nlev))
    np.testing.assert_allclose(wet.S, dry.S, rtol=1.0e-12, atol=1.0e-12)


def test_physical_bounds_salinity():
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    nus = np.full(nlev + 1, 1.0e-3, dtype=np.float64)
    for _ in range(100):
        _run_step(state, nlev, _DT, _CNPAR, nus=nus, gams=_zeros(nlev))
    assert float(np.min(state.S[1:])) >= 0.0
    assert float(np.max(state.S[1:])) < 100.0


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
    background = 35.0

    state = _make_state(nlev=nlev, depth=depth, S_init=background)
    assert state.S is not None
    state.S[:] = background + amplitude * np.cos(np.pi * z_k / depth)
    state.S[0] = background

    nus = np.full(nlev + 1, nu, dtype=np.float64)
    for _ in range(n_steps):
        _run_step(state, nlev, dt, cnpar, nus=nus, gams=_zeros(nlev))

    expected = background + amplitude * np.cos(np.pi * z_k / depth) * np.exp(
        -decay_rate * t_total
    )
    np.testing.assert_allclose(state.S[1:], expected[1:], rtol=0.02)


def test_upper_neumann_bc_salt_input():
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    top_before = float(state.S[nlev])
    _run_step(
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


def test_lower_neumann_bc_no_bottom_flux():
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    bottom_before = float(state.S[1])
    _run_step(
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


def test_zero_index_unchanged():
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    state.S[0] = -999.0
    _run_step(
        state, nlev, _DT, _CNPAR, nus=np.full(nlev + 1, 1.0e-4), gams=_zeros(nlev)
    )
    assert float(state.S[0]) == -999.0


def test_zero_all_forcing_unchanged():
    state = _make_state(S_init=35.0)
    nlev = _NLEV
    assert state.S is not None
    S_before = state.S.copy()
    _run_step(state, nlev, _DT, _CNPAR, nus=_zeros(nlev), gams=_zeros(nlev))
    np.testing.assert_allclose(state.S[1:], S_before[1:], atol=1.0e-14)


def test_patankar_no_negative_salinity():
    """S=0 exactly would divide by zero, matching the Fortran behaviour."""
    state = _make_state(S_init=1.0e-2)
    nlev = _NLEV
    for _ in range(25):
        _run_step(
            state,
            nlev,
            _DT,
            _CNPAR,
            wflux=0.0,
            sflux=1.0,
            nus=_zeros(nlev),
            gams=_zeros(nlev),
        )
    assert float(np.min(state.S[1:])) >= 0.0


def test_relaxation_towards_observed():
    nlev = _NLEV
    state = _make_state(S_init=35.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        nus=_zeros(nlev),
        gams=_zeros(nlev),
        tau_r=np.full(nlev + 1, _DT, dtype=np.float64),
        Sobs=np.full(nlev + 1, 38.0, dtype=np.float64),
    )
    assert state.S is not None
    for k in range(1, nlev + 1):
        assert float(state.S[k]) > 35.0
        assert float(state.S[k]) < 38.0


def test_multicol_parity():
    nlev = _NLEV
    batch_size = 2
    state = _make_state(S_init=35.0)
    assert state.S is not None
    nus = np.full(nlev + 1, 1.0e-3, dtype=np.float64)

    state_ref = _make_state(S_init=35.0)
    _run_step(state_ref, nlev, _DT, _CNPAR, nus=nus, gams=_zeros(nlev))
    s_single = state_ref.S.copy()

    S_b = np.tile(state.S, (batch_size, 1)).astype(np.float64)
    h_b = np.tile(state.h, (batch_size, 1)).astype(np.float64)
    w_b = np.tile(state.w, (batch_size, 1)).astype(np.float64)
    u_b = np.tile(state.u, (batch_size, 1)).astype(np.float64)
    v_b = np.tile(state.v, (batch_size, 1)).astype(np.float64)
    nus_b = np.tile(nus, (batch_size, 1)).astype(np.float64)
    gams_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    Sobs_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    tau_r_b = np.full((batch_size, nlev + 1), _LONG, dtype=np.float64)
    diff_s_up_b = np.zeros(batch_size, dtype=np.float64)
    dsdx_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
    dsdy_b = np.zeros((batch_size, nlev + 1), dtype=np.float64)
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

    step_salinity(
        batch_size,
        nlev,
        _DT,
        _CNPAR,
        state.avmolS,
        0,
        4,
        0,
        S_b,
        h_b,
        w_b,
        u_b,
        v_b,
        nus_b,
        gams_b,
        Sobs_b,
        tau_r_b,
        diff_s_up_b,
        dsdx_b,
        dsdy_b,
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

    np.testing.assert_allclose(S_b[0], s_single)
    np.testing.assert_allclose(S_b[1], s_single)


def test_no_nan_inf_typical_forcing():
    nlev = _NLEV
    state = _make_state(S_init=35.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        sflux=0.1,
        nus=np.full(nlev + 1, 1.0e-3, dtype=np.float64),
        gams=_zeros(nlev),
    )
    assert np.all(np.isfinite(state.S[1:]))


def test_no_nan_inf_zero_diffusivity():
    nlev = _NLEV
    state = _make_state(S_init=35.0)
    _run_step(state, nlev, _DT, _CNPAR, nus=_zeros(nlev), gams=_zeros(nlev))
    assert np.all(np.isfinite(state.S[1:]))


def test_no_nan_inf_patankar_path():
    nlev = _NLEV
    state = _make_state(S_init=35.0)
    _run_step(
        state,
        nlev,
        _DT,
        _CNPAR,
        sflux=0.5,
        nus=np.full(nlev + 1, 1.0e-4, dtype=np.float64),
        gams=_zeros(nlev),
    )
    assert np.all(np.isfinite(state.S[1:]))
