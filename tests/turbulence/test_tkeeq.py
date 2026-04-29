"""Tests for pygotm.turbulence.tkeeq."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.tkeeq import TKEEquationWorkspace, step_tkeeq
from pygotm.turbulence.turbulence import (
    Dirichlet,
    Neumann,
    TurbulenceState,
    init_turbulence,
    logarithmic,
    post_init_turbulence,
)

_NLEV = 12
_DT = 60.0
_DEPTH = 24.0


def make_equidistant_h(nlev: int, depth: float) -> np.ndarray:
    h = np.full(nlev + 1, depth / nlev, dtype=np.float64)
    h[0] = 0.0
    return h


def _zeros(nlev: int = _NLEV) -> np.ndarray:
    return np.zeros(nlev + 1, dtype=np.float64)


def _make_state(
    nlev: int = _NLEV,
    *,
    sig_k: float = 1.0,
    k_ubc: int = Neumann,
    k_lbc: int = Neumann,
    ubc_type: int = logarithmic,
    lbc_type: int = logarithmic,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(
        state,
        sig_k=sig_k,
        k_ubc=k_ubc,
        k_lbc=k_lbc,
        ubc_type=ubc_type,
        lbc_type=lbc_type,
    )
    post_init_turbulence(state, nlev)
    state.cm0 = state.cm0_fix
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: TKEEquationWorkspace | None = None,
    tke: np.ndarray | None = None,
    h: np.ndarray | None = None,
    P: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Px: np.ndarray | None = None,
    PSTK: np.ndarray | None = None,
    num: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> TKEEquationWorkspace:
    assert state.tke is not None
    assert state.tkeo is not None
    assert state.P is not None
    assert state.B is not None
    assert state.Px is not None
    assert state.PSTK is not None
    assert state.num is not None
    assert state.eps is not None

    ws = (
        workspace
        if workspace is not None
        else TKEEquationWorkspace(nlev, n_cols=n_cols)
    )
    profile_h = h if h is not None else make_equidistant_h(nlev, _DEPTH)
    for col in range(n_cols):
        ws.tke[col] = tke if tke is not None else state.tke
        ws.tkeo[col] = state.tkeo
        ws.h[col] = profile_h
        ws.P[col] = P if P is not None else state.P
        ws.B[col] = B if B is not None else state.B
        ws.Px[col] = Px if Px is not None else state.Px
        ws.PSTK[col] = PSTK if PSTK is not None else state.PSTK
        ws.num[col] = num if num is not None else state.num
        ws.eps[col] = eps if eps is not None else state.eps
        ws.u_taus[col, 0] = u_taus
        ws.u_taub[col, 0] = u_taub
        ws.z0s[col, 0] = z0s
        ws.z0b[col, 0] = z0b

    ws.avh.fill(0.0)
    ws.l_sour.fill(0.0)
    ws.q_sour.fill(0.0)
    ws.au.fill(0.0)
    ws.bu.fill(0.0)
    ws.cu.fill(0.0)
    ws.du.fill(0.0)
    ws.ru.fill(0.0)
    ws.qu.fill(0.0)
    return ws


def _run_step_tkeeq(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    *,
    workspace: TKEEquationWorkspace | None = None,
    tke: np.ndarray | None = None,
    h: np.ndarray | None = None,
    P: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Px: np.ndarray | None = None,
    PSTK: np.ndarray | None = None,
    num: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> TKEEquationWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        tke=tke,
        h=h,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        eps=eps,
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
        n_cols=n_cols,
    )

    step_tkeeq(
        n_cols,
        nlev,
        dt,
        state.sig_k,
        state.k_min,
        state.k_ubc,
        state.k_lbc,
        state.ubc_type,
        state.lbc_type,
        state.cm0,
        state.cmsf,
        state.cw,
        state.gen_alpha,
        state.gen_l,
        ws.tke,
        ws.tkeo,
        ws.h,
        ws.P,
        ws.B,
        ws.Px,
        ws.PSTK,
        ws.num,
        ws.eps,
        ws.avh,
        ws.l_sour,
        ws.q_sour,
        ws.u_taus,
        ws.u_taub,
        ws.z0s,
        ws.z0b,
        ws.au,
        ws.bu,
        ws.cu,
        ws.du,
        ws.ru,
        ws.qu,
    )

    assert state.tke is not None
    assert state.tkeo is not None
    state.tke[:] = ws.tke[0]
    state.tkeo[:] = ws.tkeo[0]
    return ws


def test_import() -> None:
    from pygotm.turbulence.tkeeq import step_tkeeq as _step  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = TKEEquationWorkspace(_NLEV, n_cols=2)
    assert workspace.tke.shape == (2, _NLEV + 1)
    assert workspace.au.shape == (2, _NLEV + 1)


def test_smoke_step_tkeeq() -> None:
    state = _make_state()
    _run_step_tkeeq(state, _NLEV, _DT)


def test_tkeo_saves_old_profile() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.tke is not None
    state.tke[:] = np.linspace(1.0e-4, 3.0e-4, nlev + 1)
    tke_before = state.tke.copy()

    _run_step_tkeeq(state, nlev, _DT, num=_zeros(nlev))

    assert state.tkeo is not None
    np.testing.assert_allclose(state.tkeo, tke_before, rtol=1.0e-12)


def test_avh_equals_num_over_sig_k_on_interior_levels() -> None:
    nlev = _NLEV
    state = _make_state(nlev, sig_k=0.75)
    num = np.linspace(1.0e-5, 4.0e-4, nlev + 1)

    workspace = _run_step_tkeeq(state, nlev, _DT, num=num)
    avh = workspace.avh[0]

    np.testing.assert_allclose(avh[1:nlev], num[1:nlev] / state.sig_k, rtol=1.0e-12)
    assert avh[0] == 0.0
    assert avh[nlev] == 0.0


def test_positive_source_branch_matches_analytic_no_diffusion_update() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = np.full(nlev + 1, 2.0e-4, dtype=np.float64)
    P = np.full(nlev + 1, 3.0e-6, dtype=np.float64)
    B = np.full(nlev + 1, 1.5e-6, dtype=np.float64)
    eps = np.full(nlev + 1, 4.0e-7, dtype=np.float64)

    _run_step_tkeeq(
        state,
        nlev,
        dt=30.0,
        tke=initial_tke,
        P=P,
        B=B,
        num=_zeros(nlev),
        eps=eps,
    )

    assert state.tke is not None
    expected = (initial_tke[1:nlev] + 30.0 * (P[1:nlev] + B[1:nlev])) / (
        1.0 + 30.0 * eps[1:nlev] / initial_tke[1:nlev]
    )
    np.testing.assert_allclose(state.tke[1:nlev], expected, rtol=1.0e-12)


def test_negative_buoyancy_branch_moves_sink_into_l_sour() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = np.full(nlev + 1, 2.5e-4, dtype=np.float64)
    P = np.full(nlev + 1, 1.0e-6, dtype=np.float64)
    B = np.full(nlev + 1, -2.0e-6, dtype=np.float64)
    eps = np.full(nlev + 1, 5.0e-7, dtype=np.float64)

    _run_step_tkeeq(
        state,
        nlev,
        dt=45.0,
        tke=initial_tke,
        P=P,
        B=B,
        num=_zeros(nlev),
        eps=eps,
    )

    assert state.tke is not None
    expected = (initial_tke[1:nlev] + 45.0 * P[1:nlev]) / (
        1.0 + 45.0 * (eps[1:nlev] - B[1:nlev]) / initial_tke[1:nlev]
    )
    np.testing.assert_allclose(state.tke[1:nlev], expected, rtol=1.0e-12)


def test_boundary_fill_uses_logarithmic_dirichlet_values() -> None:
    nlev = _NLEV
    state = _make_state(nlev, k_ubc=Dirichlet, k_lbc=Dirichlet)
    u_taus = 0.012
    u_taub = 0.009

    _run_step_tkeeq(
        state,
        nlev,
        _DT,
        num=_zeros(nlev),
        u_taus=u_taus,
        u_taub=u_taub,
    )

    assert state.tke is not None
    np.testing.assert_allclose(
        state.tke[nlev],
        u_taus**2 / state.cm0**2,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.tke[0],
        u_taub**2 / state.cm0**2,
        rtol=1.0e-12,
    )


def test_clips_boundaries_and_collapsing_interior_to_k_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = np.full(nlev + 1, 1.1e-8, dtype=np.float64)
    eps = np.full(nlev + 1, 1.0e-5, dtype=np.float64)

    _run_step_tkeeq(
        state,
        nlev,
        dt=60.0,
        tke=initial_tke,
        num=_zeros(nlev),
        eps=eps,
        u_taus=0.0,
        u_taub=0.0,
    )

    assert state.tke is not None
    assert np.all(state.tke >= state.k_min)
    assert np.all(state.tke <= initial_tke)
    assert state.tke[0] == state.k_min
    assert state.tke[nlev] == state.k_min


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = np.linspace(1.0e-4, 3.5e-4, nlev + 1)
    P = np.linspace(0.0, 2.0e-6, nlev + 1)
    B = np.linspace(-1.0e-6, 1.0e-6, nlev + 1)
    Px = np.linspace(0.0, 1.0e-6, nlev + 1)
    PSTK = np.linspace(0.0, 5.0e-7, nlev + 1)
    num = np.linspace(1.0e-5, 2.5e-4, nlev + 1)
    eps = np.linspace(2.0e-7, 6.0e-7, nlev + 1)

    single = _run_step_tkeeq(
        state,
        nlev,
        _DT,
        tke=initial_tke,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        eps=eps,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=1,
    )
    multi = _run_step_tkeeq(
        state,
        nlev,
        _DT,
        tke=initial_tke,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        eps=eps,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=2,
    )

    for name in ("tke", "tkeo"):
        single_arr = getattr(single, name)[0]
        multi_0 = getattr(multi, name)[0]
        multi_1 = getattr(multi, name)[1]
        np.testing.assert_allclose(multi_0, single_arr, rtol=1.0e-12)
        np.testing.assert_allclose(multi_1, single_arr, rtol=1.0e-12)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = np.linspace(1.0e-4, 2.5e-4, nlev + 1)
    P = np.linspace(0.0, 3.0e-6, nlev + 1)
    B = np.linspace(-1.0e-6, 1.0e-6, nlev + 1)
    Px = np.linspace(0.0, 8.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 4.0e-7, nlev + 1)
    num = np.linspace(1.0e-5, 2.0e-4, nlev + 1)
    eps = np.linspace(2.0e-7, 5.0e-7, nlev + 1)

    _run_step_tkeeq(
        state,
        nlev,
        _DT,
        tke=initial_tke,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        eps=eps,
        u_taus=0.01,
        u_taub=0.008,
    )

    assert state.tke is not None
    assert state.tkeo is not None
    assert np.isfinite(state.tke).all()
    assert np.isfinite(state.tkeo).all()
