"""Tests for pygotm.turbulence.q2over2eq."""

from __future__ import annotations

import numpy as np
from taichi_helpers import (
    fill_field_from_array,
    fill_field_scalar,
    make_equidistant_h,
    read_field_array,
)

from pygotm.turbulence.q2over2eq import (
    Q2Over2EquationWorkspace,
    step_q2over2eq,
)
from pygotm.turbulence.turbulence import (
    Dirichlet,
    Neumann,
    TurbulenceState,
    init_turbulence,
    logarithmic,
    post_init_turbulence,
    q2over2_bc,
)

_NLEV = 12
_DT = 60.0
_DEPTH = 24.0


def _zeros(nlev: int = _NLEV) -> np.ndarray:
    return np.zeros(nlev + 1, dtype=np.float64)


def _constant(value: float, nlev: int = _NLEV) -> np.ndarray:
    return np.full(nlev + 1, value, dtype=np.float64)


def _make_state(
    nlev: int = _NLEV,
    *,
    k_ubc: int = Neumann,
    k_lbc: int = Neumann,
    ubc_type: int = logarithmic,
    lbc_type: int = logarithmic,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(
        state,
        k_ubc=k_ubc,
        k_lbc=k_lbc,
        ubc_type=ubc_type,
        lbc_type=lbc_type,
    )
    post_init_turbulence(state, nlev)
    state.cm0 = state.cm0_fix
    state.cde = state.cm0**3
    state.b1 = 2.0**1.5 / state.cde
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: Q2Over2EquationWorkspace | None = None,
    tke: np.ndarray | None = None,
    h: np.ndarray | None = None,
    P: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Px: np.ndarray | None = None,
    PSTK: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    L: np.ndarray | None = None,
    sq_var: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> Q2Over2EquationWorkspace:
    assert state.tke is not None
    assert state.tkeo is not None
    assert state.P is not None
    assert state.B is not None
    assert state.Px is not None
    assert state.PSTK is not None
    assert state.eps is not None
    assert state.L is not None
    assert state.sq_var is not None

    ws = (
        workspace
        if workspace is not None
        else Q2Over2EquationWorkspace(nlev, n_cols=n_cols)
    )
    profile_h = h if h is not None else make_equidistant_h(nlev, _DEPTH)
    for col in range(n_cols):
        fill_field_from_array(ws.tke, tke if tke is not None else state.tke, col=col)
        fill_field_from_array(ws.tkeo, state.tkeo, col=col)
        fill_field_from_array(ws.h, profile_h, col=col)
        fill_field_from_array(ws.P, P if P is not None else state.P, col=col)
        fill_field_from_array(ws.B, B if B is not None else state.B, col=col)
        fill_field_from_array(ws.Px, Px if Px is not None else state.Px, col=col)
        fill_field_from_array(
            ws.PSTK,
            PSTK if PSTK is not None else state.PSTK,
            col=col,
        )
        fill_field_from_array(ws.eps, eps if eps is not None else state.eps, col=col)
        fill_field_from_array(ws.L, L if L is not None else state.L, col=col)
        fill_field_from_array(
            ws.sq_var,
            sq_var if sq_var is not None else state.sq_var,
            col=col,
        )
        fill_field_scalar(ws.u_taus, u_taus, col=col)
        fill_field_scalar(ws.u_taub, u_taub, col=col)
        fill_field_scalar(ws.z0s, z0s, col=col)
        fill_field_scalar(ws.z0b, z0b, col=col)

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


def _run_step_q2over2eq(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    *,
    workspace: Q2Over2EquationWorkspace | None = None,
    tke: np.ndarray | None = None,
    h: np.ndarray | None = None,
    P: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Px: np.ndarray | None = None,
    PSTK: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    L: np.ndarray | None = None,
    sq_var: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> Q2Over2EquationWorkspace:
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
        eps=eps,
        L=L,
        sq_var=sq_var,
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
        n_cols=n_cols,
    )

    step_q2over2eq(
        n_cols,
        nlev,
        dt,
        state.k_min,
        state.b1,
        state.k_ubc,
        state.k_lbc,
        state.ubc_type,
        state.lbc_type,
        state.sq,
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
        ws.eps,
        ws.L,
        ws.sq_var,
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
    state.tke[:] = read_field_array(ws.tke)
    state.tkeo[:] = read_field_array(ws.tkeo)
    return ws


def test_import() -> None:
    from pygotm.turbulence.q2over2eq import step_q2over2eq as _step  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = Q2Over2EquationWorkspace(_NLEV, n_cols=2)
    assert workspace.tke.shape == (2, _NLEV + 1)
    assert workspace.sq_var.shape == (2, _NLEV + 1)


def test_smoke_step_q2over2eq() -> None:
    state = _make_state()
    _run_step_q2over2eq(state, _NLEV, _DT)


def test_tkeo_saves_old_profile() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.tke is not None
    state.tke[:] = np.linspace(1.0e-4, 3.0e-4, nlev + 1)
    tke_before = state.tke.copy()

    _run_step_q2over2eq(
        state,
        nlev,
        _DT,
        sq_var=_zeros(nlev),
        u_taus=0.0,
        u_taub=0.0,
    )

    assert state.tkeo is not None
    np.testing.assert_allclose(state.tkeo, tke_before, rtol=1.0e-12)


def test_avh_matches_fortran_formula() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.2e-4, 2.8e-4, nlev + 1)
    L = np.linspace(0.01, 0.08, nlev + 1)
    sq_var = np.linspace(0.05, 0.3, nlev + 1)

    workspace = _run_step_q2over2eq(
        state,
        nlev,
        _DT,
        tke=tke,
        eps=_constant(2.0e-7, nlev),
        L=L,
        sq_var=sq_var,
        u_taus=0.0,
        u_taub=0.0,
    )
    avh = read_field_array(workspace.avh)

    expected = sq_var[1:nlev] * np.sqrt(2.0 * tke[1:nlev]) * L[1:nlev]
    np.testing.assert_allclose(avh[1:nlev], expected, rtol=1.0e-12)
    assert avh[0] == 0.0
    assert avh[nlev] == 0.0


def test_positive_source_branch_matches_analytic_no_diffusion_update() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = _constant(2.0e-4, nlev)
    P = _constant(3.0e-6, nlev)
    B = _constant(1.5e-6, nlev)
    eps = _constant(4.0e-7, nlev)

    _run_step_q2over2eq(
        state,
        nlev,
        dt=30.0,
        tke=initial_tke,
        P=P,
        B=B,
        eps=eps,
        L=_constant(0.05, nlev),
        sq_var=_zeros(nlev),
        u_taus=0.0,
        u_taub=0.0,
    )

    assert state.tke is not None
    expected = (initial_tke[1:nlev] + 30.0 * (P[1:nlev] + B[1:nlev])) / (
        1.0 + 30.0 * eps[1:nlev] / initial_tke[1:nlev]
    )
    np.testing.assert_allclose(state.tke[1:nlev], expected, rtol=1.0e-12)


def test_negative_buoyancy_branch_moves_sink_into_l_sour() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = _constant(2.5e-4, nlev)
    P = _constant(1.0e-6, nlev)
    B = _constant(-2.0e-6, nlev)
    eps = _constant(5.0e-7, nlev)

    _run_step_q2over2eq(
        state,
        nlev,
        dt=45.0,
        tke=initial_tke,
        P=P,
        B=B,
        eps=eps,
        L=_constant(0.05, nlev),
        sq_var=_zeros(nlev),
        u_taus=0.0,
        u_taub=0.0,
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
    z0s = 1.2e-3
    z0b = 8.0e-4

    _run_step_q2over2eq(
        state,
        nlev,
        _DT,
        sq_var=_zeros(nlev),
        L=_constant(0.05, nlev),
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
    )

    assert state.tke is not None
    np.testing.assert_allclose(
        state.tke[nlev],
        q2over2_bc(state, Dirichlet, logarithmic, z0s, z0s, u_taus),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.tke[0],
        q2over2_bc(state, Dirichlet, logarithmic, z0b, z0b, u_taub),
        rtol=1.0e-12,
    )


def test_clips_boundaries_and_collapsing_interior_to_k_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_tke = np.full(nlev + 1, 1.1e-8, dtype=np.float64)
    eps = np.full(nlev + 1, 1.0e-5, dtype=np.float64)

    _run_step_q2over2eq(
        state,
        nlev,
        dt=60.0,
        tke=initial_tke,
        eps=eps,
        L=_constant(0.05, nlev),
        sq_var=_zeros(nlev),
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
    eps = np.linspace(2.0e-7, 6.0e-7, nlev + 1)
    L = np.linspace(0.02, 0.06, nlev + 1)
    sq_var = np.linspace(0.1, 0.3, nlev + 1)

    single = _run_step_q2over2eq(
        state,
        nlev,
        _DT,
        tke=initial_tke,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        eps=eps,
        L=L,
        sq_var=sq_var,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=1,
    )
    multi = _run_step_q2over2eq(
        state,
        nlev,
        _DT,
        tke=initial_tke,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        eps=eps,
        L=L,
        sq_var=sq_var,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=2,
    )

    for name in ("tke", "tkeo"):
        single_arr = read_field_array(getattr(single, name), col=0)
        multi_0 = read_field_array(getattr(multi, name), col=0)
        multi_1 = read_field_array(getattr(multi, name), col=1)
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
    eps = np.linspace(2.0e-7, 5.0e-7, nlev + 1)
    L = np.linspace(0.02, 0.06, nlev + 1)
    sq_var = np.linspace(0.1, 0.25, nlev + 1)

    _run_step_q2over2eq(
        state,
        nlev,
        _DT,
        tke=initial_tke,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        eps=eps,
        L=L,
        sq_var=sq_var,
        u_taus=0.01,
        u_taub=0.008,
    )

    assert state.tke is not None
    assert state.tkeo is not None
    assert np.isfinite(state.tke).all()
    assert np.isfinite(state.tkeo).all()
