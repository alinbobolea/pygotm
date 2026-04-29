"""Tests for pygotm.turbulence.genericeq."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.genericeq import (
    GenericEquationWorkspace,
    step_genericeq,
)
from pygotm.turbulence.turbulence import (
    Dirichlet,
    Neumann,
    TurbulenceState,
    init_turbulence,
    logarithmic,
    post_init_turbulence,
    psi_bc,
)

_NLEV = 12
_DT = 60.0
_DEPTH = 24.0


def _zeros(nlev: int = _NLEV) -> np.ndarray:
    return np.zeros(nlev + 1, dtype=np.float64)


def _constant(value: float, nlev: int = _NLEV) -> np.ndarray:
    return np.full(nlev + 1, value, dtype=np.float64)


def make_equidistant_h(nlev: int, depth: float) -> np.ndarray:
    h = np.full(nlev + 1, depth / nlev, dtype=np.float64)
    h[0] = 0.0
    return h


def _make_state(
    nlev: int = _NLEV,
    *,
    psi_ubc: int = Neumann,
    psi_lbc: int = Neumann,
    ubc_type: int = logarithmic,
    lbc_type: int = logarithmic,
    length_lim: bool = True,
    **overrides: float | int | bool,
) -> TurbulenceState:
    state = TurbulenceState()
    override_map: dict[str, bool | int | float] = dict(overrides)
    init_turbulence(
        state,
        psi_ubc=psi_ubc,
        psi_lbc=psi_lbc,
        ubc_type=ubc_type,
        lbc_type=lbc_type,
        length_lim=length_lim,
        overrides=override_map,
    )
    post_init_turbulence(state, nlev)
    state.cm0 = state.cm0_fix
    state.cmsf = 0.55
    state.cde = state.cm0**3
    state.sig_e0 = state.sig_e
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: GenericEquationWorkspace | None = None,
    tke: np.ndarray | None = None,
    tkeo: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    L: np.ndarray | None = None,
    h: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    SS: np.ndarray | None = None,
    P: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Px: np.ndarray | None = None,
    PSTK: np.ndarray | None = None,
    num: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> GenericEquationWorkspace:
    assert state.tke is not None
    assert state.tkeo is not None
    assert state.eps is not None
    assert state.L is not None
    assert state.P is not None
    assert state.B is not None
    assert state.Px is not None
    assert state.PSTK is not None
    assert state.num is not None

    ws = (
        workspace
        if workspace is not None
        else GenericEquationWorkspace(nlev, n_cols=n_cols)
    )
    profile_h = h if h is not None else make_equidistant_h(nlev, _DEPTH)
    for col in range(n_cols):
        ws.tke[col] = tke if tke is not None else state.tke
        ws.tkeo[col] = tkeo if tkeo is not None else state.tkeo
        ws.eps[col] = eps if eps is not None else state.eps
        ws.L[col] = L if L is not None else state.L
        ws.h[col] = profile_h
        ws.NN[col] = NN if NN is not None else _zeros(nlev)
        ws.SS[col] = SS if SS is not None else _zeros(nlev)
        ws.P[col] = P if P is not None else state.P
        ws.B[col] = B if B is not None else state.B
        ws.Px[col] = Px if Px is not None else state.Px
        ws.PSTK[col] = PSTK if PSTK is not None else state.PSTK
        ws.num[col] = num if num is not None else state.num
        ws.u_taus[col, 0] = u_taus
        ws.u_taub[col, 0] = u_taub
        ws.z0s[col, 0] = z0s
        ws.z0b[col, 0] = z0b

    ws.psi.fill(0.0)
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


def _run_step_genericeq(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    *,
    workspace: GenericEquationWorkspace | None = None,
    tke: np.ndarray | None = None,
    tkeo: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    L: np.ndarray | None = None,
    h: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    SS: np.ndarray | None = None,
    P: np.ndarray | None = None,
    B: np.ndarray | None = None,
    Px: np.ndarray | None = None,
    PSTK: np.ndarray | None = None,
    num: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> GenericEquationWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        tke=tke,
        tkeo=tkeo,
        eps=eps,
        L=L,
        h=h,
        NN=NN,
        SS=SS,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
        n_cols=n_cols,
    )

    step_genericeq(
        n_cols,
        nlev,
        dt,
        state.cpsi1,
        state.cpsi2,
        state.cpsi3plus,
        state.cpsi3minus,
        state.cpsix,
        state.cpsi4,
        state.sig_psi,
        state.cm0,
        state.kappa,
        state.cde,
        state.galp,
        int(state.length_lim),
        state.eps_min,
        state.psi_ubc,
        state.psi_lbc,
        state.ubc_type,
        state.lbc_type,
        state.sig_k,
        state.cmsf,
        state.cw,
        state.gen_m,
        state.gen_n,
        state.gen_p,
        state.gen_alpha,
        state.gen_l,
        ws.tke,
        ws.tkeo,
        ws.eps,
        ws.L,
        ws.h,
        ws.NN,
        ws.SS,
        ws.P,
        ws.B,
        ws.Px,
        ws.PSTK,
        ws.num,
        ws.u_taus,
        ws.u_taub,
        ws.z0s,
        ws.z0b,
        ws.psi,
        ws.avh,
        ws.l_sour,
        ws.q_sour,
        ws.au,
        ws.bu,
        ws.cu,
        ws.du,
        ws.ru,
        ws.qu,
    )

    assert state.eps is not None
    assert state.L is not None
    state.eps[:] = ws.eps[0]
    state.L[:] = ws.L[0]
    return ws


def test_import() -> None:
    assert callable(step_genericeq)


def test_workspace_instantiates() -> None:
    workspace = GenericEquationWorkspace(_NLEV, n_cols=2)
    assert workspace.psi.shape == (2, _NLEV + 1)
    assert workspace.avh.shape == (2, _NLEV + 1)


def test_smoke_step_genericeq() -> None:
    state = _make_state()
    tke = _constant(2.0e-4)
    _run_step_genericeq(
        state,
        _NLEV,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=_constant(2.0e-7),
        L=_constant(0.05),
    )


def test_avh_matches_fortran_formula() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    num = np.linspace(1.0e-6, 2.5e-4, nlev + 1)

    workspace = _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=_constant(2.0e-4, nlev),
        tkeo=_constant(2.0e-4, nlev),
        eps=_constant(2.0e-7, nlev),
        L=_constant(0.05, nlev),
        num=num,
    )

    avh = workspace.avh[0]
    np.testing.assert_allclose(avh[1:nlev], num[1:nlev] / state.sig_psi, rtol=1.0e-12)


def test_positive_source_branch_matches_analytic_no_diffusion_update() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    tke = _constant(2.5e-4, nlev)
    initial_eps = _constant(2.0e-7, nlev)
    initial_l = _constant(0.04, nlev)
    psi_old = state.cm0**state.gen_p * tke**state.gen_m * initial_l**state.gen_n
    P = _constant(2.5e-6, nlev)
    B = _constant(5.0e-7, nlev)
    Px = _constant(4.0e-7, nlev)
    PSTK = _constant(2.0e-7, nlev)

    workspace = _run_step_genericeq(
        state,
        nlev,
        dt=45.0,
        tke=tke,
        tkeo=tke,
        eps=initial_eps,
        L=initial_l,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=_zeros(nlev),
    )

    psi = workspace.psi[0]
    psi_over_tke = psi_old[1:nlev] / tke[1:nlev]
    prod = psi_over_tke * (
        state.cpsi1 * P[1:nlev] + state.cpsix * Px[1:nlev] + state.cpsi4 * PSTK[1:nlev]
    )
    buoyan = state.cpsi3plus * psi_over_tke * B[1:nlev]
    diss = state.cpsi2 * psi_over_tke * initial_eps[1:nlev]
    expected_psi = (psi_old[1:nlev] + 45.0 * (prod + buoyan)) / (
        1.0 + 45.0 * diss / psi_old[1:nlev]
    )
    np.testing.assert_allclose(psi[2 : nlev - 1], expected_psi[1:-1], rtol=1.0e-12)


def test_negative_buoyancy_branch_moves_sink_into_l_sour() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
        cpsi3minus=1.25,
    )
    tke = _constant(2.5e-4, nlev)
    initial_eps = _constant(3.0e-7, nlev)
    initial_l = _constant(0.04, nlev)
    psi_old = state.cm0**state.gen_p * tke**state.gen_m * initial_l**state.gen_n
    P = _constant(1.0e-6, nlev)
    B = _constant(-3.0e-6, nlev)

    workspace = _run_step_genericeq(
        state,
        nlev,
        dt=45.0,
        tke=tke,
        tkeo=tke,
        eps=initial_eps,
        L=initial_l,
        P=P,
        B=B,
        num=_zeros(nlev),
    )

    psi = workspace.psi[0]
    psi_over_tke = psi_old[1:nlev] / tke[1:nlev]
    prod = psi_over_tke * state.cpsi1 * P[1:nlev]
    buoyan = state.cpsi3minus * psi_over_tke * B[1:nlev]
    diss = state.cpsi2 * psi_over_tke * initial_eps[1:nlev]
    expected_psi = (psi_old[1:nlev] + 45.0 * prod) / (
        1.0 + 45.0 * (diss - buoyan) / psi_old[1:nlev]
    )
    np.testing.assert_allclose(psi[2 : nlev - 1], expected_psi[1:-1], rtol=1.0e-12)


def test_boundary_fill_uses_logarithmic_dirichlet_values() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    tke = np.linspace(1.5e-4, 3.0e-4, nlev + 1)
    u_taus = 0.012
    u_taub = 0.009
    z0s = 1.2e-3
    z0b = 8.0e-4

    workspace = _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=_constant(2.0e-7, nlev),
        L=_constant(0.05, nlev),
        num=_zeros(nlev),
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
    )

    psi = workspace.psi[0]
    top = psi_bc(state, Dirichlet, logarithmic, z0s, tke[nlev], z0s, u_taus)
    bottom = psi_bc(state, Dirichlet, logarithmic, z0b, tke[0], z0b, u_taub)
    np.testing.assert_allclose(psi[nlev], top, rtol=1.0e-12)
    np.testing.assert_allclose(psi[0], bottom, rtol=1.0e-12)


def test_clips_to_eps_min_and_updates_length_scale() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    tke = _constant(1.5e-4, nlev)

    _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=_constant(2.0e-7, nlev),
        L=_constant(1.0e8, nlev),
        num=_zeros(nlev),
    )

    assert state.eps is not None
    assert state.L is not None
    expected_l = state.cde * np.sqrt(tke[1:nlev] ** 3) / state.eps_min
    np.testing.assert_allclose(state.eps[2 : nlev - 1], state.eps_min, rtol=1.0e-12)
    np.testing.assert_allclose(state.L[2 : nlev - 1], expected_l[1:-1], rtol=1.0e-12)


def test_galperin_length_limit_applies_under_stable_stratification() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=True,
    )
    tke = _constant(4.0e-4, nlev)
    NN = _constant(4.0e-4, nlev)

    _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=_constant(1.0e-12, nlev),
        L=_constant(5.0, nlev),
        NN=NN,
        num=_zeros(nlev),
    )

    assert state.eps is not None
    assert state.L is not None
    epslim = state.cde / np.sqrt(2.0) / state.galp * tke[1:nlev] * np.sqrt(NN[1:nlev])
    expected_l = state.cde * np.sqrt(tke[1:nlev] ** 3) / epslim
    np.testing.assert_allclose(state.eps[2 : nlev - 1], epslim[1:-1], rtol=1.0e-12)
    np.testing.assert_allclose(state.L[2 : nlev - 1], expected_l[1:-1], rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.5e-4, 3.0e-4, nlev + 1)
    L = np.linspace(0.02, 0.08, nlev + 1)
    eps = np.linspace(2.0e-7, 5.0e-7, nlev + 1)
    P = np.linspace(0.0, 2.5e-6, nlev + 1)
    B = np.linspace(-8.0e-7, 8.0e-7, nlev + 1)
    Px = np.linspace(0.0, 6.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 3.0e-7, nlev + 1)
    num = np.linspace(1.0e-5, 2.5e-4, nlev + 1)
    NN = np.linspace(-2.0e-4, 2.0e-4, nlev + 1)

    single = _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
        L=L,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        NN=NN,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=1,
    )
    multi = _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
        L=L,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        NN=NN,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=2,
    )

    for name in ("psi", "eps", "L"):
        single_arr = getattr(single, name)[0]
        multi_0 = getattr(multi, name)[0]
        multi_1 = getattr(multi, name)[1]
        np.testing.assert_allclose(multi_0, single_arr, rtol=1.0e-12)
        np.testing.assert_allclose(multi_1, single_arr, rtol=1.0e-12)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.2e-4, 2.8e-4, nlev + 1)
    L = np.linspace(0.02, 0.07, nlev + 1)
    eps = np.linspace(2.0e-7, 5.0e-7, nlev + 1)
    P = np.linspace(0.0, 2.5e-6, nlev + 1)
    B = np.linspace(-5.0e-7, 8.0e-7, nlev + 1)
    Px = np.linspace(0.0, 4.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 2.0e-7, nlev + 1)
    num = np.linspace(1.0e-5, 2.0e-4, nlev + 1)
    NN = np.linspace(-1.0e-4, 3.0e-4, nlev + 1)

    workspace = _run_step_genericeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
        L=L,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        num=num,
        NN=NN,
        u_taus=0.01,
        u_taub=0.008,
    )

    assert state.eps is not None
    assert state.L is not None
    psi = workspace.psi[0]
    assert np.isfinite(psi).all()
    assert np.isfinite(state.eps).all()
    assert np.isfinite(state.L).all()
