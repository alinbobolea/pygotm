"""Tests for pygotm.turbulence.lengthscaleeq."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.lengthscaleeq import (
    LengthScaleEquationWorkspace,
    step_lengthscaleeq,
)
from pygotm.turbulence.turbulence import (
    Dirichlet,
    Neumann,
    TurbulenceState,
    init_turbulence,
    logarithmic,
    post_init_turbulence,
    q2l_bc,
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


def _constant(value: float, nlev: int = _NLEV) -> np.ndarray:
    return np.full(nlev + 1, value, dtype=np.float64)


def _make_state(
    nlev: int = _NLEV,
    *,
    psi_ubc: int = Neumann,
    psi_lbc: int = Neumann,
    ubc_type: int = logarithmic,
    lbc_type: int = logarithmic,
    my_length: int = 1,
    length_lim: bool = True,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(
        state,
        psi_ubc=psi_ubc,
        psi_lbc=psi_lbc,
        ubc_type=ubc_type,
        lbc_type=lbc_type,
        my_length=my_length,
        length_lim=length_lim,
    )
    post_init_turbulence(state, nlev)
    state.cm0 = state.cm0_fix
    state.cmsf = 0.55
    state.cde = state.cm0**3
    state.b1 = 2.0**1.5 / state.cde
    state.sig_e0 = state.sig_e
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: LengthScaleEquationWorkspace | None = None,
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
    sl_var: np.ndarray | None = None,
    depth: float = _DEPTH,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> LengthScaleEquationWorkspace:
    assert state.tke is not None
    assert state.tkeo is not None
    assert state.eps is not None
    assert state.L is not None
    assert state.P is not None
    assert state.B is not None
    assert state.Px is not None
    assert state.PSTK is not None
    assert state.sl_var is not None

    ws = (
        workspace
        if workspace is not None
        else LengthScaleEquationWorkspace(nlev, n_cols=n_cols)
    )
    profile_h = h if h is not None else make_equidistant_h(nlev, depth)
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
        ws.sl_var[col] = sl_var if sl_var is not None else state.sl_var
        ws.depth[col, 0] = depth
        ws.u_taus[col, 0] = u_taus
        ws.u_taub[col, 0] = u_taub
        ws.z0s[col, 0] = z0s
        ws.z0b[col, 0] = z0b

    ws.q2l.fill(0.0)
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


def _run_step_lengthscaleeq(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    *,
    workspace: LengthScaleEquationWorkspace | None = None,
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
    sl_var: np.ndarray | None = None,
    depth: float = _DEPTH,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    z0s: float = 1.0e-3,
    z0b: float = 1.0e-3,
    n_cols: int = 1,
) -> LengthScaleEquationWorkspace:
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
        sl_var=sl_var,
        depth=depth,
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
        n_cols=n_cols,
    )

    step_lengthscaleeq(
        n_cols,
        nlev,
        dt,
        state.k_min,
        state.eps_min,
        state.kappa,
        state.e1,
        state.e2,
        state.e3,
        state.ex,
        state.e6,
        state.b1,
        state.cde,
        state.my_length,
        state.galp,
        int(state.length_lim),
        state.psi_ubc,
        state.psi_lbc,
        state.ubc_type,
        state.lbc_type,
        state.sl,
        state.sq,
        state.cw,
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
        ws.sl_var,
        ws.depth,
        ws.u_taus,
        ws.u_taub,
        ws.z0s,
        ws.z0b,
        ws.q2l,
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


def _compute_lz(
    my_length: int,
    h: np.ndarray,
    depth: float,
    z0s: float,
    z0b: float,
    kappa: float,
) -> np.ndarray:
    lz = np.zeros_like(h)
    db = 0.0
    for i in range(1, len(h) - 1):
        db += h[i]
        ds = depth - db
        if my_length == 1:
            lz[i] = kappa * (ds + z0s) * (db + z0b) / (ds + z0s + db + z0b)
        elif my_length == 2:
            lz[i] = kappa * min(ds + z0s, db + z0b)
        elif my_length == 3:
            lz[i] = kappa * (ds + z0s)
        else:
            raise AssertionError(f"unexpected my_length={my_length}")
    return lz


def test_import() -> None:
    assert callable(step_lengthscaleeq)


def test_workspace_instantiates() -> None:
    workspace = LengthScaleEquationWorkspace(_NLEV, n_cols=2)
    assert workspace.q2l.shape == (2, _NLEV + 1)
    assert workspace.depth.shape == (2, _NLEV + 1)


def test_smoke_step_lengthscaleeq() -> None:
    state = _make_state()
    tke = _constant(2.0e-4)
    _run_step_lengthscaleeq(
        state,
        _NLEV,
        _DT,
        tke=tke,
        tkeo=tke,
        L=_constant(0.05),
    )


def test_avh_matches_fortran_formula() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.5e-4, 3.0e-4, nlev + 1)
    L = np.linspace(0.02, 0.08, nlev + 1)
    sl_var = np.linspace(0.05, 0.25, nlev + 1)

    workspace = _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=L,
        sl_var=sl_var,
    )

    avh = workspace.avh[0]
    expected = sl_var[1:nlev] * np.sqrt(2.0 * tke[1:nlev]) * L[1:nlev]
    np.testing.assert_allclose(avh[1:nlev], expected, rtol=1.0e-12)


def test_my_length_variants_match_local_sink_update() -> None:
    nlev = _NLEV
    h = make_equidistant_h(nlev, _DEPTH)
    tke = _constant(2.0e-4, nlev)
    initial_l = _constant(0.05, nlev)
    q2l_old = 2.0 * tke * initial_l

    for my_length in (1, 2, 3):
        state = _make_state(
            nlev,
            psi_ubc=Dirichlet,
            psi_lbc=Dirichlet,
            my_length=my_length,
            length_lim=False,
        )

        workspace = _run_step_lengthscaleeq(
            state,
            nlev,
            dt=30.0,
            tke=tke,
            tkeo=tke,
            L=initial_l,
            h=h,
            sl_var=_zeros(nlev),
            z0s=1.2e-3,
            z0b=8.0e-4,
        )

        q2l = workspace.q2l[0]
        lz = _compute_lz(my_length, h, _DEPTH, 1.2e-3, 8.0e-4, state.kappa)
        q3 = np.sqrt(8.0 * tke[1:nlev] ** 3)
        diss = q3 / state.b1 * (1.0 + state.e2 * (initial_l[1:nlev] / lz[1:nlev]) ** 2)
        expected = q2l_old[1:nlev] / (1.0 + 30.0 * diss / q2l_old[1:nlev])
        np.testing.assert_allclose(q2l[2 : nlev - 1], expected[1:-1], rtol=1.0e-12)


def test_positive_source_branch_matches_analytic_no_diffusion_update() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    h = make_equidistant_h(nlev, _DEPTH)
    tke = _constant(2.5e-4, nlev)
    initial_l = _constant(0.04, nlev)
    q2l_old = 2.0 * tke * initial_l
    P = _constant(2.5e-6, nlev)
    B = _constant(5.0e-7, nlev)
    Px = _constant(4.0e-7, nlev)
    PSTK = _constant(2.0e-7, nlev)

    workspace = _run_step_lengthscaleeq(
        state,
        nlev,
        dt=45.0,
        tke=tke,
        tkeo=tke,
        L=initial_l,
        h=h,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        sl_var=_zeros(nlev),
    )

    q2l = workspace.q2l[0]
    lz = _compute_lz(state.my_length, h, _DEPTH, 1.0e-3, 1.0e-3, state.kappa)
    prod = initial_l[1:nlev] * (
        state.e1 * P[1:nlev] + state.ex * Px[1:nlev] + state.e6 * PSTK[1:nlev]
    )
    buoyan = state.e3 * initial_l[1:nlev] * B[1:nlev]
    q3 = np.sqrt(8.0 * tke[1:nlev] ** 3)
    diss = q3 / state.b1 * (1.0 + state.e2 * (initial_l[1:nlev] / lz[1:nlev]) ** 2)
    expected = (q2l_old[1:nlev] + 45.0 * (prod + buoyan)) / (
        1.0 + 45.0 * diss / q2l_old[1:nlev]
    )
    np.testing.assert_allclose(q2l[2 : nlev - 1], expected[1:-1], rtol=1.0e-12)


def test_negative_buoyancy_branch_moves_sink_into_l_sour() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    h = make_equidistant_h(nlev, _DEPTH)
    tke = _constant(2.5e-4, nlev)
    initial_l = _constant(0.04, nlev)
    q2l_old = 2.0 * tke * initial_l
    P = _constant(1.0e-6, nlev)
    B = _constant(-3.0e-6, nlev)

    workspace = _run_step_lengthscaleeq(
        state,
        nlev,
        dt=45.0,
        tke=tke,
        tkeo=tke,
        L=initial_l,
        h=h,
        P=P,
        B=B,
        sl_var=_zeros(nlev),
    )

    q2l = workspace.q2l[0]
    lz = _compute_lz(state.my_length, h, _DEPTH, 1.0e-3, 1.0e-3, state.kappa)
    prod = initial_l[1:nlev] * state.e1 * P[1:nlev]
    buoyan = state.e3 * initial_l[1:nlev] * B[1:nlev]
    q3 = np.sqrt(8.0 * tke[1:nlev] ** 3)
    diss = q3 / state.b1 * (1.0 + state.e2 * (initial_l[1:nlev] / lz[1:nlev]) ** 2)
    expected = (q2l_old[1:nlev] + 45.0 * prod) / (
        1.0 + 45.0 * (diss - buoyan) / q2l_old[1:nlev]
    )
    np.testing.assert_allclose(q2l[2 : nlev - 1], expected[1:-1], rtol=1.0e-12)


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

    workspace = _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=_constant(0.05, nlev),
        sl_var=_zeros(nlev),
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
    )

    q2l_profile = workspace.q2l[0]
    top = q2l_bc(state, Dirichlet, logarithmic, z0s, tke[nlev], z0s, u_taus)
    bottom = q2l_bc(state, Dirichlet, logarithmic, z0b, tke[0], z0b, u_taub)
    np.testing.assert_allclose(q2l_profile[nlev], top, rtol=1.0e-12)
    np.testing.assert_allclose(q2l_profile[0], bottom, rtol=1.0e-12)


def test_applies_galperin_length_limit_and_updates_dissipation() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=True,
    )
    tke = _constant(4.0e-4, nlev)
    NN = _constant(4.0e-4, nlev)

    _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=_constant(5.0, nlev),
        NN=NN,
        sl_var=_zeros(nlev),
    )

    assert state.L is not None
    assert state.eps is not None
    l_crit = np.sqrt(2.0 * state.galp**2 * tke[1:nlev] / NN[1:nlev])
    expected_eps = state.cde * np.sqrt(tke[1:nlev] ** 3) / l_crit
    np.testing.assert_allclose(state.L[2 : nlev - 1], l_crit[1:-1], rtol=1.0e-12)
    np.testing.assert_allclose(
        state.eps[2 : nlev - 1],
        expected_eps[1:-1],
        rtol=1.0e-12,
    )


def test_applies_l_min_floor() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    tke = _constant(2.0e-4, nlev)

    _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=_constant(1.0e-12, nlev),
        sl_var=_zeros(nlev),
    )

    assert state.L is not None
    assert state.eps is not None
    l_min = state.cde * state.k_min**1.5 / state.eps_min
    expected_eps = state.cde * np.sqrt(tke[1:nlev] ** 3) / l_min
    np.testing.assert_allclose(state.L[2 : nlev - 1], l_min, rtol=1.0e-12)
    np.testing.assert_allclose(
        state.eps[2 : nlev - 1],
        expected_eps[1:-1],
        rtol=1.0e-12,
    )


def test_applies_eps_min_floor() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        psi_ubc=Dirichlet,
        psi_lbc=Dirichlet,
        length_lim=False,
    )
    state.e2 = 0.0
    state.b1 = 1.0e30
    tke = _constant(2.0e-4, nlev)

    _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=_constant(1.0e8, nlev),
        sl_var=_zeros(nlev),
    )

    assert state.L is not None
    assert state.eps is not None
    expected_l = state.cde * np.sqrt(tke[1:nlev] ** 3) / state.eps_min
    np.testing.assert_allclose(state.eps[2 : nlev - 1], state.eps_min, rtol=1.0e-12)
    np.testing.assert_allclose(
        state.L[2 : nlev - 1],
        expected_l[1:-1],
        rtol=1.0e-12,
    )


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.5e-4, 3.0e-4, nlev + 1)
    L = np.linspace(0.02, 0.08, nlev + 1)
    P = np.linspace(0.0, 3.0e-6, nlev + 1)
    B = np.linspace(-8.0e-7, 8.0e-7, nlev + 1)
    Px = np.linspace(0.0, 6.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 3.0e-7, nlev + 1)
    sl_var = np.linspace(0.1, 0.25, nlev + 1)
    NN = np.linspace(-2.0e-4, 2.0e-4, nlev + 1)

    single = _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=L,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        sl_var=sl_var,
        NN=NN,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=1,
    )
    multi = _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=L,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        sl_var=sl_var,
        NN=NN,
        u_taus=0.012,
        u_taub=0.009,
        n_cols=2,
    )

    for name in ("q2l", "eps", "L"):
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
    P = np.linspace(0.0, 2.5e-6, nlev + 1)
    B = np.linspace(-5.0e-7, 8.0e-7, nlev + 1)
    Px = np.linspace(0.0, 4.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 2.0e-7, nlev + 1)
    sl_var = np.linspace(0.08, 0.22, nlev + 1)
    NN = np.linspace(-1.0e-4, 3.0e-4, nlev + 1)

    workspace = _run_step_lengthscaleeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        L=L,
        P=P,
        B=B,
        Px=Px,
        PSTK=PSTK,
        sl_var=sl_var,
        NN=NN,
        u_taus=0.01,
        u_taub=0.008,
    )

    assert state.eps is not None
    assert state.L is not None
    q2l = workspace.q2l[0]
    assert np.isfinite(q2l).all()
    assert np.isfinite(state.eps).all()
    assert np.isfinite(state.L).all()
