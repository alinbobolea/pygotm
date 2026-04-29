"""Tests for pygotm.turbulence.dissipationeq."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.dissipationeq import (
    DissipationEquationWorkspace,
    step_dissipationeq,
)
from pygotm.turbulence.turbulence import (
    Dirichlet,
    Neumann,
    TurbulenceState,
    epsilon_bc,
    init_turbulence,
    injection,
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
    psi_ubc: int = Neumann,
    psi_lbc: int = Neumann,
    ubc_type: int = logarithmic,
    lbc_type: int = logarithmic,
    sig_peps: bool = False,
    length_lim: bool = True,
    ce3minus: float = 0.0,
    compute_kappa: bool = True,
    compute_c3: bool = True,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(
        state,
        psi_ubc=psi_ubc,
        psi_lbc=psi_lbc,
        ubc_type=ubc_type,
        lbc_type=lbc_type,
        sig_peps=sig_peps,
        length_lim=length_lim,
        ce3minus=ce3minus,
        compute_kappa=compute_kappa,
        compute_c3=compute_c3,
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
    workspace: DissipationEquationWorkspace | None = None,
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
) -> DissipationEquationWorkspace:
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
        else DissipationEquationWorkspace(nlev, n_cols=n_cols)
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

    ws.avh.fill(0.0)
    ws.sig_eff.fill(0.0)
    ws.l_sour.fill(0.0)
    ws.q_sour.fill(0.0)
    ws.au.fill(0.0)
    ws.bu.fill(0.0)
    ws.cu.fill(0.0)
    ws.du.fill(0.0)
    ws.ru.fill(0.0)
    ws.qu.fill(0.0)
    return ws


def _run_step_dissipationeq(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    *,
    workspace: DissipationEquationWorkspace | None = None,
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
) -> DissipationEquationWorkspace:
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

    step_dissipationeq(
        n_cols,
        nlev,
        dt,
        state.ce1,
        state.ce2,
        state.ce3plus,
        state.ce3minus,
        state.cex,
        state.ce4,
        state.cm0,
        state.cde,
        state.kappa,
        state.galp,
        state.sig_k,
        state.sig_e,
        state.sig_e0,
        int(state.sig_peps),
        int(state.length_lim),
        state.eps_min,
        state.psi_ubc,
        state.psi_lbc,
        state.ubc_type,
        state.lbc_type,
        state.cmsf,
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
        ws.num,
        ws.avh,
        ws.sig_eff,
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

    assert state.eps is not None
    assert state.L is not None
    state.eps[:] = ws.eps[0]
    state.L[:] = ws.L[0]
    return ws


def test_import() -> None:
    from pygotm.turbulence.dissipationeq import (  # noqa: F401
        step_dissipationeq as _step,
    )

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = DissipationEquationWorkspace(_NLEV, n_cols=2)
    assert workspace.eps.shape == (2, _NLEV + 1)
    assert workspace.sig_eff.shape == (2, _NLEV + 1)


def test_epsilon_bc_logarithmic_matches_fortran_formula() -> None:
    state = _make_state()

    value = epsilon_bc(state, Dirichlet, logarithmic, 0.25, 2.0e-4, 0.01, 0.012)

    expected = state.cde * (2.0e-4**1.5) / (state.kappa * (0.25 + 0.01))
    assert value == expected
    neumann = epsilon_bc(state, Neumann, logarithmic, 0.25, 2.0e-4, 0.01, 0.012)
    expected_flux = state.cm0**4 * (2.0e-4**2) / (state.sig_e * (0.25 + 0.01))
    assert neumann == expected_flux


def test_epsilon_bc_injection_matches_fortran_formula() -> None:
    state = _make_state(ubc_type=injection, lbc_type=injection)

    zi = 0.05
    z0 = 1.0e-3
    u_tau = 0.012
    f_k = state.cw * u_tau**3
    capital_k = (-state.sig_k * f_k / (state.cmsf * state.gen_alpha * state.gen_l)) ** (
        2.0 / 3.0
    ) / z0**state.gen_alpha

    value = epsilon_bc(state, Dirichlet, injection, zi, 2.0e-4, z0, u_tau)
    expected = (
        state.cde
        * capital_k**1.5
        / state.gen_l
        * (zi + z0) ** (1.5 * state.gen_alpha - 1.0)
    )
    assert value == expected

    flux = epsilon_bc(state, Neumann, injection, zi, 2.0e-4, z0, u_tau)
    expected_flux = (
        -state.cmsf
        * state.cde
        / state.sig_e0
        * capital_k**2
        * (1.5 * state.gen_alpha - 1.0)
        * (zi + z0) ** (2.0 * state.gen_alpha - 1.0)
    )
    assert flux == expected_flux


def test_smoke_step_dissipationeq() -> None:
    state = _make_state()
    _run_step_dissipationeq(state, _NLEV, _DT)


def test_sig_eff_matches_wave_breaking_mixture() -> None:
    nlev = _NLEV
    state = _make_state(nlev, sig_peps=True, compute_kappa=False)
    state.sig_e0 = 0.9
    num = np.linspace(1.0e-5, 2.2e-4, nlev + 1)
    P = np.full(nlev + 1, 3.0e-7, dtype=np.float64)
    B = np.full(nlev + 1, 1.0e-7, dtype=np.float64)
    Px = np.full(nlev + 1, 2.0e-7, dtype=np.float64)
    eps = np.full(nlev + 1, 5.0e-7, dtype=np.float64)

    workspace = _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        num=num,
        P=P,
        B=B,
        Px=Px,
        eps=eps,
        tke=np.full(nlev + 1, 2.0e-4, dtype=np.float64),
        tkeo=np.full(nlev + 1, 2.0e-4, dtype=np.float64),
    )

    sig_eff = workspace.sig_eff[0]
    avh = workspace.avh[0]

    peps = (P[1:nlev] + Px[1:nlev] + B[1:nlev]) / eps[1:nlev]
    peps = np.minimum(peps, 1.0)
    expected_sig_eff = peps * state.sig_e + (1.0 - peps) * state.sig_e0
    np.testing.assert_allclose(sig_eff[1:nlev], expected_sig_eff, rtol=1.0e-12)
    np.testing.assert_allclose(avh[1:nlev], num[1:nlev] / expected_sig_eff)
    assert sig_eff[0] == state.sig_e
    assert sig_eff[nlev] == state.sig_e0


def test_positive_source_branch_matches_analytic_no_diffusion_update() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=False)
    initial_tke = np.full(nlev + 1, 3.0e-4, dtype=np.float64)
    initial_eps = np.full(nlev + 1, 2.0e-7, dtype=np.float64)
    P = np.full(nlev + 1, 2.5e-6, dtype=np.float64)
    B = np.full(nlev + 1, 5.0e-7, dtype=np.float64)
    Px = np.full(nlev + 1, 4.0e-7, dtype=np.float64)

    _run_step_dissipationeq(
        state,
        nlev,
        dt=30.0,
        tke=initial_tke,
        tkeo=initial_tke,
        eps=initial_eps,
        P=P,
        B=B,
        Px=Px,
        num=_zeros(nlev),
    )

    assert state.eps is not None
    eps_over_tke = initial_eps[1:nlev] / initial_tke[1:nlev]
    prod = eps_over_tke * (state.ce1 * P[1:nlev] + state.cex * Px[1:nlev])
    buoyan = state.ce3plus * eps_over_tke * B[1:nlev]
    diss = state.ce2 * eps_over_tke * initial_eps[1:nlev]
    expected = (initial_eps[1:nlev] + 30.0 * (prod + buoyan)) / (
        1.0 + 30.0 * diss / initial_eps[1:nlev]
    )
    np.testing.assert_allclose(state.eps[2 : nlev - 1], expected[1:-1], rtol=1.0e-12)


def test_negative_buoyancy_branch_moves_sink_into_l_sour() -> None:
    nlev = _NLEV
    state = _make_state(
        nlev,
        length_lim=False,
        ce3minus=1.25,
        compute_c3=False,
    )
    initial_tke = np.full(nlev + 1, 2.5e-4, dtype=np.float64)
    initial_eps = np.full(nlev + 1, 3.0e-7, dtype=np.float64)
    P = np.full(nlev + 1, 1.0e-6, dtype=np.float64)
    B = np.full(nlev + 1, -3.0e-6, dtype=np.float64)

    _run_step_dissipationeq(
        state,
        nlev,
        dt=45.0,
        tke=initial_tke,
        tkeo=initial_tke,
        eps=initial_eps,
        P=P,
        B=B,
        num=_zeros(nlev),
    )

    assert state.eps is not None
    eps_over_tke = initial_eps[1:nlev] / initial_tke[1:nlev]
    prod = eps_over_tke * state.ce1 * P[1:nlev]
    buoyan = state.ce3minus * eps_over_tke * B[1:nlev]
    diss = state.ce2 * eps_over_tke * initial_eps[1:nlev]
    expected = (initial_eps[1:nlev] + 45.0 * prod) / (
        1.0 + 45.0 * (diss - buoyan) / initial_eps[1:nlev]
    )
    np.testing.assert_allclose(state.eps[2 : nlev - 1], expected[1:-1], rtol=1.0e-12)


def test_boundary_fill_uses_logarithmic_dirichlet_values() -> None:
    nlev = _NLEV
    state = _make_state(nlev, psi_ubc=Dirichlet, psi_lbc=Dirichlet, length_lim=False)
    tke = np.full(nlev + 1, 2.0e-4, dtype=np.float64)
    u_taus = 0.012
    u_taub = 0.009
    z0s = 1.2e-3
    z0b = 8.0e-4

    _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=np.full(nlev + 1, 2.0e-7, dtype=np.float64),
        num=_zeros(nlev),
        u_taus=u_taus,
        u_taub=u_taub,
        z0s=z0s,
        z0b=z0b,
    )

    assert state.eps is not None
    top = epsilon_bc(state, Dirichlet, logarithmic, z0s, tke[nlev], z0s, u_taus)
    bottom = epsilon_bc(state, Dirichlet, logarithmic, z0b, tke[0], z0b, u_taub)
    np.testing.assert_allclose(state.eps[nlev], top, rtol=1.0e-12)
    np.testing.assert_allclose(state.eps[0], bottom, rtol=1.0e-12)


def test_clips_to_eps_min_and_updates_length_scale() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=False)
    tke = np.full(nlev + 1, 1.5e-4, dtype=np.float64)
    eps = np.full(nlev + 1, 1.0e-15, dtype=np.float64)

    _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
        num=_zeros(nlev),
    )

    assert state.eps is not None
    assert state.L is not None
    assert np.all(state.eps >= state.eps_min)
    expected_l = state.cde * np.sqrt(tke[1:nlev] ** 3) / state.eps[1:nlev]
    np.testing.assert_allclose(state.L[1:nlev], expected_l, rtol=1.0e-12)


def test_galperin_length_limit_applies_under_stable_stratification() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=True)
    tke = np.full(nlev + 1, 4.0e-4, dtype=np.float64)
    eps = np.full(nlev + 1, 1.0e-12, dtype=np.float64)
    NN = np.full(nlev + 1, 4.0e-4, dtype=np.float64)

    _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
        NN=NN,
        num=_zeros(nlev),
    )

    assert state.eps is not None
    assert state.L is not None
    epslim = state.cde / np.sqrt(2.0) / state.galp * tke[1:nlev] * np.sqrt(NN[1:nlev])
    np.testing.assert_allclose(state.eps[1:nlev], epslim, rtol=1.0e-12)
    expected_l = state.cde * np.sqrt(tke[1:nlev] ** 3) / epslim
    np.testing.assert_allclose(state.L[1:nlev], expected_l, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.5e-4, 3.0e-4, nlev + 1)
    eps = np.linspace(2.0e-7, 5.0e-7, nlev + 1)
    P = np.linspace(0.0, 2.5e-6, nlev + 1)
    B = np.linspace(-8.0e-7, 8.0e-7, nlev + 1)
    Px = np.linspace(0.0, 6.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 3.0e-7, nlev + 1)
    num = np.linspace(1.0e-5, 2.5e-4, nlev + 1)
    NN = np.linspace(-2.0e-4, 2.0e-4, nlev + 1)

    single = _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
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
    multi = _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
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

    for name in ("eps", "L"):
        single_arr = getattr(single, name)[0]
        multi_0 = getattr(multi, name)[0]
        multi_1 = getattr(multi, name)[1]
        np.testing.assert_allclose(multi_0, single_arr, rtol=1.0e-12)
        np.testing.assert_allclose(multi_1, single_arr, rtol=1.0e-12)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.2e-4, 2.8e-4, nlev + 1)
    eps = np.linspace(2.0e-7, 4.5e-7, nlev + 1)
    P = np.linspace(0.0, 2.0e-6, nlev + 1)
    B = np.linspace(-6.0e-7, 6.0e-7, nlev + 1)
    Px = np.linspace(0.0, 4.0e-7, nlev + 1)
    PSTK = np.linspace(0.0, 2.0e-7, nlev + 1)
    num = np.linspace(1.0e-5, 2.0e-4, nlev + 1)
    NN = np.linspace(-1.0e-4, 3.0e-4, nlev + 1)

    _run_step_dissipationeq(
        state,
        nlev,
        _DT,
        tke=tke,
        tkeo=tke,
        eps=eps,
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
    assert np.isfinite(state.eps).all()
    assert np.isfinite(state.L).all()
