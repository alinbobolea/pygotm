"""Tests for pygotm.turbulence.kbeq."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.kbeq import KBEquationWorkspace, step_kbeq
from pygotm.turbulence.turbulence import (
    Dirichlet,
    Neumann,
    TurbulenceState,
    init_turbulence,
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
    k_ubc: int = Neumann,
    k_lbc: int = Neumann,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, k_ubc=k_ubc, k_lbc=k_lbc)
    post_init_turbulence(state, nlev)
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: KBEquationWorkspace | None = None,
    kb: np.ndarray | None = None,
    h: np.ndarray | None = None,
    Pb: np.ndarray | None = None,
    epsb: np.ndarray | None = None,
    nuh: np.ndarray | None = None,
    n_cols: int = 1,
) -> KBEquationWorkspace:
    assert state.kb is not None
    assert state.Pb is not None
    assert state.epsb is not None
    assert state.nuh is not None

    ws = (
        workspace if workspace is not None else KBEquationWorkspace(nlev, n_cols=n_cols)
    )
    profile_h = h if h is not None else make_equidistant_h(nlev, _DEPTH)
    for col in range(n_cols):
        ws.kb[col] = kb if kb is not None else state.kb
        ws.h[col] = profile_h
        ws.Pb[col] = Pb if Pb is not None else state.Pb
        ws.epsb[col] = epsb if epsb is not None else state.epsb
        ws.nuh[col] = nuh if nuh is not None else state.nuh

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


def _run_step_kbeq(
    state: TurbulenceState,
    nlev: int,
    dt: float,
    *,
    workspace: KBEquationWorkspace | None = None,
    kb: np.ndarray | None = None,
    h: np.ndarray | None = None,
    Pb: np.ndarray | None = None,
    epsb: np.ndarray | None = None,
    nuh: np.ndarray | None = None,
    n_cols: int = 1,
) -> KBEquationWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        kb=kb,
        h=h,
        Pb=Pb,
        epsb=epsb,
        nuh=nuh,
        n_cols=n_cols,
    )

    step_kbeq(
        n_cols,
        nlev,
        dt,
        state.kb_min,
        state.k_ubc,
        state.k_lbc,
        ws.kb,
        ws.h,
        ws.Pb,
        ws.epsb,
        ws.nuh,
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

    assert state.kb is not None
    state.kb[:] = ws.kb[0]
    return ws


def test_import() -> None:
    from pygotm.turbulence.kbeq import step_kbeq as _step  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = KBEquationWorkspace(_NLEV, n_cols=2)
    assert workspace.kb.shape == (2, _NLEV + 1)
    assert workspace.au.shape == (2, _NLEV + 1)


def test_smoke_step_kbeq() -> None:
    state = _make_state()
    _run_step_kbeq(state, _NLEV, _DT)


def test_avh_matches_nuh_everywhere() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    nuh = np.linspace(1.0e-6, 3.0e-4, nlev + 1)

    workspace = _run_step_kbeq(state, nlev, _DT, nuh=nuh)
    avh = workspace.avh[0]

    np.testing.assert_allclose(avh, nuh, rtol=1.0e-12)


def test_positive_source_branch_matches_analytic_no_diffusion_update() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_kb = np.full(nlev + 1, 2.0e-4, dtype=np.float64)
    Pb = np.full(nlev + 1, 3.0e-6, dtype=np.float64)
    epsb = np.full(nlev + 1, 4.0e-7, dtype=np.float64)

    _run_step_kbeq(
        state,
        nlev,
        dt=30.0,
        kb=initial_kb,
        Pb=Pb,
        epsb=epsb,
        nuh=_zeros(nlev),
    )

    assert state.kb is not None
    expected = (initial_kb[1:nlev] + 30.0 * Pb[1:nlev]) / (
        1.0 + 30.0 * epsb[1:nlev] / initial_kb[1:nlev]
    )
    np.testing.assert_allclose(state.kb[1:nlev], expected, rtol=1.0e-12)


def test_negative_source_branch_moves_sink_into_l_sour() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_kb = np.full(nlev + 1, 2.5e-4, dtype=np.float64)
    Pb = np.full(nlev + 1, -2.0e-6, dtype=np.float64)
    epsb = np.full(nlev + 1, 5.0e-7, dtype=np.float64)

    _run_step_kbeq(
        state,
        nlev,
        dt=45.0,
        kb=initial_kb,
        Pb=Pb,
        epsb=epsb,
        nuh=_zeros(nlev),
    )

    assert state.kb is not None
    expected = initial_kb[1:nlev] / (
        1.0 + 45.0 * (epsb[1:nlev] - Pb[1:nlev]) / initial_kb[1:nlev]
    )
    np.testing.assert_allclose(state.kb[1:nlev], expected, rtol=1.0e-12)


def test_dirichlet_boundaries_force_adjacent_levels_to_kb_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev, k_ubc=Dirichlet, k_lbc=Dirichlet)

    _run_step_kbeq(
        state,
        nlev,
        _DT,
        kb=np.full(nlev + 1, 2.0e-4, dtype=np.float64),
        Pb=np.full(nlev + 1, 1.0e-6, dtype=np.float64),
        epsb=np.full(nlev + 1, 2.0e-7, dtype=np.float64),
        nuh=_zeros(nlev),
    )

    assert state.kb is not None
    assert state.kb[0] == state.kb_min
    assert state.kb[1] == state.kb_min
    assert state.kb[nlev - 1] == state.kb_min
    assert state.kb[nlev] == state.kb_min


def test_clips_boundaries_and_collapsing_interior_to_kb_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_kb = np.full(nlev + 1, 1.1e-8, dtype=np.float64)
    epsb = np.full(nlev + 1, 1.0e-5, dtype=np.float64)

    _run_step_kbeq(
        state,
        nlev,
        dt=60.0,
        kb=initial_kb,
        epsb=epsb,
        nuh=_zeros(nlev),
    )

    assert state.kb is not None
    assert np.all(state.kb >= state.kb_min)
    assert np.all(state.kb <= initial_kb)
    assert state.kb[0] == state.kb_min
    assert state.kb[nlev] == state.kb_min


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_kb = np.linspace(1.5e-4, 3.0e-4, nlev + 1)
    Pb = np.linspace(-2.0e-6, 2.0e-6, nlev + 1)
    epsb = np.linspace(2.0e-7, 6.0e-7, nlev + 1)
    nuh = np.linspace(1.0e-5, 2.5e-4, nlev + 1)

    single = _run_step_kbeq(
        state,
        nlev,
        _DT,
        kb=initial_kb,
        Pb=Pb,
        epsb=epsb,
        nuh=nuh,
        n_cols=1,
    )
    multi = _run_step_kbeq(
        state,
        nlev,
        _DT,
        kb=initial_kb,
        Pb=Pb,
        epsb=epsb,
        nuh=nuh,
        n_cols=2,
    )

    single_kb = single.kb[0]
    multi_0 = multi.kb[0]
    multi_1 = multi.kb[1]
    np.testing.assert_allclose(multi_0, single_kb, rtol=1.0e-12)
    np.testing.assert_allclose(multi_1, single_kb, rtol=1.0e-12)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    initial_kb = np.linspace(1.0e-4, 2.2e-4, nlev + 1)
    Pb = np.linspace(-1.0e-6, 4.0e-6, nlev + 1)
    epsb = np.linspace(2.0e-7, 4.5e-7, nlev + 1)
    nuh = np.linspace(1.0e-6, 1.8e-4, nlev + 1)

    _run_step_kbeq(
        state,
        nlev,
        _DT,
        kb=initial_kb,
        Pb=Pb,
        epsb=epsb,
        nuh=nuh,
    )

    assert state.kb is not None
    assert np.isfinite(state.kb).all()
