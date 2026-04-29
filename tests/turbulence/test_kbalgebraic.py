"""Tests for pygotm.turbulence.kbalgebraic."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.kbalgebraic import KBAlgebraicWorkspace, step_kbalgebraic
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12


def _make_state(nlev: int = _NLEV, *, ctt: float = 0.8) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, ctt=ctt)
    post_init_turbulence(state, nlev)
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: KBAlgebraicWorkspace | None = None,
    tke: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    kb: np.ndarray | None = None,
    Pb: np.ndarray | None = None,
    n_cols: int = 1,
) -> KBAlgebraicWorkspace:
    assert state.tke is not None
    assert state.eps is not None
    assert state.kb is not None
    assert state.Pb is not None

    ws = (
        workspace
        if workspace is not None
        else KBAlgebraicWorkspace(nlev, n_cols=n_cols)
    )
    for col in range(n_cols):
        ws.tke[col] = tke if tke is not None else state.tke
        ws.eps[col] = eps if eps is not None else state.eps
        ws.kb[col] = kb if kb is not None else state.kb
        ws.Pb[col] = Pb if Pb is not None else state.Pb
    return ws


def _run_step_kbalgebraic(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: KBAlgebraicWorkspace | None = None,
    tke: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    kb: np.ndarray | None = None,
    Pb: np.ndarray | None = None,
    n_cols: int = 1,
) -> KBAlgebraicWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        tke=tke,
        eps=eps,
        kb=kb,
        Pb=Pb,
        n_cols=n_cols,
    )

    step_kbalgebraic(
        n_cols,
        nlev,
        state.ctt,
        state.kb_min,
        ws.tke,
        ws.eps,
        ws.kb,
        ws.Pb,
    )

    assert state.kb is not None
    state.kb[:] = ws.kb[0]
    return ws


def test_import() -> None:
    from pygotm.turbulence.kbalgebraic import step_kbalgebraic as _step  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = KBAlgebraicWorkspace(_NLEV, n_cols=2)
    assert workspace.kb.shape == (2, _NLEV + 1)
    assert workspace.Pb.shape == (2, _NLEV + 1)


def test_smoke_step_kbalgebraic() -> None:
    state = _make_state()
    _run_step_kbalgebraic(state, _NLEV)


def test_formula_matches_fortran_expression() -> None:
    nlev = _NLEV
    state = _make_state(nlev, ctt=0.65)
    tke = np.linspace(2.0e-4, 5.0e-4, nlev + 1)
    eps = np.linspace(3.0e-7, 7.0e-7, nlev + 1)
    Pb = np.linspace(2.0e-7, 1.4e-6, nlev + 1)

    _run_step_kbalgebraic(state, nlev, tke=tke, eps=eps, Pb=Pb)

    assert state.kb is not None
    expected = state.ctt * tke / eps * Pb
    expected = np.maximum(expected, state.kb_min)
    np.testing.assert_allclose(state.kb, expected, rtol=1.0e-12)


def test_clips_negative_values_to_kb_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_kbalgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 2.5e-4, dtype=np.float64),
        eps=np.full(nlev + 1, 4.0e-7, dtype=np.float64),
        Pb=np.full(nlev + 1, -1.0e-6, dtype=np.float64),
    )

    assert state.kb is not None
    assert np.all(state.kb == state.kb_min)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    tke = np.linspace(1.5e-4, 3.5e-4, nlev + 1)
    eps = np.linspace(2.5e-7, 6.0e-7, nlev + 1)
    Pb = np.linspace(-5.0e-8, 1.2e-6, nlev + 1)

    state = _make_state(nlev)
    single = _run_step_kbalgebraic(state, nlev, tke=tke, eps=eps, Pb=Pb)
    single_result = single.kb[0]

    multi_state = _make_state(nlev)
    multi = _run_step_kbalgebraic(
        multi_state,
        nlev,
        tke=tke,
        eps=eps,
        Pb=Pb,
        n_cols=3,
    )

    for col in range(3):
        np.testing.assert_allclose(
            multi.kb[col],
            single_result,
            rtol=1.0e-12,
        )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_kbalgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 3.0e-4, dtype=np.float64),
        eps=np.full(nlev + 1, 5.0e-7, dtype=np.float64),
        Pb=np.linspace(1.0e-8, 8.0e-7, nlev + 1),
    )

    assert state.kb is not None
    assert np.all(np.isfinite(state.kb))
