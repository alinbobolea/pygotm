"""Tests for pygotm.turbulence.epsbalgebraic."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.turbulence.epsbalgebraic import (
    EpsBAlgebraicWorkspace,
    step_epsbalgebraic,
)
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
    workspace: EpsBAlgebraicWorkspace | None = None,
    tke: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    kb: np.ndarray | None = None,
    epsb: np.ndarray | None = None,
    n_cols: int = 1,
) -> EpsBAlgebraicWorkspace:
    assert state.tke is not None
    assert state.eps is not None
    assert state.kb is not None
    assert state.epsb is not None

    ws = (
        workspace
        if workspace is not None
        else EpsBAlgebraicWorkspace(nlev, n_cols=n_cols)
    )
    for col in range(n_cols):
        fill_field_from_array(ws.tke, tke if tke is not None else state.tke, col=col)
        fill_field_from_array(ws.eps, eps if eps is not None else state.eps, col=col)
        fill_field_from_array(ws.kb, kb if kb is not None else state.kb, col=col)
        fill_field_from_array(
            ws.epsb,
            epsb if epsb is not None else state.epsb,
            col=col,
        )
    return ws


def _run_step_epsbalgebraic(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: EpsBAlgebraicWorkspace | None = None,
    tke: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    kb: np.ndarray | None = None,
    epsb: np.ndarray | None = None,
    n_cols: int = 1,
) -> EpsBAlgebraicWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        tke=tke,
        eps=eps,
        kb=kb,
        epsb=epsb,
        n_cols=n_cols,
    )

    step_epsbalgebraic(
        n_cols,
        nlev,
        state.ctt,
        state.epsb_min,
        ws.tke,
        ws.eps,
        ws.kb,
        ws.epsb,
    )

    assert state.epsb is not None
    state.epsb[:] = read_field_array(ws.epsb)
    return ws


def test_import() -> None:
    from pygotm.turbulence.epsbalgebraic import (
        step_epsbalgebraic as _step,
    )  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = EpsBAlgebraicWorkspace(_NLEV, n_cols=2)
    assert workspace.epsb.shape == (2, _NLEV + 1)
    assert workspace.kb.shape == (2, _NLEV + 1)


def test_smoke_step_epsbalgebraic() -> None:
    state = _make_state()
    _run_step_epsbalgebraic(state, _NLEV)


def test_formula_matches_fortran_expression() -> None:
    nlev = _NLEV
    state = _make_state(nlev, ctt=0.55)
    tke = np.linspace(2.0e-4, 4.0e-4, nlev + 1)
    eps = np.linspace(2.0e-7, 8.0e-7, nlev + 1)
    kb = np.linspace(1.0e-6, 7.0e-5, nlev + 1)

    _run_step_epsbalgebraic(state, nlev, tke=tke, eps=eps, kb=kb)

    assert state.epsb is not None
    expected = (1.0 / state.ctt) * eps / tke * kb
    expected = np.maximum(expected, state.epsb_min)
    np.testing.assert_allclose(state.epsb, expected, rtol=1.0e-12)


def test_clips_small_values_to_epsb_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_epsbalgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 3.0e-4, dtype=np.float64),
        eps=np.full(nlev + 1, 1.0e-7, dtype=np.float64),
        kb=np.zeros(nlev + 1, dtype=np.float64),
    )

    assert state.epsb is not None
    assert np.all(state.epsb == state.epsb_min)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    tke = np.linspace(2.5e-4, 5.0e-4, nlev + 1)
    eps = np.linspace(3.5e-7, 9.0e-7, nlev + 1)
    kb = np.linspace(8.0e-7, 5.0e-5, nlev + 1)

    state = _make_state(nlev)
    single = _run_step_epsbalgebraic(state, nlev, tke=tke, eps=eps, kb=kb)
    single_result = read_field_array(single.epsb)

    multi_state = _make_state(nlev)
    multi = _run_step_epsbalgebraic(
        multi_state,
        nlev,
        tke=tke,
        eps=eps,
        kb=kb,
        n_cols=3,
    )

    for col in range(3):
        np.testing.assert_allclose(
            read_field_array(multi.epsb, col=col),
            single_result,
            rtol=1.0e-12,
        )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_epsbalgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 2.5e-4, dtype=np.float64),
        eps=np.full(nlev + 1, 6.0e-7, dtype=np.float64),
        kb=np.linspace(1.0e-6, 4.0e-5, nlev + 1),
    )

    assert state.epsb is not None
    assert np.all(np.isfinite(state.epsb))
