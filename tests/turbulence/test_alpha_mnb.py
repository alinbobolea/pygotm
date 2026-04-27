"""Tests for pygotm.turbulence.alpha_mnb."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.turbulence.alpha_mnb import AlphaMNBWorkspace, step_alpha_mnb
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12
_MIN_ALPHA = 1.0e-10


def _make_state(nlev: int = _NLEV) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, nlev)
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: AlphaMNBWorkspace | None = None,
    tke: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    kb: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    SS: np.ndarray | None = None,
    SSCSTK: np.ndarray | None = None,
    SSSTK: np.ndarray | None = None,
    as_: np.ndarray | None = None,
    an: np.ndarray | None = None,
    at: np.ndarray | None = None,
    av: np.ndarray | None = None,
    aw: np.ndarray | None = None,
    n_cols: int = 1,
) -> AlphaMNBWorkspace:
    assert state.tke is not None
    assert state.eps is not None
    assert state.kb is not None
    assert state.as_ is not None
    assert state.an is not None
    assert state.at is not None
    assert state.av is not None
    assert state.aw is not None

    ws = workspace if workspace is not None else AlphaMNBWorkspace(nlev, n_cols=n_cols)

    zeros = np.zeros(nlev + 1, dtype=np.float64)
    for col in range(n_cols):
        fill_field_from_array(ws.tke, tke if tke is not None else state.tke, col=col)
        fill_field_from_array(ws.eps, eps if eps is not None else state.eps, col=col)
        fill_field_from_array(ws.kb, kb if kb is not None else state.kb, col=col)
        fill_field_from_array(ws.NN, NN if NN is not None else zeros, col=col)
        fill_field_from_array(ws.SS, SS if SS is not None else zeros, col=col)
        fill_field_from_array(
            ws.SSCSTK,
            SSCSTK if SSCSTK is not None else zeros,
            col=col,
        )
        fill_field_from_array(
            ws.SSSTK,
            SSSTK if SSSTK is not None else zeros,
            col=col,
        )
        fill_field_from_array(ws.as_, as_ if as_ is not None else state.as_, col=col)
        fill_field_from_array(ws.an, an if an is not None else state.an, col=col)
        fill_field_from_array(ws.at, at if at is not None else state.at, col=col)
        fill_field_from_array(ws.av, av if av is not None else state.av, col=col)
        fill_field_from_array(ws.aw, aw if aw is not None else state.aw, col=col)
    return ws


def _run_step_alpha_mnb(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: AlphaMNBWorkspace | None = None,
    tke: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    kb: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    SS: np.ndarray | None = None,
    SSCSTK: np.ndarray | None = None,
    SSSTK: np.ndarray | None = None,
    as_: np.ndarray | None = None,
    an: np.ndarray | None = None,
    at: np.ndarray | None = None,
    av: np.ndarray | None = None,
    aw: np.ndarray | None = None,
    n_cols: int = 1,
) -> AlphaMNBWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        tke=tke,
        eps=eps,
        kb=kb,
        NN=NN,
        SS=SS,
        SSCSTK=SSCSTK,
        SSSTK=SSSTK,
        as_=as_,
        an=an,
        at=at,
        av=av,
        aw=aw,
        n_cols=n_cols,
    )

    step_alpha_mnb(
        n_cols,
        nlev,
        int(SSCSTK is not None),
        int(SSSTK is not None),
        ws.tke,
        ws.eps,
        ws.kb,
        ws.NN,
        ws.SS,
        ws.SSCSTK,
        ws.SSSTK,
        ws.as_,
        ws.an,
        ws.at,
        ws.av,
        ws.aw,
    )

    assert state.as_ is not None
    assert state.an is not None
    assert state.at is not None
    assert state.av is not None
    assert state.aw is not None
    state.as_[:] = read_field_array(ws.as_)
    state.an[:] = read_field_array(ws.an)
    state.at[:] = read_field_array(ws.at)
    state.av[:] = read_field_array(ws.av)
    state.aw[:] = read_field_array(ws.aw)
    return ws


def test_import() -> None:
    assert callable(step_alpha_mnb)


def test_workspace_instantiates() -> None:
    workspace = AlphaMNBWorkspace(_NLEV, n_cols=2)
    assert workspace.as_.shape == (2, _NLEV + 1)
    assert workspace.aw.shape == (2, _NLEV + 1)


def test_formula_matches_fortran_without_stokes() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    tke = np.linspace(2.0e-4, 4.5e-4, nlev + 1)
    eps = np.linspace(2.0e-7, 7.0e-7, nlev + 1)
    kb = np.linspace(0.0, 4.0e-5, nlev + 1)
    NN = np.linspace(-2.0e-4, 3.0e-4, nlev + 1)
    SS = np.linspace(0.0, 6.0e-5, nlev + 1)

    _run_step_alpha_mnb(state, nlev, tke=tke, eps=eps, kb=kb, NN=NN, SS=SS)
    assert state.an is not None
    assert state.as_ is not None
    assert state.at is not None

    tau2 = tke * tke / (eps * eps)
    np.testing.assert_allclose(state.an, tau2 * NN, rtol=1.0e-12)
    np.testing.assert_allclose(state.as_, np.maximum(tau2 * SS, _MIN_ALPHA))
    np.testing.assert_allclose(state.at, np.maximum(tke / eps * kb / eps, _MIN_ALPHA))


def test_stokes_terms_require_both_optional_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.av is not None
    assert state.aw is not None

    tke = np.full(nlev + 1, 3.0e-4, dtype=np.float64)
    eps = np.full(nlev + 1, 4.0e-7, dtype=np.float64)
    kb = np.full(nlev + 1, 5.0e-6, dtype=np.float64)
    SSCSTK = np.linspace(1.0e-6, 8.0e-6, nlev + 1)
    SSSTK = np.linspace(-1.0e-6, 6.0e-6, nlev + 1)

    state.av[:] = 7.0
    state.aw[:] = 9.0
    _run_step_alpha_mnb(
        state,
        nlev,
        tke=tke,
        eps=eps,
        kb=kb,
        SSCSTK=SSCSTK,
    )
    np.testing.assert_allclose(state.av, 7.0)
    np.testing.assert_allclose(state.aw, 9.0)

    _run_step_alpha_mnb(
        state,
        nlev,
        tke=tke,
        eps=eps,
        kb=kb,
        SSCSTK=SSCSTK,
        SSSTK=SSSTK,
    )

    tau2 = tke * tke / (eps * eps)
    np.testing.assert_allclose(state.av, tau2 * SSCSTK, rtol=1.0e-12)
    np.testing.assert_allclose(state.aw, np.maximum(tau2 * SSSTK, _MIN_ALPHA))


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    tke = np.linspace(2.5e-4, 5.0e-4, nlev + 1)
    eps = np.linspace(3.0e-7, 8.0e-7, nlev + 1)
    kb = np.linspace(1.0e-6, 2.0e-5, nlev + 1)
    NN = np.linspace(-1.0e-4, 2.0e-4, nlev + 1)
    SS = np.linspace(1.0e-6, 3.0e-5, nlev + 1)
    SSCSTK = np.linspace(0.0, 2.5e-5, nlev + 1)
    SSSTK = np.linspace(0.0, 1.5e-5, nlev + 1)

    single_state = _make_state(nlev)
    single = _run_step_alpha_mnb(
        single_state,
        nlev,
        tke=tke,
        eps=eps,
        kb=kb,
        NN=NN,
        SS=SS,
        SSCSTK=SSCSTK,
        SSSTK=SSSTK,
    )

    multi_state = _make_state(nlev)
    multi = _run_step_alpha_mnb(
        multi_state,
        nlev,
        tke=tke,
        eps=eps,
        kb=kb,
        NN=NN,
        SS=SS,
        SSCSTK=SSCSTK,
        SSSTK=SSSTK,
        n_cols=2,
    )

    for name in ("as_", "an", "at", "av", "aw"):
        single_arr = read_field_array(getattr(single, name), col=0)
        multi_0 = read_field_array(getattr(multi, name), col=0)
        multi_1 = read_field_array(getattr(multi, name), col=1)
        np.testing.assert_allclose(multi_0, single_arr, rtol=1.0e-12)
        np.testing.assert_allclose(multi_1, single_arr, rtol=1.0e-12)


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_alpha_mnb(
        state,
        nlev,
        tke=np.full(nlev + 1, 3.5e-4, dtype=np.float64),
        eps=np.full(nlev + 1, 5.0e-7, dtype=np.float64),
        kb=np.linspace(1.0e-6, 3.0e-5, nlev + 1),
        NN=np.linspace(-2.0e-4, 3.0e-4, nlev + 1),
        SS=np.linspace(0.0, 4.0e-5, nlev + 1),
        SSCSTK=np.linspace(0.0, 1.0e-5, nlev + 1),
        SSSTK=np.linspace(0.0, 2.0e-5, nlev + 1),
    )

    for array in (state.as_, state.an, state.at, state.av, state.aw):
        assert array is not None
        assert np.isfinite(array).all()
