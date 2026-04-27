"""Tests for pygotm.turbulence.tkealgebraic."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, fill_field_scalar, read_field_array

from pygotm.turbulence.tkealgebraic import (
    TKEAlgebraicWorkspace,
    step_tkealgebraic,
)
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12


def _make_state(nlev: int = _NLEV) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, nlev)
    state.cm0 = state.cm0_fix
    state.cde = state.cm0**3
    return state


def _prepare_workspace(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: TKEAlgebraicWorkspace | None = None,
    tke: np.ndarray | None = None,
    L: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    SS: np.ndarray | None = None,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    n_cols: int = 1,
) -> TKEAlgebraicWorkspace:
    assert state.tke is not None
    assert state.tkeo is not None
    assert state.L is not None
    assert state.cmue1 is not None
    assert state.cmue2 is not None

    ws = (
        workspace
        if workspace is not None
        else TKEAlgebraicWorkspace(nlev, n_cols=n_cols)
    )

    nn = NN if NN is not None else np.zeros(nlev + 1, dtype=np.float64)
    ss = SS if SS is not None else np.zeros(nlev + 1, dtype=np.float64)

    for col in range(n_cols):
        fill_field_from_array(ws.tke, tke if tke is not None else state.tke, col=col)
        fill_field_from_array(ws.tkeo, state.tkeo, col=col)
        fill_field_from_array(ws.L, L if L is not None else state.L, col=col)
        fill_field_from_array(ws.NN, nn, col=col)
        fill_field_from_array(ws.SS, ss, col=col)
        fill_field_from_array(
            ws.cmue1,
            cmue1 if cmue1 is not None else state.cmue1,
            col=col,
        )
        fill_field_from_array(
            ws.cmue2,
            cmue2 if cmue2 is not None else state.cmue2,
            col=col,
        )
        fill_field_scalar(ws.u_taus, u_taus, col=col)
        fill_field_scalar(ws.u_taub, u_taub, col=col)

    return ws


def _run_step_tkealgebraic(
    state: TurbulenceState,
    nlev: int,
    *,
    workspace: TKEAlgebraicWorkspace | None = None,
    tke: np.ndarray | None = None,
    L: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    SS: np.ndarray | None = None,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    u_taus: float = 0.0,
    u_taub: float = 0.0,
    n_cols: int = 1,
) -> TKEAlgebraicWorkspace:
    ws = _prepare_workspace(
        state,
        nlev,
        workspace=workspace,
        tke=tke,
        L=L,
        NN=NN,
        SS=SS,
        cmue1=cmue1,
        cmue2=cmue2,
        u_taus=u_taus,
        u_taub=u_taub,
        n_cols=n_cols,
    )

    step_tkealgebraic(
        n_cols,
        nlev,
        state.k_min,
        state.cm0,
        state.cde,
        ws.tke,
        ws.tkeo,
        ws.L,
        ws.NN,
        ws.SS,
        ws.cmue1,
        ws.cmue2,
        ws.u_taus,
        ws.u_taub,
    )

    assert state.tke is not None
    assert state.tkeo is not None
    state.tke[:] = read_field_array(ws.tke)
    state.tkeo[:] = read_field_array(ws.tkeo)
    return ws


def test_import() -> None:
    from pygotm.turbulence.tkealgebraic import step_tkealgebraic as _step  # noqa: F401

    assert callable(_step)


def test_workspace_instantiates() -> None:
    workspace = TKEAlgebraicWorkspace(_NLEV, n_cols=2)
    assert workspace.tke.shape == (2, _NLEV + 1)
    assert workspace.cmue2.shape == (2, _NLEV + 1)


def test_smoke_step_tkealgebraic() -> None:
    state = _make_state()
    _run_step_tkealgebraic(state, _NLEV)


def test_tkeo_saves_old_profile() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    assert state.tke is not None
    state.tke[:] = np.linspace(1.0e-4, 2.8e-4, nlev + 1)
    tke_before = state.tke.copy()

    _run_step_tkealgebraic(state, nlev)

    assert state.tkeo is not None
    np.testing.assert_allclose(state.tkeo, tke_before, rtol=1.0e-12)


def test_interior_formula_matches_fortran_expression() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    L = np.linspace(0.1, 1.3, nlev + 1)
    NN = np.linspace(1.0e-5, 6.0e-5, nlev + 1)
    SS = np.linspace(8.0e-5, 2.0e-4, nlev + 1)
    cmue1 = np.linspace(0.4, 0.8, nlev + 1)
    cmue2 = np.linspace(0.1, 0.3, nlev + 1)

    _run_step_tkealgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 9.0e-5, dtype=np.float64),
        L=L,
        NN=NN,
        SS=SS,
        cmue1=cmue1,
        cmue2=cmue2,
        u_taus=0.015,
        u_taub=0.01,
    )

    assert state.tke is not None
    expected = (
        L[1:nlev] ** 2
        / state.cde
        * (cmue1[1:nlev] * SS[1:nlev] - cmue2[1:nlev] * NN[1:nlev])
    )
    expected = np.maximum(expected, state.k_min)
    np.testing.assert_allclose(state.tke[1:nlev], expected, rtol=1.0e-12)


def test_boundary_values_match_fortran_formula() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    u_taus = 0.013
    u_taub = 0.009

    _run_step_tkealgebraic(state, nlev, u_taus=u_taus, u_taub=u_taub)

    assert state.tke is not None
    boundary_scale = np.sqrt(state.cm0 * state.cde)
    np.testing.assert_allclose(
        state.tke[0],
        np.maximum(u_taub**2 / boundary_scale, state.k_min),
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        state.tke[nlev],
        np.maximum(u_taus**2 / boundary_scale, state.k_min),
        rtol=1.0e-12,
    )


def test_clips_negative_and_zero_values_to_k_min() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_tkealgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 5.0e-5, dtype=np.float64),
        L=np.full(nlev + 1, 0.5, dtype=np.float64),
        NN=np.full(nlev + 1, 2.0e-4, dtype=np.float64),
        SS=np.zeros(nlev + 1, dtype=np.float64),
        cmue1=np.zeros(nlev + 1, dtype=np.float64),
        cmue2=np.ones(nlev + 1, dtype=np.float64),
        u_taus=0.0,
        u_taub=0.0,
    )

    assert state.tke is not None
    assert np.all(state.tke == state.k_min)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    tke = np.linspace(1.0e-4, 2.2e-4, nlev + 1)
    L = np.linspace(0.15, 1.25, nlev + 1)
    NN = np.linspace(2.0e-6, 4.5e-5, nlev + 1)
    SS = np.linspace(9.0e-5, 1.8e-4, nlev + 1)
    cmue1 = np.linspace(0.55, 0.85, nlev + 1)
    cmue2 = np.linspace(0.08, 0.28, nlev + 1)

    single = _run_step_tkealgebraic(
        state,
        nlev,
        tke=tke,
        L=L,
        NN=NN,
        SS=SS,
        cmue1=cmue1,
        cmue2=cmue2,
        u_taus=0.012,
        u_taub=0.007,
    )
    single_result = read_field_array(single.tke)

    multi_state = _make_state(nlev)
    multi = _run_step_tkealgebraic(
        multi_state,
        nlev,
        tke=tke,
        L=L,
        NN=NN,
        SS=SS,
        cmue1=cmue1,
        cmue2=cmue2,
        u_taus=0.012,
        u_taub=0.007,
        n_cols=3,
    )

    for col in range(3):
        np.testing.assert_allclose(
            read_field_array(multi.tke, col=col),
            single_result,
            rtol=1.0e-12,
        )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_tkealgebraic(
        state,
        nlev,
        tke=np.full(nlev + 1, 2.0e-4, dtype=np.float64),
        L=np.linspace(0.2, 1.0, nlev + 1),
        NN=np.linspace(0.0, 5.0e-5, nlev + 1),
        SS=np.linspace(4.0e-5, 1.2e-4, nlev + 1),
        cmue1=np.full(nlev + 1, 0.7, dtype=np.float64),
        cmue2=np.full(nlev + 1, 0.1, dtype=np.float64),
        u_taus=0.011,
        u_taub=0.006,
    )

    assert state.tke is not None
    assert state.tkeo is not None
    assert np.all(np.isfinite(state.tke))
    assert np.all(np.isfinite(state.tkeo))
