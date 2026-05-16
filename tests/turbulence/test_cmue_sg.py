"""Tests for pygotm.turbulence.cmue_sg."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.cmue_sg import CmueSGWorkspace, step_cmue_sg
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
    return state


def _run_step_cmue_sg(
    state: TurbulenceState,
    nlev: int,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    n_cols: int = 1,
) -> CmueSGWorkspace:
    assert state.cmue1 is not None
    assert state.cmue2 is not None

    ws = CmueSGWorkspace(nlev, n_cols=n_cols)
    for col in range(n_cols):
        ws.as_[col] = as_
        ws.an[col] = an
        ws.cmue1[col] = cmue1 if cmue1 is not None else state.cmue1
        ws.cmue2[col] = cmue2 if cmue2 is not None else state.cmue2

    step_cmue_sg(
        n_cols,
        nlev,
        state.cm0_fix,
        state.Prandtl0_fix,
        ws.as_,
        ws.an,
        ws.cmue1,
        ws.cmue2,
    )

    state.cmue1[:] = ws.cmue1[0]
    state.cmue2[:] = ws.cmue2[0]
    return ws


def test_import() -> None:
    assert callable(step_cmue_sg)


def test_workspace_instantiates() -> None:
    workspace = CmueSGWorkspace(_NLEV, n_cols=2)
    assert workspace.cmue2.shape == (2, _NLEV + 1)


def test_formula_matches_fortran_and_prandtl_limit() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    as_ = np.linspace(0.12, 0.7, nlev + 1)
    an = np.linspace(-0.03, 0.35, nlev + 1)
    sentinel1 = np.full(nlev + 1, -1.0)
    sentinel2 = np.full(nlev + 1, -2.0)

    _run_step_cmue_sg(state, nlev, as_=as_, an=an, cmue1=sentinel1, cmue2=sentinel2)
    assert state.cmue1 is not None
    assert state.cmue2 is not None

    expected_1 = sentinel1.copy()
    expected_2 = sentinel2.copy()
    for i in range(1, nlev):
        ri = an[i] / (as_[i] + 1.0e-8)
        if ri >= 1.0e-10:
            prandtl = (
                state.Prandtl0_fix * np.exp(-ri / (state.Prandtl0_fix * 0.25))
                + ri / 0.25
            )
        else:
            prandtl = state.Prandtl0_fix
        expected_1[i] = state.cm0_fix
        expected_2[i] = state.cm0_fix / min(3.0, prandtl)

    np.testing.assert_allclose(state.cmue1, expected_1, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue2, expected_2, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    as_ = np.linspace(0.2, 0.9, nlev + 1)
    an = np.linspace(-0.02, 0.25, nlev + 1)

    single_state = _make_state(nlev)
    single = _run_step_cmue_sg(single_state, nlev, as_=as_, an=an)
    multi_state = _make_state(nlev)
    multi = _run_step_cmue_sg(multi_state, nlev, as_=as_, an=an, n_cols=2)

    for name in ("cmue1", "cmue2"):
        single_arr = getattr(single, name)[0]
        for col in range(2):
            np.testing.assert_allclose(
                getattr(multi, name)[col],
                single_arr,
                rtol=1.0e-12,
            )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    _run_step_cmue_sg(
        state,
        nlev,
        as_=np.linspace(0.1, 1.3, nlev + 1),
        an=np.linspace(-0.08, 0.4, nlev + 1),
    )
    assert state.cmue1 is not None
    assert state.cmue2 is not None

    assert np.isfinite(state.cmue1).all()
    assert np.isfinite(state.cmue2).all()
