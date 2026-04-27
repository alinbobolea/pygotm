"""Tests for pygotm.turbulence.cmue_b."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array

from pygotm.turbulence.cmue_b import CmueBWorkspace, step_cmue_b
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12


def _configure_second_order_state(state: TurbulenceState) -> None:
    state.cc1 = 5.0
    state.cc2 = 0.8
    state.cc3 = 1.968
    state.cc4 = 1.136
    state.cc5 = 0.0
    state.cc6 = 0.4
    state.ct1 = 5.95
    state.ct2 = 0.6
    state.ct3 = 1.0
    state.ct4 = 0.0
    state.ct5 = 0.3333
    state.ctt = 0.72

    state.a1 = 2.0 / 3.0 - state.cc2 / 2.0
    state.a2 = 1.0 - state.cc3 / 2.0
    state.a3 = 1.0 - state.cc4 / 2.0
    state.a4 = state.cc5 / 2.0
    state.a5 = 0.5 - state.cc6 / 2.0
    state.at1 = 1.0 - state.ct2
    state.at2 = 1.0 - state.ct3
    state.at3 = 2.0 * (1.0 - state.ct4)
    state.at4 = 2.0 * (1.0 - state.ct5)
    state.at5 = 2.0 * state.ctt * (1.0 - state.ct5)

    n_val = state.cc1 / 2.0
    state.cm0 = (
        (
            state.a2 * state.a2
            - 3.0 * state.a3 * state.a3
            + 3.0 * state.a1 * n_val
        )
        / (3.0 * n_val * n_val)
    ) ** 0.25


def _make_state(nlev: int = _NLEV) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, nlev)
    _configure_second_order_state(state)
    return state


def _reference_cmue_b(
    state: TurbulenceState,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    gam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlev = as_.size - 1
    out_cmue1 = cmue1.copy()
    out_cmue2 = cmue2.copy()
    out_gam = gam.copy()

    n_val = 0.5 * state.cc1
    nt_val = state.ct1

    d0 = 36.0 * n_val**3 * nt_val**2
    d1 = 84.0 * state.a5 * state.at3 * n_val**2 * nt_val
    d2 = 9.0 * (state.at2**2 - state.at1**2) * n_val**3 - 12.0 * (
        state.a2**2 - 3.0 * state.a3**2
    ) * n_val * nt_val**2
    d3 = (
        12.0
        * state.a5
        * state.at3
        * (state.a2 * state.at1 - 3.0 * state.a3 * state.at2)
        * n_val
        + 12.0
        * state.a5
        * state.at3
        * (state.a3**2 - state.a2**2)
        * nt_val
    )
    d4 = 48.0 * state.a5**2 * state.at3**2 * n_val
    d5 = 3.0 * (state.a2**2 - 3.0 * state.a3**2) * (
        state.at1**2 - state.at2**2
    ) * n_val

    n0 = 36.0 * state.a1 * n_val**2 * nt_val**2
    n1 = -12.0 * state.a5 * state.at3 * (state.at1 + state.at2) * n_val**2 + 8.0 * (
        state.a5 * state.at3 * (6.0 * state.a1 - state.a2 - 3.0 * state.a3)
    ) * n_val * nt_val
    n2 = 9.0 * state.a1 * (state.at2**2 - state.at1**2) * n_val**2
    n3 = 12.0 * state.a5 * state.at4 * (
        3.0 * (state.at1 + state.at2) * n_val**2
        + 2.0 * (state.a2 + 3.0 * state.a3) * n_val * nt_val
    )

    nt0 = 12.0 * state.at3 * n_val**3 * nt_val
    nt1 = 12.0 * state.a5 * state.at3**2 * n_val**2
    nt2 = 9.0 * state.a1 * state.at3 * (state.at1 - state.at2) * n_val**2 + (
        6.0 * state.a1 * (state.a2 - 3.0 * state.a3)
        - 4.0 * (state.a2**2 - 3.0 * state.a3**2)
    ) * state.at3 * n_val * nt_val

    gam0 = 36.0 * state.at4 * n_val**3 * nt_val
    gam1 = 36.0 * state.a5 * state.at3 * state.at4 * n_val**2
    gam2 = -12.0 * state.at4 * (state.a2**2 - 3.0 * state.a3**2) * n_val * nt_val

    cm3_inv = 1.0 / state.cm0**3

    for i in range(1, nlev):
        d_cm = (
            d0
            + d1 * an[i]
            + d2 * as_[i]
            + d3 * an[i] * as_[i]
            + d4 * an[i] ** 2
            + d5 * as_[i] ** 2
        )
        n_cm = n0 + n1 * an[i] + n2 * as_[i] + n3 * at[i]
        n_cmp = nt0 + nt1 * an[i] + nt2 * as_[i]
        n_gam = (gam0 + gam1 * an[i] + gam2 * as_[i]) * at[i]
        out_cmue1[i] = cm3_inv * n_cm / d_cm
        out_cmue2[i] = cm3_inv * n_cmp / d_cm
        out_gam[i] = n_gam / d_cm

    return out_cmue1, out_cmue2, out_gam


def _run_step_cmue_b(
    state: TurbulenceState,
    nlev: int,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    gam: np.ndarray | None = None,
    n_cols: int = 1,
) -> CmueBWorkspace:
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.gam is not None

    ws = CmueBWorkspace(nlev, n_cols=n_cols)
    for col in range(n_cols):
        fill_field_from_array(ws.as_, as_, col=col)
        fill_field_from_array(ws.an, an, col=col)
        fill_field_from_array(ws.at, at, col=col)
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
        fill_field_from_array(ws.gam, gam if gam is not None else state.gam, col=col)

    step_cmue_b(
        n_cols,
        nlev,
        state.cm0,
        state.cc1,
        state.ct1,
        state.a1,
        state.a2,
        state.a3,
        state.a5,
        state.at1,
        state.at2,
        state.at3,
        state.at4,
        ws.as_,
        ws.an,
        ws.at,
        ws.cmue1,
        ws.cmue2,
        ws.gam,
    )

    state.cmue1[:] = read_field_array(ws.cmue1)
    state.cmue2[:] = read_field_array(ws.cmue2)
    state.gam[:] = read_field_array(ws.gam)
    return ws


def test_import() -> None:
    assert callable(step_cmue_b)


def test_workspace_instantiates() -> None:
    workspace = CmueBWorkspace(_NLEV, n_cols=2)
    assert workspace.gam.shape == (2, _NLEV + 1)


def test_formula_matches_reference_and_preserves_boundaries() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    as_ = np.linspace(0.4, 2.0, nlev + 1)
    an = np.linspace(-0.3, 0.4, nlev + 1)
    at = np.linspace(0.05, 0.25, nlev + 1)
    sentinel1 = np.full(nlev + 1, -1.0)
    sentinel2 = np.full(nlev + 1, -2.0)
    sentinel3 = np.full(nlev + 1, -3.0)

    _run_step_cmue_b(
        state,
        nlev,
        as_=as_,
        an=an,
        at=at,
        cmue1=sentinel1,
        cmue2=sentinel2,
        gam=sentinel3,
    )
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.gam is not None

    expected_1, expected_2, expected_3 = _reference_cmue_b(
        state,
        as_=as_,
        an=an,
        at=at,
        cmue1=sentinel1,
        cmue2=sentinel2,
        gam=sentinel3,
    )

    np.testing.assert_allclose(state.cmue1, expected_1, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue2, expected_2, rtol=1.0e-12)
    np.testing.assert_allclose(state.gam, expected_3, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    as_ = np.linspace(0.5, 2.2, nlev + 1)
    an = np.linspace(-0.2, 0.35, nlev + 1)
    at = np.linspace(0.03, 0.18, nlev + 1)

    single_state = _make_state(nlev)
    single = _run_step_cmue_b(single_state, nlev, as_=as_, an=an, at=at)
    multi_state = _make_state(nlev)
    multi = _run_step_cmue_b(multi_state, nlev, as_=as_, an=an, at=at, n_cols=2)

    for name in ("cmue1", "cmue2", "gam"):
        single_arr = read_field_array(getattr(single, name), col=0)
        for col in range(2):
            np.testing.assert_allclose(
                read_field_array(getattr(multi, name), col=col),
                single_arr,
                rtol=1.0e-12,
            )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    _run_step_cmue_b(
        state,
        nlev,
        as_=np.linspace(0.4, 2.5, nlev + 1),
        an=np.linspace(-0.4, 0.45, nlev + 1),
        at=np.linspace(0.02, 0.3, nlev + 1),
    )
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.gam is not None

    assert np.isfinite(state.cmue1).all()
    assert np.isfinite(state.cmue2).all()
    assert np.isfinite(state.gam).all()
