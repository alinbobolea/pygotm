"""Tests for pygotm.turbulence.cmue_d."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.cmue_d import CmueDWorkspace, step_cmue_d
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12
_AN_LIMIT_FACT = 0.5
_SMALL = 1.0e-10


def _configure_second_order_state(
    state: TurbulenceState,
    *,
    force_linear_branch: bool = False,
) -> None:
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
    if force_linear_branch:
        state.at2 = state.at1
    state.at3 = 2.0 * (1.0 - state.ct4)
    state.at4 = 2.0 * (1.0 - state.ct5)
    state.at5 = 2.0 * state.ctt * (1.0 - state.ct5)

    n_val = state.cc1 / 2.0
    state.cm0 = (
        (state.a2 * state.a2 - 3.0 * state.a3 * state.a3 + 3.0 * state.a1 * n_val)
        / (3.0 * n_val * n_val)
    ) ** 0.25


def _make_state(
    nlev: int = _NLEV,
    *,
    force_linear_branch: bool = False,
) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, nlev)
    _configure_second_order_state(state, force_linear_branch=force_linear_branch)
    return state


def _reference_cmue_d(
    state: TurbulenceState,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nlev = as_.size - 1
    out_as = as_.copy()
    out_an = an.copy()
    out_cmue1 = cmue1.copy()
    out_cmue2 = cmue2.copy()

    n_val = 0.5 * state.cc1
    nt_val = state.ct1

    d0 = 36.0 * n_val**3 * nt_val**2
    d1 = (
        84.0 * state.a5 * state.at3 * n_val**2 * nt_val
        + 36.0 * state.at5 * n_val**3 * nt_val
    )
    d2 = (
        9.0 * (state.at2**2 - state.at1**2) * n_val**3
        - 12.0 * (state.a2**2 - 3.0 * state.a3**2) * n_val * nt_val**2
    )
    d3 = (
        12.0
        * state.a5
        * state.at3
        * (state.a2 * state.at1 - 3.0 * state.a3 * state.at2)
        * n_val
        + 12.0 * state.a5 * state.at3 * (state.a3**2 - state.a2**2) * nt_val
        + 12.0 * state.at5 * (3.0 * state.a3**2 - state.a2**2) * n_val * nt_val
    )
    d4 = (
        48.0 * state.a5**2 * state.at3**2 * n_val
        + 36.0 * state.a5 * state.at3 * state.at5 * n_val**2
    )
    d5 = 3.0 * (state.a2**2 - 3.0 * state.a3**2) * (state.at1**2 - state.at2**2) * n_val

    n0 = 36.0 * state.a1 * n_val**2 * nt_val**2
    n1 = (
        -12.0 * state.a5 * state.at3 * (state.at1 + state.at2) * n_val**2
        + 8.0
        * state.a5
        * state.at3
        * (6.0 * state.a1 - state.a2 - 3.0 * state.a3)
        * n_val
        * nt_val
        + 36.0 * state.a1 * state.at5 * n_val**2 * nt_val
    )
    n2 = 9.0 * state.a1 * (state.at2**2 - state.at1**2) * n_val**2
    nt0 = 12.0 * state.at3 * n_val**3 * nt_val
    nt1 = 12.0 * state.a5 * state.at3**2 * n_val**2
    nt2 = (
        9.0 * state.a1 * state.at3 * (state.at1 - state.at2) * n_val**2
        + (
            6.0 * state.a1 * (state.a2 - 3.0 * state.a3)
            - 4.0 * (state.a2**2 - 3.0 * state.a3**2)
        )
        * state.at3
        * n_val
        * nt_val
    )

    cm3_inv = 1.0 / state.cm0**3
    an_min = (-(d1 + nt0) + np.sqrt((d1 + nt0) ** 2 - 4.0 * d0 * (d4 + nt1))) / (
        2.0 * (d4 + nt1)
    )

    for i in range(1, nlev):
        out_an[i] = max(out_an[i], _AN_LIMIT_FACT * an_min)
        tmp0 = -d0 - (d1 + nt0) * out_an[i] - (d4 + nt1) * out_an[i] ** 2
        tmp1 = -d2 + n0 + (n1 - d3 - nt2) * out_an[i]
        if abs(n2 - d5) < _SMALL:
            out_as[i] = -tmp0 / tmp1
        else:
            tmp2 = n2 - d5
            out_as[i] = (-tmp1 + np.sqrt(tmp1 * tmp1 - 4.0 * tmp0 * tmp2)) / (
                2.0 * tmp2
            )

        d_cm = (
            d0
            + d1 * out_an[i]
            + d2 * out_as[i]
            + d3 * out_an[i] * out_as[i]
            + d4 * out_an[i] ** 2
            + d5 * out_as[i] ** 2
        )
        n_cm = n0 + n1 * out_an[i] + n2 * out_as[i]
        n_cmp = nt0 + nt1 * out_an[i] + nt2 * out_as[i]
        out_cmue1[i] = cm3_inv * n_cm / d_cm
        out_cmue2[i] = cm3_inv * n_cmp / d_cm

    return out_as, out_an, out_cmue1, out_cmue2


def _run_step_cmue_d(
    state: TurbulenceState,
    nlev: int,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    n_cols: int = 1,
) -> CmueDWorkspace:
    assert state.cmue1 is not None
    assert state.cmue2 is not None

    ws = CmueDWorkspace(nlev, n_cols=n_cols)
    for col in range(n_cols):
        ws.as_[col] = as_
        ws.an[col] = an
        ws.cmue1[col] = cmue1 if cmue1 is not None else state.cmue1
        ws.cmue2[col] = cmue2 if cmue2 is not None else state.cmue2

    step_cmue_d(
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
        state.at5,
        ws.as_,
        ws.an,
        ws.cmue1,
        ws.cmue2,
    )

    assert state.as_ is not None
    assert state.an is not None
    state.as_[:] = ws.as_[0]
    state.an[:] = ws.an[0]
    state.cmue1[:] = ws.cmue1[0]
    state.cmue2[:] = ws.cmue2[0]
    return ws


def test_import() -> None:
    assert callable(step_cmue_d)


def test_workspace_instantiates() -> None:
    workspace = CmueDWorkspace(_NLEV, n_cols=2)
    assert workspace.cmue1.shape == (2, _NLEV + 1)


def test_general_branch_matches_reference() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    as_ = np.full(nlev + 1, 0.5, dtype=np.float64)
    an = np.full(nlev + 1, -6.0, dtype=np.float64)
    cmue1 = np.full(nlev + 1, -1.0)
    cmue2 = np.full(nlev + 1, -2.0)

    _run_step_cmue_d(state, nlev, as_=as_, an=an, cmue1=cmue1, cmue2=cmue2)
    assert state.as_ is not None
    assert state.an is not None
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    expected_as, expected_an, expected_1, expected_2 = _reference_cmue_d(
        state,
        as_=as_,
        an=an,
        cmue1=cmue1,
        cmue2=cmue2,
    )

    np.testing.assert_allclose(state.as_, expected_as, rtol=1.0e-12)
    np.testing.assert_allclose(state.an, expected_an, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue1, expected_1, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue2, expected_2, rtol=1.0e-12)


def test_special_linear_branch_matches_reference() -> None:
    nlev = _NLEV
    state = _make_state(nlev, force_linear_branch=True)
    as_ = np.full(nlev + 1, 0.6, dtype=np.float64)
    an = np.full(nlev + 1, -6.0, dtype=np.float64)

    _run_step_cmue_d(state, nlev, as_=as_, an=an)
    assert state.as_ is not None
    assert state.an is not None
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    expected_as, expected_an, expected_1, expected_2 = _reference_cmue_d(
        state,
        as_=as_,
        an=an,
        cmue1=np.zeros(nlev + 1, dtype=np.float64),
        cmue2=np.zeros(nlev + 1, dtype=np.float64),
    )

    np.testing.assert_allclose(state.as_, expected_as, rtol=1.0e-12)
    np.testing.assert_allclose(state.an, expected_an, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue1, expected_1, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue2, expected_2, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    as_ = np.full(nlev + 1, 0.55, dtype=np.float64)
    an = np.full(nlev + 1, -5.0, dtype=np.float64)

    single_state = _make_state(nlev)
    single = _run_step_cmue_d(single_state, nlev, as_=as_, an=an)
    multi_state = _make_state(nlev)
    multi = _run_step_cmue_d(multi_state, nlev, as_=as_, an=an, n_cols=2)

    for name in ("as_", "an", "cmue1", "cmue2"):
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
    _run_step_cmue_d(
        state,
        nlev,
        as_=np.full(nlev + 1, 0.7, dtype=np.float64),
        an=np.full(nlev + 1, -4.0, dtype=np.float64),
    )

    for array in (state.as_, state.an, state.cmue1, state.cmue2):
        assert array is not None
        assert np.isfinite(array).all()
