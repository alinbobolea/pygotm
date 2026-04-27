"""Tests for pygotm.turbulence.variances."""

from __future__ import annotations

import numpy as np
from taichi_helpers import fill_field_from_array, read_field_array
from turbulence_model_analysis_reference import configure_second_order_state

from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)
from pygotm.turbulence.variances import VariancesWorkspace, step_variances

_NLEV = 8


def _make_state(nlev: int = _NLEV) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state)
    post_init_turbulence(state, nlev)
    configure_second_order_state(state)
    return state


def _reference_variances(
    state: TurbulenceState,
    *,
    tke: np.ndarray,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    num: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_value = 0.5 * state.cc1
    fac1 = 2.0 / 3.0
    fac3 = state.a2 / 3.0 + state.a3
    fac4 = state.a2 / 3.0 - state.a3
    fac5 = 2.0 / 3.0 * state.a2
    fac2 = 1.0 / (n_value * eps)

    uu = tke * (
        fac1 + fac2 * (fac3 * num * SSU - fac5 * num * SSV - 4.0 / 3.0 * state.a5 * B)
    )
    vv = tke * (
        fac1 + fac2 * (fac3 * num * SSV - fac5 * num * SSU - 4.0 / 3.0 * state.a5 * B)
    )
    ww = tke * (
        fac1 + fac2 * (fac4 * (P + Px) + 8.0 / 3.0 * state.a5 * B)
    )
    return uu, vv, ww


def _run_step(
    state: TurbulenceState,
    *,
    tke: np.ndarray,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    num: np.ndarray,
    SSU: np.ndarray,
    SSV: np.ndarray,
    n_cols: int = 1,
) -> VariancesWorkspace:
    workspace = VariancesWorkspace(_NLEV, n_cols=n_cols)
    for col in range(n_cols):
        fill_field_from_array(workspace.tke, tke, col=col)
        fill_field_from_array(workspace.eps, eps, col=col)
        fill_field_from_array(workspace.P, P, col=col)
        fill_field_from_array(workspace.B, B, col=col)
        fill_field_from_array(workspace.Px, Px, col=col)
        fill_field_from_array(workspace.num, num, col=col)
        fill_field_from_array(workspace.SSU, SSU, col=col)
        fill_field_from_array(workspace.SSV, SSV, col=col)

    step_variances(
        n_cols,
        _NLEV,
        state.cc1,
        state.ct1,
        state.a2,
        state.a3,
        state.a5,
        workspace.tke,
        workspace.eps,
        workspace.P,
        workspace.B,
        workspace.Px,
        workspace.num,
        workspace.SSU,
        workspace.SSV,
        workspace.uu,
        workspace.vv,
        workspace.ww,
    )
    return workspace


def test_import() -> None:
    assert callable(step_variances)


def test_workspace_instantiates() -> None:
    workspace = VariancesWorkspace(_NLEV, n_cols=2)
    assert workspace.uu.shape == (2, _NLEV + 1)


def test_matches_reference_translation() -> None:
    state = _make_state()
    tke = np.linspace(2.0e-4, 5.0e-4, _NLEV + 1)
    eps = np.linspace(2.0e-6, 6.0e-6, _NLEV + 1)
    P = np.linspace(1.0e-6, 5.0e-6, _NLEV + 1)
    B = np.linspace(-2.0e-6, -5.0e-7, _NLEV + 1)
    Px = np.linspace(3.0e-7, 1.2e-6, _NLEV + 1)
    num = np.linspace(1.0e-4, 4.0e-4, _NLEV + 1)
    SSU = np.linspace(1.0e-4, 6.0e-4, _NLEV + 1)
    SSV = np.linspace(2.0e-4, 7.0e-4, _NLEV + 1)

    workspace = _run_step(
        state,
        tke=tke,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        num=num,
        SSU=SSU,
        SSV=SSV,
    )
    expected = _reference_variances(
        state,
        tke=tke,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        num=num,
        SSU=SSU,
        SSV=SSV,
    )

    np.testing.assert_allclose(
        read_field_array(workspace.uu), expected[0], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        read_field_array(workspace.vv), expected[1], rtol=1.0e-12
    )
    np.testing.assert_allclose(
        read_field_array(workspace.ww), expected[2], rtol=1.0e-12
    )


def test_variances_remain_positive_for_physically_valid_inputs() -> None:
    state = _make_state()
    tke = np.full(_NLEV + 1, 4.0e-4)
    eps = np.full(_NLEV + 1, 4.0e-6)
    P = np.full(_NLEV + 1, 2.0e-6)
    B = np.full(_NLEV + 1, -5.0e-7)
    Px = np.full(_NLEV + 1, 1.0e-7)
    num = np.full(_NLEV + 1, 2.5e-4)
    SSU = np.full(_NLEV + 1, 3.0e-4)
    SSV = np.full(_NLEV + 1, 2.0e-4)

    workspace = _run_step(
        state,
        tke=tke,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        num=num,
        SSU=SSU,
        SSV=SSV,
    )

    for name in ("uu", "vv", "ww"):
        assert np.all(read_field_array(getattr(workspace, name)) > 0.0)


def test_multicolumn_parity_for_identical_columns() -> None:
    state = _make_state()
    tke = np.linspace(2.0e-4, 5.0e-4, _NLEV + 1)
    eps = np.linspace(2.0e-6, 6.0e-6, _NLEV + 1)
    P = np.linspace(1.0e-6, 5.0e-6, _NLEV + 1)
    B = np.linspace(-2.0e-6, -5.0e-7, _NLEV + 1)
    Px = np.linspace(3.0e-7, 1.2e-6, _NLEV + 1)
    num = np.linspace(1.0e-4, 4.0e-4, _NLEV + 1)
    SSU = np.linspace(1.0e-4, 6.0e-4, _NLEV + 1)
    SSV = np.linspace(2.0e-4, 7.0e-4, _NLEV + 1)

    single = _run_step(
        state,
        tke=tke,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        num=num,
        SSU=SSU,
        SSV=SSV,
    )
    multi = _run_step(
        state,
        tke=tke,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        num=num,
        SSU=SSU,
        SSV=SSV,
        n_cols=2,
    )

    for name in ("uu", "vv", "ww"):
        expected = read_field_array(getattr(single, name))
        for col in range(2):
            np.testing.assert_allclose(
                read_field_array(getattr(multi, name), col=col),
                expected,
            )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    state = _make_state()
    tke = np.full(_NLEV + 1, 4.0e-4)
    eps = np.full(_NLEV + 1, 4.0e-6)
    P = np.full(_NLEV + 1, 2.0e-6)
    B = np.full(_NLEV + 1, -5.0e-7)
    Px = np.full(_NLEV + 1, 1.0e-7)
    num = np.full(_NLEV + 1, 2.5e-4)
    SSU = np.full(_NLEV + 1, 3.0e-4)
    SSV = np.full(_NLEV + 1, 2.0e-4)

    workspace = _run_step(
        state,
        tke=tke,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        num=num,
        SSU=SSU,
        SSV=SSV,
    )

    for name in ("uu", "vv", "ww"):
        assert np.isfinite(read_field_array(getattr(workspace, name))).all()
