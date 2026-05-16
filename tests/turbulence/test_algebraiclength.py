"""Tests for pygotm.turbulence.algebraiclength."""

from __future__ import annotations

import numpy as np
import pytest

from pygotm.turbulence.algebraiclength import (
    AlgebraicLengthWorkspace,
    step_algebraiclength,
)
from pygotm.turbulence.turbulence import (
    Blackadar,
    Parabolic,
    Robert_Ouellet,
    Triangular,
    TurbulenceState,
    Xing_Davies,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12
_DEPTH = 24.0
_Z0B = 2.0e-3
_Z0S = 1.0e-3


def make_equidistant_h(nlev: int, depth: float) -> np.ndarray:
    h = np.full(nlev + 1, depth / nlev, dtype=np.float64)
    h[0] = 0.0
    return h


def _make_state(nlev: int = _NLEV, *, length_lim: bool = True) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, length_lim=length_lim)
    post_init_turbulence(state, nlev)
    state.cde = state.cm0_fix**3
    return state


def _reference_algebraiclength(
    method: int,
    *,
    tke: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    depth: float,
    z0b: float,
    z0s: float,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: bool,
    eps_min: float,
) -> tuple[np.ndarray, np.ndarray]:
    nlev = tke.size - 1
    L = np.zeros(nlev + 1, dtype=np.float64)
    eps = np.zeros(nlev + 1, dtype=np.float64)

    db = 0.0
    if method == Parabolic:
        for i in range(1, nlev):
            db += h[i]
            ds = depth - db
            L[i] = kappa * (ds + z0s) * (db + z0b) / (ds + db + z0b + z0s)
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s
    elif method == Triangular:
        for i in range(1, nlev):
            db += h[i]
            ds = depth - db
            L[i] = kappa * min(ds + z0s, db + z0b)
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s
    elif method == Xing_Davies:
        for i in range(1, nlev):
            db += h[i]
            ds = depth - db
            db_xing = db * np.exp(-2.0 * db / depth)
            L[i] = kappa * (ds + z0s) * (db_xing + z0b) / (ds + db_xing + z0s + z0b)
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s
    elif method == Robert_Ouellet:
        for i in range(1, nlev):
            db += h[i]
            L[i] = kappa * (db + z0b) * np.sqrt(1.0 - (db - z0s) / depth)
        L[0] = kappa * z0b
        L[nlev] = kappa * (depth + z0b) * np.sqrt(z0s / depth)
    elif method == Blackadar:
        int_qz = 0.0
        int_q = 0.0
        for i in range(1, nlev):
            db += h[i]
            root_tke = np.sqrt(tke[i])
            int_qz += root_tke * (db + z0b) * h[i]
            int_q += root_tke * h[i]
        la = 0.2 * int_qz / int_q
        db = 0.0
        for i in range(1, nlev):
            db += h[i]
            ds = depth - db
            L[i] = 1.0 / (
                1.0 / (kappa * (ds + z0s)) + 1.0 / (kappa * (db + z0b)) + 1.0 / la
            )
        L[0] = kappa * z0b
        L[nlev] = kappa * z0s
    else:
        raise AssertionError(f"unexpected method={method}")

    for i in range(nlev + 1):
        if NN[i] > 0.0 and length_lim:
            lcrit = np.sqrt(2.0 * galp * galp * tke[i] / NN[i])
            L[i] = min(L[i], lcrit)
        eps[i] = cde * np.sqrt(tke[i] ** 3) / L[i]
        if eps[i] < eps_min:
            eps[i] = eps_min
            L[i] = cde * np.sqrt(tke[i] ** 3) / eps_min

    return L, eps


def _run_step_algebraiclength(
    state: TurbulenceState,
    method: int,
    nlev: int,
    *,
    tke: np.ndarray,
    h: np.ndarray | None = None,
    NN: np.ndarray | None = None,
    L: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    depth: float = _DEPTH,
    z0b: float = _Z0B,
    z0s: float = _Z0S,
    n_cols: int = 1,
) -> AlgebraicLengthWorkspace:
    assert state.L is not None
    assert state.eps is not None

    ws = AlgebraicLengthWorkspace(nlev, n_cols=n_cols)
    profile_h = h if h is not None else make_equidistant_h(nlev, depth)
    profile_nn = NN if NN is not None else np.zeros(nlev + 1, dtype=np.float64)
    for col in range(n_cols):
        ws.tke[col] = tke
        ws.eps[col] = eps if eps is not None else state.eps
        ws.L[col] = L if L is not None else state.L
        ws.h[col] = profile_h
        ws.NN[col] = profile_nn
        ws.depth[col, 0] = depth
        ws.z0b[col, 0] = z0b
        ws.z0s[col, 0] = z0s

    step_algebraiclength(
        n_cols,
        method,
        nlev,
        state.kappa,
        state.cde,
        state.galp,
        int(state.length_lim),
        state.eps_min,
        ws.tke,
        ws.eps,
        ws.L,
        ws.h,
        ws.NN,
        ws.depth,
        ws.z0b,
        ws.z0s,
    )

    state.L[:] = ws.L[0]
    state.eps[:] = ws.eps[0]
    return ws


def test_import() -> None:
    assert callable(step_algebraiclength)


def test_workspace_instantiates() -> None:
    workspace = AlgebraicLengthWorkspace(_NLEV, n_cols=2)
    assert workspace.L.shape == (2, _NLEV + 1)
    assert workspace.depth.shape == (2, _NLEV + 1)


@pytest.mark.parametrize(
    "method",
    [Parabolic, Triangular, Xing_Davies, Robert_Ouellet, Blackadar],
)
def test_profiles_match_fortran_reference_for_all_methods(method: int) -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=False)
    h = make_equidistant_h(nlev, _DEPTH)
    tke = np.linspace(2.0e-4, 4.0e-4, nlev + 1)
    NN = np.linspace(-2.0e-4, 2.0e-4, nlev + 1)

    _run_step_algebraiclength(state, method, nlev, tke=tke, h=h, NN=NN)
    assert state.L is not None
    assert state.eps is not None
    expected_L, expected_eps = _reference_algebraiclength(
        method,
        tke=tke,
        h=h,
        NN=NN,
        depth=_DEPTH,
        z0b=_Z0B,
        z0s=_Z0S,
        kappa=state.kappa,
        cde=state.cde,
        galp=state.galp,
        length_lim=False,
        eps_min=state.eps_min,
    )

    np.testing.assert_allclose(state.L, expected_L, rtol=1.0e-12)
    np.testing.assert_allclose(state.eps, expected_eps, rtol=1.0e-12)


def test_stable_length_limit_caps_profile() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=True)
    h = make_equidistant_h(nlev, _DEPTH)
    tke = np.full(nlev + 1, 3.0e-4, dtype=np.float64)
    NN = np.full(nlev + 1, 5.0e-3, dtype=np.float64)

    _run_step_algebraiclength(state, Triangular, nlev, tke=tke, h=h, NN=NN)
    assert state.L is not None

    lcrit = np.sqrt(2.0 * state.galp * state.galp * tke / NN)
    assert np.all(state.L <= lcrit + 1.0e-15)


def test_eps_min_backstop_replaces_too_small_dissipation() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=False)
    h = make_equidistant_h(nlev, _DEPTH)
    tke = np.full(nlev + 1, 1.0e-10, dtype=np.float64)
    NN = np.zeros(nlev + 1, dtype=np.float64)

    _run_step_algebraiclength(state, Parabolic, nlev, tke=tke, h=h, NN=NN)
    assert state.eps is not None
    assert state.L is not None

    assert np.all(state.eps == state.eps_min)
    expected_L = state.cde * np.sqrt(tke**3) / state.eps_min
    np.testing.assert_allclose(state.L, expected_L, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=True)
    h = make_equidistant_h(nlev, _DEPTH)
    tke = np.linspace(2.5e-4, 4.5e-4, nlev + 1)
    NN = np.linspace(-5.0e-4, 3.0e-4, nlev + 1)

    single = _run_step_algebraiclength(state, Xing_Davies, nlev, tke=tke, h=h, NN=NN)

    multi_state = _make_state(nlev, length_lim=True)
    multi = _run_step_algebraiclength(
        multi_state,
        Xing_Davies,
        nlev,
        tke=tke,
        h=h,
        NN=NN,
        n_cols=2,
    )

    for name in ("L", "eps"):
        single_arr = getattr(single, name)[0]
        for col in range(2):
            np.testing.assert_allclose(
                getattr(multi, name)[col],
                single_arr,
                rtol=1.0e-12,
            )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=True)
    h = make_equidistant_h(nlev, _DEPTH)
    tke = np.linspace(1.5e-4, 6.0e-4, nlev + 1)
    NN = np.linspace(-1.0e-3, 2.0e-3, nlev + 1)

    _run_step_algebraiclength(state, Blackadar, nlev, tke=tke, h=h, NN=NN)
    assert state.L is not None
    assert state.eps is not None

    assert np.isfinite(state.L).all()
    assert np.isfinite(state.eps).all()
