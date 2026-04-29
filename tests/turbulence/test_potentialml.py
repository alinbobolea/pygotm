"""Tests for pygotm.turbulence.potentialml."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.potentialml import PotentialMLWorkspace, step_potentialml
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 6
_DEPTH = 18.0
_Z0B = 2.0e-3
_Z0S = 1.0e-3


def make_equidistant_h(nlev: int, depth: float) -> np.ndarray:
    h = np.full(nlev + 1, depth / nlev, dtype=np.float64)
    h[0] = 0.0
    return h


def _make_state(nlev: int, *, length_lim: bool) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, length_lim=length_lim)
    post_init_turbulence(state, nlev)
    state.cde = state.cm0_fix**3
    return state


def _reference_potentialml(
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
    lu = np.zeros(nlev + 1, dtype=np.float64)
    ld = np.zeros(nlev + 1, dtype=np.float64)
    L = np.zeros(nlev + 1, dtype=np.float64)
    eps = np.zeros(nlev + 1, dtype=np.float64)

    for i in range(1, nlev):
        integral = 0.0
        buoydiff = 0.0
        for j in range(i + 1, nlev + 1):
            buoydiff += NN[j - 1] * 0.5 * (h[j] + h[j - 1])
            integral += buoydiff * h[j]
            if integral >= tke[i]:
                if j != nlev:
                    if j != i + 1:
                        lu[i] = lu[i] - (integral - tke[i]) / buoydiff
                    else:
                        if NN[i] > 1.0e-8:
                            lu[i] = np.sqrt(2.0) * np.sqrt(tke[i]) / np.sqrt(NN[i])
                        else:
                            lu[i] = h[i]
                    break
            lu[i] += h[j]

    for i in range(nlev - 1, 0, -1):
        integral = 0.0
        buoydiff = 0.0
        for j in range(i - 1, 0, -1):
            buoydiff += NN[j] * 0.5 * (h[j + 1] + h[j])
            integral -= buoydiff * h[j]
            if integral >= tke[i]:
                if j != i - 1:
                    ld[i] = ld[i] - (integral - tke[i]) / buoydiff
                else:
                    if NN[i] > 1.0e-8:
                        ld[i] = np.sqrt(2.0) * np.sqrt(tke[i]) / np.sqrt(NN[i])
                    else:
                        ld[i] = h[i]
                break
            ld[i] += h[j]

    for i in range(1, nlev):
        L[i] = np.sqrt(lu[i] * ld[i])

    L[0] = kappa * z0b
    L[nlev] = kappa * z0s

    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(nlev + 1):
            if NN[i] > 0.0 and length_lim:
                lcrit = np.sqrt(2.0 * galp * galp * tke[i] / NN[i])
                L[i] = min(L[i], lcrit)
            eps[i] = cde * np.sqrt(tke[i] ** 3) / L[i]
            if eps[i] < eps_min:
                eps[i] = eps_min
                L[i] = cde * np.sqrt(tke[i] ** 3) / eps_min

    return L, eps


def _run_step(
    state: TurbulenceState,
    nlev: int,
    *,
    tke: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    depth: float = _DEPTH,
    z0b: float = _Z0B,
    z0s: float = _Z0S,
    n_cols: int = 1,
) -> PotentialMLWorkspace:
    assert state.L is not None
    assert state.eps is not None

    workspace = PotentialMLWorkspace(nlev, n_cols=n_cols)
    for col in range(n_cols):
        workspace.tke[col] = tke
        workspace.eps[col] = state.eps
        workspace.L[col] = state.L
        workspace.h[col] = h
        workspace.NN[col] = NN
        workspace.depth[col, 0] = depth
        workspace.z0b[col, 0] = z0b
        workspace.z0s[col, 0] = z0s

    step_potentialml(
        n_cols,
        nlev,
        state.kappa,
        state.cde,
        state.galp,
        int(state.length_lim),
        state.eps_min,
        workspace.tke,
        workspace.eps,
        workspace.L,
        workspace.h,
        workspace.NN,
        workspace.depth,
        workspace.z0b,
        workspace.z0s,
    )
    return workspace


def test_import() -> None:
    assert callable(step_potentialml)


def test_workspace_instantiates() -> None:
    workspace = PotentialMLWorkspace(_NLEV, n_cols=2)
    assert workspace.L.shape == (2, _NLEV + 1)


def test_matches_reference_translation() -> None:
    state = _make_state(_NLEV, length_lim=False)
    tke = np.linspace(2.0e-4, 5.0e-4, _NLEV + 1)
    h = make_equidistant_h(_NLEV, _DEPTH)
    NN = np.linspace(5.0e-5, 3.0e-4, _NLEV + 1)

    workspace = _run_step(state, _NLEV, tke=tke, h=h, NN=NN)
    expected_L, expected_eps = _reference_potentialml(
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

    np.testing.assert_allclose(
        workspace.L[0], expected_L, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        workspace.eps[0], expected_eps, rtol=1.0e-12
    )


def test_stable_length_limit_caps_profile() -> None:
    state = _make_state(_NLEV, length_lim=True)
    tke = np.full(_NLEV + 1, 4.0e-4)
    h = make_equidistant_h(_NLEV, _DEPTH)
    NN = np.full(_NLEV + 1, 1.0e-2)

    workspace = _run_step(state, _NLEV, tke=tke, h=h, NN=NN)
    result = workspace.L[0]
    lcrit = np.sqrt(2.0 * state.galp * state.galp * tke / NN)

    assert np.all(result <= lcrit + 1.0e-15)


def test_eps_min_backstop_replaces_too_small_dissipation() -> None:
    state = _make_state(1, length_lim=False)
    tke = np.full(2, 1.0e-10)
    h = make_equidistant_h(1, 1.0)
    NN = np.zeros(2)

    workspace = _run_step(state, 1, tke=tke, h=h, NN=NN, depth=1.0, z0b=0.2, z0s=0.3)
    expected_L = state.cde * np.sqrt(tke**3) / state.eps_min

    np.testing.assert_allclose(
        workspace.eps[0], np.full(2, state.eps_min)
    )
    np.testing.assert_allclose(workspace.L[0], expected_L, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    state = _make_state(_NLEV, length_lim=False)
    tke = np.linspace(2.0e-4, 5.0e-4, _NLEV + 1)
    h = make_equidistant_h(_NLEV, _DEPTH)
    NN = np.linspace(5.0e-5, 3.0e-4, _NLEV + 1)

    single = _run_step(state, _NLEV, tke=tke, h=h, NN=NN)
    multi = _run_step(state, _NLEV, tke=tke, h=h, NN=NN, n_cols=2)

    for name in ("L", "eps"):
        expected = getattr(single, name)[0]
        for col in range(2):
            np.testing.assert_allclose(
                getattr(multi, name)[col],
                expected,
            )


def test_no_nan_or_inf_for_boundary_only_case() -> None:
    state = _make_state(1, length_lim=True)
    tke = np.full(2, 3.0e-4)
    h = make_equidistant_h(1, 1.0)
    NN = np.full(2, 1.0e-4)

    workspace = _run_step(state, 1, tke=tke, h=h, NN=NN, depth=1.0, z0b=0.2, z0s=0.3)

    assert np.isfinite(workspace.L[0]).all()
    assert np.isfinite(workspace.eps[0]).all()
