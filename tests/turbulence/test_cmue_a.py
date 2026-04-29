"""Tests for pygotm.turbulence.cmue_a."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.cmue_a import CmueAWorkspace, step_cmue_a
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


def _reference_cmue_a(
    state: TurbulenceState,
    *,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    Pb: np.ndarray,
    epsb: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    r: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    gam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nlev = eps.size - 1
    out_cmue1 = cmue1.copy()
    out_cmue2 = cmue2.copy()
    out_gam = gam.copy()

    cm3_inv = 1.0 / state.cm0**3
    xd0 = 36.0
    xd1 = 84.0 * state.a5 * state.at3
    xd2 = 9.0 * (state.at2**2 - state.at1**2)
    xd3 = -12.0 * (state.a2**2 - 3.0 * state.a3**2)
    xd4 = 12.0 * state.a5 * state.at3 * (
        state.a2 * state.at1 - 3.0 * state.a3 * state.at2
    )
    xd5 = 12.0 * state.a5 * state.at3 * (state.a3**2 - state.a2**2)
    xd6 = 48.0 * state.a5**2 * state.at3**2
    xd7 = 3.0 * (state.a2**2 - 3.0 * state.a3**2) * (
        state.at1**2 - state.at2**2
    )

    xn0 = 36.0 * state.a1
    xn1 = -12.0 * state.a5 * state.at3 * (state.at1 + state.at2)
    xn2 = 8.0 * state.a5 * state.at3 * (6.0 * state.a1 - state.a2 - 3.0 * state.a3)
    xn3 = 9.0 * state.a1 * (state.at2**2 - state.at1**2)
    xn4 = 36.0 * state.a5 * state.at4 * (state.at1 + state.at2)
    xn5 = 24.0 * state.a5 * state.at4 * (state.a2 + 3.0 * state.a3)

    xt0 = 12.0 * state.at3
    xt1 = 12.0 * state.a5 * state.at3**2
    xt2 = 9.0 * state.a1 * state.at3 * (state.at1 - state.at2)
    xt3 = (
        6.0 * state.a1 * (state.a2 - 3.0 * state.a3)
        - 4.0 * (state.a2**2 - 3.0 * state.a3**2)
    ) * state.at3

    xg0 = 36.0 * state.at4
    xg1 = 36.0 * state.a5 * state.at3 * state.at4
    xg2 = -12.0 * state.at4 * (state.a2**2 - 3.0 * state.a3**2)

    for i in range(1, nlev):
        pe = (P[i] + Px[i] + B[i]) / eps[i]
        pbeb = Pb[i] / epsb[i]
        r_i = 1.0 / r[i]
        n_val = pe + 0.5 * state.cc1 - 1.0
        nt_val = 0.5 * (pe - 1.0) + state.ct1 + 0.5 * r_i * (pbeb - 1.0)
        nt_val = (pe - 1.0) + state.ct1

        d0 = xd0 * n_val**3 * nt_val**2
        d1 = xd1 * n_val**2 * nt_val
        d2 = xd2 * n_val**3 + xd3 * n_val * nt_val**2
        d3 = xd4 * n_val + xd5 * nt_val
        d4 = xd6 * n_val
        d5 = xd7 * n_val

        n0 = xn0 * n_val**2 * nt_val**2
        n1 = xn1 * n_val**2 + xn2 * n_val * nt_val
        n2 = xn3 * n_val**2
        n3 = xn4 * n_val**2 + xn5 * n_val * nt_val

        nt0 = xt0 * n_val**3 * nt_val
        nt1 = xt1 * n_val**2
        nt2 = xt2 * n_val**2 + xt3 * n_val * nt_val

        gam0 = xg0 * n_val**3 * nt_val
        gam1 = xg1 * n_val**2
        gam2 = xg2 * n_val * nt_val

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


def _run_step_cmue_a(
    state: TurbulenceState,
    nlev: int,
    *,
    eps: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    Pb: np.ndarray,
    epsb: np.ndarray,
    as_: np.ndarray,
    an: np.ndarray,
    at: np.ndarray,
    r: np.ndarray,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    gam: np.ndarray | None = None,
    n_cols: int = 1,
) -> CmueAWorkspace:
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.gam is not None

    ws = CmueAWorkspace(nlev, n_cols=n_cols)
    for col in range(n_cols):
        ws.eps[col] = eps
        ws.P[col] = P
        ws.B[col] = B
        ws.Px[col] = Px
        ws.Pb[col] = Pb
        ws.epsb[col] = epsb
        ws.as_[col] = as_
        ws.an[col] = an
        ws.at[col] = at
        ws.r[col] = r
        ws.cmue1[col] = cmue1 if cmue1 is not None else state.cmue1
        ws.cmue2[col] = cmue2 if cmue2 is not None else state.cmue2
        ws.gam[col] = gam if gam is not None else state.gam

    step_cmue_a(
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
        ws.eps,
        ws.P,
        ws.B,
        ws.Px,
        ws.Pb,
        ws.epsb,
        ws.as_,
        ws.an,
        ws.at,
        ws.r,
        ws.cmue1,
        ws.cmue2,
        ws.gam,
    )

    state.cmue1[:] = ws.cmue1[0]
    state.cmue2[:] = ws.cmue2[0]
    state.gam[:] = ws.gam[0]
    return ws


def test_import() -> None:
    assert callable(step_cmue_a)


def test_workspace_instantiates() -> None:
    workspace = CmueAWorkspace(_NLEV, n_cols=2)
    assert workspace.gam.shape == (2, _NLEV + 1)


def test_formula_matches_reference_and_preserves_boundaries() -> None:
    nlev = _NLEV
    state = _make_state(nlev)

    eps = np.linspace(3.0e-7, 9.0e-7, nlev + 1)
    P = np.linspace(2.0e-7, 4.0e-6, nlev + 1)
    B = np.linspace(-1.0e-6, 1.5e-6, nlev + 1)
    Px = np.linspace(0.0, 8.0e-7, nlev + 1)
    Pb = np.linspace(2.0e-9, 8.0e-8, nlev + 1)
    epsb = np.linspace(4.0e-9, 1.6e-7, nlev + 1)
    as_ = np.linspace(0.4, 1.8, nlev + 1)
    an = np.linspace(-0.3, 0.35, nlev + 1)
    at = np.linspace(0.04, 0.18, nlev + 1)
    r = np.linspace(0.5, 1.5, nlev + 1)
    sentinel1 = np.full(nlev + 1, -1.0)
    sentinel2 = np.full(nlev + 1, -2.0)
    sentinel3 = np.full(nlev + 1, -3.0)

    _run_step_cmue_a(
        state,
        nlev,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        Pb=Pb,
        epsb=epsb,
        as_=as_,
        an=an,
        at=at,
        r=r,
        cmue1=sentinel1,
        cmue2=sentinel2,
        gam=sentinel3,
    )
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.gam is not None

    expected_1, expected_2, expected_3 = _reference_cmue_a(
        state,
        eps=eps,
        P=P,
        B=B,
        Px=Px,
        Pb=Pb,
        epsb=epsb,
        as_=as_,
        an=an,
        at=at,
        r=r,
        cmue1=sentinel1,
        cmue2=sentinel2,
        gam=sentinel3,
    )

    np.testing.assert_allclose(state.cmue1, expected_1, rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue2, expected_2, rtol=1.0e-12)
    np.testing.assert_allclose(state.gam, expected_3, rtol=1.0e-12)


def test_weak_equilibrium_overwrite_makes_pbeb_and_r_inert() -> None:
    nlev = _NLEV
    common = dict(
        eps=np.linspace(3.0e-7, 9.0e-7, nlev + 1),
        P=np.linspace(1.0e-7, 3.0e-6, nlev + 1),
        B=np.linspace(-8.0e-7, 1.0e-6, nlev + 1),
        Px=np.linspace(0.0, 5.0e-7, nlev + 1),
        as_=np.linspace(0.3, 1.5, nlev + 1),
        an=np.linspace(-0.2, 0.25, nlev + 1),
        at=np.linspace(0.03, 0.12, nlev + 1),
    )

    baseline = _make_state(nlev)
    _run_step_cmue_a(
        baseline,
        nlev,
        eps=common["eps"],
        P=common["P"],
        B=common["B"],
        Px=common["Px"],
        Pb=np.linspace(2.0e-9, 5.0e-8, nlev + 1),
        epsb=np.linspace(4.0e-9, 9.0e-8, nlev + 1),
        as_=common["as_"],
        an=common["an"],
        at=common["at"],
        r=np.linspace(0.7, 1.1, nlev + 1),
    )

    perturbed = _make_state(nlev)
    _run_step_cmue_a(
        perturbed,
        nlev,
        eps=common["eps"],
        P=common["P"],
        B=common["B"],
        Px=common["Px"],
        Pb=np.linspace(8.0e-8, 3.0e-7, nlev + 1),
        epsb=np.linspace(2.0e-9, 2.5e-8, nlev + 1),
        as_=common["as_"],
        an=common["an"],
        at=common["at"],
        r=np.linspace(1.8, 3.2, nlev + 1),
    )
    assert baseline.cmue1 is not None
    assert baseline.cmue2 is not None
    assert baseline.gam is not None
    assert perturbed.cmue1 is not None
    assert perturbed.cmue2 is not None
    assert perturbed.gam is not None

    np.testing.assert_allclose(perturbed.cmue1, baseline.cmue1, rtol=1.0e-12)
    np.testing.assert_allclose(perturbed.cmue2, baseline.cmue2, rtol=1.0e-12)
    np.testing.assert_allclose(perturbed.gam, baseline.gam, rtol=1.0e-12)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    kwargs = dict(
        eps=np.linspace(3.0e-7, 9.0e-7, nlev + 1),
        P=np.linspace(2.0e-7, 4.0e-6, nlev + 1),
        B=np.linspace(-1.0e-6, 1.5e-6, nlev + 1),
        Px=np.linspace(0.0, 8.0e-7, nlev + 1),
        Pb=np.linspace(2.0e-9, 8.0e-8, nlev + 1),
        epsb=np.linspace(4.0e-9, 1.6e-7, nlev + 1),
        as_=np.linspace(0.4, 1.8, nlev + 1),
        an=np.linspace(-0.3, 0.35, nlev + 1),
        at=np.linspace(0.04, 0.18, nlev + 1),
        r=np.linspace(0.5, 1.5, nlev + 1),
    )

    single_state = _make_state(nlev)
    single = _run_step_cmue_a(
        single_state,
        nlev,
        eps=kwargs["eps"],
        P=kwargs["P"],
        B=kwargs["B"],
        Px=kwargs["Px"],
        Pb=kwargs["Pb"],
        epsb=kwargs["epsb"],
        as_=kwargs["as_"],
        an=kwargs["an"],
        at=kwargs["at"],
        r=kwargs["r"],
    )
    multi_state = _make_state(nlev)
    multi = _run_step_cmue_a(
        multi_state,
        nlev,
        eps=kwargs["eps"],
        P=kwargs["P"],
        B=kwargs["B"],
        Px=kwargs["Px"],
        Pb=kwargs["Pb"],
        epsb=kwargs["epsb"],
        as_=kwargs["as_"],
        an=kwargs["an"],
        at=kwargs["at"],
        r=kwargs["r"],
        n_cols=2,
    )

    for name in ("cmue1", "cmue2", "gam"):
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
    _run_step_cmue_a(
        state,
        nlev,
        eps=np.linspace(3.0e-7, 9.0e-7, nlev + 1),
        P=np.linspace(2.0e-7, 4.0e-6, nlev + 1),
        B=np.linspace(-1.0e-6, 1.5e-6, nlev + 1),
        Px=np.linspace(0.0, 8.0e-7, nlev + 1),
        Pb=np.linspace(2.0e-9, 8.0e-8, nlev + 1),
        epsb=np.linspace(4.0e-9, 1.6e-7, nlev + 1),
        as_=np.linspace(0.4, 1.8, nlev + 1),
        an=np.linspace(-0.3, 0.35, nlev + 1),
        at=np.linspace(0.04, 0.18, nlev + 1),
        r=np.linspace(0.5, 1.5, nlev + 1),
    )
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.gam is not None

    assert np.isfinite(state.cmue1).all()
    assert np.isfinite(state.cmue2).all()
    assert np.isfinite(state.gam).all()
