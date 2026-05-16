"""Tests for pygotm.turbulence.cmue_d_h15."""

from __future__ import annotations

import numpy as np

from pygotm.turbulence.cmue_d_h15 import CmueDH15Workspace, step_cmue_d_h15
from pygotm.turbulence.turbulence import (
    TurbulenceState,
    init_turbulence,
    post_init_turbulence,
)

_NLEV = 12
_SMALL = 1.0e-8
_SQRT2 = np.sqrt(2.0)
_VK = 0.41
_MY_A1 = 0.92
_MY_A2 = 0.74
_MY_B1 = 16.6
_MY_B2 = 10.1
_MY_C1 = 0.08
_MY_C2 = 0.7
_MY_C3 = 0.2
_GHMIN = -0.28
_GHOFF = 0.003
_GVOFF = 0.006
_SXMAX = 2.12
_SHN0 = _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
_SHNH = -9.0 * _MY_A1 * _MY_A2 * (_MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1))
_SHNS = 9.0 * _MY_A1 * _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1) * (2.0 * _MY_A1 + _MY_A2)
_SHNV = (
    9.0
    * _MY_A1
    * _MY_A2
    * (
        _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1 - 3.0 * _MY_C1)
        - 2.0 * _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1 + 3.0 * _MY_C1)
    )
)
_SHDAH = -9.0 * _MY_A1 * _MY_A2
_SHDAV = -36.0 * _MY_A1 * _MY_A1
_SHDBH = -3.0 * _MY_A2 * (6.0 * _MY_A1 + _MY_B2 * (1.0 - _MY_C3))
_SHDV = -9.0 * _MY_A2 * _MY_A2 * (1.0 - _MY_C2)
_SHDVH = -162.0 * _MY_A1 * _MY_A1 * _MY_A2 * (2.0 * _MY_A1 + (2.0 - _MY_C2) * _MY_A2)
_SHDVV = 324.0 * _MY_A1 * _MY_A1 * _MY_A2 * _MY_A2 * (1.0 - _MY_C2)
_SSN0 = _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
_SSDH = -9.0 * _MY_A1 * _MY_A2
_SSDV = -9.0 * _MY_A1 * _MY_A1
_SMN0 = _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1 - 3.0 * _MY_C1)
_SMNHSH = 9.0 * _MY_A1 * (2.0 * _MY_A1 + _MY_A2 * (1.0 - _MY_C2))
_SMNSSS = 27.0 * _MY_A1 * _MY_A1
_SMDH = -9.0 * _MY_A1 * _MY_A2
_SMDV = -36.0 * _MY_A1 * _MY_A1
_SCALE = 4.0 / (_MY_B1 * _MY_B1)


def _make_state(nlev: int = _NLEV, *, length_lim: bool = True) -> TurbulenceState:
    state = TurbulenceState()
    init_turbulence(state, length_lim=length_lim)
    post_init_turbulence(state, nlev)
    return state


def _reference_cmue_d_h15(
    state: TurbulenceState,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    SPF: np.ndarray,
    cmue1: np.ndarray,
    cmue2: np.ndarray,
    cmue3: np.ndarray,
    sq_var: np.ndarray,
    sl_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nlev = as_.size - 1
    out_1 = cmue1.copy()
    out_2 = cmue2.copy()
    out_3 = cmue3.copy()
    out_sq = sq_var.copy()
    out_sl = sl_var.copy()

    for i in range(1, nlev):
        gh = -_SCALE * an[i]
        gm = _SCALE * as_[i]
        gv = _SCALE * av[i]
        gs = _SCALE * aw[i]

        if not state.length_lim:
            tmp1 = 1.0
            tmp2 = _GHMIN / min(_GHMIN, gh)
            tmp1 = min(tmp1, tmp2)
            if tmp1 < 1.0:
                gh *= tmp1
                gv *= tmp1
                gs *= tmp1

        tmp0 = 2.0
        if gv > 0.0:
            tmp1 = (_SHDAH + _SHDBH) * gh + (_SHDAV + _SHDV) * gv
            tmp1 += (_SHDAH * _GHOFF + _SHDAV * _GVOFF) * (_SHDBH * gh) + (
                _SHDVH * _GHOFF + _SHDVV * _GVOFF
            ) * gv
            tmp1 += (_SHDAH * gh + _SHDAV * gv) * (_SHDBH * _GHOFF) + (
                _SHDVH * gh + _SHDVV * gv
            ) * _GVOFF
            tmp2 = (_SHDAH * gh + _SHDAV * gv) * (_SHDBH * gh) + (
                _SHDVH * gh + _SHDVV * gv
            ) * gv
            tmp4 = (
                1.0
                + (_SHDAH + _SHDBH) * _GHOFF
                + (_SHDAV + _SHDV) * _GVOFF
                + (_SHDAH * _GHOFF + _SHDAV * _GVOFF) * (_SHDBH * _GHOFF)
                + (_SHDVH * _GHOFF + _SHDVV * _GVOFF) * _GVOFF
            )
            tmp3 = tmp1 * tmp1 - 4.0 * tmp2 * tmp4
            if tmp3 >= 0.0 and tmp2 < 0.0:
                tmp3 = (-tmp1 + np.sqrt(tmp3)) / (2.0 * tmp2)
            elif tmp3 >= 0.0 and tmp3 > 0.0:
                tmp3 = (-tmp1 - np.sqrt(tmp3)) / (2.0 * tmp2)
            else:
                tmp3 = 2.0
            if 0.0 < tmp3 < 1.0:
                tmp0 = min(tmp0, tmp3)

        gv *= SPF[i]
        gs *= SPF[i] ** 2

        if gh > 0.0:
            tmp1 = 2.0 * (_SHDAH + _SHDBH) * gh + (_SHDAV + _SHDV) * gv
            tmp2 = (2.0 * _SHDAH * gh + _SHDAV * gv) * (2.0 * _SHDBH * gh) + (
                2.0 * _SHDVH * gh + _SHDVV * gv
            ) * gv
            tmp4 = 1.0
            tmp3 = tmp1 * tmp1 - 4.0 * tmp2 * tmp4
            if tmp3 >= 0.0 and tmp2 < 0.0:
                tmp3 = (-tmp1 + np.sqrt(tmp3)) / (2.0 * tmp2)
            elif tmp3 >= 0.0 and tmp3 > 0.0:
                tmp3 = (-tmp1 - np.sqrt(tmp3)) / (2.0 * tmp2)
            else:
                tmp3 = 2.0
            if 0.0 < tmp3 < 1.0:
                tmp0 = min(tmp0, tmp3)

        if 0.0 < tmp0 < 1.0:
            gh *= tmp0
            gm *= tmp0
            gv *= tmp0
            gs *= tmp0

        tmp1 = _SHN0 + _SHNH * gh + _SHNS * gs + _SHNV * gv
        if tmp1 < 0.0:
            sh = _SMALL
        else:
            tmp2 = (1.0 + _SHDAH * gh + _SHDAV * gv) * (1.0 + _SHDBH * gh) + (
                _SHDV + _SHDVH * gh + _SHDVV * gv
            ) * gv
            if tmp2 <= 0.0:
                sh = _SXMAX
            else:
                sh = min(max(_SMALL, tmp1 / tmp2), _SXMAX)

        tmp2 = 1.0 + _SSDH * gh + _SSDV * gv
        if tmp2 < 0.0:
            ss = _SXMAX
        else:
            ss = min(max(_SMALL, _SSN0 / tmp2), _SXMAX)

        tmp1 = _SMN0 + _SMNHSH * gh * sh + _SMNSSS * gs * ss
        if 0.0 <= tmp1 < _SMALL:
            gh += _SMALL
            gv += _SMALL
            tmp1 = _SMN0 + _SMNHSH * gh * sh + _SMNSSS * gs * ss
        elif -_SMALL < tmp1 < 0.0:
            gh -= _SMALL
            gv -= _SMALL
            tmp1 = _SMN0 + _SMNHSH * gh * sh + _SMNSSS * gs * ss

        if tmp1 < 0.0:
            sm = _SMALL
        else:
            tmp2 = 1.0 + _SMDH * gh + _SMDV * gv
            if tmp2 <= 0.0:
                sm = _SXMAX
            else:
                sm = min(max(_SMALL, tmp1 / tmp2), _SXMAX)

        ss *= SPF[i]
        out_1[i] = _SQRT2 * sm
        out_2[i] = _SQRT2 * sh
        out_3[i] = _SQRT2 * ss
        out_sq[i] = np.sqrt(state.sq**2 + (_VK * sh) ** 2)
        out_sl[i] = np.sqrt(state.sl**2 + (_VK * sh) ** 2)

    return out_1, out_2, out_3, out_sq, out_sl


def _run_step_cmue_d_h15(
    state: TurbulenceState,
    nlev: int,
    *,
    as_: np.ndarray,
    an: np.ndarray,
    av: np.ndarray,
    aw: np.ndarray,
    SPF: np.ndarray,
    cmue1: np.ndarray | None = None,
    cmue2: np.ndarray | None = None,
    cmue3: np.ndarray | None = None,
    sq_var: np.ndarray | None = None,
    sl_var: np.ndarray | None = None,
    n_cols: int = 1,
) -> CmueDH15Workspace:
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.cmue3 is not None
    assert state.sq_var is not None
    assert state.sl_var is not None

    ws = CmueDH15Workspace(nlev, n_cols=n_cols)
    for col in range(n_cols):
        ws.as_[col] = as_
        ws.an[col] = an
        ws.av[col] = av
        ws.aw[col] = aw
        ws.SPF[col] = SPF
        ws.cmue1[col] = cmue1 if cmue1 is not None else state.cmue1
        ws.cmue2[col] = cmue2 if cmue2 is not None else state.cmue2
        ws.cmue3[col] = cmue3 if cmue3 is not None else state.cmue3
        ws.sq_var[col] = sq_var if sq_var is not None else state.sq_var
        ws.sl_var[col] = sl_var if sl_var is not None else state.sl_var

    step_cmue_d_h15(
        n_cols,
        nlev,
        int(state.length_lim),
        state.sq,
        state.sl,
        ws.as_,
        ws.an,
        ws.av,
        ws.aw,
        ws.SPF,
        ws.cmue1,
        ws.cmue2,
        ws.cmue3,
        ws.sq_var,
        ws.sl_var,
    )

    state.cmue1[:] = ws.cmue1[0]
    state.cmue2[:] = ws.cmue2[0]
    state.cmue3[:] = ws.cmue3[0]
    state.sq_var[:] = ws.sq_var[0]
    state.sl_var[:] = ws.sl_var[0]
    return ws


def test_import() -> None:
    assert callable(step_cmue_d_h15)


def test_workspace_instantiates() -> None:
    workspace = CmueDH15Workspace(_NLEV, n_cols=2)
    assert workspace.cmue3.shape == (2, _NLEV + 1)


def test_formula_matches_reference_and_preserves_boundaries() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=False)

    as_ = np.linspace(0.4, 1.5, nlev + 1)
    an = np.linspace(-0.08, 0.25, nlev + 1)
    av = np.linspace(0.0, 0.9, nlev + 1)
    aw = np.linspace(0.1, 0.7, nlev + 1)
    SPF = np.linspace(0.2, 1.0, nlev + 1)
    sentinel1 = np.full(nlev + 1, -1.0)
    sentinel2 = np.full(nlev + 1, -2.0)
    sentinel3 = np.full(nlev + 1, -3.0)
    sentinel4 = np.full(nlev + 1, -4.0)
    sentinel5 = np.full(nlev + 1, -5.0)

    _run_step_cmue_d_h15(
        state,
        nlev,
        as_=as_,
        an=an,
        av=av,
        aw=aw,
        SPF=SPF,
        cmue1=sentinel1,
        cmue2=sentinel2,
        cmue3=sentinel3,
        sq_var=sentinel4,
        sl_var=sentinel5,
    )
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.cmue3 is not None
    assert state.sq_var is not None
    assert state.sl_var is not None

    expected = _reference_cmue_d_h15(
        state,
        as_=as_,
        an=an,
        av=av,
        aw=aw,
        SPF=SPF,
        cmue1=sentinel1,
        cmue2=sentinel2,
        cmue3=sentinel3,
        sq_var=sentinel4,
        sl_var=sentinel5,
    )

    np.testing.assert_allclose(state.cmue1, expected[0], rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue2, expected[1], rtol=1.0e-12)
    np.testing.assert_allclose(state.cmue3, expected[2], rtol=1.0e-12)
    np.testing.assert_allclose(state.sq_var, expected[3], rtol=1.0e-12)
    np.testing.assert_allclose(state.sl_var, expected[4], rtol=1.0e-12)


def test_zero_surface_proximity_forces_zero_cmue3() -> None:
    nlev = _NLEV
    state = _make_state(nlev)
    as_ = np.linspace(0.4, 1.0, nlev + 1)
    an = np.linspace(-0.05, 0.15, nlev + 1)
    av = np.linspace(0.0, 0.5, nlev + 1)
    aw = np.linspace(0.2, 0.8, nlev + 1)
    SPF = np.zeros(nlev + 1, dtype=np.float64)

    _run_step_cmue_d_h15(state, nlev, as_=as_, an=an, av=av, aw=aw, SPF=SPF)

    assert state.cmue3 is not None
    np.testing.assert_allclose(state.cmue3[1:nlev], 0.0, atol=1.0e-15)


def test_multicolumn_parity_for_identical_columns() -> None:
    nlev = _NLEV
    kwargs = dict(
        as_=np.linspace(0.4, 1.2, nlev + 1),
        an=np.linspace(-0.06, 0.2, nlev + 1),
        av=np.linspace(0.0, 0.8, nlev + 1),
        aw=np.linspace(0.1, 0.6, nlev + 1),
        SPF=np.linspace(0.3, 1.0, nlev + 1),
    )

    single_state = _make_state(nlev, length_lim=False)
    single = _run_step_cmue_d_h15(
        single_state,
        nlev,
        as_=kwargs["as_"],
        an=kwargs["an"],
        av=kwargs["av"],
        aw=kwargs["aw"],
        SPF=kwargs["SPF"],
    )
    multi_state = _make_state(nlev, length_lim=False)
    multi = _run_step_cmue_d_h15(
        multi_state,
        nlev,
        as_=kwargs["as_"],
        an=kwargs["an"],
        av=kwargs["av"],
        aw=kwargs["aw"],
        SPF=kwargs["SPF"],
        n_cols=2,
    )

    for name in ("cmue1", "cmue2", "cmue3", "sq_var", "sl_var"):
        single_arr = getattr(single, name)[0]
        for col in range(2):
            np.testing.assert_allclose(
                getattr(multi, name)[col],
                single_arr,
                rtol=1.0e-12,
            )


def test_no_nan_or_inf_for_valid_inputs() -> None:
    nlev = _NLEV
    state = _make_state(nlev, length_lim=False)
    _run_step_cmue_d_h15(
        state,
        nlev,
        as_=np.linspace(0.4, 1.2, nlev + 1),
        an=np.linspace(-0.06, 0.2, nlev + 1),
        av=np.linspace(0.0, 0.8, nlev + 1),
        aw=np.linspace(0.1, 0.6, nlev + 1),
        SPF=np.linspace(0.3, 1.0, nlev + 1),
    )

    for array in (
        state.cmue1,
        state.cmue2,
        state.cmue3,
        state.sq_var,
        state.sl_var,
    ):
        assert array is not None
        assert np.isfinite(array).all()
