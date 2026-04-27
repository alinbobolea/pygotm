from __future__ import annotations

from typing import cast

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.turbulence.turbulence import TurbulenceState
from pygotm.util.density import DensityState

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


class ReadyDensityState(DensityState):
    alpha: FloatArray
    beta: FloatArray
    rho: FloatArray
    rho_p: FloatArray


class ReadyMeanflowState(MeanflowState):
    ga: FloatArray
    z: FloatArray
    zi: FloatArray
    h: FloatArray
    ho: FloatArray
    u: FloatArray
    v: FloatArray
    w: FloatArray
    uo: FloatArray
    vo: FloatArray
    T: FloatArray
    S: FloatArray
    Tp: FloatArray
    Sp: FloatArray
    Ti: FloatArray
    Tobs: FloatArray
    Sobs: FloatArray
    NN: FloatArray
    NNT: FloatArray
    NNS: FloatArray
    SS: FloatArray
    SSU: FloatArray
    SSV: FloatArray
    SSCSTK: FloatArray
    SSSTK: FloatArray
    buoy: FloatArray
    rad: FloatArray
    xP: FloatArray
    avh: FloatArray
    fric: FloatArray
    drag: FloatArray
    bioshade: FloatArray


class ReadyTurbulenceState(TurbulenceState):
    tke: FloatArray
    eps: FloatArray
    omega: FloatArray
    L: FloatArray
    tkeo: FloatArray
    kb: FloatArray
    epsb: FloatArray
    P: FloatArray
    B: FloatArray
    Pb: FloatArray
    Px: FloatArray
    PSTK: FloatArray
    num: FloatArray
    nuh: FloatArray
    nus: FloatArray
    nucl: FloatArray
    gamu: FloatArray
    gamv: FloatArray
    gamb: FloatArray
    gamh: FloatArray
    gams: FloatArray
    cmue1: FloatArray
    cmue2: FloatArray
    cmue3: FloatArray
    sq_var: FloatArray
    sl_var: FloatArray
    gam: FloatArray
    as_: FloatArray
    an: FloatArray
    at: FloatArray
    av: FloatArray
    aw: FloatArray
    SPF: FloatArray
    r: FloatArray
    Rig: FloatArray
    xRf: FloatArray
    uu: FloatArray
    vv: FloatArray
    ww: FloatArray


def require_density_state(state: DensityState) -> ReadyDensityState:
    assert state.alpha is not None
    assert state.beta is not None
    assert state.rho is not None
    assert state.rho_p is not None
    return cast(ReadyDensityState, state)


def require_meanflow_state(state: MeanflowState) -> ReadyMeanflowState:
    assert state.ga is not None
    assert state.z is not None
    assert state.zi is not None
    assert state.h is not None
    assert state.ho is not None
    assert state.u is not None
    assert state.v is not None
    assert state.w is not None
    assert state.uo is not None
    assert state.vo is not None
    assert state.T is not None
    assert state.S is not None
    assert state.Tp is not None
    assert state.Sp is not None
    assert state.Ti is not None
    assert state.Tobs is not None
    assert state.Sobs is not None
    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None
    assert state.SS is not None
    assert state.SSU is not None
    assert state.SSV is not None
    assert state.SSCSTK is not None
    assert state.SSSTK is not None
    assert state.buoy is not None
    assert state.rad is not None
    assert state.xP is not None
    assert state.avh is not None
    assert state.fric is not None
    assert state.drag is not None
    assert state.bioshade is not None
    return cast(ReadyMeanflowState, state)


def require_turbulence_state(state: TurbulenceState) -> ReadyTurbulenceState:
    assert state.tke is not None
    assert state.eps is not None
    assert state.omega is not None
    assert state.L is not None
    assert state.tkeo is not None
    assert state.kb is not None
    assert state.epsb is not None
    assert state.P is not None
    assert state.B is not None
    assert state.Pb is not None
    assert state.Px is not None
    assert state.PSTK is not None
    assert state.num is not None
    assert state.nuh is not None
    assert state.nus is not None
    assert state.nucl is not None
    assert state.gamu is not None
    assert state.gamv is not None
    assert state.gamb is not None
    assert state.gamh is not None
    assert state.gams is not None
    assert state.cmue1 is not None
    assert state.cmue2 is not None
    assert state.cmue3 is not None
    assert state.sq_var is not None
    assert state.sl_var is not None
    assert state.gam is not None
    assert state.as_ is not None
    assert state.an is not None
    assert state.at is not None
    assert state.av is not None
    assert state.aw is not None
    assert state.SPF is not None
    assert state.r is not None
    assert state.Rig is not None
    assert state.xRf is not None
    assert state.uu is not None
    assert state.vv is not None
    assert state.ww is not None
    return cast(ReadyTurbulenceState, state)
