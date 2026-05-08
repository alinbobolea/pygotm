"""Runtime state arrays for the compiled single-column GOTM runner."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pygotm.arrays import make_column_array

FloatArray = NDArray[np.float64]

__all__ = ["FloatArray", "RuntimeState", "allocate_runtime_state"]

_PROFILE_FIELDS = (
    "ga",
    "z",
    "zi",
    "h",
    "ho",
    "u",
    "uo",
    "v",
    "vo",
    "w",
    "T",
    "S",
    "Tp",
    "Sp",
    "Ti",
    "Tobs",
    "Sobs",
    "NN",
    "NNT",
    "NNS",
    "SS",
    "SSU",
    "SSV",
    "SSCSTK",
    "SSSTK",
    "buoy",
    "alpha",
    "beta",
    "rho_p",
    "rho",
    "rad",
    "xP",
    "avh",
    "fric",
    "drag",
    "bioshade",
    "tke",
    "tkeo",
    "eps",
    "omega",
    "L",
    "kb",
    "epsb",
    "P",
    "B",
    "Pb",
    "Px",
    "PSTK",
    "num",
    "nuh",
    "nus",
    "nucl",
    "gamu",
    "gamv",
    "gamb",
    "gamh",
    "gams",
    "cmue1",
    "cmue2",
    "cmue3",
    "sq_var",
    "sl_var",
    "gam",
    "as_",
    "an",
    "at",
    "av",
    "aw",
    "SPF",
    "r",
    "Rig",
    "xRf",
    "uu",
    "vv",
    "ww",
)

_SCALAR_FIELDS = (
    "z0b",
    "z0s",
    "za",
    "u_taub",
    "u_taubo",
    "u_taus",
    "taub",
    "tx",
    "ty",
)


def _new_profile(nlev: int) -> FloatArray:
    return make_column_array(nlev)


def _validate_nlev(nlev: int) -> None:
    if nlev < 1:
        msg = f"nlev must be at least 1, got {nlev}"
        raise ValueError(msg)


@dataclass(slots=True)
class RuntimeState:
    """Flat 1D state arrays owned by the compiled single-column runtime."""

    nlev: int

    # Meanflow arrays.
    ga: FloatArray
    z: FloatArray
    zi: FloatArray
    h: FloatArray
    ho: FloatArray
    u: FloatArray
    uo: FloatArray
    v: FloatArray
    vo: FloatArray
    w: FloatArray
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
    alpha: FloatArray
    beta: FloatArray
    rho_p: FloatArray
    rho: FloatArray
    rad: FloatArray
    xP: FloatArray
    avh: FloatArray
    fric: FloatArray
    drag: FloatArray
    bioshade: FloatArray

    # Turbulence arrays.
    tke: FloatArray
    tkeo: FloatArray
    eps: FloatArray
    omega: FloatArray
    L: FloatArray
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

    # Mutable scalar runtime values stored as length-1 arrays for Numba.
    z0b: FloatArray
    z0s: FloatArray
    za: FloatArray
    u_taub: FloatArray
    u_taubo: FloatArray
    u_taus: FloatArray
    taub: FloatArray
    tx: FloatArray
    ty: FloatArray

    def iter_profile_arrays(self) -> Iterator[tuple[str, FloatArray]]:
        """Yield every physical profile array in stable declaration order."""

        for name in _PROFILE_FIELDS:
            yield name, getattr(self, name)

    def iter_scalar_arrays(self) -> Iterator[tuple[str, FloatArray]]:
        """Yield mutable scalar arrays in stable declaration order."""

        for name in _SCALAR_FIELDS:
            yield name, getattr(self, name)

    def validate(self) -> None:
        """Raise if any runtime profile violates compiled-runner assumptions."""

        _validate_nlev(self.nlev)
        expected_shape = (self.nlev + 1,)
        for name, array in self.iter_profile_arrays():
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != expected_shape:
                msg = f"{name} must have shape {expected_shape}, got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)
        for name, array in self.iter_scalar_arrays():
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != (1,):
                msg = f"{name} must have shape (1,), got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)


def allocate_runtime_state(nlev: int) -> RuntimeState:
    """Allocate zero-filled 1D state arrays with GOTM 0:nlev indexing."""

    _validate_nlev(nlev)
    state = RuntimeState(
        nlev=nlev,
        ga=_new_profile(nlev),
        z=_new_profile(nlev),
        zi=_new_profile(nlev),
        h=_new_profile(nlev),
        ho=_new_profile(nlev),
        u=_new_profile(nlev),
        uo=_new_profile(nlev),
        v=_new_profile(nlev),
        vo=_new_profile(nlev),
        w=_new_profile(nlev),
        T=_new_profile(nlev),
        S=_new_profile(nlev),
        Tp=_new_profile(nlev),
        Sp=_new_profile(nlev),
        Ti=_new_profile(nlev),
        Tobs=_new_profile(nlev),
        Sobs=_new_profile(nlev),
        NN=_new_profile(nlev),
        NNT=_new_profile(nlev),
        NNS=_new_profile(nlev),
        SS=_new_profile(nlev),
        SSU=_new_profile(nlev),
        SSV=_new_profile(nlev),
        SSCSTK=_new_profile(nlev),
        SSSTK=_new_profile(nlev),
        buoy=_new_profile(nlev),
        alpha=_new_profile(nlev),
        beta=_new_profile(nlev),
        rho_p=_new_profile(nlev),
        rho=_new_profile(nlev),
        rad=_new_profile(nlev),
        xP=_new_profile(nlev),
        avh=_new_profile(nlev),
        fric=_new_profile(nlev),
        drag=_new_profile(nlev),
        bioshade=_new_profile(nlev),
        tke=_new_profile(nlev),
        tkeo=_new_profile(nlev),
        eps=_new_profile(nlev),
        omega=_new_profile(nlev),
        L=_new_profile(nlev),
        kb=_new_profile(nlev),
        epsb=_new_profile(nlev),
        P=_new_profile(nlev),
        B=_new_profile(nlev),
        Pb=_new_profile(nlev),
        Px=_new_profile(nlev),
        PSTK=_new_profile(nlev),
        num=_new_profile(nlev),
        nuh=_new_profile(nlev),
        nus=_new_profile(nlev),
        nucl=_new_profile(nlev),
        gamu=_new_profile(nlev),
        gamv=_new_profile(nlev),
        gamb=_new_profile(nlev),
        gamh=_new_profile(nlev),
        gams=_new_profile(nlev),
        cmue1=_new_profile(nlev),
        cmue2=_new_profile(nlev),
        cmue3=_new_profile(nlev),
        sq_var=_new_profile(nlev),
        sl_var=_new_profile(nlev),
        gam=_new_profile(nlev),
        as_=_new_profile(nlev),
        an=_new_profile(nlev),
        at=_new_profile(nlev),
        av=_new_profile(nlev),
        aw=_new_profile(nlev),
        SPF=_new_profile(nlev),
        r=_new_profile(nlev),
        Rig=_new_profile(nlev),
        xRf=_new_profile(nlev),
        uu=_new_profile(nlev),
        vv=_new_profile(nlev),
        ww=_new_profile(nlev),
        z0b=np.zeros(1, dtype=np.float64),
        z0s=np.zeros(1, dtype=np.float64),
        za=np.zeros(1, dtype=np.float64),
        u_taub=np.zeros(1, dtype=np.float64),
        u_taubo=np.zeros(1, dtype=np.float64),
        u_taus=np.zeros(1, dtype=np.float64),
        taub=np.zeros(1, dtype=np.float64),
        tx=np.zeros(1, dtype=np.float64),
        ty=np.zeros(1, dtype=np.float64),
    )
    state.validate()
    return state
