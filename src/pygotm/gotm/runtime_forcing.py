"""Preprocessed forcing arrays for compiled single-column GOTM integration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pygotm.gotm.runtime_state import FloatArray

IntArray = NDArray[np.int64]

__all__ = ["IntArray", "RuntimeForcing", "allocate_runtime_forcing"]

_SCALAR_SERIES = (
    "time",
    "secondsofday",
    "zeta",
    "dpdx",
    "dpdy",
    "h_press",
    "tx",
    "ty",
    "heat",
    "swr",
    "airp",
    "airt",
    "hum",
    "cloud",
    "u10",
    "v10",
    "precip",
    "longwave",
    "sst_obs",
    "sss_obs",
    "w_adv",
    "w_height",
    "us0",
    "vs0",
    "ds",
    "light_A",
    "light_g1",
    "light_g2",
)

_PROFILE_SERIES = (
    "Tobs",
    "Sobs",
    "Tprof",
    "Sprof",
    "epsprof",
    "uprof",
    "vprof",
    "dtdx",
    "dtdy",
    "dsdx",
    "dsdy",
    "us",
    "vs",
    "dusdz",
    "dvsdz",
)


def _scalar_series(nt: int) -> FloatArray:
    return np.zeros(nt + 1, dtype=np.float64)


def _profile_series(nt: int, nlev: int) -> FloatArray:
    return np.zeros((nt + 1, nlev + 1), dtype=np.float64)


@dataclass(slots=True)
class RuntimeForcing:
    """Dense forcing and observation inputs prepared before Numba integration."""

    nlev: int
    nt: int
    yearday: IntArray

    time: FloatArray
    secondsofday: FloatArray
    zeta: FloatArray
    dpdx: FloatArray
    dpdy: FloatArray
    h_press: FloatArray
    tx: FloatArray
    ty: FloatArray
    heat: FloatArray
    swr: FloatArray
    airp: FloatArray
    airt: FloatArray
    hum: FloatArray
    cloud: FloatArray
    u10: FloatArray
    v10: FloatArray
    precip: FloatArray
    longwave: FloatArray
    sst_obs: FloatArray
    sss_obs: FloatArray
    w_adv: FloatArray
    w_height: FloatArray
    us0: FloatArray
    vs0: FloatArray
    ds: FloatArray
    light_A: FloatArray
    light_g1: FloatArray
    light_g2: FloatArray

    Tobs: FloatArray
    Sobs: FloatArray
    Tprof: FloatArray
    Sprof: FloatArray
    epsprof: FloatArray
    uprof: FloatArray
    vprof: FloatArray
    dtdx: FloatArray
    dtdy: FloatArray
    dsdx: FloatArray
    dsdy: FloatArray
    us: FloatArray
    vs: FloatArray
    dusdz: FloatArray
    dvsdz: FloatArray

    def iter_scalar_series(self) -> Iterator[tuple[str, FloatArray]]:
        """Yield scalar forcing series in stable declaration order."""

        for name in _SCALAR_SERIES:
            yield name, getattr(self, name)

    def iter_profile_series(self) -> Iterator[tuple[str, FloatArray]]:
        """Yield profile forcing series in stable declaration order."""

        for name in _PROFILE_SERIES:
            yield name, getattr(self, name)

    def validate(self) -> None:
        """Raise if forcing arrays violate compiled-runner assumptions."""

        if self.nlev < 1:
            msg = f"nlev must be at least 1, got {self.nlev}"
            raise ValueError(msg)
        if self.nt < 0:
            msg = f"nt must be non-negative, got {self.nt}"
            raise ValueError(msg)
        if self.yearday.dtype != np.int64:
            msg = f"yearday must have dtype int64, got {self.yearday.dtype}"
            raise TypeError(msg)
        if self.yearday.shape != (self.nt + 1,):
            msg = f"yearday must have shape ({self.nt + 1},), got {self.yearday.shape}"
            raise ValueError(msg)
        if not self.yearday.flags.c_contiguous:
            msg = "yearday must be C-contiguous"
            raise ValueError(msg)

        scalar_shape = (self.nt + 1,)
        for name, array in self.iter_scalar_series():
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != scalar_shape:
                msg = f"{name} must have shape {scalar_shape}, got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)

        profile_shape = (self.nt + 1, self.nlev + 1)
        for name, array in self.iter_profile_series():
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != profile_shape:
                msg = f"{name} must have shape {profile_shape}, got {array.shape}"
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)


def allocate_runtime_forcing(nlev: int, nt: int) -> RuntimeForcing:
    """Allocate dense runtime forcing arrays for steps 0:nt."""

    forcing = RuntimeForcing(
        nlev=nlev,
        nt=nt,
        yearday=np.zeros(nt + 1, dtype=np.int64),
        time=_scalar_series(nt),
        secondsofday=_scalar_series(nt),
        zeta=_scalar_series(nt),
        dpdx=_scalar_series(nt),
        dpdy=_scalar_series(nt),
        h_press=_scalar_series(nt),
        tx=_scalar_series(nt),
        ty=_scalar_series(nt),
        heat=_scalar_series(nt),
        swr=_scalar_series(nt),
        airp=_scalar_series(nt),
        airt=_scalar_series(nt),
        hum=_scalar_series(nt),
        cloud=_scalar_series(nt),
        u10=_scalar_series(nt),
        v10=_scalar_series(nt),
        precip=_scalar_series(nt),
        longwave=_scalar_series(nt),
        sst_obs=np.full(nt + 1, np.nan, dtype=np.float64),
        sss_obs=np.full(nt + 1, np.nan, dtype=np.float64),
        w_adv=_scalar_series(nt),
        w_height=_scalar_series(nt),
        us0=_scalar_series(nt),
        vs0=_scalar_series(nt),
        ds=_scalar_series(nt),
        light_A=_scalar_series(nt),
        light_g1=_scalar_series(nt),
        light_g2=_scalar_series(nt),
        Tobs=_profile_series(nt, nlev),
        Sobs=_profile_series(nt, nlev),
        Tprof=_profile_series(nt, nlev),
        Sprof=_profile_series(nt, nlev),
        epsprof=_profile_series(nt, nlev),
        uprof=_profile_series(nt, nlev),
        vprof=_profile_series(nt, nlev),
        dtdx=_profile_series(nt, nlev),
        dtdy=_profile_series(nt, nlev),
        dsdx=_profile_series(nt, nlev),
        dsdy=_profile_series(nt, nlev),
        us=_profile_series(nt, nlev),
        vs=_profile_series(nt, nlev),
        dusdz=_profile_series(nt, nlev),
        dvsdz=_profile_series(nt, nlev),
    )
    forcing.validate()
    return forcing
