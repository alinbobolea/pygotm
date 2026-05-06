"""Dense output buffers for compiled single-column GOTM integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pygotm.gotm.runtime_state import FloatArray

IntArray = NDArray[np.int64]

__all__ = ["IntArray", "RuntimeOutput", "allocate_runtime_output"]


def _nout(nt: int, output_every: int, enabled: bool, force_final: bool) -> int:
    if not enabled:
        return 0
    if nt < 0:
        msg = f"nt must be non-negative, got {nt}"
        raise ValueError(msg)
    if output_every < 1:
        msg = f"output_every must be at least 1, got {output_every}"
        raise ValueError(msg)

    periodic = nt // output_every
    final_extra = 1 if force_final and nt % output_every else 0
    return 1 + periodic + final_extra


def _output_profile(nout: int, nlev: int) -> FloatArray:
    return np.zeros((nout, nlev + 1), dtype=np.float64)


@dataclass(slots=True)
class RuntimeOutput:
    """Preallocated dense output buffers filled inside the compiled loop."""

    enabled: bool
    output_every: int
    force_final: bool
    nout: int
    output_step: IntArray

    time: FloatArray
    u: FloatArray
    v: FloatArray
    T: FloatArray
    S: FloatArray
    tke: FloatArray
    eps: FloatArray
    num: FloatArray
    nuh: FloatArray
    h: FloatArray
    xP: FloatArray
    fric: FloatArray
    drag: FloatArray
    avh: FloatArray
    bioshade: FloatArray
    ga: FloatArray
    SS: FloatArray
    P: FloatArray
    B: FloatArray
    Pb: FloatArray
    kb: FloatArray
    epsb: FloatArray
    L: FloatArray
    PSTK: FloatArray
    cmue1: FloatArray
    cmue2: FloatArray
    as_: FloatArray
    an: FloatArray
    at: FloatArray
    nus: FloatArray
    nucl: FloatArray
    z: FloatArray
    zi: FloatArray

    def validate(self, nlev: int) -> None:
        """Raise if output buffers do not match the runtime grid."""

        if self.output_every < 1:
            msg = f"output_every must be at least 1, got {self.output_every}"
            raise ValueError(msg)
        if self.nout != self.output_step.shape[0]:
            msg = "nout must equal output_step length"
            raise ValueError(msg)
        if self.time.shape != (self.nout,):
            msg = f"time must have shape ({self.nout},), got {self.time.shape}"
            raise ValueError(msg)
        expected_profile_shape = (self.nout, nlev + 1)
        for name in (
            "u",
            "v",
            "T",
            "S",
            "tke",
            "eps",
            "num",
            "nuh",
            "h",
            "xP",
            "fric",
            "drag",
            "avh",
            "bioshade",
            "ga",
            "SS",
            "P",
            "B",
            "Pb",
            "kb",
            "epsb",
            "L",
            "PSTK",
            "cmue1",
            "cmue2",
            "as_",
            "an",
            "at",
            "nus",
            "nucl",
            "z",
            "zi",
        ):
            array = getattr(self, name)
            if array.dtype != np.float64:
                msg = f"{name} must have dtype float64, got {array.dtype}"
                raise TypeError(msg)
            if array.shape != expected_profile_shape:
                msg = (
                    f"{name} must have shape {expected_profile_shape}, "
                    f"got {array.shape}"
                )
                raise ValueError(msg)
            if not array.flags.c_contiguous:
                msg = f"{name} must be C-contiguous"
                raise ValueError(msg)


def allocate_runtime_output(
    nlev: int,
    nt: int,
    *,
    enabled: bool = True,
    output_every: int = 1,
    force_final: bool = True,
) -> RuntimeOutput:
    """Allocate dense output arrays for initial, periodic, and final states."""

    if nlev < 1:
        msg = f"nlev must be at least 1, got {nlev}"
        raise ValueError(msg)
    nout = _nout(nt, output_every, enabled, force_final)
    output = RuntimeOutput(
        enabled=enabled,
        output_every=output_every,
        force_final=force_final,
        nout=nout,
        output_step=np.full(nout, -1, dtype=np.int64),
        time=np.full(nout, np.nan, dtype=np.float64),
        u=_output_profile(nout, nlev),
        v=_output_profile(nout, nlev),
        T=_output_profile(nout, nlev),
        S=_output_profile(nout, nlev),
        tke=_output_profile(nout, nlev),
        eps=_output_profile(nout, nlev),
        num=_output_profile(nout, nlev),
        nuh=_output_profile(nout, nlev),
        h=_output_profile(nout, nlev),
        xP=_output_profile(nout, nlev),
        fric=_output_profile(nout, nlev),
        drag=_output_profile(nout, nlev),
        avh=_output_profile(nout, nlev),
        bioshade=_output_profile(nout, nlev),
        ga=_output_profile(nout, nlev),
        SS=_output_profile(nout, nlev),
        P=_output_profile(nout, nlev),
        B=_output_profile(nout, nlev),
        Pb=_output_profile(nout, nlev),
        kb=_output_profile(nout, nlev),
        epsb=_output_profile(nout, nlev),
        L=_output_profile(nout, nlev),
        PSTK=_output_profile(nout, nlev),
        cmue1=_output_profile(nout, nlev),
        cmue2=_output_profile(nout, nlev),
        as_=_output_profile(nout, nlev),
        an=_output_profile(nout, nlev),
        at=_output_profile(nout, nlev),
        nus=_output_profile(nout, nlev),
        nucl=_output_profile(nout, nlev),
        z=_output_profile(nout, nlev),
        zi=_output_profile(nout, nlev),
    )
    output.validate(nlev)
    return output
