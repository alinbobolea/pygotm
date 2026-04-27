"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: diagnostics --- additional diagnostics
!
! !INTERFACE:
!   module diagnostics
!
! !DESCRIPTION:
!  This module calculates different diagnostics. It is very easy to extend
!  the number of diagnostics calculated - and have those newly defined
!  values saved in a file.
!
! !USES:
!   IMPLICIT NONE
!   private
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_diagnostics, do_diagnostics, clean_diagnostics
!
! !PUBLIC DATA MEMBERS:
!   REALTYPE, public                    :: ekin,epot,eturb
!   REALTYPE                            :: epot0
!   REALTYPE, public, allocatable       :: taux(:),tauy(:)
!   integer, public                     :: mld_method=2
!   REALTYPE, public                    :: mld_surf,mld_bott
!   REALTYPE                            :: diff_k = 1e-05
!   REALTYPE                            :: Ri_crit = 0.5
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding, Jorn Bruggeman and Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DiagnosticsState",
    "clean_diagnostics",
    "do_diagnostics",
    "init_diagnostics",
]


@dataclass
class DiagnosticsState:
    """Diagnostic and integrated quantities from ``diagnostics.F90``."""

    ekin: float = 0.0
    epot: float = 0.0
    eturb: float = 0.0
    epot0: float = 0.0
    taux: np.ndarray | None = None
    tauy: np.ndarray | None = None
    Rig: np.ndarray | None = None
    mld_method: int = 2
    mld_surf: float = 0.0
    mld_bott: float = 0.0
    diff_k: float = 1.0e-5
    Ri_crit: float = 0.5


def init_diagnostics(state: DiagnosticsState, nlev: int) -> None:
    """Allocate diagnostic work arrays."""

    state.epot0 = state.epot
    state.taux = np.zeros(nlev + 1, dtype=np.float64)
    state.tauy = np.zeros(nlev + 1, dtype=np.float64)
    state.Rig = np.zeros(nlev + 1, dtype=np.float64)


def do_diagnostics(
    state: DiagnosticsState,
    nlev: int,
    *,
    tx: float,
    ty: float,
    drag: np.ndarray,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    buoy: np.ndarray,
    tke: np.ndarray,
    num: np.ndarray,
    nucl: np.ndarray,
    dusdz: np.ndarray | None = None,
    dvsdz: np.ndarray | None = None,
    turb_method: int = 0,
    rho0: float = 1027.0,
) -> None:
    """Compute mixed-layer depth, stresses, and integrated energies."""

    if state.taux is None or state.tauy is None or state.Rig is None:
        msg = "call init_diagnostics before do_diagnostics"
        raise RuntimeError(msg)

    dus = np.zeros(nlev + 1, dtype=np.float64) if dusdz is None else dusdz
    dvs = np.zeros(nlev + 1, dtype=np.float64) if dvsdz is None else dvsdz
    state.Rig[:] = NN / (SS + 1.0e-10)

    if state.mld_method == 1:
        if turb_method != 100:
            state.mld_surf = 0.0
            for i in range(nlev, 0, -1):
                if tke[i] < state.diff_k:
                    break
                state.mld_surf += h[i]
            state.mld_bott = 0.0
            for i in range(1, nlev + 1):
                if tke[i] < state.diff_k:
                    break
                state.mld_bott += h[i]
        else:
            state.mld_surf = 0.0
            state.mld_bott = 0.0
    elif state.mld_method == 2:
        state.mld_surf = h[nlev]
        state.mld_bott = 0.0
        for i in range(nlev - 1, 0, -1):
            if state.Rig[i] > state.Ri_crit:
                break
            state.mld_surf += h[i]
    elif state.mld_method == 3:
        index = int(np.argmax(NN[1 : nlev + 1])) + 1
        state.mld_surf = float(np.sum(h[index : nlev + 1]))
        state.mld_bott = 0.0

    state.taux[nlev] = -tx
    state.tauy[nlev] = -ty
    for i in range(nlev - 1, 0, -1):
        spacing = 0.5 * (h[i + 1] + h[i])
        state.taux[i] = -num[i] * (u[i + 1] - u[i]) / spacing - nucl[i] * dus[i]
        state.tauy[i] = -num[i] * (v[i + 1] - v[i]) / spacing - nucl[i] * dvs[i]
    speed = math.hypot(u[1], v[1])
    state.taux[0] = -drag[1] * u[1] * speed
    state.tauy[0] = -drag[1] * v[1] * speed

    ekin = 0.0
    epot = 0.0
    eturb = 0.0
    zloc = 0.0
    for i in range(1, nlev + 1):
        zloc -= 0.5 * h[i]
        ekin += 0.5 * h[i] * (u[i] ** 2 + v[i] ** 2)
        eturb += h[i] * (tke[i] + tke[i - 1])
        epot += h[i] * buoy[i] * zloc
        zloc -= 0.5 * h[i]

    state.ekin = ekin * rho0
    state.epot = epot * rho0 - state.epot0
    state.eturb = eturb * rho0


def clean_diagnostics(state: DiagnosticsState) -> None:
    """Release diagnostic work arrays."""

    state.taux = None
    state.tauy = None
    state.Rig = None
