# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !MODULE: sediment --- suspended sediment dynamics
!
! !DESCRIPTION:
!  This subroutine computes the transport of sediment, given by its
!  concentration. Settling is advective and turbulent mixing is represented
!  by diffusion. The sinking speed is negative by definition.
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pygotm.util.adv_center import CONSERVATIVE, adv_center
from pygotm.util.diff_center import diff_center
from pygotm.util.lagrange import lagrange
from pygotm.util.util import Dirichlet, Neumann, flux, oneSided

__all__ = [
    "NoFlux",
    "SedimentState",
    "SmithMcLean",
    "do_sediment",
    "end_sediment",
    "init_sediment",
    "save_sediment",
    "sediment_eulerian",
    "sediment_lagrangian",
    "settling_velocity_zanke",
]

NoFlux: int = 1
SmithMcLean: int = 2
_LONG: float = 1.0e15


@dataclass
class SedimentState:
    """State for suspended sediment transport."""

    sedi_calc: bool = False
    sedi_eulerian: bool = True
    sedi_dens: bool = False
    rho_sed: float = 2650.0
    size: float = 62.5e-6
    init_conc: float = 0.0001
    adv_method: int = 1
    cnpar: float = 0.5
    sedi_method: int = NoFlux
    z0b_method: int = 1
    sedi_npar: int = 10000
    take_mean: bool = True
    C: np.ndarray | None = None
    wc: np.ndarray | None = None
    Cobs: np.ndarray | None = None
    Qsour: np.ndarray | None = None
    Lsour: np.ndarray | None = None
    RelaxT: np.ndarray | None = None
    zp: np.ndarray | None = None
    zi: np.ndarray | None = None
    DiffBcup: int = Neumann
    DiffBcdw: int = Neumann
    AdvBcup: int = flux
    AdvBcdw: int = flux
    DiffCup: float = 0.0
    DiffCdw: float = 0.0
    AdvCup: float = 0.0
    AdvCdw: float = 0.0
    ustarc: float = 0.0
    gs: float = 0.0
    za: float = 0.0
    au: np.ndarray | None = None
    bu: np.ndarray | None = None
    cu: np.ndarray | None = None
    du: np.ndarray | None = None
    ru: np.ndarray | None = None
    qu: np.ndarray | None = None
    adv_cu: np.ndarray | None = None
    _lagrange_count: int = 0
    _lagrange_set_c_zero: bool = True


def settling_velocity_zanke(
    size: float,
    gravity: float,
    rho_sed: float,
    rho0: float,
    *,
    avmolu: float = 1.3e-6,
) -> tuple[float, float]:
    """Return Zanke fall velocity and reduced gravity."""

    gs = gravity * (rho_sed - rho0) / rho0
    wc = (
        -10.0
        * avmolu
        / size
        * (np.sqrt(1.0 + (0.01 * gs * size**3) / avmolu / avmolu) - 1.0)
    )
    return float(wc), float(gs)


def init_sediment(
    state: SedimentState,
    nlev: int,
    gravity: float,
    rho0: float,
    *,
    depth: float = 1.0,
    h: np.ndarray | None = None,
    sedi_calc: bool | None = None,
    sedi_eulerian: bool | None = None,
    sedi_method: int | None = None,
) -> None:
    """Initialise sediment vectors and derived parameters."""

    if sedi_calc is not None:
        state.sedi_calc = bool(sedi_calc)
    if sedi_eulerian is not None:
        state.sedi_eulerian = bool(sedi_eulerian)
    if sedi_method is not None:
        state.sedi_method = int(sedi_method)

    if not state.sedi_calc:
        return

    state.C = np.full(nlev + 1, state.init_conc, dtype=np.float64)
    state.C[0] = 0.0
    state.wc = np.zeros(nlev + 1, dtype=np.float64)
    state.Cobs = np.zeros(nlev + 1, dtype=np.float64)
    state.Qsour = np.zeros(nlev + 1, dtype=np.float64)
    state.Lsour = np.zeros(nlev + 1, dtype=np.float64)
    state.RelaxT = np.full(nlev + 1, _LONG, dtype=np.float64)
    state.au = np.zeros(nlev + 1, dtype=np.float64)
    state.bu = np.zeros(nlev + 1, dtype=np.float64)
    state.cu = np.zeros(nlev + 1, dtype=np.float64)
    state.du = np.zeros(nlev + 1, dtype=np.float64)
    state.ru = np.zeros(nlev + 1, dtype=np.float64)
    state.qu = np.zeros(nlev + 1, dtype=np.float64)
    state.adv_cu = np.zeros(nlev + 1, dtype=np.float64)

    wc, state.gs = settling_velocity_zanke(
        state.size,
        gravity,
        state.rho_sed,
        rho0,
    )
    state.wc[:] = wc

    if state.sedi_method == SmithMcLean:
        dsize = state.size * (state.gs / (1.3e-6 * 1.3e-6)) ** (1.0 / 3.0)
        state.ustarc = -0.4 * wc if dsize > 10.0 else -4.0 / dsize * wc

    if not state.sedi_eulerian:
        state.zp = np.zeros(state.sedi_npar, dtype=np.float64)
        state.zi = np.zeros(state.sedi_npar, dtype=np.int64)
        if h is None:
            h = np.full(nlev + 1, depth / nlev, dtype=np.float64)
        zlev = np.zeros(nlev + 1, dtype=np.float64)
        zlev[0] = -depth
        for n in range(1, nlev + 1):
            zlev[n] = zlev[n - 1] + h[n]
        for n in range(state.sedi_npar):
            state.zp[n] = -depth + (n + 1) / float(state.sedi_npar + 1) * depth
            for i in range(1, nlev + 1):
                if zlev[i] > state.zp[n]:
                    state.zi[n] = i
                    break


def _configure_eulerian_boundary(
    state: SedimentState,
    h: np.ndarray,
    u_taub: float,
    kappa: float,
    z0b: float,
) -> None:
    if state.sedi_method == NoFlux:
        state.za = 0.0
        state.DiffBcup = Neumann
        state.DiffBcdw = Neumann
        state.DiffCup = 0.0
        state.DiffCdw = 0.0
        state.AdvBcup = flux
        state.AdvBcdw = flux
        state.AdvCup = 0.0
        state.AdvCdw = 0.0
        return

    if state.sedi_method != SmithMcLean:
        msg = f"invalid sediment method {state.sedi_method}"
        raise ValueError(msg)

    cbott = 0.0
    rouse = 0.0
    if u_taub >= state.ustarc and state.ustarc > 0.0:
        if state.z0b_method != 1:
            state.za = 26.3 / state.gs * (u_taub**2 - state.ustarc**2)
        else:
            state.za = 0.0
        cbott = 1.56e-3 * ((u_taub / state.ustarc) ** 2 - 1.0)
        assert state.wc is not None
        rouse = state.wc[1] / kappa / u_taub
    else:
        state.za = 0.0

    state.DiffBcup = Neumann
    state.DiffBcdw = Dirichlet
    state.DiffCup = 0.0
    state.DiffCdw = cbott * ((0.5 * h[1] + z0b) / z0b) ** rouse
    state.AdvBcup = flux
    state.AdvBcdw = oneSided
    state.AdvCup = 0.0
    state.AdvCdw = 0.0


def sediment_eulerian(
    state: SedimentState,
    nlev: int,
    dt: float,
    h: np.ndarray,
    num: np.ndarray,
    *,
    u_taub: float = 0.0,
    kappa: float = 0.4,
    z0b: float = 0.01,
    w_adv_method: int = 0,
    grid_method: int = 0,
) -> None:
    """Eulerian sediment transport update."""

    if w_adv_method != 0:
        raise ValueError("w_adv_method=0 is required for sediment")
    if grid_method == 3:
        raise ValueError("adaptive grids do not yet work with sediment")

    assert state.C is not None
    assert state.wc is not None
    assert state.Cobs is not None
    assert state.Qsour is not None
    assert state.Lsour is not None
    assert state.RelaxT is not None
    assert state.au is not None
    assert state.bu is not None
    assert state.cu is not None
    assert state.du is not None
    assert state.ru is not None
    assert state.qu is not None
    assert state.adv_cu is not None

    _configure_eulerian_boundary(state, h, u_taub, kappa, z0b)
    adv_center(
        nlev,
        dt,
        h,
        h,
        state.wc,
        state.AdvBcup,
        state.AdvBcdw,
        state.AdvCup,
        state.AdvCdw,
        state.adv_method,
        CONSERVATIVE,
        state.C,
        state.adv_cu,
    )
    diff_center(
        nlev,
        dt,
        state.cnpar,
        1,
        h,
        state.DiffBcup,
        state.DiffBcdw,
        state.DiffCup,
        state.DiffCdw,
        num,
        state.Lsour,
        state.Qsour,
        state.RelaxT,
        state.Cobs,
        state.C,
        state.au,
        state.bu,
        state.cu,
        state.du,
        state.ru,
        state.qu,
    )


def sediment_lagrangian(
    state: SedimentState,
    nlev: int,
    dt: float,
    zlev: np.ndarray,
    nuh: np.ndarray,
    h: np.ndarray,
    *,
    write_results: bool = False,
    rng: np.random.Generator | None = None,
) -> None:
    """Lagrangian sediment particle update."""

    assert state.C is not None
    assert state.wc is not None
    assert state.zp is not None
    assert state.zi is not None
    active = np.ones(state.sedi_npar, dtype=bool)
    lagrange(
        nlev,
        dt,
        zlev,
        nuh,
        state.wc[1],
        state.sedi_npar,
        active,
        state.zi,
        state.zp,
        rng=rng,
    )
    active[:] = True
    if write_results or state.take_mean:
        if state._lagrange_set_c_zero:
            state.C[:] = 0.0
            state._lagrange_set_c_zero = False
        for npart in range(state.sedi_npar):
            if active[npart]:
                layer = state.zi[npart]
                state.C[layer] += 1.0
                active[npart] = False
        state._lagrange_count = state._lagrange_count + 1 if state.take_mean else 1
        if write_results:
            depth = -zlev[0]
            for n in range(1, nlev + 1):
                state.C[n] = (
                    state.init_conc
                    * state.C[n]
                    / state.sedi_npar
                    * depth
                    / h[n]
                    / state._lagrange_count
                )
            state._lagrange_count = 0
            state._lagrange_set_c_zero = True


def do_sediment(
    state: SedimentState,
    nlev: int,
    dt: float,
    h: np.ndarray,
    num: np.ndarray,
    *,
    u_taub: float = 0.0,
    kappa: float = 0.4,
    z0b: float = 0.01,
) -> None:
    """Dispatch one sediment update."""

    if not state.sedi_calc:
        return
    if state.sedi_eulerian:
        sediment_eulerian(state, nlev, dt, h, num, u_taub=u_taub, kappa=kappa, z0b=z0b)


def save_sediment(state: SedimentState, path: str | Path | None = None) -> np.ndarray:
    """Return or optionally write the current sediment concentration."""

    assert state.C is not None
    values = state.C.copy()
    if path is not None:
        np.savetxt(path, values)
    return values


def end_sediment(state: SedimentState) -> None:
    """Release sediment arrays."""

    state.C = None
    state.wc = None
    state.Cobs = None
    state.Qsour = None
    state.Lsour = None
    state.RelaxT = None
    state.zp = None
    state.zi = None
    state.au = None
    state.bu = None
    state.cu = None
    state.du = None
    state.ru = None
    state.qu = None
    state.adv_cu = None
