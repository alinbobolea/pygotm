r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The V-momentum equation\label{sec:vequation}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import numba
import numpy as np

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.util.adv_center import adv_center
from pygotm.util.diff_center import diff_center
from pygotm.util.util import Neumann as _NEUMANN
from pygotm.util.util import oneSided as _ONE_SIDED

__all__ = [
    "vequation",
    "step_vequation",
]

_ADV_MODE: int = 0
_POS_CONC: int = 0
_LONG: float = 1.0e15


@numba.njit(cache=True)
def _step_vequation(
    nlev: int,
    dt: float,
    cnpar: float,
    avmolu: float,
    gravity: float,
    ext_method: int,
    w_adv_active: int,
    w_adv_discr: int,
    plume_active: int,
    ty_val: float,
    dzetady_val: float,
    v: np.ndarray,
    vo: np.ndarray,
    u: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    drag: np.ndarray,
    num: np.ndarray,
    nucl: np.ndarray,
    dvsdz: np.ndarray,
    idpdy: np.ndarray,
    vprof: np.ndarray,
    tau_r: np.ndarray,
    avh: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    av: np.ndarray,
    bv: np.ndarray,
    cv: np.ndarray,
    dv: np.ndarray,
    rv: np.ndarray,
    qv: np.ndarray,
    adv_cv: np.ndarray,
) -> None:
    for k in range(nlev + 1):
        vo[k] = v[k]

    for k in range(nlev + 1):
        avh[k] = num[k] + avmolu

    dzy = 0.0
    if ext_method == 0:
        dzy = dzetady_val

    for k in range(1, nlev + 1):
        q_sour[k] = 0.0
        l_sour[k] = 0.0
        q_sour[k] += -gravity * dzy + idpdy[k]
        q_sour[k] += (nucl[k] * dvsdz[k] - nucl[k - 1] * dvsdz[k - 1]) / h[k]

    speed1 = math.sqrt(u[1] * u[1] + v[1] * v[1])
    l_sour[1] = -drag[1] / h[1] * speed1

    if plume_active == 1:
        speed_top = math.sqrt(u[nlev] * u[nlev] + v[nlev] * v[nlev])
        l_sour[nlev] = -drag[nlev] / h[nlev] * speed_top

    if w_adv_active == 1:
        adv_center(
            nlev, dt, h, h, w,
            _ONE_SIDED, _ONE_SIDED, 0.0, 0.0,
            w_adv_discr, _ADV_MODE, v, adv_cv,
        )

    diff_center(
        nlev, dt, cnpar, _POS_CONC, h,
        _NEUMANN, _NEUMANN, ty_val, 0.0,
        avh, l_sour, q_sour, tau_r, vprof, v,
        av, bv, cv, dv, rv, qv,
    )


@numba.njit(parallel=True, cache=True)
def step_vequation(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    avmolu: float,
    gravity: float,
    ext_method: int,
    w_adv_active: int,
    w_adv_discr: int,
    plume_active: int,
    ty: np.ndarray,
    dzetady: np.ndarray,
    v: np.ndarray,
    vo: np.ndarray,
    u: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    drag: np.ndarray,
    num: np.ndarray,
    nucl: np.ndarray,
    dvsdz: np.ndarray,
    idpdy: np.ndarray,
    vprof: np.ndarray,
    tau_r: np.ndarray,
    avh: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    av: np.ndarray,
    bv: np.ndarray,
    cv: np.ndarray,
    dv: np.ndarray,
    rv: np.ndarray,
    qv: np.ndarray,
    adv_cv: np.ndarray,
) -> None:
    """Batch variant: process batch_size columns in parallel."""
    for b in numba.prange(batch_size):
        _step_vequation(
            nlev, dt, cnpar, avmolu, gravity,
            ext_method, w_adv_active, w_adv_discr, plume_active,
            ty[b], dzetady[b],
            v[b], vo[b], u[b], h[b], w[b], drag[b],
            num[b], nucl[b], dvsdz[b], idpdy[b], vprof[b], tau_r[b],
            avh[b], q_sour[b], l_sour[b],
            av[b], bv[b], cv[b], dv[b], rv[b], qv[b], adv_cv[b],
        )


def vequation(
    state: MeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    ty: float,
    num: np.ndarray,
    nucl: np.ndarray,
    gamv: np.ndarray,
    ext_method: int = 0,
    dpdy: float = 0.0,
    idpdy: np.ndarray | None = None,
    dvsdz: np.ndarray | None = None,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
    vel_relax_tau: np.ndarray | None = None,
    vel_relax_ramp: float = _LONG,
    vprof: np.ndarray | None = None,
    plume_active: bool = False,
) -> None:
    """Advance the V-momentum equation for one column."""
    _ = gamv

    assert state.h is not None
    assert state.v is not None
    assert state.vo is not None
    assert state.u is not None
    assert state.w is not None
    assert state.drag is not None
    assert state.avh is not None

    n = nlev + 1

    if vel_relax_tau is None:
        v_relax_tau = np.full(n, _LONG, dtype=np.float64)
    else:
        v_relax_tau = vel_relax_tau.astype(np.float64, copy=True)

    if vel_relax_ramp < _LONG:
        state.runtimev += dt
        if state.runtimev < vel_relax_ramp:
            v_relax_tau *= vel_relax_ramp / (vel_relax_ramp - state.runtimev)

    _idpdy = idpdy if idpdy is not None else np.zeros(n, dtype=np.float64)
    _dvsdz = dvsdz if dvsdz is not None else np.zeros(n, dtype=np.float64)
    _vprof = vprof if vprof is not None else np.zeros(n, dtype=np.float64)

    q_sour = np.zeros(n, dtype=np.float64)
    l_sour = np.zeros(n, dtype=np.float64)
    av = np.zeros(n, dtype=np.float64)
    bv = np.zeros(n, dtype=np.float64)
    cv = np.zeros(n, dtype=np.float64)
    dv = np.zeros(n, dtype=np.float64)
    rv = np.zeros(n, dtype=np.float64)
    qv = np.zeros(n, dtype=np.float64)
    adv_cv = np.zeros(n, dtype=np.float64)

    _step_vequation(
        nlev, dt, cnpar, state.avmolu, state.gravity,
        ext_method, int(w_adv_active), w_adv_discr, int(plume_active),
        ty, dpdy,
        state.v, state.vo, state.u, state.h, state.w, state.drag,
        num, nucl, _dvsdz, _idpdy, _vprof, v_relax_tau,
        state.avh, q_sour, l_sour, av, bv, cv, dv, rv, qv, adv_cv,
    )
