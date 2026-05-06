"""
U-momentum (east–west) equation.

Implements GOTM Section 3.2.5 — advances the depth-varying eastward velocity
:math:`U` by one timestep using a Crank–Nicolson scheme.

Governing equation
------------------

The horizontally-averaged, incompressible U-momentum equation reads (Eq. 12):

.. math::

   \\frac{\\partial U}{\\partial t} - f V =
     -g \\frac{\\partial \\zeta}{\\partial x}
     + \\frac{\\partial}{\\partial z}
       \\left[ (\\nu_t + \\nu) \\frac{\\partial U}{\\partial z}
               - \\tilde{\\Gamma}_U \\right]
     + \\int_z^\\eta \\frac{\\partial B}{\\partial x}\\,dz'
     - \\frac{1}{\\tau_R}(U - U_{\\mathrm{obs}})
     - C_f U \\sqrt{U^2 + V^2} \\comma

where :math:`f` is the Coriolis parameter (handled separately by
:mod:`pygotm.meanflow.coriolis`), :math:`\\nu_t` is the turbulent eddy
viscosity, :math:`\\nu` is the molecular viscosity, :math:`\\tilde{\\Gamma}_U`
is the counter-gradient momentum flux (Langmuir/Stokes correction),
:math:`B = -g(\\rho - \\rho_0)/\\rho_0` is the buoyancy, and :math:`C_f` is
a quadratic bottom drag coefficient.

The external (barotropic) pressure gradient from sea-surface slope is
:math:`-g \\partial\\zeta/\\partial x` (active when ``ext_method = 0``).
Internal (baroclinic) pressure gradients from horizontal density gradients
enter through ``idpdx``.

Numerics
--------

The Crank–Nicolson scheme (implicitness :math:`\\sigma`, set to 1 for fully
implicit) discretises the vertical diffusion term, producing an unconditionally
stable tridiagonal system solved by the Thomas algorithm at each time step
(see :mod:`pygotm.util.diff_center`).

Bottom drag is treated as a linearised source term
:math:`l_1 = -C_f \\sqrt{U_1^2 + V_1^2} / h_1` applied at the lowest layer.

When ``w_adv_active`` is True, vertical advection is applied using the
scheme given by ``w_adv_discr`` (upwind, Superbee, etc.) from
:mod:`pygotm.util.adv_center`.

Author (original Fortran): Lars Umlauf.
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
    "uequation",
    "step_uequation",
    "step_uequation_single",
]

_ADV_MODE: int = 0
_POS_CONC: int = 0
_LONG: float = 1.0e15


@numba.njit(cache=True)
def _step_uequation(
    nlev: int,
    dt: float,
    cnpar: float,
    avmolu: float,
    gravity: float,
    ext_method: int,
    w_adv_active: int,
    w_adv_discr: int,
    seagrass_active: int,
    plume_active: int,
    tx_val: float,
    dzetadx_val: float,
    u: np.ndarray,
    uo: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    drag: np.ndarray,
    num: np.ndarray,
    nucl: np.ndarray,
    dusdz: np.ndarray,
    idpdx: np.ndarray,
    uprof: np.ndarray,
    tau_r: np.ndarray,
    avh: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
    adv_cu: np.ndarray,
) -> None:
    for k in range(nlev + 1):
        uo[k] = u[k]

    for k in range(nlev + 1):
        avh[k] = num[k] + avmolu

    dzx = 0.0
    if ext_method == 0:
        dzx = dzetadx_val

    for k in range(1, nlev + 1):
        q_sour[k] = 0.0
        l_sour[k] = 0.0
        q_sour[k] += -gravity * dzx + idpdx[k]
        q_sour[k] += (nucl[k] * dusdz[k] - nucl[k - 1] * dusdz[k - 1]) / h[k]

    if seagrass_active == 1:
        for k in range(1, nlev + 1):
            speed = math.sqrt(u[k] * u[k] + v[k] * v[k])
            l_sour[k] = -drag[k] / h[k] * speed

    speed1 = math.sqrt(u[1] * u[1] + v[1] * v[1])
    l_sour[1] = -drag[1] / h[1] * speed1

    if plume_active == 1:
        speed_top = math.sqrt(u[nlev] * u[nlev] + v[nlev] * v[nlev])
        l_sour[nlev] = -drag[nlev] / h[nlev] * speed_top

    if w_adv_active == 1:
        adv_center(
            nlev,
            dt,
            h,
            h,
            w,
            _ONE_SIDED,
            _ONE_SIDED,
            0.0,
            0.0,
            w_adv_discr,
            _ADV_MODE,
            u,
            adv_cu,
        )

    diff_center(
        nlev,
        dt,
        cnpar,
        _POS_CONC,
        h,
        _NEUMANN,
        _NEUMANN,
        tx_val,
        0.0,
        avh,
        l_sour,
        q_sour,
        tau_r,
        uprof,
        u,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )


@numba.njit(parallel=True, cache=True)
def step_uequation(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    avmolu: float,
    gravity: float,
    ext_method: int,
    w_adv_active: int,
    w_adv_discr: int,
    seagrass_active: int,
    plume_active: int,
    tx: np.ndarray,
    dzetadx: np.ndarray,
    u: np.ndarray,
    uo: np.ndarray,
    v: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    drag: np.ndarray,
    num: np.ndarray,
    nucl: np.ndarray,
    dusdz: np.ndarray,
    idpdx: np.ndarray,
    uprof: np.ndarray,
    tau_r: np.ndarray,
    avh: np.ndarray,
    q_sour: np.ndarray,
    l_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
    adv_cu: np.ndarray,
) -> None:
    """Batch variant: process batch_size columns in parallel."""
    for b in numba.prange(batch_size):
        _step_uequation(
            nlev,
            dt,
            cnpar,
            avmolu,
            gravity,
            ext_method,
            w_adv_active,
            w_adv_discr,
            seagrass_active,
            plume_active,
            tx[b],
            dzetadx[b],
            u[b],
            uo[b],
            v[b],
            h[b],
            w[b],
            drag[b],
            num[b],
            nucl[b],
            dusdz[b],
            idpdx[b],
            uprof[b],
            tau_r[b],
            avh[b],
            q_sour[b],
            l_sour[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
            adv_cu[b],
        )


step_uequation_single = _step_uequation


def uequation(
    state: MeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    tx: float,
    num: np.ndarray,
    nucl: np.ndarray,
    gamu: np.ndarray,
    ext_method: int = 0,
    dpdx: float = 0.0,
    idpdx: np.ndarray | None = None,
    dusdz: np.ndarray | None = None,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
    vel_relax_tau: np.ndarray | None = None,
    vel_relax_ramp: float = _LONG,
    uprof: np.ndarray | None = None,
    plume_active: bool = False,
    seagrass_active: bool = False,
) -> None:
    """Advance the U-momentum equation for one column.

    Parameters
    ----------
    state:
        MeanflowState with h, u, uo, v, w, drag, avh, gravity, avmolu.
    nlev:
        Number of model layers.
    dt:
        Time step [s].
    cnpar:
        Crank–Nicolson implicitness (0=explicit, 1=fully implicit).
    tx:
        Surface wind stress in x [m² s⁻²] (Neumann BC at surface).
    num:
        Turbulent eddy viscosity [m² s⁻¹], shape (nlev+1,).
    nucl:
        Non-local eddy viscosity (Stokes/Langmuir correction) [m² s⁻¹],
        shape (nlev+1,).
    gamu:
        Counter-gradient momentum flux (not currently used; reserved).
    ext_method:
        External pressure treatment (0 = barotropic slope from dpdx).
    dpdx:
        Barotropic pressure gradient :math:`g\\partial\\zeta/\\partial x`
        [m s⁻²].
    idpdx:
        Baroclinic pressure gradient profile [m s⁻²], shape (nlev+1,).
    dusdz:
        Stokes drift shear in x [s⁻¹], shape (nlev+1,).
    """
    _ = gamu

    assert state.h is not None
    assert state.u is not None
    assert state.uo is not None
    assert state.v is not None
    assert state.w is not None
    assert state.drag is not None
    assert state.avh is not None

    n = nlev + 1

    if vel_relax_tau is None:
        u_relax_tau = np.full(n, _LONG, dtype=np.float64)
    else:
        u_relax_tau = vel_relax_tau.astype(np.float64, copy=True)

    if vel_relax_ramp < _LONG:
        state.runtimeu += dt
        if state.runtimeu < vel_relax_ramp:
            u_relax_tau *= vel_relax_ramp / (vel_relax_ramp - state.runtimeu)

    _idpdx = idpdx if idpdx is not None else np.zeros(n, dtype=np.float64)
    _dusdz = dusdz if dusdz is not None else np.zeros(n, dtype=np.float64)
    _uprof = uprof if uprof is not None else np.zeros(n, dtype=np.float64)

    q_sour = np.zeros(n, dtype=np.float64)
    l_sour = np.zeros(n, dtype=np.float64)
    au = np.zeros(n, dtype=np.float64)
    bu = np.zeros(n, dtype=np.float64)
    cu = np.zeros(n, dtype=np.float64)
    du = np.zeros(n, dtype=np.float64)
    ru = np.zeros(n, dtype=np.float64)
    qu = np.zeros(n, dtype=np.float64)
    adv_cu = np.zeros(n, dtype=np.float64)

    _step_uequation(
        nlev,
        dt,
        cnpar,
        state.avmolu,
        state.gravity,
        ext_method,
        int(w_adv_active),
        w_adv_discr,
        int(seagrass_active),
        int(plume_active),
        tx,
        dpdx,
        state.u,
        state.uo,
        state.v,
        state.h,
        state.w,
        state.drag,
        num,
        nucl,
        _dusdz,
        _idpdx,
        _uprof,
        u_relax_tau,
        state.avh,
        q_sour,
        l_sour,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
        adv_cu,
    )
