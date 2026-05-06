# ruff: noqa: E501
"""
Vertical friction and boundary-layer drag.

Implements GOTM Section 3.2.9 — computes bottom friction velocity
:math:`u_{\\tau b}` from a logarithmic drag law and surface stress from
wind forcing, then applies them as Neumann boundary conditions for the
momentum equations.

Bottom roughness and friction velocity
--------------------------------------

The bottom roughness length combines hydraulically smooth, hydraulically rough,
and moveable-bed (sediment) contributions (Eq. 23):

.. math::

   z_{0b} = \\frac{0.1\\nu}{u_{\\tau b}} + 0.03 h_{0b} + z_a \\comma

where :math:`\\nu` is the molecular viscosity, :math:`h_{0b}` is the physical
roughness height, and :math:`z_a` is a moveable bed roughness.  When
:math:`\\nu \\le 0` the smooth-wall term is omitted.

The bottom friction velocity is (Eq. 24):

.. math::

   u_{\\tau b} = r \\sqrt{U_1^2 + V_1^2} \\comma

where :math:`U_1, V_1` are the velocities in the lowest model layer, and the
drag coefficient :math:`r` is (Eq. 25):

.. math::

   r = \\frac{\\kappa}{\\ln\\bigl((z_{0b} + h_1/2) / z_{0b}\\bigr)} \\comma

with :math:`h_1` the thickness of the lowest model layer and
:math:`\\kappa = 0.4` the von Kármán constant.
Because :math:`z_{0b}` itself depends on :math:`u_{\\tau b}` through
the smooth-wall term, the equations are iterated up to ``MaxItz0b``
times.

Surface roughness
-----------------

When Charnock's formula is activated (Eq. 26):

.. math::

   z_{0s} = \\frac{\\alpha_c\\, u_{\\tau s}^2}{g} \\comma

where :math:`\\alpha_c` is the Charnock coefficient (``charnock_val``) and
:math:`g` is gravitational acceleration.  Otherwise :math:`z_{0s}` is set
to the fixed minimum value ``z0s_min``.

When ``plume_type = 1`` the surface friction velocity is computed
analogously to the bottom, from the top-layer velocity.  Otherwise it is
derived from the prescribed wind-stress components :math:`(\\tau_x, \\tau_y)`:

.. math::

   u_{\\tau s} = (\\tau_x^2 + \\tau_y^2)^{1/4} \\point

Authors (original Fortran): Hans Burchard, Karsten Bolding.
"""

import math

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace
from pygotm.meanflow.meanflow import MeanflowState

__all__ = [
    "FrictionWorkspace",
    "KAPPA",
    "friction",
    "step_friction_batch",
    "step_friction_single",
]

KAPPA: float = 0.4
_RHO0: float = 1027.0


class FrictionWorkspace(ColumnWorkspace):
    """Batch friction workspace — profile arrays shape (batch_size, nlev+1),
    scalar arrays shape (batch_size,)."""

    def __init__(self, nlev: int, *, batch_size: int = 1) -> None:
        super().__init__(nlev, n_cols=batch_size)
        n = (batch_size, nlev + 1)
        self.h = np.zeros(n, dtype=np.float64)
        self.u = np.zeros(n, dtype=np.float64)
        self.v = np.zeros(n, dtype=np.float64)
        self.drag = np.zeros(n, dtype=np.float64)
        self.z0b = np.zeros(batch_size, dtype=np.float64)
        self.z0s = np.zeros(batch_size, dtype=np.float64)
        self.za = np.zeros(batch_size, dtype=np.float64)
        self.u_taub = np.zeros(batch_size, dtype=np.float64)
        self.u_taubo = np.zeros(batch_size, dtype=np.float64)
        self.u_taus = np.zeros(batch_size, dtype=np.float64)
        self.taub = np.zeros(batch_size, dtype=np.float64)
        self.tx = np.zeros(batch_size, dtype=np.float64)
        self.ty = np.zeros(batch_size, dtype=np.float64)


@numba.njit(cache=True)
def _step_friction_kernel(
    nlev: int,
    kappa: float,
    avmolu: float,
    rho0: float,
    gravity: float,
    h0b: float,
    z0s_min: float,
    charnock: int,
    charnock_val: float,
    calc_bottom_stress: int,
    MaxItz0b: int,
    plume_type: int,
    first: int,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    drag: np.ndarray,
    z0b: np.ndarray,
    z0s: np.ndarray,
    za: np.ndarray,
    u_taub: np.ndarray,
    u_taubo: np.ndarray,
    u_taus: np.ndarray,
    taub: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
) -> None:
    """Single-column friction kernel. Scalar outputs use length-1 arrays."""
    for k in range(nlev + 1):
        drag[k] = 0.0

    rr_b = 0.0
    rr_s = 0.0

    z0s_val = z0s_min
    if charnock == 1:
        z0s_val = charnock_val * u_taus[0] * u_taus[0] / gravity
        if z0s_val < z0s_min:
            z0s_val = z0s_min
    z0s[0] = z0s_val

    if calc_bottom_stress == 1:
        if first == 1:
            u_taub[0] = u_taubo[0]
        else:
            u_taubo[0] = u_taub[0]
        for _ in range(MaxItz0b):
            z0b_val = 0.0
            if avmolu <= 0.0:
                z0b_val = 0.03 * h0b + za[0]
            else:
                denom = avmolu if avmolu > u_taub[0] else u_taub[0]
                z0b_val = 0.1 * avmolu / denom + 0.03 * h0b + za[0]
            z0b[0] = z0b_val
            rr_b = kappa / math.log((z0b_val + h[1] / 2.0) / z0b_val)
            speed_b = math.sqrt(u[1] * u[1] + v[1] * v[1])
            u_taub[0] = rr_b * speed_b

    if plume_type == 1:
        rr_s = kappa / math.log((z0s_val + h[nlev] / 2.0) / z0s_val)

    taub[0] = u_taub[0] * u_taub[0] * rho0
    drag[1] += rr_b * rr_b

    if plume_type == 1:
        drag[nlev] += rr_s * rr_s
        speed_s = math.sqrt(u[nlev] * u[nlev] + v[nlev] * v[nlev])
        u_taus[0] = rr_s * speed_s
    else:
        tx_val = tx[0]
        ty_val = ty[0]
        u_taus[0] = (tx_val * tx_val + ty_val * ty_val) ** 0.25


step_friction_single = _step_friction_kernel


@numba.njit(parallel=True, cache=True)
def step_friction_batch(
    batch_size: int,
    nlev: int,
    kappa: float,
    avmolu: float,
    rho0: float,
    gravity: float,
    h0b: float,
    z0s_min: float,
    charnock: int,
    charnock_val: float,
    calc_bottom_stress: int,
    MaxItz0b: int,
    plume_type: int,
    first: int,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    drag: np.ndarray,
    z0b: np.ndarray,
    z0s: np.ndarray,
    za: np.ndarray,
    u_taub: np.ndarray,
    u_taubo: np.ndarray,
    u_taus: np.ndarray,
    taub: np.ndarray,
    tx: np.ndarray,
    ty: np.ndarray,
) -> None:
    """Batch variant: process batch_size columns in parallel."""
    for b in numba.prange(batch_size):
        _step_friction_kernel(
            nlev,
            kappa,
            avmolu,
            rho0,
            gravity,
            h0b,
            z0s_min,
            charnock,
            charnock_val,
            calc_bottom_stress,
            MaxItz0b,
            plume_type,
            first,
            h[b],
            u[b],
            v[b],
            drag[b],
            z0b[b : b + 1],
            z0s[b : b + 1],
            za[b : b + 1],
            u_taub[b : b + 1],
            u_taubo[b : b + 1],
            u_taus[b : b + 1],
            taub[b : b + 1],
            tx[b : b + 1],
            ty[b : b + 1],
        )


def friction(
    state: MeanflowState,
    nlev: int,
    *,
    kappa: float = KAPPA,
    avmolu: float | None = None,
    tx: float = 0.0,
    ty: float = 0.0,
    plume_type: int = 0,
    rho0: float = _RHO0,
    _first: list[bool] | None = None,
) -> None:
    """Update bottom roughness and compute friction velocities and drag."""
    assert state.h is not None
    assert state.u is not None
    assert state.v is not None
    assert state.drag is not None

    if avmolu is None:
        avmolu = state.avmolu

    if _first is None:
        _first = [True]

    h = state.h
    u = state.u
    v = state.v

    state.drag[:] = 0.0

    rr_s: float = 0.0
    rr_b: float = 0.0

    if state.charnock:
        z0s = state.charnock_val * state.u_taus**2 / state.gravity
        if z0s < state.z0s_min:
            z0s = state.z0s_min
    else:
        z0s = state.z0s_min
    state.z0s = z0s

    if state.calc_bottom_stress:
        if _first[0]:
            state.u_taub = state.u_taubo
            _first[0] = False
        else:
            state.u_taubo = state.u_taub

        for _ in range(state.MaxItz0b):
            if avmolu <= 0.0:
                z0b = 0.03 * state.h0b + state.za
            else:
                z0b = (
                    0.1 * avmolu / max(avmolu, state.u_taub)
                    + 0.03 * state.h0b
                    + state.za
                )
            state.z0b = z0b
            rr_b = kappa / math.log((z0b + h[1] / 2.0) / z0b)
            state.u_taub = rr_b * math.sqrt(u[1] ** 2 + v[1] ** 2)

    if plume_type == 1:
        rr_s = kappa / math.log((state.z0s + h[nlev] / 2.0) / state.z0s)

    state.taub = state.u_taub**2 * rho0
    state.drag[1] += rr_b * rr_b

    if plume_type == 1:
        state.drag[nlev] += rr_s * rr_s

    if plume_type == 1:
        state.u_taus = rr_s * math.sqrt(u[nlev] ** 2 + v[nlev] ** 2)
    else:
        state.u_taus = (tx**2 + ty**2) ** 0.25
