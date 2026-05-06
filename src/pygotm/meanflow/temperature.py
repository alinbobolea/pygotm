# ruff: noqa: E501
"""
Temperature (potential) transport equation.

Implements GOTM Section 3.2.10 — advances potential temperature :math:`\\Theta`
by one timestep using a Crank–Nicolson advection–diffusion solver.

Transport equation
------------------

The temperature satisfies (Eq. 27):

.. math::

   \\dot{\\Theta} = \\mathcal{D}_\\Theta
     - \\frac{1}{\\tau_R^\\Theta}(\\Theta - \\Theta_{\\mathrm{obs}})
     + \\frac{1}{C_p \\rho_0} \\frac{\\partial I}{\\partial z} \\comma

where :math:`\\mathcal{D}_\\Theta` is the turbulent diffusion operator (Eq. 28):

.. math::

   \\mathcal{D}_\\Theta = \\frac{\\partial}{\\partial z}
       \\left( (\\nu_t^\\Theta + \\nu^\\Theta)
       \\frac{\\partial \\Theta}{\\partial z}
       - \\tilde{\\Gamma}_\\Theta \\right) \\comma

with turbulent scalar diffusivity :math:`\\nu_t^\\Theta`, molecular diffusivity
:math:`\\nu^\\Theta = \\nu_{\\mathrm{mol}}^T` (``avmolT``), and counter-gradient
flux :math:`\\tilde{\\Gamma}_\\Theta` (``gamh``).

Short-wave radiation absorption
---------------------------------

The radiative heating term uses the Paulson and Simpson (1977) double-exponential
profile (Eq. 29):

.. math::

   I(z) = I_0 \\left[ A\\,e^{-z/\\eta_1}
        + (1 - A)\\,e^{-z/\\eta_2}\\,\\mathrm{bioshade} \\right] \\comma

where :math:`I_0` is the surface short-wave irradiance [W m⁻²],
:math:`A = 0.58` is the fraction in the red/near-infrared band,
:math:`\\eta_1 = 0.35\\,\\mathrm{m}` and :math:`\\eta_2 = 23.0\\,\\mathrm{m}` are
the extinction depths (defaults from Paulson & Simpson 1977), and
``bioshade`` is an optional biological shading factor.

The divergence :math:`\\partial I/\\partial z` contributes to ``q_sour`` at
each layer.  The surface layer receives the residual flux absorbed in the
topmost cell.

Surface boundary condition
--------------------------

At the sea surface a Neumann (prescribed flux) condition is applied:

.. math::

   (\\nu_t^\\Theta + \\nu^\\Theta)
   \\frac{\\partial \\Theta}{\\partial z}\\Bigg|_{z=\\zeta}
   = \\frac{Q_{\\mathrm{net}}}{C_p \\rho_0} \\comma

where :math:`Q_{\\mathrm{net}}` is the net non-penetrative heat flux (positive into
the ocean).  In the code, ``hflux`` carries the **opposite sign** — it follows the
atmospheric convention (positive = heat loss from ocean, negative = heat gain) — so
the Neumann value passed to the diffusion solver is
``diff_t_up = -hflux / (rho0 * cp)``,
which is positive (warming) when ``hflux < 0``.  This matches the calling convention
in both :mod:`pygotm.gotm.gotm` and Fortran GOTM v7, where
``shf = -heat_input%value`` before calling ``temperature()``.

A sea-ice correction suppresses the warming flux when the sea-surface
temperature is at or below the saline freezing point
(:math:`T \\le -0.0575\\,S`).

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
    "temperature",
    "step_temperature",
    "step_temperature_single",
]

_ADV_MODE: int = 0
_POS_CONC: int = 0

_A_DEFAULT: float = 0.58
_G1_DEFAULT: float = 0.35
_G2_DEFAULT: float = 23.0
_FREEZE_SLOPE: float = 0.0575
_LONG: float = 1.0e15


@numba.njit(cache=True)
def _step_temperature(
    nlev: int,
    dt: float,
    cnpar: float,
    avmolT: float,
    rho0: float,
    cp: float,
    A: float,
    g1: float,
    g2: float,
    w_adv_active: int,
    w_adv_discr: int,
    t_adv: int,
    T: np.ndarray,
    S: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nuh: np.ndarray,
    gamh: np.ndarray,
    bioshade: np.ndarray,
    rad: np.ndarray,
    Tobs: np.ndarray,
    tau_r: np.ndarray,
    i_0: float,
    diff_t_up: float,
    dtdx: np.ndarray,
    dtdy: np.ndarray,
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
    # sea ice correction: suppress warming flux (diff_t_up > 0) when SST <= freezing T
    if T[nlev] <= -_FREEZE_SLOPE * S[nlev]:
        if diff_t_up > 0.0:
            diff_t_up = 0.0

    # compute radiation profile and total diffusivity (iterate from surface down)
    rad[nlev] = i_0
    z = 0.0
    for j in range(nlev):
        i = nlev - 1 - j
        z += h[i + 1]
        rad[i] = i_0 * (
            A * math.exp(-z / g1) + (1.0 - A) * math.exp(-z / g2) * bioshade[i + 1]
        )
        avh[i] = nuh[i] + avmolT

    for k in range(nlev + 1):
        q_sour[k] = 0.0
        l_sour[k] = 0.0

    q_sour[nlev] = (i_0 - rad[nlev - 1]) / (rho0 * cp * h[nlev])
    for k in range(1, nlev):
        q_sour[k] = (rad[k] - rad[k - 1]) / (rho0 * cp * h[k])

    for k in range(1, nlev + 1):
        q_sour[k] -= (gamh[k] - gamh[k - 1]) / h[k]

    if t_adv == 1:
        for k in range(1, nlev + 1):
            q_sour[k] -= u[k] * dtdx[k] + v[k] * dtdy[k]

    if w_adv_active == 1:
        adv_center(
            nlev, dt, h, h, w,
            _ONE_SIDED, _ONE_SIDED, 0.0, 0.0,
            w_adv_discr, _ADV_MODE, T, adv_cu,
        )

    diff_center(
        nlev, dt, cnpar, _POS_CONC, h,
        _NEUMANN, _NEUMANN, diff_t_up, 0.0,
        avh, l_sour, q_sour, tau_r, Tobs, T,
        au, bu, cu, du, ru, qu,
    )


@numba.njit(parallel=True, cache=True)
def step_temperature(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    avmolT: float,
    rho0: float,
    cp: float,
    A: float,
    g1: float,
    g2: float,
    w_adv_active: int,
    w_adv_discr: int,
    t_adv: int,
    T: np.ndarray,
    S: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nuh: np.ndarray,
    gamh: np.ndarray,
    bioshade: np.ndarray,
    rad: np.ndarray,
    Tobs: np.ndarray,
    tau_r: np.ndarray,
    i_0: np.ndarray,
    diff_t_up: np.ndarray,
    dtdx: np.ndarray,
    dtdy: np.ndarray,
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
        _step_temperature(
            nlev, dt, cnpar, avmolT, rho0, cp, A, g1, g2,
            w_adv_active, w_adv_discr, t_adv,
            T[b], S[b], h[b], w[b], u[b], v[b],
            nuh[b], gamh[b], bioshade[b], rad[b], Tobs[b], tau_r[b],
            i_0[b], diff_t_up[b],
            dtdx[b], dtdy[b],
            avh[b], q_sour[b], l_sour[b],
            au[b], bu[b], cu[b], du[b], ru[b], qu[b], adv_cu[b],
        )


def temperature(
    state: MeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    I_0: float,
    wflux: float,
    hflux: float,
    nuh: np.ndarray,
    gamh: np.ndarray,
    *,
    rho0: float,
    cp: float,
    A: float = _A_DEFAULT,
    g1: float = _G1_DEFAULT,
    g2: float = _G2_DEFAULT,
    Tobs: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    dtdx: np.ndarray | None = None,
    dtdy: np.ndarray | None = None,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
    t_adv: bool = False,
) -> None:
    """Advance the temperature equation for one column.

    Parameters
    ----------
    state:
        MeanflowState with T, S, h, w, u, v, bioshade, rad, Tobs, avh, avmolT.
    nlev:
        Number of model layers.
    dt:
        Time step [s].
    cnpar:
        Crank–Nicolson implicitness (1 = fully implicit).
    I_0:
        Surface short-wave irradiance [W m⁻²].
    wflux:
        Freshwater flux [m s⁻¹] (reserved; not currently used in heat equation).
    hflux:
        Net non-penetrative surface heat flux [W m⁻²], **atmospheric convention
        (positive = heat loss from ocean; negative = heat gain into ocean)**.
        Identical convention to Fortran GOTM v7 ``shf = -heat_input%value``.
        Converted to Neumann BC as ``diff_t_up = -hflux / (rho0 * cp)``; negative
        ``hflux`` produces a positive (warming) Neumann value.
    nuh:
        Turbulent heat diffusivity profile [m² s⁻¹], shape (nlev+1,).
    gamh:
        Counter-gradient heat flux :math:`\\tilde{\\Gamma}_\\Theta`, shape (nlev+1,).
    rho0:
        Reference density [kg m⁻³].
    cp:
        Specific heat capacity [J kg⁻¹ K⁻¹].
    A:
        Fraction of short-wave in the red/near-IR band (Paulson & Simpson 1977).
        Default 0.58.
    g1:
        Extinction depth for band 1 [m]. Default 0.35 m.
    g2:
        Extinction depth for band 2 [m]. Default 23.0 m.
    """
    assert state.T is not None
    assert state.S is not None
    assert state.h is not None
    assert state.w is not None
    assert state.u is not None
    assert state.v is not None
    assert state.bioshade is not None
    assert state.rad is not None
    assert state.Tobs is not None
    assert state.avh is not None

    n = nlev + 1
    _Tobs = Tobs if Tobs is not None else state.Tobs
    _tau_r = tau_r if tau_r is not None else np.full(n, _LONG, dtype=np.float64)
    _dtdx = dtdx if dtdx is not None else np.zeros(n, dtype=np.float64)
    _dtdy = dtdy if dtdy is not None else np.zeros(n, dtype=np.float64)

    diff_t_up = -hflux / (rho0 * cp)

    q_sour = np.zeros(n, dtype=np.float64)
    l_sour = np.zeros(n, dtype=np.float64)
    au = np.zeros(n, dtype=np.float64)
    bu = np.zeros(n, dtype=np.float64)
    cu = np.zeros(n, dtype=np.float64)
    du = np.zeros(n, dtype=np.float64)
    ru = np.zeros(n, dtype=np.float64)
    qu = np.zeros(n, dtype=np.float64)
    adv_cu = np.zeros(n, dtype=np.float64)

    _step_temperature(
        nlev, dt, cnpar, state.avmolT, rho0, cp, A, g1, g2,
        int(w_adv_active), w_adv_discr, int(t_adv),
        state.T, state.S, state.h, state.w, state.u, state.v,
        nuh, gamh, state.bioshade, state.rad,
        _Tobs, _tau_r,
        I_0, diff_t_up,
        _dtdx, _dtdy,
        state.avh, q_sour, l_sour, au, bu, cu, du, ru, qu, adv_cu,
    )


step_temperature_single = _step_temperature
