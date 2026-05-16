"""
Salinity transport equation.

Implements GOTM Section 3.2.11 — advances salinity :math:`S` by one timestep
using an advection–diffusion solver with Neumann surface and bottom boundary
conditions.

The salinity satisfies the general scalar transport equation (Eq. 30):

.. math::

   \\dot{S} = \\mathcal{D}_S - \\frac{1}{\\tau_R^S}(S - S_{\\mathrm{obs}}) \\comma

where :math:`\\mathcal{D}_S` is the turbulent diffusion operator (Eq. 31):

.. math::

   \\mathcal{D}_S = \\frac{\\partial}{\\partial z}
       \\left( (\\nu_t^S + \\nu^S) \\frac{\\partial S}{\\partial z}
       - \\tilde{\\Gamma}_S \\right) \\comma

with turbulent scalar diffusivity :math:`\\nu_t^S = \\nu_t^{\\Theta}` (the same
as for temperature in the current GOTM implementation), molecular diffusivity
:math:`\\nu^S`, and counter-gradient flux :math:`\\tilde{\\Gamma}_S`.

The optional :math:`-\\tau_R^{-1}(S - S_{\\mathrm{obs}})` term relaxes the
modelled salinity towards an observed profile :math:`S_{\\mathrm{obs}}` with
time scale :math:`\\tau_R^S`.

Surface boundary condition
--------------------------

At the sea surface :math:`z = \\zeta` a Neumann (prescribed flux) condition
is applied (Eq. 32):

.. math::

   (\\nu_t^S + \\nu^S) \\frac{\\partial S}{\\partial z} = S (P - E) \\comma

where :math:`P - E` is precipitation minus evaporation [m s⁻¹].  In the code
this is passed as ``sflux = S_sfc * (P - E)`` [psu m s⁻¹] —
**positive when precipitation exceeds evaporation** (P > E, dilution regime),
**negative when evaporation dominates** (E > P, concentration regime).

.. note::

   **Sign convention vs. the GOTM manual (Eq. 32).**  The manual writes the
   surface BC as :math:`(\\nu_t^S + \\nu^S)\\,\\partial S/\\partial z = S(P-E)`
   at :math:`z = \\zeta`, where the right-hand side is the diffusive flux in the
   *upward* (+z) direction — positive means salt leaves the water column (P > E,
   dilution).  The ``diff_center`` kernel, however, uses the opposite sign
   convention: the upper Neumann argument ``Yup`` is the flux *entering* the top
   cell (positive = salt gains the surface layer).  Because these two conventions
   have opposite signs, the code negates ``sflux`` before passing it to the
   kernel::

       diff_s_up = -sflux = -S*(P-E) = S*(E-P)

   For evaporation (E > P): ``diff_s_up > 0`` → flux entering the top cell →
   surface salinity increases. ✓
   For precipitation (P > E): ``diff_s_up < 0`` → flux leaving the top cell →
   surface salinity decreases. ✓

   This matches the Fortran verbatim (``DiffSup = -sflux`` in
   ``salinity.F90``).

Both surface and bottom boundary conditions are of Neumann type; the bottom
flux is zero.

Authors (original Fortran): Lars Umlauf (scalar transport), Hans Burchard,
Karsten Bolding.
"""

import numba
import numpy as np

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.util.adv_center import adv_center
from pygotm.util.diff_center import diff_center
from pygotm.util.util import Neumann as _NEUMANN
from pygotm.util.util import oneSided as _ONE_SIDED

__all__ = [
    "salinity",
    "step_salinity",
    "step_salinity_single",
]

_ADV_MODE: int = 0
_POS_CONC: int = 1
_LONG: float = 1.0e15


@numba.njit(cache=True)
def _step_salinity(
    nlev: int,
    dt: float,
    cnpar: float,
    avmolS: float,
    w_adv_active: int,
    w_adv_discr: int,
    s_adv: int,
    S: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nus: np.ndarray,
    gams: np.ndarray,
    Sobs: np.ndarray,
    tau_r: np.ndarray,
    diff_s_up: float,
    dsdx: np.ndarray,
    dsdy: np.ndarray,
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
        avh[k] = nus[k] + avmolS
        q_sour[k] = 0.0
        l_sour[k] = 0.0

    for k in range(1, nlev + 1):
        q_sour[k] -= (gams[k] - gams[k - 1]) / h[k]

    if s_adv == 1:
        for k in range(1, nlev + 1):
            q_sour[k] -= u[k] * dsdx[k] + v[k] * dsdy[k]

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
            S,
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
        diff_s_up,
        0.0,
        avh,
        l_sour,
        q_sour,
        tau_r,
        Sobs,
        S,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )


@numba.njit(parallel=True, cache=True)
def step_salinity(
    batch_size: int,
    nlev: int,
    dt: float,
    cnpar: float,
    avmolS: float,
    w_adv_active: int,
    w_adv_discr: int,
    s_adv: int,
    S: np.ndarray,
    h: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    nus: np.ndarray,
    gams: np.ndarray,
    Sobs: np.ndarray,
    tau_r: np.ndarray,
    diff_s_up: np.ndarray,
    dsdx: np.ndarray,
    dsdy: np.ndarray,
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
        _step_salinity(
            nlev,
            dt,
            cnpar,
            avmolS,
            w_adv_active,
            w_adv_discr,
            s_adv,
            S[b],
            h[b],
            w[b],
            u[b],
            v[b],
            nus[b],
            gams[b],
            Sobs[b],
            tau_r[b],
            diff_s_up[b],
            dsdx[b],
            dsdy[b],
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


def salinity(
    state: MeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    wflux: float,
    sflux: float,
    nus: np.ndarray,
    gams: np.ndarray,
    *,
    Sobs: np.ndarray | None = None,
    tau_r: np.ndarray | None = None,
    dsdx: np.ndarray | None = None,
    dsdy: np.ndarray | None = None,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
    s_adv: bool = False,
) -> None:
    """Advance the salinity equation for one column.

    Parameters
    ----------
    state:
        MeanflowState with S, h, w, u, v, Sobs, avh, avmolS.
    nlev:
        Number of model layers.
    dt:
        Time step [s].
    cnpar:
        Crank–Nicolson implicitness (0 = explicit, 1 = fully implicit).
    wflux:
        Freshwater flux at the surface [m s⁻¹] (positive = precipitation). Not
        used directly; kept for API symmetry with ``temperature()``.
    sflux:
        Virtual salinity flux ``S_surface · (P − E)`` [psu m s⁻¹].
        Positive when P > E (precipitation-dominated, dilution);
        negative when E > P (evaporation-dominated, concentration).
        The kernel receives ``-sflux`` as the upper Neumann boundary condition;
        see the module docstring for the sign-convention note.
    nus:
        Turbulent salinity diffusivity profile [m² s⁻¹], shape (nlev+1,).
    gams:
        Counter-gradient salinity flux :math:`\\tilde{\\Gamma}_S`, shape (nlev+1,).
    """
    assert state.S is not None
    assert state.h is not None
    assert state.w is not None
    assert state.u is not None
    assert state.v is not None
    assert state.Sobs is not None
    assert state.avh is not None

    n = nlev + 1
    _Sobs = Sobs if Sobs is not None else state.Sobs
    _tau_r = tau_r if tau_r is not None else np.full(n, _LONG, dtype=np.float64)
    _dsdx = dsdx if dsdx is not None else np.zeros(n, dtype=np.float64)
    _dsdy = dsdy if dsdy is not None else np.zeros(n, dtype=np.float64)

    q_sour = np.zeros(n, dtype=np.float64)
    l_sour = np.zeros(n, dtype=np.float64)
    au = np.zeros(n, dtype=np.float64)
    bu = np.zeros(n, dtype=np.float64)
    cu = np.zeros(n, dtype=np.float64)
    du = np.zeros(n, dtype=np.float64)
    ru = np.zeros(n, dtype=np.float64)
    qu = np.zeros(n, dtype=np.float64)
    adv_cu = np.zeros(n, dtype=np.float64)

    _step_salinity(
        nlev,
        dt,
        cnpar,
        state.avmolS,
        int(w_adv_active),
        w_adv_discr,
        int(s_adv),
        state.S,
        state.h,
        state.w,
        state.u,
        state.v,
        nus,
        gams,
        _Sobs,
        _tau_r,
        -sflux,
        _dsdx,
        _dsdy,
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


step_salinity_single = _step_salinity
