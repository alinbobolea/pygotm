# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The vertical friction \label{sec:friction}
!
! !INTERFACE:
!   subroutine friction(nlev,kappa,avmolu,tx,ty,plume_type)
!
! !DESCRIPTION:
!  This subroutine updates the bottom roughness
!  \begin{equation}
!    \label{Defz0b}
!    z_0^b = 0.1 \frac{\nu}{u_*^b} + 0.03 h_0^b + z_a \point
!  \end{equation}
!  The first term on the right hand side of \eq{Defz0b} represents
!  the limit for hydraulically smooth surfaces, the second term the limit
!  for completely rough surfaces. Note that the third term, $z_a$,
!  is the contribution of suspended sediments to the
!  roughness length, see \cite{SmithMcLean77}. It is updated during calls
!  to the sediment-routines.
!
! The law-of-the-wall relations are used to compute the friction velocity
! \begin{equation}
!  \label{uStar}
!   u_*^b = r \sqrt{U_1^2 + V_1^2}
!   \comma
! \end{equation}
! where $U_1$ and $V_1$ are the components of the mean velocity
! at the center of the lowest cell.
! We used the abbreviation
!  \begin{equation}
!    \label{rParam}
!    r=\frac{\kappa}{\ln \left( \frac{0.5h_1+z_0^b}{z^b_0} \right)}
!    \comma
!  \end{equation}
!  where $\kappa$ is the von K{\'a}rm{\'a}n constant and
!  the index `1' indicates values at the center of the first
!  grid box at the bottom (version 1). Another expression for $r$ can be
!  derived using the mean value of the velocity in the lowest
!  grid box, and not its value in the middle of the box (version 2). Also
!  this method is supported in {\tt friction()} and can be activated by
!  uncommenting one line in the code.
!
!  If no breaking surface waves are considered, the law of the wall
!  also holds at the surface. The surface roughness length may
!  be calculated according to the \cite{Charnock55} formula,
!  \begin{equation}
!   \label{Charnock}
!    z_0^s=\alpha \frac{(u_*^s)^2}{g}
!   \point
!  \end{equation}
!  The model constant $\alpha$ is read in as {\tt charnock\_val} from
!  the {\tt gotm.yaml}.
!
! !USES:
!   use density,       only: rho0
!   use meanflow,      only: h,z0b,h0b,MaxItz0b,z0s,za
!   use meanflow,      only: u,v,gravity
!   use meanflow,      only: u_taub,u_taubo,u_taus,drag,taub
!   use meanflow,      only: calc_bottom_stress
!   use meanflow,      only: charnock,charnock_val,z0s_min
!
! !INPUT PARAMETERS:
!  number of vertical layers
!   integer, intent(in)                 :: nlev
!   REALTYPE, intent(in)                :: kappa,avmolu,tx,ty
!   integer, intent(in)                 :: plume_type
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
!BOC
!
!  drag = _ZERO_
!  rr_s = _ZERO_
!  rr_b = _ZERO_
!
!  use the Charnock formula to compute the surface roughness
!  if (charnock) then
!     z0s=charnock_val*u_taus**2/gravity
!     if (z0s.lt.z0s_min) z0s=z0s_min
!  else
!     z0s=z0s_min
!  end if
!
!  if (calc_bottom_stress) then
!     if (first) then
!        u_taub = u_taubo
!        first = .false.
!     else
!        u_taubo = u_taub
!     end if
!     iterate bottom roughness length MaxItz0b times
!     do i=1,MaxItz0b
!        if (avmolu.le.0) then
!           z0b=0.03*h0b + za
!        else
!           z0b=0.1*avmolu/max(avmolu,u_taub)+0.03*h0b + za
!        end if
!        compute the factor r (version 1, with log-law)
!        rr_b=kappa/(log((z0b+h(1)/2)/z0b))
!        compute the friction velocity at the bottom
!        u_taub = rr_b*sqrt( u(1)*u(1) + v(1)*v(1) )
!     end do
!  end if
!
!  compute the factor r (version 1, with log-law)
!  if (plume_type .eq. 1) rr_s=kappa/(log((z0s+h(nlev)/2)/z0s))
!
!  calculate bottom stress, which is used by sediment resuspension models
!  taub = u_taub*u_taub*rho0
!
!  add bottom friction as source term for the momentum equation
!  drag(1) = drag(1) +  rr_b*rr_b
!
!  add for surface plume scenario surface friction as source term for the momentum equation
!  if (plume_type .eq. 1) drag(nlev) = drag(nlev) +  rr_s*rr_s
!
!  be careful: tx and ty are the surface shear-stresses
!  already divided by rho!
!  if (plume_type == 1) then
!     u_taus=rr_s*sqrt( u(nlev)*u(nlev) + v(nlev)*v(nlev) )
!  else
!     u_taus=(tx**2+ty**2)**(1./4.)
!  endif
!
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.meanflow.meanflow import MeanflowState
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = [
    "FrictionWorkspace",
    "KAPPA",
    "friction",
    "step_friction",
]

# Von Kármán constant (GOTM default; passed as argument in Fortran but
# universally 0.4 in all GOTM configurations).
KAPPA: float = 0.4

# Reference seawater density [kg/m³] — mirrors density.F90 default rho0=1027.
_RHO0: float = 1027.0


class FrictionWorkspace(TaichiFieldCollection):
    """Taichi fields for the friction kernel."""

    h: ti.Field
    u: ti.Field
    v: ti.Field
    drag: ti.Field
    z0b: ti.Field
    z0s: ti.Field
    za: ti.Field
    u_taub: ti.Field
    u_taubo: ti.Field
    u_taus: ti.Field
    taub: ti.Field
    tx: ti.Field
    ty: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("h", "u", "v", "drag"))
        self.allocate_many(
            ("z0b", "z0s", "za", "u_taub", "u_taubo", "u_taus", "taub", "tx", "ty")
        )


@ti_kernel
def step_friction(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    kappa: ti.f64,
    avmolu: ti.f64,
    rho0: ti.f64,
    gravity: ti.f64,
    h0b: ti.f64,
    z0s_min: ti.f64,
    charnock: ti.i32,
    charnock_val: ti.f64,
    calc_bottom_stress: ti.i32,
    MaxItz0b: ti.i32,
    plume_type: ti.i32,
    first: ti.i32,
    h: TemplateArg,
    u: TemplateArg,
    v: TemplateArg,
    drag: TemplateArg,
    z0b: TemplateArg,
    z0s: TemplateArg,
    za: TemplateArg,
    u_taub: TemplateArg,
    u_taubo: TemplateArg,
    u_taus: TemplateArg,
    taub: TemplateArg,
    tx: TemplateArg,
    ty: TemplateArg,
):
    """Compute roughness lengths, friction velocities, and drag coefficients.

    ``first`` mirrors the saved Fortran flag: on the first call, ``u_taub`` is
    initialised from ``u_taubo`` before the roughness iteration. On later
    calls, ``u_taubo`` stores the previous call's ``u_taub``.
    """

    for col in range(n_cols):
        for k in range(nlev + 1):
            drag[col, k] = 0.0

        rr_s = 0.0
        rr_b = 0.0

        z0s_val = z0s_min
        if charnock == 1:
            z0s_val = charnock_val * u_taus[col, 0] * u_taus[col, 0] / gravity
            if z0s_val < z0s_min:
                z0s_val = z0s_min
        z0s[col, 0] = z0s_val

        if calc_bottom_stress == 1:
            if first == 1:
                u_taub[col, 0] = u_taubo[col, 0]
            else:
                u_taubo[col, 0] = u_taub[col, 0]
            for _ in range(MaxItz0b):
                z0b_val = 0.0
                if avmolu <= 0.0:
                    z0b_val = 0.03 * h0b + za[col, 0]
                else:
                    denom = ti.max(avmolu, u_taub[col, 0])
                    z0b_val = 0.1 * avmolu / denom + 0.03 * h0b + za[col, 0]
                z0b[col, 0] = z0b_val
                rr_b = kappa / ti.log((z0b_val + h[col, 1] / 2.0) / z0b_val)
                speed_b = ti.sqrt(u[col, 1] * u[col, 1] + v[col, 1] * v[col, 1])
                u_taub[col, 0] = rr_b * speed_b

        if plume_type == 1:
            rr_s = kappa / ti.log((z0s_val + h[col, nlev] / 2.0) / z0s_val)

        taub[col, 0] = u_taub[col, 0] * u_taub[col, 0] * rho0
        drag[col, 1] += rr_b * rr_b

        if plume_type == 1:
            drag[col, nlev] += rr_s * rr_s
            speed_s = ti.sqrt(u[col, nlev] * u[col, nlev] + v[col, nlev] * v[col, nlev])
            u_taus[col, 0] = rr_s * speed_s
        else:
            tx_val = tx[col, 0]
            ty_val = ty[col, 0]
            u_taus[col, 0] = (tx_val * tx_val + ty_val * ty_val) ** 0.25


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
    """Update bottom roughness and compute friction velocities and drag coefficients.

    Translates friction.F90 verbatim.  Updates ``state.drag``, ``state.z0b``,
    ``state.z0s``, ``state.u_taub``, ``state.u_taubo``, ``state.u_taus``, and
    ``state.taub`` in-place.

    Parameters
    ----------
    state:
        MeanflowState carrying h, u, v, drag, z0b, z0s, za, h0b, MaxItz0b,
        u_taub, u_taubo, u_taus, taub, calc_bottom_stress, charnock,
        charnock_val, z0s_min, gravity.
    nlev:
        Number of vertical layers.
    kappa:
        Von Kármán constant [-] (default 0.4).
    avmolu:
        Molecular kinematic viscosity [m²/s].  Defaults to ``state.avmolu``.
    tx:
        Surface wind stress in x divided by rho_0 [m²/s²].
    ty:
        Surface wind stress in y divided by rho_0 [m²/s²].
    plume_type:
        0 → standard configuration; 1 → surface-plume scenario where the
        law-of-the-wall also applies at the surface.
    rho0:
        Reference density [kg/m³] used to compute bottom stress ``taub``.
    _first:
        Mutable single-element list used to carry the Fortran ``first`` flag
        across repeated calls.  Callers sharing a persistent state object
        should pass the same list on every call.  Defaults to a new list
        (``[True]``) so that standalone calls default to the initialisation
        branch (``u_taub = u_taubo``).
    """
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

    # Reset drag to zero (mirrors: drag = _ZERO_)
    state.drag[:] = 0.0

    rr_s: float = 0.0
    rr_b: float = 0.0

    # --- surface roughness ---
    # use the Charnock formula to compute the surface roughness
    if state.charnock:
        z0s = state.charnock_val * state.u_taus**2 / state.gravity
        if z0s < state.z0s_min:
            z0s = state.z0s_min
    else:
        z0s = state.z0s_min
    state.z0s = z0s

    # --- bottom roughness iteration and friction velocity ---
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

            # compute factor r (version 1, with log-law)
            rr_b = kappa / math.log((z0b + h[1] / 2.0) / z0b)

            # compute friction velocity at the bottom
            state.u_taub = rr_b * math.sqrt(u[1] ** 2 + v[1] ** 2)

    # --- surface log-law factor for plume scenario ---
    if plume_type == 1:
        rr_s = kappa / math.log((state.z0s + h[nlev] / 2.0) / state.z0s)

    # calculate bottom stress, which is used by sediment resuspension models
    state.taub = state.u_taub**2 * rho0

    # add bottom friction as source term for the momentum equation
    state.drag[1] += rr_b * rr_b

    # add for surface plume scenario surface friction as source term
    if plume_type == 1:
        state.drag[nlev] += rr_s * rr_s

    # be careful: tx and ty are the surface shear-stresses already divided by rho!
    if plume_type == 1:
        state.u_taus = rr_s * math.sqrt(u[nlev] ** 2 + v[nlev] ** 2)
    else:
        state.u_taus = (tx**2 + ty**2) ** 0.25
