# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic q2/2-equation \label{sec:q2over2eq}
!
! !INTERFACE:
!   subroutine q2over2eq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The transport equation for the TKE $q^2/2=k$ can be written as
! \begin{equation}
!   \label{tkeB}
!   \dot{\overline{q^2/2}}
!   =
!   {\cal D}_q +  P + G + P_x + P_s - \epsilon
!   \comma
! \end{equation}
! where $\dot{\overline{q^2/2}}$ denotes the material derivative of $q^2/2$.
! With $P$ and $G$ following from \eq{PandG}, evidently, this equation is
! formally identical to \eq{tkeA}. The only reason why it is discretized
! seperately here, is the slightly different down-gradient model for the
! transport term,
! \begin{equation}
!   \label{diffusionMYTKE}
!   {\cal D}_q = \frstder{z} \left( q l S_q \partder{q^2/2}{z} \right)
!  \comma
! \end{equation}
! where $S_q$ is a model constant. The notation has been chosen according
! to that introduced by \cite{MellorYamada82}. Using their notation,
! also \eq{epsilon} can be expressed in mathematically identical form
! as
! \begin{equation}
!   \label{epsilonMY}
!   \epsilon = \frac{q^3}{B_1 l}
!   \comma
! \end{equation}
! where $B_1$ is a constant of the model. Note, that the equivalence of
! \eq{epsilon} and \eq{epsilonMY} requires that
! \begin{equation}
!   \label{B1}
!   (c_\mu^0)^{-2} = \frac{1}{2} B_1^\frac{2}{3}
!   \point
! \end{equation}
!
! !USES:
!   use turbulence,   only: P,B,Px,PSTK
!   use turbulence,   only: tke,tkeo,k_min,eps,L
!   use turbulence,   only: q2over2_bc, k_ubc, k_lbc, ubc_type, lbc_type
!   use turbulence,   only: sq_var
!   use util,         only: Dirichlet,Neumann
!
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
!  time step (s)
!   REALTYPE, intent(in)                :: dt
!
!  surface and bottom
!  friction velocity (m/s)
!   REALTYPE, intent(in)                :: u_taus,u_taub
!
!  surface and bottom
!  roughness length (m)
!   REALTYPE, intent(in)                :: z0s,z0b
!
!  layer thickness (m)
!   REALTYPE, intent(in)                :: h(0:nlev)
!
!  square of shear and buoyancy
!  frequency (1/s^2)
!   REALTYPE, intent(in)                :: NN(0:nlev),SS(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffKup,DiffKdw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  position for upper BC
!
!  obtain BC for upper boundary of type "ubc_type"
!
!  position for lower BC
!
!  obtain BC for lower boundary of type "lbc_type"
!
!  do diffusion step
!
!  fill top and bottom value with something nice
!  (only for output)
!
!  clip at k_min
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.turbulence.turbulence import Dirichlet as _DIRICHLET
from pygotm.turbulence.turbulence import Neumann as _NEUMANN
from pygotm.turbulence.turbulence import injection as _INJECTION
from pygotm.turbulence.turbulence import logarithmic as _LOGARITHMIC
from pygotm.util.diff_face import diff_face_column

__all__ = [
    "Q2Over2EquationWorkspace",
    "step_q2over2eq",
]

_CNPAR: float = 1.0
_SQRT2: float = 1.4142135623730951


class Q2Over2EquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated Mellor-Yamada q2/2 equation."""

    tke: ti.Field
    tkeo: ti.Field
    h: ti.Field
    P: ti.Field
    B: ti.Field
    Px: ti.Field
    PSTK: ti.Field
    eps: ti.Field
    L: ti.Field
    sq_var: ti.Field
    avh: ti.Field
    l_sour: ti.Field
    q_sour: ti.Field
    u_taus: ti.Field
    u_taub: ti.Field
    z0s: ti.Field
    z0b: ti.Field
    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "tkeo", "h"))
        self.allocate_many(("P", "B", "Px", "PSTK", "eps", "L", "sq_var"))
        self.allocate_many(("avh", "l_sour", "q_sour"))
        self.allocate_many(("u_taus", "u_taub", "z0s", "z0b"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti.func
def _fk_craig(u_tau, eta):  # type: ignore[no-untyped-def]
    return eta * u_tau**3


@ti.func
def _q2over2_bc_value(  # type: ignore[no-untyped-def]
    bc,
    type_,
    zi,
    z0,
    u_tau,
    b1,
    sq,
    cw,
    gen_alpha,
    gen_l,
):
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = u_tau**2 * ti.pow(b1, 2.0 / 3.0) / 2.0
        else:
            value = 0.0

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = ti.pow(
            -f_k / (_SQRT2 * sq * gen_alpha * gen_l),
            2.0 / 3.0,
        ) / ti.pow(z0, gen_alpha)

        if bc == _DIRICHLET:
            value = capital_k * ti.pow(zi + z0, gen_alpha)
        else:
            value = (
                -_SQRT2
                * sq
                * ti.pow(capital_k, 1.5)
                * gen_alpha
                * gen_l
                * ti.pow(zi + z0, 1.5 * gen_alpha)
            )

    return value


@ti_kernel
def step_q2over2eq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    k_min: ti.f64,
    b1: ti.f64,
    k_ubc: ti.i32,
    k_lbc: ti.i32,
    ubc_type: ti.i32,
    lbc_type: ti.i32,
    sq: ti.f64,
    cw: ti.f64,
    gen_alpha: ti.f64,
    gen_l: ti.f64,
    tke: TemplateArg,
    tkeo: TemplateArg,
    h: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Px: TemplateArg,
    PSTK: TemplateArg,
    eps: TemplateArg,
    L: TemplateArg,
    sq_var: TemplateArg,
    avh: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    u_taus: TemplateArg,
    u_taub: TemplateArg,
    z0s: TemplateArg,
    z0b: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    r"""Advance the dynamic q2/2-equation for one or more columns."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            tkeo[col, i] = tke[col, i]
            avh[col, i] = 0.0
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        for i in range(1, nlev):
            avh[col, i] = sq_var[col, i] * ti.sqrt(2.0 * tke[col, i]) * L[col, i]

            prod = P[col, i] + Px[col, i] + PSTK[col, i]
            buoyan = B[col, i]
            diss = eps[col, i]

            if prod + buoyan > 0.0:
                q_sour[col, i] = prod + buoyan
                l_sour[col, i] = -diss / tke[col, i]
            else:
                q_sour[col, i] = prod
                l_sour[col, i] = -(diss - buoyan) / tke[col, i]

        pos_bc = h[col, nlev]
        if k_ubc == _NEUMANN:
            pos_bc = 0.5 * h[col, nlev]
        diff_k_up = _q2over2_bc_value(
            k_ubc,
            ubc_type,
            pos_bc,
            z0s[col, 0],
            u_taus[col, 0],
            b1,
            sq,
            cw,
            gen_alpha,
            gen_l,
        )

        pos_bc = h[col, 1]
        if k_lbc == _NEUMANN:
            pos_bc = 0.5 * h[col, 1]
        diff_k_down = _q2over2_bc_value(
            k_lbc,
            lbc_type,
            pos_bc,
            z0b[col, 0],
            u_taub[col, 0],
            b1,
            sq,
            cw,
            gen_alpha,
            gen_l,
        )

        diff_face_column(
            col,
            nlev,
            dt,
            _CNPAR,
            h,
            k_ubc,
            k_lbc,
            diff_k_up,
            diff_k_down,
            avh,
            l_sour,
            q_sour,
            tke,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )

        tke[col, nlev] = _q2over2_bc_value(
            _DIRICHLET,
            ubc_type,
            z0s[col, 0],
            z0s[col, 0],
            u_taus[col, 0],
            b1,
            sq,
            cw,
            gen_alpha,
            gen_l,
        )
        tke[col, 0] = _q2over2_bc_value(
            _DIRICHLET,
            lbc_type,
            z0b[col, 0],
            z0b[col, 0],
            u_taub[col, 0],
            b1,
            sq,
            cw,
            gen_alpha,
            gen_l,
        )

        for i in range(nlev + 1):
            if tke[col, i] < k_min:
                tke[col, i] = k_min
