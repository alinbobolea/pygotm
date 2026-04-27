# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic q2l-equation\label{sec:lengthscaleeq}
!
! !INTERFACE:
!   subroutine lengthscaleeq(nlev,dt,depth,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! Following suggestions of \cite{Rotta51a}, \cite{MellorYamada82}
! proposed an equation for the product $q^2 l$ expressed by
! \begin{equation}
!   \label{MY}
!   \dot{\overline{q^2 l}}
!   = {\cal D}_l + l ( E_1  P + E_3 G + E_x P_x + E_6 P_s - E_2  F \epsilon )
!   \comma
! \end{equation}
! where $\dot{\overline{q^2 l}}$ denotes the material derivative of $q^2 l$.
! The production terms $P$ and $G$ follow from \eq{PandG}.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! $\epsilon$
! can be computed either directly from \eq{epsilonMY}, or from \eq{epsilon}
! with the help \eq{B1}.
!
! The so-called wall function, $F$, appearing in \eq{MY} is defined by
! \begin{equation}
!   \label{F}
!   F = 1 + E_2 \left( \dfrac{l}{\kappa {\cal L}_z} \right)^2
!   \comma
! \end{equation}
! $\kappa$ being the von K{\'a}rm{\'a}n constant and ${\cal L}_z$ some
! measure for the distance from the wall. Different possiblities
! for  ${\cal L}_z$ are implemented in GOTM, which can be activated
! be setting the parameter {\tt MY\_length} in {\tt gotm.yaml} to
! appropriate values. Close to the wall, however, one always has
! ${\cal L}_z= \overline{z}$, where $\overline{z}$ is the distance from
! the wall.
!
! For horizontally homogeneous flows, the transport term ${\cal D}_l$
! appearing in \eq{MY} is expressed by a simple gradient formulation,
! \begin{equation}
!   \label{diffusionMYlength}
!   {\cal D}_l = \frstder{z} \left( q l S_l \partder{q^2 l}{z} \right)
!  \comma
! \end{equation}
! where $S_l$ is a constant of the model. The values for the model
! constants recommended by \cite{MellorYamada82} are displayed in
! \tab{tab:MY_constants}. They can be set in {\tt gotm.yaml}. Note,
! that the parameter $E_3$ in stably stratifed flows is in principle
! a function of the so-called steady state Richardson-number,
! as discussed by \cite{Burchard2001c}, see discussion in the context
! of \eq{Ri_st}.
! \begin{table}[ht]
!   \begin{center}
! \begin{tabular}{ccccccc}
!                           & $B_1$  & $S_q$ & $S_l$ & $E_1$ & $E_2$ & $E_3$    \\[1mm]
!      \hline
!     \cite{MellorYamada82} & $16.6$ & $0.2$ & $0.2$ & $1.8$ & $1.33$ & $1.8$\\
!   \end{tabular}
!   \caption{\label{tab:MY_constants} Constants appearing in \eq{MY}
!     and \eq{epsilonMY}}
!   \end{center}
! \end{table}
!
! At the end of this routine the length-scale can be constrained according to a
! suggestion of \cite{Galperinetal88}. This feature is optional and can be activated
! by setting {\tt length\_lim = .true.} in {\tt gotm.yaml}.
!
! !USES:
!   use turbulence, only: P,B,Px,PSTK
!   use turbulence, only: tke,tkeo,k_min,eps,eps_min,L
!   use turbulence, only: kappa,e1,e2,e3,ex,e6,b1
!   use turbulence, only: MY_length,cm0,cde,galp,length_lim
!   use turbulence, only: q2l_bc, psi_ubc, psi_lbc, ubc_type, lbc_type
!   use turbulence, only: sl_var
!   use util,       only: Dirichlet,Neumann
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
!  local water depth (m)
!   REALTYPE, intent(in)                :: depth
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
!                     (re-write after first version of
!                      H. Burchard and K. Bolding
!EOP
!------------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                  :: DiffQ2lup,DiffQ2ldw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: ki,epslim,NN_pos
!   REALTYPE                  :: ds,db,Lcrit
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: q2l(0:nlev),q3(0:nlev)
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lz(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!
!   REALTYPE                  :: l_min
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
! compute lower bound for length scale
!
!  some quantities in Mellor-Yamada notation
!
!  diagnostic length scale for wall function
!
! prepare the production terms
!
!     compute diffusivity
!
!     compute production terms in q^2 l - equation
!
!     compute positive and negative parts of RHS
!
!  TKE and position for upper BC
!
!  obtain BC for upper boundary of type "ubc_type"
!
!  TKE and position for lower BC
!
!  obtain BC for lower boundary of type "lbc_type"
!
!  do diffusion step
!
!  fill top and bottom value with something nice
!  (only for output)
!
! compute L and epsilon
!
!    apply the length-scale clipping of Galperin et al. (1988)
!
!    check for very small lengh scale
!
!    compute dissipation rate
!
!    substitute minimum value
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
    "LengthScaleEquationWorkspace",
    "step_lengthscaleeq",
]

_CNPAR: float = 1.0
_SQRT2: float = 1.4142135623730951


class LengthScaleEquationWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated q2l length-scale equation."""

    tke: ti.Field
    tkeo: ti.Field
    eps: ti.Field
    L: ti.Field
    h: ti.Field
    NN: ti.Field
    SS: ti.Field
    P: ti.Field
    B: ti.Field
    Px: ti.Field
    PSTK: ti.Field
    sl_var: ti.Field
    depth: ti.Field
    u_taus: ti.Field
    u_taub: ti.Field
    z0s: ti.Field
    z0b: ti.Field
    q2l: ti.Field
    avh: ti.Field
    l_sour: ti.Field
    q_sour: ti.Field
    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "tkeo", "eps", "L", "h", "NN", "SS"))
        self.allocate_many(("P", "B", "Px", "PSTK", "sl_var"))
        self.allocate_many(("depth", "u_taus", "u_taub", "z0s", "z0b"))
        self.allocate_many(("q2l", "avh", "l_sour", "q_sour"))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))


@ti.func
def _fk_craig(u_tau, eta):  # type: ignore[no-untyped-def]
    return eta * u_tau**3


@ti_kernel
def step_lengthscaleeq(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    k_min: ti.f64,
    eps_min: ti.f64,
    kappa: ti.f64,
    e1: ti.f64,
    e2: ti.f64,
    e3: ti.f64,
    ex: ti.f64,
    e6: ti.f64,
    b1: ti.f64,
    cde: ti.f64,
    my_length: ti.i32,
    galp: ti.f64,
    length_lim: ti.i32,
    psi_ubc: ti.i32,
    psi_lbc: ti.i32,
    ubc_type: ti.i32,
    lbc_type: ti.i32,
    sl: ti.f64,
    sq: ti.f64,
    cw: ti.f64,
    gen_alpha: ti.f64,
    gen_l: ti.f64,
    tke: TemplateArg,
    tkeo: TemplateArg,
    eps: TemplateArg,
    L: TemplateArg,
    h: TemplateArg,
    NN: TemplateArg,
    SS: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Px: TemplateArg,
    PSTK: TemplateArg,
    sl_var: TemplateArg,
    depth: TemplateArg,
    u_taus: TemplateArg,
    u_taub: TemplateArg,
    z0s: TemplateArg,
    z0b: TemplateArg,
    q2l: TemplateArg,
    avh: TemplateArg,
    l_sour: TemplateArg,
    q_sour: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
):
    r"""Advance the dynamic q2l-equation for one or more columns."""

    for col in range(n_cols):
        l_min = cde * ti.sqrt(k_min * k_min * k_min) / eps_min

        for i in range(nlev + 1):
            q2l[col, i] = 0.0
            avh[col, i] = 0.0
            l_sour[col, i] = 0.0
            q_sour[col, i] = 0.0

        for i in range(1, nlev):
            q2l[col, i] = 2.0 * tkeo[col, i] * L[col, i]

        db = 0.0
        for i in range(1, nlev):
            db += h[col, i]
            ds = depth[col, 0] - db

            lz = 0.0
            if my_length == 1:
                lz = (
                    kappa
                    * (ds + z0s[col, 0])
                    * (db + z0b[col, 0])
                    / (ds + z0s[col, 0] + db + z0b[col, 0])
                )
            if my_length == 2:
                lz = kappa * ti.min(ds + z0s[col, 0], db + z0b[col, 0])
            if my_length == 3:
                lz = kappa * (ds + z0s[col, 0])

            avh[col, i] = sl_var[col, i] * ti.sqrt(2.0 * tkeo[col, i]) * L[col, i]

            prod = L[col, i] * (e1 * P[col, i] + ex * Px[col, i] + e6 * PSTK[col, i])
            buoyan = e3 * L[col, i] * B[col, i]
            q3 = ti.sqrt(8.0 * tkeo[col, i] * tkeo[col, i] * tkeo[col, i])
            diss = q3 / b1 * (1.0 + e2 * (L[col, i] / lz) * (L[col, i] / lz))

            if prod + buoyan > 0.0:
                q_sour[col, i] = prod + buoyan
                l_sour[col, i] = -diss / q2l[col, i]
            else:
                q_sour[col, i] = prod
                l_sour[col, i] = -(diss - buoyan) / q2l[col, i]

        ki = tke[col, nlev - 1]
        pos_bc = h[col, nlev]
        if psi_ubc == _NEUMANN:
            pos_bc = 0.5 * h[col, nlev]
        diff_q2l_up = 0.0
        if ubc_type == _LOGARITHMIC:
            if psi_ubc == _DIRICHLET:
                diff_q2l_up = 2.0 * kappa * ki * (pos_bc + z0s[col, 0])
            else:
                diff_q2l_up = (
                    -2.0
                    * _SQRT2
                    * sl
                    * kappa
                    * kappa
                    * ti.pow(ki, 1.5)
                    * (pos_bc + z0s[col, 0])
                )
        if ubc_type == _INJECTION:
            f_k = _fk_craig(u_taus[col, 0], cw)
            capital_k = ti.pow(
                -f_k / (_SQRT2 * sq * gen_alpha * gen_l),
                2.0 / 3.0,
            ) / ti.pow(z0s[col, 0], gen_alpha)
            if psi_ubc == _DIRICHLET:
                diff_q2l_up = (
                    2.0
                    * capital_k
                    * gen_l
                    * ti.pow(pos_bc + z0s[col, 0], gen_alpha + 1.0)
                )
            else:
                diff_q2l_up = (
                    -2.0
                    * _SQRT2
                    * sl
                    * (gen_alpha + 1.0)
                    * ti.pow(capital_k, 1.5)
                    * gen_l
                    * gen_l
                    * ti.pow(pos_bc + z0s[col, 0], 1.5 * gen_alpha + 1.0)
                )

        ki = tke[col, 1]
        pos_bc = h[col, 1]
        if psi_lbc == _NEUMANN:
            pos_bc = 0.5 * h[col, 1]
        diff_q2l_down = 0.0
        if lbc_type == _LOGARITHMIC:
            if psi_lbc == _DIRICHLET:
                diff_q2l_down = 2.0 * kappa * ki * (pos_bc + z0b[col, 0])
            else:
                diff_q2l_down = (
                    -2.0
                    * _SQRT2
                    * sl
                    * kappa
                    * kappa
                    * ti.pow(ki, 1.5)
                    * (pos_bc + z0b[col, 0])
                )
        if lbc_type == _INJECTION:
            f_k = _fk_craig(u_taub[col, 0], cw)
            capital_k = ti.pow(
                -f_k / (_SQRT2 * sq * gen_alpha * gen_l),
                2.0 / 3.0,
            ) / ti.pow(z0b[col, 0], gen_alpha)
            if psi_lbc == _DIRICHLET:
                diff_q2l_down = (
                    2.0
                    * capital_k
                    * gen_l
                    * ti.pow(pos_bc + z0b[col, 0], gen_alpha + 1.0)
                )
            else:
                diff_q2l_down = (
                    -2.0
                    * _SQRT2
                    * sl
                    * (gen_alpha + 1.0)
                    * ti.pow(capital_k, 1.5)
                    * gen_l
                    * gen_l
                    * ti.pow(pos_bc + z0b[col, 0], 1.5 * gen_alpha + 1.0)
                )

        diff_face_column(
            col,
            nlev,
            dt,
            _CNPAR,
            h,
            psi_ubc,
            psi_lbc,
            diff_q2l_up,
            diff_q2l_down,
            avh,
            l_sour,
            q_sour,
            q2l,
            au,
            bu,
            cu,
            du,
            ru,
            qu,
        )

        q2l[col, nlev] = 0.0
        if ubc_type == _LOGARITHMIC:
            q2l[col, nlev] = 2.0 * kappa * tke[col, nlev] * (z0s[col, 0] + z0s[col, 0])
        if ubc_type == _INJECTION:
            f_k = _fk_craig(u_taus[col, 0], cw)
            capital_k = ti.pow(
                -f_k / (_SQRT2 * sq * gen_alpha * gen_l),
                2.0 / 3.0,
            ) / ti.pow(z0s[col, 0], gen_alpha)
            q2l[col, nlev] = (
                2.0
                * capital_k
                * gen_l
                * ti.pow(z0s[col, 0] + z0s[col, 0], gen_alpha + 1.0)
            )

        q2l[col, 0] = 0.0
        if lbc_type == _LOGARITHMIC:
            q2l[col, 0] = 2.0 * kappa * tke[col, 0] * (z0b[col, 0] + z0b[col, 0])
        if lbc_type == _INJECTION:
            f_k = _fk_craig(u_taub[col, 0], cw)
            capital_k = ti.pow(
                -f_k / (_SQRT2 * sq * gen_alpha * gen_l),
                2.0 / 3.0,
            ) / ti.pow(z0b[col, 0], gen_alpha)
            q2l[col, 0] = (
                2.0
                * capital_k
                * gen_l
                * ti.pow(z0b[col, 0] + z0b[col, 0], gen_alpha + 1.0)
            )

        for i in range(nlev + 1):
            L[col, i] = q2l[col, i] / (2.0 * tke[col, i])

            if NN[col, i] > 0.0 and length_lim != 0:
                l_crit = ti.sqrt(2.0 * galp * galp * tke[col, i] / NN[col, i])
                if L[col, i] > l_crit:
                    L[col, i] = l_crit

            if L[col, i] < l_min:
                L[col, i] = l_min

            eps[col, i] = (
                cde * ti.sqrt(tke[col, i] * tke[col, i] * tke[col, i]) / L[col, i]
            )

            if eps[col, i] < eps_min:
                eps[col, i] = eps_min
                L[col, i] = (
                    cde * ti.sqrt(tke[col, i] * tke[col, i] * tke[col, i]) / eps_min
                )
