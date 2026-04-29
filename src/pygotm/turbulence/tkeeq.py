# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic k-equation \label{sec:tkeeq}
!
! !INTERFACE:
!   subroutine tkeeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The transport equation for the turbulent kinetic energy, $k$,
! follows immediately from the contraction of the Reynolds-stress
! tensor. In the case of a Boussinesq-fluid, this equation can
! be written as
! \begin{equation}
!   \label{tkeA}
!   \dot{k}
!   =
!   {\cal D}_k +  P + G + P_x + P_s - \epsilon
!   \comma
! \end{equation}
! where $\dot{k}$ denotes the material derivative of $k$. $P$ and $G$ are
! the production of $k$ by mean shear and buoyancy, respectively, and
! $\epsilon$ the rate of dissipation.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! ${\cal D}_k$ represents the sum of
! the viscous and turbulent transport terms.
! For horizontally homogeneous flows, the transport term ${\cal D}_k$
! appearing in \eq{tkeA} is presently expressed by a simple
! gradient formulation,
! \begin{equation}
!   \label{diffusionTKE}
!   {\cal D}_k = \frstder{z} \left( \dfrac{\nu_t}{\sigma_k} \partder{k}{z} \right)
!  \comma
! \end{equation}
! where $\sigma_k$ is the constant Schmidt-number for $k$.
!
! In horizontally homogeneous flows, the shear and the buoyancy
! production, $P$ and $G$, can be written as
! \begin{equation}
!   \label{PandG}
!   \begin{array}{rcl}
!   P &=& - \mean{u'w'} \partder{U}{z} - \mean{v'w'} \partder{V}{z}  \comma \\[3mm]
!   G &=&  \mean{w'b'}                                               \comma
!   \end{array}
! \end{equation}
! see \eq{PG}. Their computation is discussed in \sect{sec:production}.
!
! The rate of dissipation, $\epsilon$, can be either obtained directly
! from its parameterised transport equation as discussed in
! \sect{sec:dissipationeq}, or from any other model yielding
! an appropriate description of the dissipative length-scale, $l$.
! Then, $\epsilon$ follows from the well-known cascading relation
! of turbulence,
! \begin{equation}
!   \label{epsilon}
!   \epsilon = (c_\mu^0)^3 \frac{k^{\frac{3}{2}}}{l}
!   \comma
! \end{equation}
! where $c_\mu^0$ is a constant of the model.
!
! !USES:
!   use turbulence,   only: P,B,Px,PSTK,num
!   use turbulence,   only: tke,tkeo,k_min,eps
!   use turbulence,   only: k_bc, k_ubc, k_lbc, ubc_type, lbc_type
!   use turbulence,   only: sig_k
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
!                     (re-write after first version of
!                      H. Burchard and K. Bolding)
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
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!   tkeo=tke
!
!   do i=1,nlev-1
!
! !     compute diffusivity
!      avh(i) = num(i)/sig_k
!
! !     compute production terms in k-equation
!      prod     = P(i) + Px(i) + PSTK(i)
!      buoyan   = B(i)
!      diss     = eps(i)
!
!
!      if (prod+buoyan.gt.0) then
!         Qsour(i)  = prod+buoyan
!         Lsour(i) = -diss/tke(i)
!      else
!         Qsour(i)  = prod
!         Lsour(i) = -(diss-buoyan)/tke(i)
!      end if
!
!   end do
!
!
! !  position for upper BC
!   if (k_ubc.eq.Neumann) then
! !     flux at center "nlev"
!      pos_bc = 0.5*h(nlev)
!   else
! !     value at face "nlev-1"
!      pos_bc = h(nlev)
!   end if
!
! !  obtain BC for upper boundary of type "ubc_type"
!   DiffKup  = k_bc(k_ubc,ubc_type,pos_bc,z0s,u_taus)
!
!
! !  position for lower BC
!   if (k_lbc.eq.Neumann) then
! !     flux at center "1"
!      pos_bc = 0.5*h(1)
!   else
! !     value at face "1"
!      pos_bc = h(1)
!   end if
!
! !  obtain BC for lower boundary of type "lbc_type"
!   DiffKdw  = k_bc(k_lbc,lbc_type,pos_bc,z0b,u_taub)
!
!
! !  do diffusion step
!   call diff_face(nlev,dt,cnpar,h,k_ubc,k_lbc,                          &
!                  DiffKup,DiffKdw,avh,Lsour,Qsour,tke)
!
!
! !  fill top and bottom value with something nice
! !  (only for output)
!   tke(nlev)  = k_bc(Dirichlet,ubc_type,z0s,z0s,u_taus)
!   tke(0   )  = k_bc(Dirichlet,lbc_type,z0b,z0b,u_taub)
!
! !  clip at k_min
!   do i=0,nlev
!      tke(i) = max(tke(i),k_min)
!   enddo
!
!   return
!   end subroutine tkeeq
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array
from pygotm.turbulence.turbulence import (
    Dirichlet as _DIRICHLET,
)
from pygotm.turbulence.turbulence import (
    Neumann as _NEUMANN,
)
from pygotm.turbulence.turbulence import (
    injection as _INJECTION,
)
from pygotm.turbulence.turbulence import (
    logarithmic as _LOGARITHMIC,
)
from pygotm.util.diff_face import diff_face

__all__ = [
    "TKEEquationWorkspace",
    "step_tkeeq",
]

_CNPAR: float = 1.0


class TKEEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated TKE equation."""

    tke: np.ndarray
    tkeo: np.ndarray
    h: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    PSTK: np.ndarray
    num: np.ndarray
    eps: np.ndarray
    avh: np.ndarray
    l_sour: np.ndarray
    q_sour: np.ndarray
    u_taus: np.ndarray
    u_taub: np.ndarray
    z0s: np.ndarray
    z0b: np.ndarray
    au: np.ndarray
    bu: np.ndarray
    cu: np.ndarray
    du: np.ndarray
    ru: np.ndarray
    qu: np.ndarray

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(nlev, n_cols=n_cols)
        self.tke = make_column_array(nlev, n_cols=n_cols)
        self.tkeo = make_column_array(nlev, n_cols=n_cols)
        self.h = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.PSTK = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.avh = make_column_array(nlev, n_cols=n_cols)
        self.l_sour = make_column_array(nlev, n_cols=n_cols)
        self.q_sour = make_column_array(nlev, n_cols=n_cols)
        self.u_taus = make_column_array(nlev, n_cols=n_cols)
        self.u_taub = make_column_array(nlev, n_cols=n_cols)
        self.z0s = make_column_array(nlev, n_cols=n_cols)
        self.z0b = make_column_array(nlev, n_cols=n_cols)
        self.au = make_column_array(nlev, n_cols=n_cols)
        self.bu = make_column_array(nlev, n_cols=n_cols)
        self.cu = make_column_array(nlev, n_cols=n_cols)
        self.du = make_column_array(nlev, n_cols=n_cols)
        self.ru = make_column_array(nlev, n_cols=n_cols)
        self.qu = make_column_array(nlev, n_cols=n_cols)


@numba.njit(cache=True)
def _fk_craig(u_tau: float, eta: float) -> float:
    return eta * u_tau**3


@numba.njit(cache=True)
def _k_bc_value(
    bc: int,
    type_: int,
    zi: float,
    z0: float,
    u_tau: float,
    cm0: float,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = u_tau**2 / cm0**2
        else:
            value = 0.0

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = (
            (-sig_k * f_k / (cmsf * gen_alpha * gen_l)) ** (2.0 / 3.0)
        ) / (z0**gen_alpha)

        if bc == _DIRICHLET:
            value = capital_k * (zi + z0) ** gen_alpha
        else:
            value = (
                -cmsf
                / sig_k
                * capital_k**1.5
                * gen_alpha
                * gen_l
                * (zi + z0) ** (1.5 * gen_alpha)
            )

    return value


@numba.njit(cache=True)
def _step_tkeeq(
    nlev: int,
    dt: float,
    sig_k: float,
    k_min: float,
    k_ubc: int,
    k_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cm0: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    h: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    eps: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic k-equation for a single column."""

    for i in range(nlev + 1):
        tkeo[i] = tke[i]
        avh[i] = 0.0
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        avh[i] = num[i] / sig_k

        prod = P[i] + Px[i] + PSTK[i]
        buoyan = B[i]
        diss = eps[i]

        if prod + buoyan > 0.0:
            q_sour[i] = prod + buoyan
            l_sour[i] = -diss / tke[i]
        else:
            q_sour[i] = prod
            l_sour[i] = -(diss - buoyan) / tke[i]

    pos_bc = h[nlev]
    if k_ubc == _NEUMANN:
        pos_bc = 0.5 * h[nlev]
    diff_k_up = _k_bc_value(
        k_ubc,
        ubc_type,
        pos_bc,
        z0s,
        u_taus,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    pos_bc = h[1]
    if k_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_k_down = _k_bc_value(
        k_lbc,
        lbc_type,
        pos_bc,
        z0b,
        u_taub,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    diff_face(
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

    tke[nlev] = _k_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        z0s,
        u_taus,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )
    tke[0] = _k_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        z0b,
        u_taub,
        cm0,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        if tke[i] < k_min:
            tke[i] = k_min


@numba.njit(parallel=True, cache=True)
def step_tkeeq(
    batch_size: int,
    nlev: int,
    dt: float,
    sig_k: float,
    k_min: float,
    k_ubc: int,
    k_lbc: int,
    ubc_type: int,
    lbc_type: int,
    cm0: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    h: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    eps: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    u_taus: np.ndarray,
    u_taub: np.ndarray,
    z0s: np.ndarray,
    z0b: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic k-equation for one or more columns."""

    for col in numba.prange(batch_size):
        _step_tkeeq(
            nlev,
            dt,
            sig_k,
            k_min,
            k_ubc,
            k_lbc,
            ubc_type,
            lbc_type,
            cm0,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke[col],
            tkeo[col],
            h[col],
            P[col],
            B[col],
            Px[col],
            PSTK[col],
            num[col],
            eps[col],
            avh[col],
            l_sour[col],
            q_sour[col],
            u_taus[col, 0],
            u_taub[col, 0],
            z0s[col, 0],
            z0b[col, 0],
            au[col],
            bu[col],
            cu[col],
            du[col],
            ru[col],
            qu[col],
        )
