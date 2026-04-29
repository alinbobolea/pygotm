# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The dynamic epsilon-equation \label{sec:dissipationeq}
!
! !INTERFACE:
!   subroutine omegaeq(nlev,dt,u_taus,u_taub,z0s,z0b,h,NN,SS)
!
! !DESCRIPTION:
! The $k$-$\omega$ model described by \cite{UmlaufEtAl3003} solves
! a transport equation for the inverse turbulence time scale,
! $ \omega = (c_\mu^0)^4 \varepsilon /k$, of the following form:
! \begin{equation}
!   \label{omega}
!   \dot{\omega}
!   =
!   {\cal D}_\omega
!   + \frac{\omega}{k} ( c_{\omega 1} P + c_{\omega 3} G
!                        + c_{\omega x} P_x
!                        + c_{\omega 4} P_s
!                        - c_{\omega 2} \varepsilon )
!   \comma
! \end{equation}
! where $\dot{\omega}$ denotes the material derivative of $\omega$.
! The production terms $P$ and $G$ follow from \eq{PandG}.
! $P_s$ is Stokes shear production defined in \eq{computePs}
! and $P_x$ accounts for extra turbulence production.
! ${\cal D}_\omega$ represents the sum of the viscous and turbulent
! transport terms.
!
! For horizontally homogeneous flows, the transport term ${\cal D}_\omega$
! appearing in \eq{dissipation} is presently expressed by a simple
! gradient formulation,
! \begin{equation}
!   \label{diffusionOmega}
!   {\cal D}_\omega = \frstder{z}
!    \left( \dfrac{\nu_t}{\sigma_\omega} \partder{\omega}{z} \right)
!  \comma
! \end{equation}
! where $\sigma_\omega$ is the constant Schmidt-number for $\omega$.
!
! Model constants are summarized in \tab{tab:KW_constants}. Similar
! to the two-equations models, the model parameter $c_{omega 3}$
! determines the value of the stationory Richardson number. It is
! computed numerically by solving \eq{Ri_st}.
! \begin{table}[ht]
!   \begin{center}
! \begin{tabular}{cccccc}
!     & $c_\mu^0$ & $\sigma_k$  & $\sigma_\omega$
!     & $c_{\omega 1}$ & $c_{\omega 2}$  \\[1mm] \hline
!     \cite{Rodi87} & $0.55$ & $2.0$ &  $2.0$ & $0.56$ & $0.83$ \\
!   \end{tabular}
!   \caption{\label{tab:KW_constants} Constants appearing in
!    \eq{omega} and \eq{diffusionOmega}.}
!   \end{center}
! \end{table}
!
! At the end of this routine the length-scale can be constrained according to a
! suggestion of \cite{Galperinetal88}. This feature is optional and can be activated
! by setting {\tt length\_lim = .true.} in {\tt gotm.yaml}.
!
! !USES:
!   use turbulence, only: P,B,Px,PSTK,num
!   use turbulence, only: tke,tkeo,k_min,eps,eps_min,L
!   use turbulence, only: cw1,cw2,cw3plus,cw3minus,cwx,cw4
!   use turbulence, only: cm0,cde,galp,length_lim
!   use turbulence, only: omega_bc, psi_ubc, psi_lbc, ubc_type, lbc_type
!   use turbulence, only: sig_w
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
!   REALTYPE                  :: DiffOmgUp,DiffOmgDw,pos_bc
!   REALTYPE                  :: prod,buoyan,diss
!   REALTYPE                  :: prod_pos,prod_neg,buoyan_pos,buoyan_neg
!   REALTYPE                  :: ki,epslim,OmgOverTke,NN_pos
!   REALTYPE                  :: cnpar=_ONE_
!   REALTYPE                  :: omega(0:nlev)
!   REALTYPE                  :: avh(0:nlev)
!   REALTYPE                  :: Lsour(0:nlev),Qsour(0:nlev)
!   REALTYPE                  :: cw3
!
!   integer                   :: i
!
!------------------------------------------------------------------------
!BOC
!
!  re-construct omega at "old" timestep
!
!  compute RHS
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
!     recover dissipation rate from k and omega
!
!     clip at eps_min
!
!  limit dissipation rate under stable stratification,
!  see Galperin et al. (1988)
!
!        look for N^2 > 0
!
!        compute limit
!
!        clip at limit
!
!        compute dissipative scale
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import math

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array
from pygotm.turbulence.turbulence import Dirichlet as _DIRICHLET
from pygotm.turbulence.turbulence import Neumann as _NEUMANN
from pygotm.turbulence.turbulence import injection as _INJECTION
from pygotm.turbulence.turbulence import logarithmic as _LOGARITHMIC
from pygotm.util.diff_face import diff_face

__all__ = [
    "OmegaEquationWorkspace",
    "step_omegaeq",
]

_CNPAR: float = 1.0


class OmegaEquationWorkspace(ColumnWorkspace):
    """Workspace arrays for the translated omega equation."""

    tke: np.ndarray
    tkeo: np.ndarray
    eps: np.ndarray
    L: np.ndarray
    h: np.ndarray
    NN: np.ndarray
    SS: np.ndarray
    P: np.ndarray
    B: np.ndarray
    Px: np.ndarray
    PSTK: np.ndarray
    num: np.ndarray
    u_taus: np.ndarray
    u_taub: np.ndarray
    z0s: np.ndarray
    z0b: np.ndarray
    omega: np.ndarray
    avh: np.ndarray
    l_sour: np.ndarray
    q_sour: np.ndarray
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
        self.eps = make_column_array(nlev, n_cols=n_cols)
        self.L = make_column_array(nlev, n_cols=n_cols)
        self.h = make_column_array(nlev, n_cols=n_cols)
        self.NN = make_column_array(nlev, n_cols=n_cols)
        self.SS = make_column_array(nlev, n_cols=n_cols)
        self.P = make_column_array(nlev, n_cols=n_cols)
        self.B = make_column_array(nlev, n_cols=n_cols)
        self.Px = make_column_array(nlev, n_cols=n_cols)
        self.PSTK = make_column_array(nlev, n_cols=n_cols)
        self.num = make_column_array(nlev, n_cols=n_cols)
        self.u_taus = make_column_array(nlev, n_cols=n_cols)
        self.u_taub = make_column_array(nlev, n_cols=n_cols)
        self.z0s = make_column_array(nlev, n_cols=n_cols)
        self.z0b = make_column_array(nlev, n_cols=n_cols)
        self.omega = make_column_array(nlev, n_cols=n_cols)
        self.avh = make_column_array(nlev, n_cols=n_cols)
        self.l_sour = make_column_array(nlev, n_cols=n_cols)
        self.q_sour = make_column_array(nlev, n_cols=n_cols)
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
def _omega_bc_value(
    bc: int,
    type_: int,
    zi: float,
    ki: float,
    z0: float,
    u_tau: float,
    cm0: float,
    kappa: float,
    sig_w: float,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
) -> float:
    value = 0.0

    if type_ == _LOGARITHMIC:
        if bc == _DIRICHLET:
            value = math.sqrt(ki) / (cm0 * kappa * (zi + z0))
        else:
            value = ki / (sig_w * (zi + z0))

    if type_ == _INJECTION:
        f_k = _fk_craig(u_tau, cw)
        capital_k = (
            (-sig_k * f_k / (cmsf * gen_alpha * gen_l)) ** (2.0 / 3.0)
        ) / (z0**gen_alpha)

        if bc == _DIRICHLET:
            value = (
                math.sqrt(capital_k)
                / (cm0 * gen_l)
                * (zi + z0) ** (0.5 * gen_alpha - 1.0)
            )
        else:
            value = (
                -cmsf
                * capital_k
                * (0.5 * gen_alpha - 1.0)
                / (sig_w * cm0)
                * (zi + z0) ** (gen_alpha - 1.0)
            )

    return value


@numba.njit(cache=True)
def _step_omegaeq(
    nlev: int,
    dt: float,
    cw1: float,
    cw2: float,
    cw3plus: float,
    cw3minus: float,
    cwx: float,
    cw4: float,
    sig_w: float,
    cm0: float,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: int,
    eps_min: float,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    L: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    u_taus: float,
    u_taub: float,
    z0s: float,
    z0b: float,
    omega: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic omega-equation for a single column."""

    for i in range(nlev + 1):
        omega[i] = math.sqrt(tkeo[i]) / (cm0 * L[i])
        avh[i] = 0.0
        l_sour[i] = 0.0
        q_sour[i] = 0.0

    for i in range(1, nlev):
        avh[i] = num[i] / sig_w

        cw3 = cw3minus
        if B[i] > 0.0:
            cw3 = cw3plus

        omg_over_tke = omega[i] / tkeo[i]
        prod = omg_over_tke * (cw1 * P[i] + cwx * Px[i] + cw4 * PSTK[i])
        buoyan = cw3 * omg_over_tke * B[i]
        diss = cw2 * omg_over_tke * eps[i]

        if prod + buoyan > 0.0:
            q_sour[i] = prod + buoyan
            l_sour[i] = -diss / omega[i]
        else:
            q_sour[i] = prod
            l_sour[i] = -(diss - buoyan) / omega[i]

    ki = tke[nlev - 1]
    pos_bc = h[nlev]
    if psi_ubc == _NEUMANN:
        pos_bc = 0.5 * h[nlev]
    diff_omega_up = _omega_bc_value(
        psi_ubc,
        ubc_type,
        pos_bc,
        ki,
        z0s,
        u_taus,
        cm0,
        kappa,
        sig_w,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    ki = tke[1]
    pos_bc = h[1]
    if psi_lbc == _NEUMANN:
        pos_bc = 0.5 * h[1]
    diff_omega_down = _omega_bc_value(
        psi_lbc,
        lbc_type,
        pos_bc,
        ki,
        z0b,
        u_taub,
        cm0,
        kappa,
        sig_w,
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
        psi_ubc,
        psi_lbc,
        diff_omega_up,
        diff_omega_down,
        avh,
        l_sour,
        q_sour,
        omega,
        au,
        bu,
        cu,
        du,
        ru,
        qu,
    )

    omega[nlev] = _omega_bc_value(
        _DIRICHLET,
        ubc_type,
        z0s,
        tke[nlev],
        z0s,
        u_taus,
        cm0,
        kappa,
        sig_w,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )
    omega[0] = _omega_bc_value(
        _DIRICHLET,
        lbc_type,
        z0b,
        tke[0],
        z0b,
        u_taub,
        cm0,
        kappa,
        sig_w,
        sig_k,
        cmsf,
        cw,
        gen_alpha,
        gen_l,
    )

    for i in range(nlev + 1):
        eps[i] = cm0**4 * tke[i] * omega[i]
        if eps[i] < eps_min:
            eps[i] = eps_min

    if length_lim != 0:
        for i in range(nlev + 1):
            nn_pos = 0.5 * (NN[i] + abs(NN[i]))
            epslim = cde / math.sqrt(2.0) / galp * tke[i] * math.sqrt(nn_pos)
            if eps[i] < epslim:
                eps[i] = epslim

    for i in range(nlev + 1):
        L[i] = cde * math.sqrt(tke[i] ** 3) / eps[i]


@numba.njit(parallel=True, cache=True)
def step_omegaeq(
    batch_size: int,
    nlev: int,
    dt: float,
    cw1: float,
    cw2: float,
    cw3plus: float,
    cw3minus: float,
    cwx: float,
    cw4: float,
    sig_w: float,
    cm0: float,
    kappa: float,
    cde: float,
    galp: float,
    length_lim: int,
    eps_min: float,
    psi_ubc: int,
    psi_lbc: int,
    ubc_type: int,
    lbc_type: int,
    sig_k: float,
    cmsf: float,
    cw: float,
    gen_alpha: float,
    gen_l: float,
    tke: np.ndarray,
    tkeo: np.ndarray,
    eps: np.ndarray,
    L: np.ndarray,
    h: np.ndarray,
    NN: np.ndarray,
    SS: np.ndarray,
    P: np.ndarray,
    B: np.ndarray,
    Px: np.ndarray,
    PSTK: np.ndarray,
    num: np.ndarray,
    u_taus: np.ndarray,
    u_taub: np.ndarray,
    z0s: np.ndarray,
    z0b: np.ndarray,
    omega: np.ndarray,
    avh: np.ndarray,
    l_sour: np.ndarray,
    q_sour: np.ndarray,
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
) -> None:
    r"""Advance the dynamic omega-equation for one or more columns."""
    for b in numba.prange(batch_size):
        _step_omegaeq(
            nlev,
            dt,
            cw1,
            cw2,
            cw3plus,
            cw3minus,
            cwx,
            cw4,
            sig_w,
            cm0,
            kappa,
            cde,
            galp,
            length_lim,
            eps_min,
            psi_ubc,
            psi_lbc,
            ubc_type,
            lbc_type,
            sig_k,
            cmsf,
            cw,
            gen_alpha,
            gen_l,
            tke[b],
            tkeo[b],
            eps[b],
            L[b],
            h[b],
            NN[b],
            SS[b],
            P[b],
            B[b],
            Px[b],
            PSTK[b],
            num[b],
            u_taus[b, 0],
            u_taub[b, 0],
            z0s[b, 0],
            z0b[b, 0],
            omega[b],
            avh[b],
            l_sour[b],
            q_sour[b],
            au[b],
            bu[b],
            cu[b],
            du[b],
            ru[b],
            qu[b],
        )
