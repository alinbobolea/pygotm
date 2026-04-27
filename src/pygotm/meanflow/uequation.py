r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The U-momentum equation\label{sec:uequation}
!
! !INTERFACE:
!   subroutine uequation(nlev,dt,cnpar,tx,num,nucl,gamu,ext_method)
!
! !DESCRIPTION:
!  This subroutine computes the transport of momentum in
!  $x$-direction according to
!  \begin{equation}
!   \label{uEq}
!    \dot{U}
!    = {\cal D}_U
!    - g \partder{\zeta}{x} + \int_z^{\zeta} \partder{B}{x} \,dz'
!    - \frac{1}{\tau^U_R}(U-U_{obs})-C_f U \sqrt{U^2+V^2}
!    \comma
!  \end{equation}
!  where $\dot{U}$ denotes the material derivative of $U$, $\zeta$
!  the free surface elevation and $B$ the mean buoyancy defined
!  in  \eq{DefBuoyancy}. ${\cal D}_U$ is the sum of the turbulent
!  and viscous transport terms modelled according to
!  \begin{equation}
!   \label{Du}
!    {\cal D}_U
!    = \frstder{z}
!     \left(
!        \left( \nu_t + \nu \right) \partder{U}{z}
!               - \tilde{\Gamma}_U
!      \right)
!    \point
!  \end{equation}
!  In this equation, $\nu_t$ and $\nu$ are the turbulent and
!  molecular diffusivities of momentum, respectively, and
!  $\tilde{\Gamma}_U$ denotes the non-local flux of momentum,
!  see \sect{sec:turbulenceIntro}.
!
!  Coriolis rotation is accounted for as described in
!  \sect{sec:coriolis}.
!  The external pressure gradient (second term on right hand side)
!  is applied here only if surface slopes are
!  directly given. Otherwise, the gradient is computed as
!   described in \sect{sec:extpressure}, see \cite{Burchard99}.
!  The internal pressure gradient (third
!  term on right hand side) is calculated in {\tt intpressure.F90}, see
!  \sect{sec:intpressure}.
!  The fifth term on the right hand side allows for nudging the velocity
!  to observed profiles with the relaxation time scale $\tau^U_R$.
!  This is useful for initialising
!  velocity profiles in case of significant inertial oscillations.
!  Bottom friction is implemented implicitly using the fourth term
!  on the right hand side. Implicit friction may  be
!  applied on all levels in order to allow for inner friction terms such
!  as seagrass friction (see \sect{sec:seagrass}).
!
!  Diffusion is numerically treated implicitly, see equations \eq{sigmafirst}-
!  \eq{sigmalast}.
!  The tri-diagonal matrix is solved then by a simplified Gauss elimination.
!  Vertical advection is included, and it must be non-conservative,
!  which is ensured by setting the local variable {\tt adv\_mode=0},
!  see section \ref{sec:advectionMean} on page \pageref{sec:advectionMean}.
!
! !USES:
!   use meanflow,     only: gravity,avmolu
!   use meanflow,     only: h,u,uo,v,w,avh
!   use meanflow,     only: drag,SS,runtimeu
!   use stokes_drift, only: dusdz
!   use observations, only: w_adv_input,w_adv_discr
!   use observations, only: uprof_input,vel_relax_tau,vel_relax_ramp
!   use observations, only: int_press_type
!   use observations, only: plume_type
!   use observations, only: idpdx,dpdx_input
!   use util,         only: Dirichlet,Neumann
!   use util,         only: oneSided,zeroDivergence
!
! !INPUT PARAMETERS:
!  number of vertical layers
!   integer, intent(in)                 :: nlev
!  time step (s)
!   REALTYPE, intent(in)                :: dt
!  numerical "implicitness" parameter
!   REALTYPE, intent(in)                :: cnpar
!  wind stress in x-direction divided by rho_0 (m^2/s^2)
!   REALTYPE, intent(in)                :: tx
!  diffusivity of momentum (m^2/s)
!   REALTYPE, intent(in)                :: num(0:nlev)
!  eddy coefficient of momentum flux down the Stokes gradient (m^2/s)
!   REALTYPE, intent(in)                :: nucl(0:nlev)
!  non-local flux of momentum (m^2/s^2)
!   REALTYPE, intent(in)                :: gamu(0:nlev)
!  method to compute external pressure gradient
!   integer, intent(in)                 :: ext_method
!
! !DEFINED PARAMETERS:
!   REALTYPE, parameter                 :: long=1.0D15
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!                      (re-write after first version of
!                       Hans Burchard and Karsten Bolding)
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: adv_mode=0
!   integer                   :: posconc=0
!   integer                   :: i
!   integer                   :: DiffBcup,DiffBcdw
!   integer                   :: AdvBcup,AdvBcdw
!   REALTYPE                  :: DiffUup,DiffUdw
!   REALTYPE                  :: AdvUup,AdvUdw
!   REALTYPE                  :: dzetadx
!   REALTYPE                  :: Lsour(0:nlev)
!   REALTYPE                  :: Qsour(0:nlev)
!   REALTYPE                  :: URelaxTau(0:nlev)
!
!-----------------------------------------------------------------------
!BOC
!  save old value
!   uo = u
!
!  set boundary conditions
!   DiffBcup       = Neumann
!   DiffBcdw       = Neumann
!   DiffUup        = tx
!   DiffUdw        = _ZERO_   ! bottom friction treated as a source term
!
!   AdvBcup        = oneSided
!   AdvBcdw        = oneSided
!   AdvUup         = _ZERO_
!   AdvUdw         = _ZERO_
!
!  set external pressure gradient
!   if (ext_method .eq. 0) then
!      dzetadx = dpdx_input%value
!   else
!      dzetadx = _ZERO_
!   endif
!
!  set vector of relaxation times
!   if (vel_relax_ramp .lt. long) then
!      runtimeu=runtimeu+dt
!      if (runtimeu .lt. vel_relax_ramp) then
!         URelaxTau=vel_relax_tau*vel_relax_ramp/(vel_relax_ramp-runtimeu)
!      else
!         URelaxTau=vel_relax_tau
!      end if
!   else
!      URelaxTau=vel_relax_tau
!   end if
!
!  compute total diffusivity
!   avh=num+avmolu
!
!   do i=1,nlev
!      Qsour(i) = _ZERO_
!      Lsour(i) = _ZERO_
!
!     add external and internal pressure gradients
!      Qsour(i) = Qsour(i) - gravity*dzetadx + idpdx(i)
!
!     add down Stokes gradient fluxes
!      Qsour(i) = Qsour(i) + ( nucl(i  )*dusdz%data(i  )
!                             -nucl(i-1)*dusdz%data(i-1) )/h(i)
!   end do
!
!  implement bottom friction as source term
!   Lsour(1) = - drag(1)/h(1)*sqrt(u(1)*u(1)+v(1)*v(1))
!
!  do advection step
!   if (w_adv_input%method.ne.0) then
!      call adv_center(nlev,dt,h,h,w,AdvBcup,AdvBcdw,
!                          AdvUup,AdvUdw,w_adv_discr,adv_mode,U)
!   end if
!
!  do diffusion step
!   call diff_center(nlev,dt,cnpar,posconc,h,DiffBcup,DiffBcdw,
!                    DiffUup,DiffUdw,avh,Lsour,Qsour,URelaxTau,uprof_input%data,U)
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from typing import cast

import numpy as np
import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.meanflow.meanflow import MeanflowState
from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.util.adv_center import adv_center_column
from pygotm.util.diff_center import diff_center_column
from pygotm.util.util import Neumann as _NEUMANN
from pygotm.util.util import oneSided as _ONE_SIDED

__all__ = [
    "UEquationWorkspace",
    "step_uequation",
    "uequation",
]

# Advection mode 0 = non-conservative (correct for vertical velocity in 1-D model).
# Conservative mode (1) is for settling/rising concentrations; momentum uses 0.
_ADV_MODE: int = 0
_POS_CONC: int = 0  # momentum can be negative; no Patankar source linearisation

# Relaxation threshold: vel_relax_ramp >= long means "no ramp" in the Fortran.
_LONG: float = 1.0e15


class UEquationWorkspace(TaichiFieldCollection):
    """Taichi fields needed by the U-momentum kernel."""

    u: ti.Field
    uo: ti.Field
    v: ti.Field
    h: ti.Field
    w: ti.Field
    drag: ti.Field
    num: ti.Field
    nucl: ti.Field
    avh: ti.Field
    dusdz: ti.Field
    q_sour: ti.Field
    l_sour: ti.Field
    idpdx: ti.Field
    uprof: ti.Field
    tau_r: ti.Field
    tx: ti.Field
    dzetadx: ti.Field
    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field
    adv_cu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        # prognostic velocity (read/write)
        self.allocate_many(("u", "uo", "v", "h", "w", "drag"))
        # diffusivity inputs and work array
        self.allocate_many(("num", "nucl", "avh"))
        # Stokes drift shear in x
        self.allocate("dusdz")
        # source term arrays
        self.allocate_many(("q_sour", "l_sour"))
        # internal pressure gradient (baroclinic, 1:nlev)
        self.allocate("idpdx")
        # observed U profile for relaxation
        self.allocate("uprof")
        # relaxation time scale
        self.allocate("tau_r")
        # surface stress per column
        self.allocate("tx")
        # external pressure gradient (barotropic dζ/dx) per column
        self.allocate("dzetadx")
        # tridiagonal work arrays (Thomas algorithm)
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))
        # advection flux work array (reused per call)
        self.allocate("adv_cu")


@ti_kernel
def step_uequation(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    avmolu: ti.f64,
    gravity: ti.f64,
    ext_method: ti.i32,
    w_adv_active: ti.i32,
    w_adv_discr: ti.i32,
    seagrass_active: ti.i32,
    plume_active: ti.i32,
    u: TemplateArg,
    uo: TemplateArg,
    v: TemplateArg,
    h: TemplateArg,
    w: TemplateArg,
    drag: TemplateArg,
    num: TemplateArg,
    nucl: TemplateArg,
    dusdz: TemplateArg,
    idpdx: TemplateArg,
    uprof: TemplateArg,
    tau_r: TemplateArg,
    tx: TemplateArg,
    dzetadx: TemplateArg,
    avh: TemplateArg,
    q_sour: TemplateArg,
    l_sour: TemplateArg,
    au: TemplateArg,
    bu: TemplateArg,
    cu: TemplateArg,
    du: TemplateArg,
    ru: TemplateArg,
    qu: TemplateArg,
    adv_cu: TemplateArg,
):
    r"""Advance the U-momentum equation for all columns.

    Outer loop is parallel across columns (GPU threads).  The inner physics
    (Thomas algorithm, advection Courant iterations) remain serial within
    each column, which is required by the sequential nature of the Thomas
    algorithm.

    Boundary conditions mirror uequation.F90:
      - Upper: Neumann  — flux = tx (wind stress / rho_0)
      - Lower: Neumann  — flux = 0  (bottom friction enters via Lsour[1])

    Bottom friction is treated implicitly as a linear source term at k=1:
      Lsour[1] = -drag[1]/h[1] * |U_h|

    The non-local gamu term is deliberately excluded: it is commented out in
    the Fortran reference source (uequation.F90 line ~181).
    """
    for col in range(n_cols):  # parallel across columns
        # --- save old velocity ---
        for k in range(nlev + 1):
            uo[col, k] = u[col, k]

        # --- total diffusivity: avh = num + avmolu ---
        for k in range(nlev + 1):
            avh[col, k] = num[col, k] + avmolu

        # --- assemble source terms ---
        # dzetadx is zero when ext_method != 0; otherwise it is the barotropic
        # pressure gradient set by the caller before invoking the kernel.
        dzx = 0.0
        if ext_method == 0:
            dzx = dzetadx[col, 0]  # scalar stored at index 0 for this column

        for k in range(1, nlev + 1):
            q_sour[col, k] = 0.0
            l_sour[col, k] = 0.0
            # external + internal pressure gradients
            q_sour[col, k] += -gravity * dzx + idpdx[col, k]
            # Stokes gradient flux divergence
            q_sour[col, k] += (
                nucl[col, k] * dusdz[col, k] - nucl[col, k - 1] * dusdz[col, k - 1]
            ) / h[col, k]

        if seagrass_active == 1:
            for k in range(1, nlev + 1):
                speed = ti.sqrt(u[col, k] * u[col, k] + v[col, k] * v[col, k])
                l_sour[col, k] = -drag[col, k] / h[col, k] * speed

        # --- bottom friction as implicit linear source at k=1 ---
        speed = ti.sqrt(u[col, 1] * u[col, 1] + v[col, 1] * v[col, 1])
        l_sour[col, 1] = -drag[col, 1] / h[col, 1] * speed

        # --- surface plume friction at k=nlev (int_press_type==2, plume_type==1) ---
        if plume_active == 1:
            speed_top = ti.sqrt(
                u[col, nlev] * u[col, nlev] + v[col, nlev] * v[col, nlev]
            )
            l_sour[col, nlev] = -drag[col, nlev] / h[col, nlev] * speed_top

        # --- optional advection step (non-conservative, adv_mode=0) ---
        if w_adv_active == 1:
            adv_center_column(
                col,
                nlev,
                dt,
                h,
                h,  # ho = h for mean flow (grid not changing within step)
                w,
                _ONE_SIDED,  # AdvBcup
                _ONE_SIDED,  # AdvBcdw
                0.0,  # AdvUup
                0.0,  # AdvUdw
                w_adv_discr,
                _ADV_MODE,
                u,
                adv_cu,
            )

        # --- Crank-Nicolson diffusion step ---
        diff_center_column(
            col,
            nlev,
            dt,
            cnpar,
            _POS_CONC,
            h,
            _NEUMANN,  # DiffBcup
            _NEUMANN,  # DiffBcdw
            tx[col, 0],  # DiffUup = wind stress / rho_0 (scalar at index 0)
            0.0,  # DiffUdw = 0 (bottom flux; friction is implicit in Lsour)
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


def _write_profile(field: ti.Field, values: np.ndarray) -> None:
    data = np.zeros(field.shape, dtype=np.float64)
    if data.ndim == 1:
        data[:] = values
    else:
        data[0, :] = values
    field.from_numpy(data)


def _write_scalar(field: ti.Field, value: float) -> None:
    field.fill(0.0)
    if len(field.shape) == 1:
        field[0] = value
    else:
        field[0, 0] = value


def _read_profile(field: ti.Field) -> np.ndarray:
    data = np.asarray(field.to_numpy(), dtype=np.float64)
    if data.ndim == 1:
        return data.copy()
    return np.asarray(data[0, :], dtype=np.float64).copy()


def _ensure_workspace(state: MeanflowState, nlev: int) -> UEquationWorkspace:
    if state._kernel_nlev != nlev:
        state._kernel_workspaces.clear()
        state._kernel_nlev = nlev

    workspace = state._kernel_workspaces.get("uequation")
    if workspace is None:
        workspace = UEquationWorkspace(nlev=nlev, n_cols=1)
        state._kernel_workspaces["uequation"] = workspace

    return cast(UEquationWorkspace, workspace)


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
    """Advance the U-momentum equation for one column via ``step_uequation``."""

    _ = gamu

    assert state.h is not None
    assert state.u is not None
    assert state.uo is not None
    assert state.v is not None
    assert state.w is not None
    assert state.drag is not None
    assert state.avh is not None

    n = nlev + 1
    workspace = _ensure_workspace(state, nlev)

    if vel_relax_tau is None:
        u_relax_tau = np.full(n, _LONG, dtype=np.float64)
    else:
        u_relax_tau = vel_relax_tau.astype(np.float64, copy=True)

    if vel_relax_ramp < _LONG:
        state.runtimeu += dt
        if state.runtimeu < vel_relax_ramp:
            u_relax_tau *= vel_relax_ramp / (vel_relax_ramp - state.runtimeu)

    _write_profile(workspace.u, state.u)
    _write_profile(workspace.uo, state.uo)
    _write_profile(workspace.v, state.v)
    _write_profile(workspace.h, state.h)
    _write_profile(workspace.w, state.w)
    _write_profile(workspace.drag, state.drag)
    _write_profile(workspace.num, num)
    _write_profile(workspace.nucl, nucl)
    _write_profile(
        workspace.dusdz,
        dusdz if dusdz is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(
        workspace.idpdx,
        idpdx if idpdx is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(
        workspace.uprof,
        uprof if uprof is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(workspace.tau_r, u_relax_tau)

    workspace.avh.fill(0.0)
    workspace.q_sour.fill(0.0)
    workspace.l_sour.fill(0.0)
    workspace.au.fill(0.0)
    workspace.bu.fill(0.0)
    workspace.cu.fill(0.0)
    workspace.du.fill(0.0)
    workspace.ru.fill(0.0)
    workspace.qu.fill(0.0)
    workspace.adv_cu.fill(0.0)

    _write_scalar(workspace.tx, tx)
    _write_scalar(workspace.dzetadx, dpdx)

    step_uequation(
        1,
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
        workspace.u,
        workspace.uo,
        workspace.v,
        workspace.h,
        workspace.w,
        workspace.drag,
        workspace.num,
        workspace.nucl,
        workspace.dusdz,
        workspace.idpdx,
        workspace.uprof,
        workspace.tau_r,
        workspace.tx,
        workspace.dzetadx,
        workspace.avh,
        workspace.q_sour,
        workspace.l_sour,
        workspace.au,
        workspace.bu,
        workspace.cu,
        workspace.du,
        workspace.ru,
        workspace.qu,
        workspace.adv_cu,
    )

    state.u[:] = _read_profile(workspace.u)
    state.uo[:] = _read_profile(workspace.uo)
    state.avh[:] = _read_profile(workspace.avh)
