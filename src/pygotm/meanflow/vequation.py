r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The V-momentum equation\label{sec:vequation}
!
! !INTERFACE:
!   subroutine vequation(nlev,dt,cnpar,ty,num, nucl, gamv,ext_method)
!
! !DESCRIPTION:
!  This subroutine computes the transport of momentum in
!  $y$-direction according to
!  \begin{equation}
!   \label{vEq}
!    \dot{V}
!    = {\cal D}_V
!    - g \partder{\zeta}{y} + \int_z^{\zeta} \partder{B}{y} \,dz'
!    - \frac{1}{\tau^V_R}(V-V_{obs})-C_f V \sqrt{U^2+V^2}
!    \comma
!  \end{equation}
!  where $\dot{V}$ denotes the material derivative of $V$, $\zeta$
!  the free surface elevation and $B$ the mean buoyancy defined
!  in  \eq{DefBuoyancy}. ${\cal D}_V$ is the sum of the turbulent
!  and viscous transport terms modelled according to
!  \begin{equation}
!   \label{Dv}
!    {\cal D}_V
!    = \frstder{z}
!     \left(
!        \left( \nu_t + \nu \right) \partder{V}{z}
!               - \tilde{\Gamma}_V
!      \right)
!    \point
!  \end{equation}
!  In this equation, $\nu_t$ and $\nu$ are the turbulent and
!  molecular diffusivities of momentum, respectively, and
!  $\tilde{\Gamma}_V$ denotes the non-local flux of momentum,
!  see \sect{sec:turbulenceIntro}.
!
!  Coriolis rotation is accounted for as described in
!  \sect{sec:coriolis}. All other terms are completely analogous
!  to those described in \sect{sec:uequation}.
!
! !USES:
!   use meanflow,     only: gravity,avmolu
!   use meanflow,     only: h,v,vo,u,w,avh
!   use meanflow,     only: drag,SS,runtimev
!   use stokes_drift, only: dvsdz
!   use observations, only: w_adv_input,w_adv_discr
!   use observations, only: vprof_input,vel_relax_tau,vel_relax_ramp
!   use observations, only: int_press_type
!   use observations, only: plume_type
!   use observations, only: idpdy,dpdy_input
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
!  wind stress in y-direction
!  divided by rho_0 (m^2/s^2)
!   REALTYPE, intent(in)                :: ty
!  diffusivity of momentum (m^2/s)
!   REALTYPE, intent(in)                :: num(0:nlev)
!  eddy coefficient of momentum flux down the Stokes gradient (m^2/s)
!   REALTYPE, intent(in)                :: nucl(0:nlev)
!  non-local flux of momentum (m^2/s^2)
!   REALTYPE, intent(in)                :: gamv(0:nlev)
!  method to compute external
!  pressure gradient
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
!   REALTYPE                  :: DiffVup,DiffVdw
!   REALTYPE                  :: AdvVup,AdvVdw
!   REALTYPE                  :: dzetady
!   REALTYPE                  :: Lsour(0:nlev)
!   REALTYPE                  :: Qsour(0:nlev)
!   REALTYPE                  :: VRelaxTau(0:nlev)
!
!-----------------------------------------------------------------------
!BOC
!  save old value
!   vo = v
!
!  set boundary conditions
!   DiffBcup       = Neumann
!   DiffBcdw       = Neumann
!   DiffVup        = ty
!   DiffVdw        = _ZERO_   ! bottom friction treated as a source term
!
!   AdvBcup        = oneSided
!   AdvBcdw        = oneSided
!   AdvVup         = _ZERO_
!   AdvVdw         = _ZERO_
!
!  set external pressure gradient
!   if (ext_method .eq. 0) then
!      dzetady = dpdy_input%value
!   else
!      dzetady = _ZERO_
!   endif
!
!  set vector of relaxation times
!   if (vel_relax_ramp .lt. long) then
!      runtimev=runtimev+dt
!      if (runtimev .lt. vel_relax_ramp) then
!         VRelaxTau=vel_relax_tau*vel_relax_ramp/(vel_relax_ramp-runtimev)
!      else
!         VRelaxTau=vel_relax_tau
!      end if
!   else
!      VRelaxTau=vel_relax_tau
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
!      Qsour(i) = Qsour(i) - gravity*dzetady + idpdy(i)
!
!     add down Stokes gradient fluxes
!      Qsour(i) = Qsour(i) + ( nucl(i  )*dvsdz%data(i  )
!                             -nucl(i-1)*dvsdz%data(i-1) )/h(i)
!
!   end do
!
!  implement bottom friction as source term
!   Lsour(1) = - drag(1)/h(1)*sqrt(u(1)*u(1)+v(1)*v(1))
!
!  for surface plumes implement surface friction as source term
!   if (int_press_type == 2 .and. plume_type .eq. 1) then
!      Lsour(nlev) = - drag(nlev)/h(nlev)*sqrt(u(nlev)*u(nlev)+v(nlev)*v(nlev))
!   end if
!
!  do advection step
!   if (w_adv_input%method.ne.0) then
!      call adv_center(nlev,dt,h,h,w,AdvBcup,AdvBcdw,
!                      AdvVup,AdvVdw,w_adv_discr,adv_mode,V)
!   end if
!
!  do diffusion step
!   call diff_center(nlev,dt,cnpar,posconc,h,DiffBcup,DiffBcdw,
!                    DiffVup,DiffVdw,avh,Lsour,Qsour,VRelaxTau,vprof_input%data,V)
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
    "VEquationWorkspace",
    "step_vequation",
    "vequation",
]

# Advection mode 0 = non-conservative (correct for vertical velocity in 1-D model).
_ADV_MODE: int = 0
_POS_CONC: int = 0  # momentum can be negative; no Patankar source linearisation

# Relaxation threshold: vel_relax_ramp >= long means "no ramp" in the Fortran.
_LONG: float = 1.0e15


class VEquationWorkspace(TaichiFieldCollection):
    """Taichi fields needed by the V-momentum kernel."""

    v: ti.Field
    vo: ti.Field
    u: ti.Field
    h: ti.Field
    w: ti.Field
    drag: ti.Field
    num: ti.Field
    nucl: ti.Field
    avh: ti.Field
    dvsdz: ti.Field
    q_sour: ti.Field
    l_sour: ti.Field
    idpdy: ti.Field
    vprof: ti.Field
    tau_r: ti.Field
    ty: ti.Field
    dzetady: ti.Field
    av: ti.Field
    bv: ti.Field
    cv: ti.Field
    dv: ti.Field
    rv: ti.Field
    qv: ti.Field
    adv_cv: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        # prognostic velocity (read/write)
        self.allocate_many(("v", "vo", "u", "h", "w", "drag"))
        # diffusivity inputs and work array
        self.allocate_many(("num", "nucl", "avh"))
        # Stokes drift shear in y
        self.allocate("dvsdz")
        # source term arrays
        self.allocate_many(("q_sour", "l_sour"))
        # internal pressure gradient (baroclinic, 1:nlev)
        self.allocate("idpdy")
        # observed V profile for relaxation
        self.allocate("vprof")
        # relaxation time scale
        self.allocate("tau_r")
        # surface stress per column
        self.allocate("ty")
        # external pressure gradient (barotropic dζ/dy) per column
        self.allocate("dzetady")
        # tridiagonal work arrays (Thomas algorithm)
        self.allocate_many(("av", "bv", "cv", "dv", "rv", "qv"))
        # advection flux work array (reused per call)
        self.allocate("adv_cv")


@ti_kernel
def step_vequation(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    avmolu: ti.f64,
    gravity: ti.f64,
    ext_method: ti.i32,
    w_adv_active: ti.i32,
    w_adv_discr: ti.i32,
    plume_active: ti.i32,
    v: TemplateArg,
    vo: TemplateArg,
    u: TemplateArg,
    h: TemplateArg,
    w: TemplateArg,
    drag: TemplateArg,
    num: TemplateArg,
    nucl: TemplateArg,
    dvsdz: TemplateArg,
    idpdy: TemplateArg,
    vprof: TemplateArg,
    tau_r: TemplateArg,
    ty: TemplateArg,
    dzetady: TemplateArg,
    avh: TemplateArg,
    q_sour: TemplateArg,
    l_sour: TemplateArg,
    av: TemplateArg,
    bv: TemplateArg,
    cv: TemplateArg,
    dv: TemplateArg,
    rv: TemplateArg,
    qv: TemplateArg,
    adv_cv: TemplateArg,
):
    r"""Advance the V-momentum equation for all columns.

    Outer loop is parallel across columns (GPU threads).  The inner physics
    (Thomas algorithm, advection Courant iterations) remain serial within
    each column, which is required by the sequential nature of the Thomas
    algorithm.

    Boundary conditions mirror vequation.F90:
      - Upper: Neumann  — flux = ty (wind stress / rho_0 in y)
      - Lower: Neumann  — flux = 0  (bottom friction enters via Lsour[1])

    Bottom friction is treated implicitly as a linear source term at k=1:
      Lsour[1] = -drag[1]/h[1] * sqrt(U[1]^2 + V[1]^2)

    Surface plume friction (int_press_type==2, plume_type==1) adds an
    analogous implicit source at k=nlev when plume_active==1.

    The non-local gamv term is commented out in the Fortran reference source
    (vequation.F90) and is therefore not implemented here.
    """
    for col in range(n_cols):  # parallel across columns
        # --- save old velocity ---
        for k in range(nlev + 1):
            vo[col, k] = v[col, k]

        # --- total diffusivity: avh = num + avmolu ---
        for k in range(nlev + 1):
            avh[col, k] = num[col, k] + avmolu

        # --- assemble source terms ---
        dzy = 0.0
        if ext_method == 0:
            dzy = dzetady[col, 0]  # scalar stored at index 0 for this column

        for k in range(1, nlev + 1):
            q_sour[col, k] = 0.0
            l_sour[col, k] = 0.0
            # external + internal pressure gradients
            q_sour[col, k] += -gravity * dzy + idpdy[col, k]
            # Stokes gradient flux divergence
            q_sour[col, k] += (
                nucl[col, k] * dvsdz[col, k] - nucl[col, k - 1] * dvsdz[col, k - 1]
            ) / h[col, k]

        # --- bottom friction as implicit linear source at k=1 ---
        speed1 = ti.sqrt(u[col, 1] * u[col, 1] + v[col, 1] * v[col, 1])
        l_sour[col, 1] = -drag[col, 1] / h[col, 1] * speed1

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
                0.0,  # AdvVup
                0.0,  # AdvVdw
                w_adv_discr,
                _ADV_MODE,
                v,
                adv_cv,
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
            ty[col, 0],  # DiffVup = wind stress / rho_0 (scalar at index 0)
            0.0,  # DiffVdw = 0 (bottom flux; friction is implicit in Lsour)
            avh,
            l_sour,
            q_sour,
            tau_r,
            vprof,
            v,
            av,
            bv,
            cv,
            dv,
            rv,
            qv,
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


def _ensure_workspace(state: MeanflowState, nlev: int) -> VEquationWorkspace:
    if state._kernel_nlev != nlev:
        state._kernel_workspaces.clear()
        state._kernel_nlev = nlev

    workspace = state._kernel_workspaces.get("vequation")
    if workspace is None:
        workspace = VEquationWorkspace(nlev=nlev, n_cols=1)
        state._kernel_workspaces["vequation"] = workspace

    return cast(VEquationWorkspace, workspace)


def vequation(
    state: MeanflowState,
    nlev: int,
    dt: float,
    cnpar: float,
    ty: float,
    num: np.ndarray,
    nucl: np.ndarray,
    gamv: np.ndarray,
    ext_method: int = 0,
    dpdy: float = 0.0,
    idpdy: np.ndarray | None = None,
    dvsdz: np.ndarray | None = None,
    w_adv_active: bool = False,
    w_adv_discr: int = 4,
    vel_relax_tau: np.ndarray | None = None,
    vel_relax_ramp: float = _LONG,
    vprof: np.ndarray | None = None,
    plume_active: bool = False,
) -> None:
    """Advance the V-momentum equation for one column via ``step_vequation``.

    Parameters
    ----------
    state:
        MeanflowState carrying h, v, vo, u, w, drag, avmolu, gravity,
        runtimev.  ``state.v`` and ``state.avh`` are updated in-place.
    nlev:
        Number of model layers.
    dt:
        Time step [s].
    cnpar:
        Crank-Nicolson implicitness parameter (0 = explicit, 1 = implicit,
        0.5 = Crank-Nicolson).
    ty:
        Wind stress in y divided by rho_0 [m²/s²] — Neumann upper BC.
    num:
        Turbulent momentum diffusivity at interfaces, shape (nlev+1,) [m²/s].
    nucl:
        Langmuir-cell Stokes-gradient diffusivity, shape (nlev+1,) [m²/s].
    gamv:
        Non-local momentum flux, shape (nlev+1,) [m²/s²].  Passed for
        interface compatibility; the Fortran source comments this term out.
    ext_method:
        0 → use *dpdy* as the barotropic pressure gradient; else → zero.
    dpdy:
        External (barotropic) surface pressure gradient [m/s²] applied via
        Qsour when ``ext_method == 0``.
    idpdy:
        Internal (baroclinic) pressure gradient at cell centres 1..nlev
        [m/s²].  Defaults to zeros.
    dvsdz:
        Stokes drift shear in y at interfaces, shape (nlev+1,) [s⁻¹].
        Defaults to zeros.
    w_adv_active:
        Not implemented in the numpy path; advection is handled by the
        multi-column kernel.  Included for interface symmetry.
    w_adv_discr:
        Advection scheme selector (unused in numpy path).
    vel_relax_tau:
        Relaxation time scale profile, shape (nlev+1,) [s].  Defaults to
        ``_LONG`` (no relaxation).
    vel_relax_ramp:
        Ramp duration [s]; when >= ``_LONG`` no ramp is applied.
    vprof:
        Observed V profile for relaxation, shape (nlev+1,) [m/s].  Defaults
        to zeros.
    plume_active:
        When True, adds surface plume friction at k=nlev (mirrors the Fortran
        condition ``int_press_type == 2 .and. plume_type == 1``).
    """
    _ = gamv

    assert state.h is not None
    assert state.v is not None
    assert state.vo is not None
    assert state.u is not None
    assert state.w is not None
    assert state.drag is not None
    assert state.avh is not None

    n = nlev + 1
    workspace = _ensure_workspace(state, nlev)

    if vel_relax_tau is None:
        v_relax_tau = np.full(n, _LONG, dtype=np.float64)
    else:
        v_relax_tau = vel_relax_tau.astype(np.float64, copy=True)

    if vel_relax_ramp < _LONG:
        state.runtimev += dt
        if state.runtimev < vel_relax_ramp:
            v_relax_tau *= vel_relax_ramp / (vel_relax_ramp - state.runtimev)

    _write_profile(workspace.v, state.v)
    _write_profile(workspace.vo, state.vo)
    _write_profile(workspace.u, state.u)
    _write_profile(workspace.h, state.h)
    _write_profile(workspace.w, state.w)
    _write_profile(workspace.drag, state.drag)
    _write_profile(workspace.num, num)
    _write_profile(workspace.nucl, nucl)
    _write_profile(
        workspace.dvsdz,
        dvsdz if dvsdz is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(
        workspace.idpdy,
        idpdy if idpdy is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(
        workspace.vprof,
        vprof if vprof is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(workspace.tau_r, v_relax_tau)

    workspace.avh.fill(0.0)
    workspace.q_sour.fill(0.0)
    workspace.l_sour.fill(0.0)
    workspace.av.fill(0.0)
    workspace.bv.fill(0.0)
    workspace.cv.fill(0.0)
    workspace.dv.fill(0.0)
    workspace.rv.fill(0.0)
    workspace.qv.fill(0.0)
    workspace.adv_cv.fill(0.0)

    _write_scalar(workspace.ty, ty)
    _write_scalar(workspace.dzetady, dpdy)

    step_vequation(
        1,
        nlev,
        dt,
        cnpar,
        state.avmolu,
        state.gravity,
        ext_method,
        int(w_adv_active),
        w_adv_discr,
        int(plume_active),
        workspace.v,
        workspace.vo,
        workspace.u,
        workspace.h,
        workspace.w,
        workspace.drag,
        workspace.num,
        workspace.nucl,
        workspace.dvsdz,
        workspace.idpdy,
        workspace.vprof,
        workspace.tau_r,
        workspace.ty,
        workspace.dzetady,
        workspace.avh,
        workspace.q_sour,
        workspace.l_sour,
        workspace.av,
        workspace.bv,
        workspace.cv,
        workspace.dv,
        workspace.rv,
        workspace.qv,
        workspace.adv_cv,
    )

    state.v[:] = _read_profile(workspace.v)
    state.vo[:] = _read_profile(workspace.vo)
    state.avh[:] = _read_profile(workspace.avh)
