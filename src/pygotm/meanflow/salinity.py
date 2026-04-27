r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The salinity equation \label{sec:salinity}
!
! !INTERFACE:
!   subroutine salinity(nlev,dt,cnpar,wflux,sflux,nus,gams)
!
! !DESCRIPTION:
! This subroutine computes the balance of salinity in the form
!  \begin{equation}
!   \label{SEq}
!    \dot{S}
!    = {\cal D}_S
!    - \frac{1}{\tau^S_R}(S-S_{obs})
!    \comma
!  \end{equation}
!  where $\dot{S}$ denotes the material derivative of the salinity $S$, and
!  ${\cal D}_S$ is the sum of the turbulent and viscous transport
!  terms modelled according to
!  \begin{equation}
!   \label{DS}
!    {\cal D}_S
!    = \frstder{z}
!     \left(
!        \left( \nu^S_t + \nu^S \right) \partder{S}{z} - \tilde{\Gamma}_S
!        \right)
!    \point
!  \end{equation}
!  In this equation, $\nu^S_t$ and $\nu^S$ are the turbulent and
!  molecular diffusivities of salinity, respectively,
!  and $\tilde{\Gamma}_S$
!  denotes the non-local flux of salinity, see
!  \sect{sec:turbulenceIntro}. In the current version of GOTM,
!  we set $\nu^S_t = \nu^\Theta_t$ for simplicity.
!
!  Horizontal advection is optionally
!  included  (see {\tt gotm.yaml}) by means of prescribed
!  horizontal gradients $\partial_xS$ and $\partial_yS$ and
!  calculated horizontal mean velocities $U$ and $V$.
!  Relaxation with the time scale $\tau^S_R$
!  towards a precribed (changing in time)
!  profile $S_{obs}$ is possible.

!  Inner sources or sinks are not considered.
!  The surface freshwater flux is given by means of the precipitation
!  - evaporation data read in as $P-E$ through the {\tt airsea.nml} namelist:
!  \begin{equation}
!     \label{S_sbc}
!    {\cal D}_S =  S (P-E),
!    \qquad \mbox{at } z=\zeta,
!  \end{equation}
!  with $P-E$ given as a velocity (note that ${\cal D}_S$ is the flux in the
!  direction of $z$, and thus positive for a \emph{loss} of salinity) .
!  Diffusion is numerically treated implicitly,
!  see equations (\ref{sigmafirst})-(\ref{sigmalast}).
!  The tri-diagonal matrix is solved then by a simplified Gauss elimination.
!  Vertical advection is included, and it must be non-conservative,
!  which is ensured by setting the local variable {\tt adv\_mode=0},
!  see section \ref{sec:advectionMean} on page \pageref{sec:advectionMean}.
!
! !USES:
!   use meanflow,     only: avmolS
!   use meanflow,     only: h,u,v,w,S,avh
!   use meanflow,     only: Sobs
!   use observations, only: dsdx_input,dsdy_input,s_adv
!   use observations, only: w_adv_discr,w_adv_input
!   use observations, only: sprof_input,SRelaxTau
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
!  precip-evap - or - melt rate under glacial ice (m/s)
!   REALTYPE, intent(in)                :: wflux
!  upward surface salinity flux under glacial ice (psu m/s)
!   REALTYPE, intent(in)                :: sflux
!  diffusivity of salinity (m^2/s)
!   REALTYPE, intent(in)                :: nus(0:nlev)
!  non-local salinity flux (psu m/s)
!   REALTYPE, intent(in)                :: gams(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
!
!BOC
!
!  set boundary conditions
!   DiffBcup       = Neumann
!   DiffBcdw       = Neumann
!   DiffSup        = -sflux
!   DiffSdw        = _ZERO_
!
!   AdvBcup       = oneSided
!   AdvBcdw       = oneSided
!   AdvSup        = _ZERO_
!   AdvSdw        = _ZERO_
!
!  compute total diffusivity
!   do i=0,nlev
!      avh(i)=nus(i)+avmolS
!   end do
!
!  add contributions to source term
!   Lsour=_ZERO_
!   Qsour=_ZERO_
!
!   do i=1,nlev
!     from non-local turbulence
!      Qsour(i) = Qsour(i) - ( gams(i) - gams(i-1) )/h(i)
!   end do
!
!  ... and from lateral advection
!   if (s_adv) then
!      do i=1,nlev
!         Qsour(i) = Qsour(i) - u(i)*dsdx_input%data(i) - v(i)*dsdy_input%data(i)
!      end do
!   end if
!
!  do advection step
!   if (w_adv_input%method .ne. 0) then
!      call adv_center(nlev,dt,h,h,w,AdvBcup,AdvBcdw,
!                          AdvSup,AdvSdw,w_adv_discr,adv_mode,S)
!   end if
!
!  do diffusion step
!   call diff_center(nlev,dt,cnpar,posconc,h,DiffBcup,DiffBcdw,
!                    DiffSup,DiffSdw,avh,LSour,Qsour,SRelaxTau,Sobs,S)
!EOC
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
    "SalinityWorkspace",
    "step_salinity",
    "salinity",
]

# Advection mode 0 = non-conservative (correct for tracers in 1-D model).
_ADV_MODE: int = 0
# posconc=1: salinity is a non-negative concentration → Patankar linearisation.
_POS_CONC: int = 1

# Large relaxation time → no relaxation
_LONG: float = 1.0e15


class SalinityWorkspace(TaichiFieldCollection):
    """Taichi fields needed by the salinity kernel."""

    S: ti.Field
    h: ti.Field
    w: ti.Field
    u: ti.Field
    v: ti.Field
    nus: ti.Field
    avh: ti.Field
    gams: ti.Field
    q_sour: ti.Field
    l_sour: ti.Field
    Sobs: ti.Field
    tau_r: ti.Field
    wflux: ti.Field
    sflux: ti.Field
    dsdx: ti.Field
    dsdy: ti.Field
    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field
    adv_cu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        # prognostic salinity (read/write) and horizontal velocities
        self.allocate_many(("S", "h", "w", "u", "v"))
        # diffusivity inputs and work array
        self.allocate_many(("nus", "avh"))
        # non-local salinity flux
        self.allocate("gams")
        # source term arrays
        self.allocate_many(("q_sour", "l_sour"))
        # observed S profile for relaxation
        self.allocate("Sobs")
        # relaxation time scale
        self.allocate("tau_r")
        # scalar inputs per column (stored at index 0): surface fluxes
        self.allocate_many(("wflux", "sflux"))
        # lateral salinity gradients (optional)
        self.allocate_many(("dsdx", "dsdy"))
        # tridiagonal work arrays (Thomas algorithm)
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))
        # advection flux work array
        self.allocate("adv_cu")


@ti_kernel
def step_salinity(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    avmolS: ti.f64,
    w_adv_active: ti.i32,
    w_adv_discr: ti.i32,
    s_adv: ti.i32,
    S: TemplateArg,
    h: TemplateArg,
    w: TemplateArg,
    u: TemplateArg,
    v: TemplateArg,
    nus: TemplateArg,
    gams: TemplateArg,
    Sobs: TemplateArg,
    tau_r: TemplateArg,
    wflux: TemplateArg,
    sflux: TemplateArg,
    dsdx: TemplateArg,
    dsdy: TemplateArg,
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
    r"""Advance the salinity equation for all columns.

    Outer loop is parallel across columns (GPU threads).  Inner vertical
    loops are serial (required by the Thomas algorithm).

    Boundary conditions mirror salinity.F90:
      - Upper: Neumann — flux = -sflux, with Patankar linearisation when flux < 0
      - Lower: Neumann — flux = 0

    posconc=1: salinity is a non-negative concentration.  When DiffSup < 0
    (salt leaving through surface) the Patankar (1980) trick is applied:
    the negative flux is moved to the main diagonal as an implicit linear
    source, guaranteeing non-negativity.
    """
    for col in range(n_cols):  # parallel across columns
        _ = wflux[col, 0]
        sflx = sflux[col, 0]

        # Surface salinity flux BC: DiffSup = -sflux
        diff_s_up = -sflx

        # --- compute total diffusivity ---
        for k in range(nlev + 1):
            avh[col, k] = nus[col, k] + avmolS

        # --- assemble source terms ---
        for k in range(nlev + 1):
            q_sour[col, k] = 0.0
            l_sour[col, k] = 0.0

        # non-local salinity flux divergence
        for k in range(1, nlev + 1):
            q_sour[col, k] -= (gams[col, k] - gams[col, k - 1]) / h[col, k]

        # optional lateral advection
        if s_adv == 1:
            for k in range(1, nlev + 1):
                q_sour[col, k] -= (
                    u[col, k] * dsdx[col, k] + v[col, k] * dsdy[col, k]
                )

        # --- optional vertical advection step (non-conservative, adv_mode=0) ---
        if w_adv_active == 1:
            adv_center_column(
                col,
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

        # --- Crank-Nicolson diffusion step (mirrors diff_center.F90) ---
        diff_center_column(
            col,
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


def _ensure_workspace(state: MeanflowState, nlev: int) -> SalinityWorkspace:
    if state._kernel_nlev != nlev:
        state._kernel_workspaces.clear()
        state._kernel_nlev = nlev

    workspace = state._kernel_workspaces.get("salinity")
    if workspace is None:
        workspace = SalinityWorkspace(nlev=nlev, n_cols=1)
        state._kernel_workspaces["salinity"] = workspace

    return cast(SalinityWorkspace, workspace)


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
    """Advance the salinity equation for one column via ``step_salinity``."""

    assert state.S is not None
    assert state.h is not None
    assert state.w is not None
    assert state.u is not None
    assert state.v is not None
    assert state.Sobs is not None
    assert state.avh is not None

    n = nlev + 1
    workspace = _ensure_workspace(state, nlev)

    _write_profile(workspace.S, state.S)
    _write_profile(workspace.h, state.h)
    _write_profile(workspace.w, state.w)
    _write_profile(workspace.u, state.u)
    _write_profile(workspace.v, state.v)
    _write_profile(workspace.nus, nus)
    _write_profile(workspace.gams, gams)
    _write_profile(workspace.Sobs, Sobs if Sobs is not None else state.Sobs)
    _write_profile(
        workspace.tau_r,
        tau_r if tau_r is not None else np.full(n, _LONG, dtype=np.float64),
    )
    _write_profile(
        workspace.dsdx,
        dsdx if dsdx is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(
        workspace.dsdy,
        dsdy if dsdy is not None else np.zeros(n, dtype=np.float64),
    )

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

    _write_scalar(workspace.wflux, wflux)
    _write_scalar(workspace.sflux, sflux)

    step_salinity(
        1,
        nlev,
        dt,
        cnpar,
        state.avmolS,
        int(w_adv_active),
        w_adv_discr,
        int(s_adv),
        workspace.S,
        workspace.h,
        workspace.w,
        workspace.u,
        workspace.v,
        workspace.nus,
        workspace.gams,
        workspace.Sobs,
        workspace.tau_r,
        workspace.wflux,
        workspace.sflux,
        workspace.dsdx,
        workspace.dsdy,
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

    state.S[:] = _read_profile(workspace.S)
    state.avh[:] = _read_profile(workspace.avh)
