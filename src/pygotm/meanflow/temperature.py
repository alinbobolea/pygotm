# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The temperature equation \label{sec:temperature}
!
! !INTERFACE:
!   subroutine temperature(nlev,dt,cnpar,I_0,wflux,hflux,nuh,gamh,rad)
!
! !DESCRIPTION:
! This subroutine computes the balance of heat in the form
!  \begin{equation}
!   \label{TEq}
!    \dot{\Theta}
!    = {\cal D}_\Theta
!    - \frac{1}{\tau^\Theta_R}(\Theta-\Theta_{obs})
!    + \frac{1}{C_p \rho_0} \partder{I}{z}
!    \comma
!  \end{equation}
!  where $\dot{\Theta}$ denotes the material derivative of the mean  potential
!  temperature $\Theta$, and
!  ${\cal D}_\Theta$ is the sum of the turbulent and viscous transport
!  terms modelled according to
!  \begin{equation}
!   \label{DT}
!    {\cal D}_\Theta
!    = \frstder{z}
!     \left(
!        \left( \nu^\Theta_t + \nu^\Theta \right) \partder{\Theta}{z}
!               - \tilde{\Gamma}_\Theta
!        \right)
!    \point
!  \end{equation}
!  In this equation, $\nu^\Theta_t$ and $\nu^\Theta$ are the turbulent and
!  molecular diffusivities of heat, respectively, and $\tilde{\Gamma}_\Theta$
!  denotes the non-local flux of heat, see \sect{sec:turbulenceIntro}.
!
!  Horizontal advection is optionally
!  included  (see {\tt gotm.yaml}) by means of prescribed
!  horizontal gradients $\partial_x\Theta$ and $\partial_y\Theta$ and
!  calculated horizontal mean velocities $U$ and $V$.
!  Relaxation with the time scale $\tau^\Theta_R$
!  towards a precribed profile $\Theta_{obs}$, changing in time, is possible.
!
!  The sum of latent, sensible, and longwave radiation is treated
!  as a boundary condition. Solar radiation is treated as an inner
!  source, $I(z)$. It is computed according the
!  exponential law (see \cite{PaulsonSimpson77})
!  \begin{equation}
!    \label{Iz}
!    I(z) = I_0 \bigg(Ae^{z/\eta_1}+(1-A)e^{z/\eta_2}\bigg)B(z).
!  \end{equation}
!  The absorbtion coefficients $\eta_1$ and $\eta_2$ depend on the water type
!  and have to be prescribed either by means of choosing a \cite{Jerlov68} class
!  (see \cite{PaulsonSimpson77}) or by reading in a file through the namelist
!  {\tt extinct} in {\tt gotm.yaml}. The damping term due to bioturbidity,
!  $B(z)$ is calculated in the biogeochemical routines, see section
!  \ref{sec:bio-intro}.
!
!  Diffusion is numerically treated implicitly, see equations (\ref{sigmafirst})-
!  (\ref{sigmalast}).
!  The tri-diagonal matrix is solved then by a simplified Gauss elimination.
!  Vertical advection is included, and it must be non-conservative,
!  which is ensured by setting the local variable {\tt adv\_mode=0},
!  see section \ref{sec:advectionMean} on page \pageref{sec:advectionMean}.
!
! !USES:
!   use density,      only: rho0,cp
!   use meanflow,     only: avmolT
!   use meanflow,     only: h,u,v,w,T,S,avh
!   use meanflow,     only: Tobs
!   use meanflow,     only: bioshade
!   use meanflow,     only: Hice
!   use observations, only: dtdx_input,dtdy_input,t_adv
!   use observations, only: w_adv_discr,w_adv_input
!   use observations, only: tprof_input,TRelaxTau
!   use observations, only: A_input,g1_input,g2_input
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
!  surface short waves radiation  (W/m^2)
!   REALTYPE, intent(in)                :: I_0
!  surface water flux (m/s) - e.g. precip-evap or melt rate under ice
!   REALTYPE, intent(in)                :: wflux
!  surface heat flux (W/m^2)  (negative for heat loss)
!   REALTYPE, intent(in)                :: hflux
!  diffusivity of heat (m^2/s)
!   REALTYPE, intent(in)                :: nuh(0:nlev)
!  non-local heat flux (Km/s)
!   REALTYPE, intent(in)                :: gamh(0:nlev)
!
! !OUTPUT PARAMETERS:
!  shortwave radiation profile (W/m^2)
!   REALTYPE                            :: rad(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: adv_mode=0
!   integer                   :: posconc=0
!   integer                   :: i
!   integer                   :: DiffBcup,DiffBcdw
!   integer                   :: AdvBcup,AdvBcdw
!   REALTYPE                  :: DiffTup,DiffTdw
!   REALTYPE                  :: AdvTup,AdvTdw
!   REALTYPE                  :: Lsour(0:nlev)
!   REALTYPE                  :: Qsour(0:nlev)
!   REALTYPE                  :: z
!-----------------------------------------------------------------------
!BOC
!
!  set boundary conditions
!   DiffBcup       = Neumann
!   DiffBcdw       = Neumann
!
!  For the open ocean the surface temperature flux is only the diffusive
!  component.  In case of ice cover, hflux is defined as the sum of a diffusive
!  and advective component.
!   DiffTup = -hflux/(rho0*cp)
!   ! simple sea ice model: surface heat flux switched off for sst < freezing temp
!   if (T(nlev) .le. -0.0575*S(nlev)) then
!       DiffTup    = min(_ZERO_,DiffTup)
!   end if
!   DiffTdw        = _ZERO_
!
!   AdvBcup        = oneSided
!   AdvBcdw        = oneSided
!   AdvTup         = _ZERO_
!   AdvTdw         = _ZERO_
!
!  initalize radiation
!   rad(nlev)  = I_0
!   z          =_ZERO_
!
!   do i=nlev-1,0,-1
!      z=z+h(i+1)
!      ! compute short wave radiation
!      rad(i)=I_0*(A_input%value*exp(-z/g1_input%value)+(1.-A_input%value)*exp(-z/g2_input%value)*bioshade(i+1))
!      ! compute total diffusivity
!      avh(i)=nuh(i)+avmolT
!   end do
!
!  add contributions to source term
!   Lsour=_ZERO_
!   Qsour=_ZERO_
!
!   Qsour(nlev)=(I_0-rad(nlev-1))/(rho0*cp*h(nlev))
!   do i=1,nlev-1
!      ! from radiation
!      Qsour(i) = (rad(i)-rad(i-1))/(rho0*cp*h(i))
!   enddo
!
!   do i=1,nlev
!      ! from non-local turbulence
!      Qsour(i) = Qsour(i) - ( gamh(i) - gamh(i-1) )/h(i)
!   end do
!
!  ... and from lateral advection
!   if (t_adv) then
!      do i=1,nlev
!         Qsour(i) = Qsour(i) - u(i)*dtdx_input%data(i) - v(i)*dtdy_input%data(i)
!      end do
!   end if
!
!  do advection step
!   if (w_adv_input%method.ne.0) then
!      call adv_center(nlev,dt,h,h,w,AdvBcup,AdvBcdw,
!                          AdvTup,AdvTdw,w_adv_discr,adv_mode,T)
!   end if
!
!  do diffusion step
!   call diff_center(nlev,dt,cnpar,posconc,h,DiffBcup,DiffBcdw,
!                    DiffTup,DiffTdw,avh,Lsour,Qsour,TRelaxTau,Tobs,T)
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------

Python note:
The verbatim Fortran comment above says "negative for heat loss". In this
codebase, ``hflux`` follows the atmospheric sign convention documented in
``AGENTS.md``: ``hflux > 0`` means the ocean loses heat, and
``DiffTup = -hflux / (rho0 * cp)`` is applied exactly as in GOTM.
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
    "TemperatureWorkspace",
    "step_temperature",
    "temperature",
]

# Advection mode 0 = non-conservative (correct for tracers in 1-D model).
_ADV_MODE: int = 0
_POS_CONC: int = 0  # temperature can be negative; no Patankar linearisation

# Paulson-Simpson (1977) Jerlov type I defaults (open ocean clear water).
# A: fraction of UV/blue in the surface irradiance [-]
# g1: attenuation length for UV component [m]
# g2: attenuation length for visible component [m]
_A_DEFAULT: float = 0.58
_G1_DEFAULT: float = 0.35
_G2_DEFAULT: float = 23.0

# Freezing temperature slope for seawater [°C / (g/kg)]
# Tf = -0.0575 * S  (approximate linear relation used in GOTM ice model)
_FREEZE_SLOPE: float = 0.0575

# Large relaxation time → no relaxation
_LONG: float = 1.0e15


class TemperatureWorkspace(TaichiFieldCollection):
    """Taichi fields needed by the temperature kernel."""

    T: ti.Field
    S: ti.Field
    h: ti.Field
    w: ti.Field
    u: ti.Field
    v: ti.Field
    nuh: ti.Field
    avh: ti.Field
    gamh: ti.Field
    bioshade: ti.Field
    rad: ti.Field
    q_sour: ti.Field
    l_sour: ti.Field
    Tobs: ti.Field
    tau_r: ti.Field
    I_0: ti.Field
    wflux: ti.Field
    hflux: ti.Field
    dtdx: ti.Field
    dtdy: ti.Field
    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field
    adv_cu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        # prognostic temperature and salinity (read/write)
        self.allocate_many(("T", "S", "h", "w"))
        # velocity for lateral advection
        self.allocate_many(("u", "v"))
        # diffusivity inputs and work array
        self.allocate_many(("nuh", "avh"))
        # non-local heat flux
        self.allocate("gamh")
        # biological shading (1 = no shading)
        self.allocate("bioshade")
        # shortwave radiation profile (output)
        self.allocate("rad")
        # source term arrays
        self.allocate_many(("q_sour", "l_sour"))
        # observed T profile for relaxation
        self.allocate("Tobs")
        # relaxation time scale
        self.allocate("tau_r")
        # scalar inputs per column (stored at index 0)
        self.allocate_many(("I_0", "wflux", "hflux"))
        # lateral temperature gradients (optional)
        self.allocate_many(("dtdx", "dtdy"))
        # tridiagonal work arrays (Thomas algorithm)
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))
        # advection flux work array
        self.allocate("adv_cu")


@ti_kernel
def step_temperature(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    dt: ti.f64,
    cnpar: ti.f64,
    avmolT: ti.f64,
    rho0: ti.f64,
    cp: ti.f64,
    A: ti.f64,
    g1: ti.f64,
    g2: ti.f64,
    w_adv_active: ti.i32,
    w_adv_discr: ti.i32,
    t_adv: ti.i32,
    T: TemplateArg,
    S: TemplateArg,
    h: TemplateArg,
    w: TemplateArg,
    u: TemplateArg,
    v: TemplateArg,
    nuh: TemplateArg,
    gamh: TemplateArg,
    bioshade: TemplateArg,
    rad: TemplateArg,
    Tobs: TemplateArg,
    tau_r: TemplateArg,
    I_0: TemplateArg,
    wflux: TemplateArg,
    hflux: TemplateArg,
    dtdx: TemplateArg,
    dtdy: TemplateArg,
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
    r"""Advance the temperature equation for all columns.

    Outer loop is parallel across columns (GPU threads).  Inner vertical
    loops are serial (required by the Thomas algorithm).

    Boundary conditions mirror temperature.F90:
      - Upper: Neumann — DiffTup = -hflux/(rho0*cp)
        with atmospheric sign convention:
        hflux > 0 = ocean losing heat, hflux < 0 = ocean gaining heat
      - Lower: Neumann — flux = 0

    Solar radiation uses the Paulson-Simpson (1977) two-band exponential:
      I(z) = I_0 * (A*exp(-z/g1) + (1-A)*exp(-z/g2)*bioshade(k))
    """
    for col in range(n_cols):  # parallel across columns
        i_0 = I_0[col, 0]
        hflx = hflux[col, 0]

        # --- surface heat flux BC with simple sea ice correction ---
        diff_t_up = -hflx / (rho0 * cp)
        # simple sea ice model: suppress upward heat flux when SST <= freezing T
        if T[col, nlev] <= -_FREEZE_SLOPE * S[col, nlev]:
            if diff_t_up > 0.0:
                diff_t_up = 0.0

        # --- compute radiation profile and total diffusivity ---
        # Iterate j=0..nlev-1; layer index i = nlev-1-j goes from nlev-1 down to 0.
        # Taichi kernels do not support 3-argument range() (no step parameter).
        rad[col, nlev] = i_0
        z = 0.0
        for j in range(nlev):
            i = nlev - 1 - j
            z += h[col, i + 1]
            rad[col, i] = i_0 * (
                A * ti.exp(-z / g1) + (1.0 - A) * ti.exp(-z / g2) * bioshade[col, i + 1]
            )
            avh[col, i] = nuh[col, i] + avmolT

        # --- assemble source terms (layers 1..nlev) ---
        for k in range(nlev + 1):
            q_sour[col, k] = 0.0
            l_sour[col, k] = 0.0

        # radiation divergence: surface layer absorbs I_0 - rad(nlev-1)
        q_sour[col, nlev] = (i_0 - rad[col, nlev - 1]) / (rho0 * cp * h[col, nlev])
        for k in range(1, nlev):
            q_sour[col, k] = (rad[col, k] - rad[col, k - 1]) / (rho0 * cp * h[col, k])

        # non-local heat flux divergence
        for k in range(1, nlev + 1):
            q_sour[col, k] -= (gamh[col, k] - gamh[col, k - 1]) / h[col, k]

        # optional lateral advection
        if t_adv == 1:
            for k in range(1, nlev + 1):
                q_sour[col, k] -= u[col, k] * dtdx[col, k] + v[col, k] * dtdy[col, k]

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
                T,
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
            _NEUMANN,
            _NEUMANN,
            diff_t_up,
            0.0,
            avh,
            l_sour,
            q_sour,
            tau_r,
            Tobs,
            T,
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


def _ensure_workspace(state: MeanflowState, nlev: int) -> TemperatureWorkspace:
    if state._kernel_nlev != nlev:
        state._kernel_workspaces.clear()
        state._kernel_nlev = nlev

    workspace = state._kernel_workspaces.get("temperature")
    if workspace is None:
        workspace = TemperatureWorkspace(nlev=nlev, n_cols=1)
        state._kernel_workspaces["temperature"] = workspace

    return cast(TemperatureWorkspace, workspace)


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
    """Advance the temperature equation for one column via ``step_temperature``."""

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
    workspace = _ensure_workspace(state, nlev)

    _write_profile(workspace.T, state.T)
    _write_profile(workspace.S, state.S)
    _write_profile(workspace.h, state.h)
    _write_profile(workspace.w, state.w)
    _write_profile(workspace.u, state.u)
    _write_profile(workspace.v, state.v)
    _write_profile(workspace.nuh, nuh)
    _write_profile(workspace.gamh, gamh)
    _write_profile(workspace.bioshade, state.bioshade)
    _write_profile(workspace.Tobs, Tobs if Tobs is not None else state.Tobs)
    _write_profile(
        workspace.tau_r,
        tau_r if tau_r is not None else np.full(n, _LONG, dtype=np.float64),
    )
    _write_profile(
        workspace.dtdx,
        dtdx if dtdx is not None else np.zeros(n, dtype=np.float64),
    )
    _write_profile(
        workspace.dtdy,
        dtdy if dtdy is not None else np.zeros(n, dtype=np.float64),
    )

    workspace.avh.fill(0.0)
    workspace.rad.fill(0.0)
    workspace.q_sour.fill(0.0)
    workspace.l_sour.fill(0.0)
    workspace.au.fill(0.0)
    workspace.bu.fill(0.0)
    workspace.cu.fill(0.0)
    workspace.du.fill(0.0)
    workspace.ru.fill(0.0)
    workspace.qu.fill(0.0)
    workspace.adv_cu.fill(0.0)

    _write_scalar(workspace.I_0, I_0)
    _write_scalar(workspace.wflux, wflux)
    _write_scalar(workspace.hflux, hflux)

    step_temperature(
        1,
        nlev,
        dt,
        cnpar,
        state.avmolT,
        rho0,
        cp,
        A,
        g1,
        g2,
        int(w_adv_active),
        w_adv_discr,
        int(t_adv),
        workspace.T,
        workspace.S,
        workspace.h,
        workspace.w,
        workspace.u,
        workspace.v,
        workspace.nuh,
        workspace.gamh,
        workspace.bioshade,
        workspace.rad,
        workspace.Tobs,
        workspace.tau_r,
        workspace.I_0,
        workspace.wflux,
        workspace.hflux,
        workspace.dtdx,
        workspace.dtdy,
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

    state.T[:] = _read_profile(workspace.T)
    state.rad[:] = _read_profile(workspace.rad)
    state.avh[:] = _read_profile(workspace.avh)
