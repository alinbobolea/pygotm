r"""!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Advection schemes --- grid centers\label{sec:advectionMean}
!
! !INTERFACE:
!
! !DESCRIPTION:
!
!     This subroutine solves a one-dimensional advection equation. There are two
!     options, depending whether the advection should be conservative or not.
!     Conservative advection has to be applied when settling of sediment or
!     rising of phytoplankton is considered. In this case the advection is of
!     the form
!      \begin{equation}
!       \label{Yadvection_cons}
!        \partder{Y}{t} = - \partder{F}{z}
!        \comma
!      \end{equation}
!     where $F=wY$ is the flux caused by the advective velocity, $w$.
!
!     Non-conservative advective transport has to be applied, when the water
!     has a non-zero vertical velocity. In three-dimensional applications,
!     this transport would be conservative, since vertical divergence would be
!     compensated by horizontal convergence and vice versa. However, the
!     key assumption of one-dimensional modelling is horizontal homogeneity,
!     such that we indeed have to apply a vertically non-conservative method,
!     which is of the form
!      \begin{equation}
!       \label{Yadvection_noncons}
!        \partder{Y}{t} = - w\partder{Y}{z}
!                       = - \left(\partder{F}{z} - Y\partder{w}{z} \right).
!      \end{equation}
!
!     The discretized form of \eq{Yadvection_cons} is
!      \begin{equation}
!       \label{advDiscretized_cons}
!       Y_i^{n+1} = Y_i^n
!       - \dfrac{\Delta t}{h_i}
!        \left( F^n_{i} - F^n_{i-1} \right)
!       \comma
!      \end{equation}
!     where the integers $n$ and $i$ correspond to the present time and space
!     level, respectively.
!
!     For the non-conservative form \eq{Yadvection_noncons},
!     an extra term needs to be included:
!      \begin{equation}
!       \label{advDiscretized_noncons}
!       Y_i^{n+1} = Y_i^n
!       - \dfrac{\Delta t}{h_i}
!        \left( F^n_{i} - F^n_{i-1} -Y_i^n \left(w_k-w_{k-1}  \right)\right).
!      \end{equation}
!
!      Which advection method is applied is decided by the flag {\tt mode},
!      which gives conservative advection \eq{advDiscretized_cons}
!      for {\tt mode=1} and
!      non-conservative advection \eq{advDiscretized_noncons} for {\tt mode=0}.
!
!     Fluxes are defined at the grid faces, the variable $Y_i$ is defined at the
!      grid centers. The fluxes are computed in an upstream-biased way,
!      \begin{equation}
!       \label{upstream}
!       F^n_{i} = \dfrac{1}{\Delta t}
!       \int_{z^\text{Face}_{i} - w \Delta t}^{z^\text{Face}_{i}} Y(z') dz'
!       \point
!      \end{equation}
!      For a third-order polynomial approximation of $Y$ (see \cite{Pietrzak98}),
!      these fluxes can be written the in so-called Lax-Wendroff form as
!      \begin{equation}
!       \label{fluxDiscretized}
!        \begin{array}{rcll}
!          F_{i} &=& w_{i} \left(Y_i +  \dfrac{1}{2} \Phi^+_{i}
!          \left(1-\magn{c_{i}} \right) \left( Y_{i+1} - Y_i \right) \right)
!          \quad & \text{for} \quad w_{i} > 0
!          \comma  \\[5mm]
!          F_{i} &=& w_{i} \left(Y_{i+1} +  \dfrac{1}{2} \Phi^-_{i}
!          \left(1-\magn{c_{i}} \right) \left( Y_i - Y_{i+1} \right) \right)
!          \quad & \text{for} \quad w_{i} < 0
!          \comma
!        \end{array}
!      \end{equation}
!      where $c_{i} = 2 w_{i} \Delta t / (h_i+h_{i+1})$ is the Courant number.
!      The factors appearing in \eq{fluxDiscretized} are defined as
!      \begin{equation}
!       \label{phiDiscretized}
!      \Phi^+_{i} =  \alpha_{i} +  \beta_{i}  r^+_{i}
!      \comma
!      \Phi^-_{i} =  \alpha_{i} +  \beta_{i}  r^-_{i}
!      \comma
!      \end{equation}
!      where
!      \begin{equation}
!       \label{alphaDiscretized}
!       \alpha_{i} = \dfrac{1}{2}
!        + \dfrac{1}{6} \left( 1- 2 \magn{c_{i}} \right) \comma
!       \beta_{i} = \dfrac{1}{2}
!        - \dfrac{1}{6} \left( 1- 2 \magn{c_{i}} \right)
!      \point
!      \end{equation}
!      The upstream and downstream slope parameters are
!      \begin{equation}
!       \label{slopeDiscretized}
!       r^+_{i} = \dfrac{Y_i - Y_{i-1}}{Y_{i+1}-Y_{i}}  \comma
!       r^-_{i} = \dfrac{Y_{i+2} - Y_{i+1}}{Y_{i+1}-Y_{i}}
!      \point
!      \end{equation}
!
!      To obtain monotonic and positive schemes also in the presence of strong
!      gradients, so-called slope limiters are aplied for the factors $\Phi^+_{i}$
!      and $\Phi^-_{i}$. The two most obvious cases are
!      the first-order upstream discretisation with $\Phi^+_{i}=\Phi^-_{i}=0$
!      and the Lax-Wendroff scheme with  $\Phi^+_{i}=\Phi^-_{i}=1$.
!      The subroutine {\tt adv\_center.F90} provides six different slope-limiters,
!      all discussed in detail by \cite{Pietrzak98}:
!
!     \begin{itemize}
!      \item first-order upstream ({\tt method=UPSTREAM})
!      \item second-order upstream-biased polynomial scheme ({\tt method=P1},
!            not yet implemented)
!      \item third-order upstream-biased polynomial scheme ({\tt method=P2})
!      \item third-order scheme (TVD) with Superbee limiter ({\tt method=Superbee})
!      \item third-order scheme (TVD) with MUSCL limiter ({\tt method=MUSCL})
!      \item third-order scheme (TVD) with ULTIMATE QUICKEST limiter
!            ({\tt method=P2\_PDM})
!     \end{itemize}
!
!     If during a certain time step the maximum Courant number is larger
!     than one, a split iteration will be carried out which guarantees that the
!     split step Courant numbers are just smaller than 1.
!
!     Several kinds of boundary conditions are implemented for the upper
!     and lower boundaries. They are set by the integer values {\tt Bcup}
!     and {\tt Bcdw}, that have to correspond to the parameters defined
!     in the module {\tt util}, see \sect{sec:utils}. The
!     following choices exist at the moment:
!
!     For the value {\tt flux}, the boundary values {\tt Yup} and {\tt Ydw} are
!     interpreted as specified fluxes at the uppermost and lowest interface.
!     Fluxes into the boundary cells are counted positive by convention.
!     For the value {\tt value}, {\tt Yup} and {\tt Ydw} specify the value
!     of $Y$ at the interfaces, and the flux is computed by multiplying with
!     the (known) speed  at the interface. For the value {\tt oneSided},
!     {\tt Yup} and {\tt Ydw} are ignored and the flux is computed
!     from a one-sided first-order upstream discretisation using the speed
!     at the interface and the value of $Y$ at the center of the boundary cell.
!     For the value {\tt zeroDivergence}, the fluxes into and out of the
!     respective boundary cell are set equal.
!     This corresponds to a zero-gradient formulation, or to zero
!     flux divergence in the boundary cells.
!
!     Be careful that your boundary conditions are mathematically well defined.
!     For example, specifying an inflow into the boundary cell with the
!     speed at the boundary being directed outward does not make sense.
!
! !USES:
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: N
!
!  time step (s)
!   REALTYPE, intent(in)                :: dt
!
!  layer thickness (m)
!   REALTYPE, intent(in)                :: h(0:N)
!
!  old layer thickness (m)
!   REALTYPE, intent(in)                :: ho(0:N)
!
!  vertical advection speed
!   REALTYPE, intent(in)                :: ww(0:N)
!
!  type of upper BC
!   integer,  intent(in)                :: Bcup
!
!  type of lower BC
!   integer,  intent(in)                :: Bcdw
!
!  value of upper BC
!   REALTYPE, intent(in)                :: Yup
!
!  value of lower BC
!   REALTYPE, intent(in)                :: Ydw
!
!  type of advection scheme
!   integer,  intent(in)                :: method
!
!  advection mode (0: non-conservative, 1: conservative)
!   integer,  intent(in)                :: mode
!
! !INPUT/OUTPUT PARAMETERS:
!   REALTYPE, intent(inout)             :: Y(0:N)
!
! !DEFINED PARAMETERS:
!   REALTYPE,     parameter             :: one6th=1.0d0/6.0d0
!   integer,      parameter             :: itmax=100
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                              :: i,k,it
!   REALTYPE                             :: Yu,Yc,Yd
!   REALTYPE                             :: c,cmax
!   REALTYPE                             :: cu(0:N)
!
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: adv_reconstruct -
!
! !INTERFACE:
!
! !USES:
!
! !INPUT PARAMETERS:
!   integer,intent(in)  :: scheme
!   REALTYPE,intent(in) :: cfl,fuu,fu,fd
!
! !DEFINED PARAMETERS:
!   REALTYPE,parameter :: one3rd=_ONE_/3
!   REALTYPE,parameter :: one6th=_ONE_/6
!
! !REVISION HISTORY:
!  Original author(s): Knut Klingbeil
!
!EOP
!
! !LOCAL VARIABLES:
!   REALTYPE           :: ratio,limiter,x,deltaf,deltafu
!
!-----------------------------------------------------------------------
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.util.util import (
    CENTRAL,
    MUSCL,
    P1,
    P2,
    P2_PDM,
    SPLMAX13,
    UPSTREAM,
)
from pygotm.util.util import (
    Superbee as SUPERBEE,
)
from pygotm.util.util import (
    flux as FLUX,
)
from pygotm.util.util import (
    oneSided as ONE_SIDED,
)
from pygotm.util.util import (
    value as VALUE,
)
from pygotm.util.util import (
    zeroDivergence as ZERO_DIVERGENCE,
)

__all__ = [
    "AdvectionWorkspace",
    "CENTRAL",
    "CONSERVATIVE",
    "FLUX",
    "MUSCL",
    "NON_CONSERVATIVE",
    "ONE_SIDED",
    "P1",
    "P2",
    "P2_PDM",
    "SPLMAX13",
    "SUPERBEE",
    "UPSTREAM",
    "VALUE",
    "ZERO_DIVERGENCE",
    "adv_center",
    "adv_center_column",
    "clean_adv_center",
    "init_adv_center",
]


NON_CONSERVATIVE = 0
CONSERVATIVE = 1

_HALF = 0.5
_ONE_THIRD = 1.0 / 3.0
_ONE_SIXTH = 1.0 / 6.0
_ITMAX = 100


class AdvectionWorkspace(TaichiFieldCollection):
    """Allocate GOTM-style face flux work arrays for advection kernels."""

    cu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate("cu")

    def clear(self) -> None:
        """Drop references to the allocated Taichi fields."""

        for name in tuple(self._fields):
            delattr(self, name)
        self._fields.clear()


def init_adv_center(nlev: int, *, n_cols: int | None = None) -> AdvectionWorkspace:
    """Allocate the temporary face-flux array used by `adv_center`."""

    return AdvectionWorkspace(nlev=nlev, n_cols=n_cols)


def clean_adv_center(workspace: AdvectionWorkspace) -> None:
    """Release Python references to the advection work fields."""

    workspace.clear()


@ti.func
def _adv_reconstruct(  # type: ignore[no-untyped-def]
    scheme,
    cfl,
    fuu,
    fu,
    fd,
):
    """Reconstruct the upstream-biased interface value with the GOTM limiter."""

    deltaf = fd - fu
    deltafu = fu - fuu
    result = fu
    limiter = 0.0
    x = 0.0

    if deltaf * deltafu > 0.0:
        ratio = deltafu / deltaf

        if scheme == SUPERBEE:
            limiter = ti.max(
                ti.min(2.0 * ratio, 1.0),
                ti.min(ratio, 2.0),
            )
        elif scheme == P2_PDM:
            x = _ONE_SIXTH * (1.0 - 2.0 * cfl)
            limiter = (0.5 + x) + (0.5 - x) * ratio
            limiter = ti.min(
                2.0 * ratio / (cfl + 1.0e-10),
                ti.min(limiter, 2.0 / (1.0 - cfl)),
            )
        elif scheme == SPLMAX13:
            limiter = ti.min(
                2.0 * ratio,
                ti.min(
                    _ONE_THIRD * ti.max(1.0 + 2.0 * ratio, 2.0 + ratio),
                    2.0,
                ),
            )
        elif scheme == MUSCL:
            limiter = ti.min(
                2.0 * ratio,
                ti.min(_HALF * (1.0 + ratio), 2.0),
            )
        elif scheme == P2:
            x = _ONE_SIXTH * (1.0 - 2.0 * cfl)
            limiter = (0.5 + x) + (0.5 - x) * ratio
        elif scheme == CENTRAL:
            limiter = 1.0 / (1.0 - cfl)
        else:
            # UPSTREAM and P1 (not yet implemented in GOTM) use first-order
            # upstream.
            limiter = 0.0

        result = fu + _HALF * limiter * (1.0 - cfl) * deltaf
    else:
        if scheme == P2:
            x = _ONE_SIXTH * (1.0 - 2.0 * cfl)
            result = fu + _HALF * (1.0 - cfl) * (
                (0.5 + x) * deltaf + (0.5 - x) * deltafu
            )
        elif scheme == CENTRAL:
            result = _HALF * (fu + fd)
        else:
            result = fu  # UPSTREAM and P1 fall back to upstream value

    return result


@ti.func
def adv_center(  # type: ignore[no-untyped-def]
    nlev,
    dt,
    h,
    ho,
    ww,
    bc_up,
    bc_down,
    y_up,
    y_down,
    method,
    mode,
    y,
    cu,
):
    """Advance a single-column tracer on cell centers using GOTM advection."""

    cmax = 0.0
    for level in range(nlev + 1):
        cu[level] = 0.0

    for k in range(1, nlev):
        courant = ti.abs(ww[k]) * dt / (0.5 * (h[k] + h[k + 1]))
        if courant > cmax:
            cmax = courant

    iterations = ti.min(_ITMAX, ti.cast(cmax, ti.i32) + 1)
    iterations_f = ti.cast(iterations, ti.f64)

    for _ in range(iterations):
        for k in range(1, nlev):
            courant = 0.0
            y_upstream = 0.0
            y_central = 0.0
            y_downstream = 0.0
            if ww[k] > 0.0:
                courant = ww[k] / iterations_f * dt / (0.5 * (h[k] + h[k + 1]))
                if k > 1:
                    y_upstream = y[k - 1]
                else:
                    y_upstream = y[k]
                y_central = y[k]
                y_downstream = y[k + 1]
            else:
                courant = -ww[k] / iterations_f * dt / (0.5 * (h[k] + h[k + 1]))
                if k < nlev - 1:
                    y_upstream = y[k + 2]
                else:
                    y_upstream = y[k + 1]
                y_central = y[k + 1]
                y_downstream = y[k]

            reconstructed = _adv_reconstruct(
                method,
                courant,
                y_upstream,
                y_central,
                y_downstream,
            )
            cu[k] = ww[k] * reconstructed

        if bc_up == FLUX:
            cu[nlev] = -y_up
        elif bc_up == VALUE:
            cu[nlev] = ww[nlev] * y_up
        elif bc_up == ONE_SIDED:
            if ww[nlev] >= 0.0:
                cu[nlev] = ww[nlev] * y[nlev]
            else:
                cu[nlev] = 0.0
        else:
            cu[nlev] = cu[nlev - 1]

        if bc_down == FLUX:
            cu[0] = y_down
        elif bc_down == VALUE:
            cu[0] = ww[0] * y_down
        elif bc_down == ONE_SIDED:
            if ww[0] <= 0.0:
                cu[0] = ww[0] * y[1]
            else:
                cu[0] = 0.0
        else:
            cu[0] = cu[1]

        if mode == NON_CONSERVATIVE:
            for k in range(1, nlev + 1):
                y[k] = y[k] - dt / iterations_f * (
                    (cu[k] - cu[k - 1]) / h[k] - y[k] * (ww[k] - ww[k - 1]) / h[k]
                )
        else:
            for k in range(1, nlev + 1):
                y[k] = y[k] - dt / iterations_f * ((cu[k] - cu[k - 1]) / h[k])


@ti.func
def adv_center_column(  # type: ignore[no-untyped-def]
    col,
    nlev,
    dt,
    h,
    ho,
    ww,
    bc_up,
    bc_down,
    y_up,
    y_down,
    method,
    mode,
    y,
    cu,
):
    """Advance one column within a multi-column advection kernel."""

    cmax = 0.0
    for level in range(nlev + 1):
        cu[col, level] = 0.0

    for k in range(1, nlev):
        courant = ti.abs(ww[col, k]) * dt / (0.5 * (h[col, k] + h[col, k + 1]))
        if courant > cmax:
            cmax = courant

    iterations = ti.min(_ITMAX, ti.cast(cmax, ti.i32) + 1)
    iterations_f = ti.cast(iterations, ti.f64)

    for _ in range(iterations):
        for k in range(1, nlev):
            courant = 0.0
            y_upstream = 0.0
            y_central = 0.0
            y_downstream = 0.0
            if ww[col, k] > 0.0:
                courant = (
                    ww[col, k] / iterations_f * dt / (0.5 * (h[col, k] + h[col, k + 1]))
                )
                if k > 1:
                    y_upstream = y[col, k - 1]
                else:
                    y_upstream = y[col, k]
                y_central = y[col, k]
                y_downstream = y[col, k + 1]
            else:
                courant = (
                    -ww[col, k]
                    / iterations_f
                    * dt
                    / (0.5 * (h[col, k] + h[col, k + 1]))
                )
                if k < nlev - 1:
                    y_upstream = y[col, k + 2]
                else:
                    y_upstream = y[col, k + 1]
                y_central = y[col, k + 1]
                y_downstream = y[col, k]

            reconstructed = _adv_reconstruct(
                method,
                courant,
                y_upstream,
                y_central,
                y_downstream,
            )
            cu[col, k] = ww[col, k] * reconstructed

        if bc_up == FLUX:
            cu[col, nlev] = -y_up
        elif bc_up == VALUE:
            cu[col, nlev] = ww[col, nlev] * y_up
        elif bc_up == ONE_SIDED:
            if ww[col, nlev] >= 0.0:
                cu[col, nlev] = ww[col, nlev] * y[col, nlev]
            else:
                cu[col, nlev] = 0.0
        else:
            cu[col, nlev] = cu[col, nlev - 1]

        if bc_down == FLUX:
            cu[col, 0] = y_down
        elif bc_down == VALUE:
            cu[col, 0] = ww[col, 0] * y_down
        elif bc_down == ONE_SIDED:
            if ww[col, 0] <= 0.0:
                cu[col, 0] = ww[col, 0] * y[col, 1]
            else:
                cu[col, 0] = 0.0
        else:
            cu[col, 0] = cu[col, 1]

        if mode == NON_CONSERVATIVE:
            for k in range(1, nlev + 1):
                y[col, k] = y[col, k] - dt / iterations_f * (
                    (cu[col, k] - cu[col, k - 1]) / h[col, k]
                    - y[col, k] * (ww[col, k] - ww[col, k - 1]) / h[col, k]
                )
        else:
            for k in range(1, nlev + 1):
                y[col, k] = y[col, k] - dt / iterations_f * (
                    (cu[col, k] - cu[col, k - 1]) / h[col, k]
                )
