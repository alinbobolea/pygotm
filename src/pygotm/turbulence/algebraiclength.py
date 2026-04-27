# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Some algebraic length-scale relations \label{sec:algebraiclength}
!
! !INTERFACE:
!   subroutine algebraiclength(method,nlev,z0b,z0s,depth,h,NN)
!
! !DESCRIPTION:
! This subroutine computes the vertical profile of the turbulent
! scale $l$ from different types of analytical expressions. These
! range from simple geometrical forms to more complicated expressions
! taking into account the effects of stratification and shear. The
! users can select their method in the input file {\tt gotm.yaml}.
! For convenience, we define here $d_b$ and $d_s$ as the distance
! from the bottom and the surface, respectively. The water
! depth is then given by $H=d_b+d_s$, and $z_0^b$ and
! $z_0^s$ are the repective roughness lengths. With these
! abbreviations, the expressions implemented in GOTM are as follows.
! \begin{enumerate}
!  \item The parabolic profile is defined according to
!    \begin{equation}
!      l=\kappa \frac{(d_s+z_0^s) (d_b+z_0^b)}
!                    {d_s+d_b+z_0^b+z_0^s}
!    \comma
!    \end{equation}
!    where it should be noted that only for large water depth
!    this equation converges to $\kappa(z+z_0)$ near the bottom
!   or near the surface.
!  \item The triangular profile is defined according to
!    \begin{equation}
!       l = \kappa \, \min(d_s+z_0^s,d_b+z_0^b)
!    \comma
!    \end{equation}
!    which converges always to $\kappa(z+z_0)$ near the bottom
!   or near the surface.
!  \item A distorted parabola can be constructed by
!     using a slightly modified form of the equation
!     used by \cite{XingDavies95},
!     \begin{equation}
!        l = \kappa \frac{(d_s+z_0^s)(d_b^\text{Xing}+z_0^b)}
!                      {d_s+d_b^\text{Xing}+z_0^s+z_0^b}
!        \comma
!        d_b^\text{Xing} =
!        d_b \exp{\left(-\beta \frac{d_b}{H} \right)}
!    \comma
!    \end{equation}
!    where it should be noted that only for large water depth
!    this equation converges to $\kappa(z+z_0)$ near the bottom
!   or near the surface. The constant $\beta$ is a form parameter
!   determining the distortion of the profile. Currently we use
!   $\beta = 2$ in GOTM.
!  \item A distorted parabola can be constructed by
!     using a slightly modified form of the equation
!     used by \cite{RobertOuellet87},
!    \begin{equation}
!       l = \kappa (d_b+z_0^b)
!           \sqrt{1-\frac{d_b-z_0^s}{H}}
!    \comma
!    \end{equation}
!    where it should be noted that only for large water depth
!    this equation converges to $\kappa(z+z_0)$ near the bottom.
!    Near the surface, the slope of $l$ is always different from
!    the law of the wall, a fact that becomes important when model
!    solutions for the case of breaking waves are computed, see
!    \sect{sec:analyse}.
!  \item Also the famous formula of \cite{Blackadar62} is based on
!     a parabolic shape, extended by an extra length--scale $l_a$.
!    Using the form of \cite{Luytenetal96}, the algebraic relation
!    is expressed by
!     \begin{equation}
!        l = \left( \frac{1}{\kappa (d_s+z_0^s)}
!                  +\frac{1}{\kappa (d_b+z_0^b)}
!                  +\dfrac{1}{l_a} \right)
!    \comma
!     \end{equation}
!    where
!   \begin{equation}
!        l_a = \gamma_0 \frac{\int_{-H}^\eta k^\frac{1}{2} z dz}
!                            {\int_{-H}^\eta k^\frac{1}{2} dz}
! \end{equation}
!    is the natural kinetic energy scale resulting from the
!    first moment of the rms turbulent velocity. The constant
!    $\gamma_0$ usually takes the value $\gamma_0 = 0.2$.
!    It should be noted that this expression for $l$
!    converges to $\kappa(z+z_0)$ at the surface and the bottom
!    only for large water depth, and when $l_a$ plays only a
!    minor role.
! \end{enumerate}
! After the length--scale has been computed, it is optionally
! limited by the method suggested by \cite{Galperinetal88}. This
! option can be activated in {\tt gotm.yaml} by setting
! {\tt length\_lim = .true.} The rate of dissipation is computed
! according to \eq{epsilon}.
!
! !USES:
!   use turbulence, only: L,eps,tke,k_min,eps_min
!   use turbulence, only: cde,galp,kappa,length_lim
!   use turbulence, only: Parabolic,Triangular,Xing_Davies,Robert_Ouellet,Blackadar
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  type of length scale
!   integer,  intent(in)                :: method
!
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
!  surface and bottom roughness (m)
!   REALTYPE, intent(in)                :: z0b,z0s
!
!  local depth (m)
!   REALTYPE, intent(in)                :: depth
!
!  layer thicknesses (m)
!   REALTYPE, intent(in)                :: h(0:nlev)
!
!  buoyancy frequency (1/s^2)
!   REALTYPE, intent(in)                :: NN(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s):  Manuel Ruiz Villarreal, Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!   integer                 :: i
!   REALTYPE                :: ds,db,dbxing
!   REALTYPE                :: beta,gamma,La,int_qz,int_q
!   REALTYPE                :: Lcrit,L_min
!
!-----------------------------------------------------------------------
!BOC
!
! distance from bottom and surface initialised
!
! parabolic shape
!
! triangular shape
!
! modified Xing and Davies (1995)
! modification of parabolic mixing length
!
! modified Robert and Ouellet(1987)
! modification of parabolic mixing length
!
! Blackadar (1962).
! In the form suggested by Luyten et al. (1996) for two boundary layers.
!
!     clip the length-scale at the Galperin et al. (1988) value
!     under stable stratifcitation
!
!     compute the dissipation rate
!
!     substitute minimum value
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel
from pygotm.turbulence.turbulence import Blackadar as _BLACKADAR
from pygotm.turbulence.turbulence import Parabolic as _PARABOLIC
from pygotm.turbulence.turbulence import Robert_Ouellet as _ROBERT_OUELLET
from pygotm.turbulence.turbulence import Triangular as _TRIANGULAR
from pygotm.turbulence.turbulence import Xing_Davies as _XING_DAVIES

__all__ = [
    "AlgebraicLengthWorkspace",
    "step_algebraiclength",
]

_BETA_XING: float = 2.0
_GAMMA_BLACKADAR: float = 0.2


class AlgebraicLengthWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated algebraic length-scale closures."""

    tke: ti.Field
    eps: ti.Field
    L: ti.Field
    h: ti.Field
    NN: ti.Field
    depth: ti.Field
    z0s: ti.Field
    z0b: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "L", "h", "NN"))
        self.allocate_many(("depth", "z0s", "z0b"))


@ti_kernel
def step_algebraiclength(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    method: ti.i32,
    nlev: ti.i32,
    kappa: ti.f64,
    cde: ti.f64,
    galp: ti.f64,
    length_lim: ti.i32,
    eps_min: ti.f64,
    tke: TemplateArg,
    eps: TemplateArg,
    L: TemplateArg,
    h: TemplateArg,
    NN: TemplateArg,
    depth: TemplateArg,
    z0b: TemplateArg,
    z0s: TemplateArg,
):
    r"""Update algebraic mixing-length profiles and dissipation."""

    for col in range(n_cols):
        local_depth = depth[col, 0]
        local_z0b = z0b[col, 0]
        local_z0s = z0s[col, 0]
        db = 0.0
        ds = 0.0

        if method == _PARABOLIC:
            for i in range(1, nlev):
                db = db + h[col, i]
                ds = local_depth - db
                L[col, i] = (
                    kappa
                    * (ds + local_z0s)
                    * (db + local_z0b)
                    / (ds + db + local_z0b + local_z0s)
                )

            L[col, 0] = kappa * local_z0b
            L[col, nlev] = kappa * local_z0s

        elif method == _TRIANGULAR:
            for i in range(1, nlev):
                db = db + h[col, i]
                ds = local_depth - db
                L[col, i] = kappa * ti.min(ds + local_z0s, db + local_z0b)

            L[col, 0] = kappa * local_z0b
            L[col, nlev] = kappa * local_z0s

        elif method == _XING_DAVIES:
            for i in range(1, nlev):
                db = db + h[col, i]
                ds = local_depth - db
                db_xing = db * ti.exp(-_BETA_XING * db / local_depth)
                L[col, i] = (
                    kappa
                    * (ds + local_z0s)
                    * (db_xing + local_z0b)
                    / (ds + db_xing + local_z0s + local_z0b)
                )

            L[col, 0] = kappa * local_z0b
            L[col, nlev] = kappa * local_z0s

        elif method == _ROBERT_OUELLET:
            for i in range(1, nlev):
                db = db + h[col, i]
                ds = local_depth - db
                L[col, i] = (
                    kappa
                    * (db + local_z0b)
                    * ti.sqrt(1.0 - (db - local_z0s) / local_depth)
                )

            L[col, 0] = kappa * local_z0b
            L[col, nlev] = (
                kappa
                * (local_depth + local_z0b)
                * ti.sqrt(local_z0s / local_depth)
            )

        elif method == _BLACKADAR:
            int_qz = 0.0
            int_q = 0.0

            for i in range(1, nlev):
                db = db + h[col, i]
                root_tke = ti.sqrt(tke[col, i])
                int_qz = int_qz + root_tke * (db + local_z0b) * h[col, i]
                int_q = int_q + root_tke * h[col, i]

            la = _GAMMA_BLACKADAR * int_qz / int_q

            db = 0.0
            for i in range(1, nlev):
                db = db + h[col, i]
                ds = local_depth - db
                L[col, i] = 1.0 / (
                    1.0 / (kappa * (ds + local_z0s))
                    + 1.0 / (kappa * (db + local_z0b))
                    + 1.0 / la
                )

            L[col, 0] = kappa * local_z0b
            L[col, nlev] = kappa * local_z0s

        for i in range(nlev + 1):
            if NN[col, i] > 0.0 and length_lim != 0:
                lcrit = ti.sqrt(2.0 * galp * galp * tke[col, i] / NN[col, i])
                if L[col, i] > lcrit:
                    L[col, i] = lcrit

            tke32 = ti.sqrt(tke[col, i] * tke[col, i] * tke[col, i])
            eps[col, i] = cde * tke32 / L[col, i]

            if eps[col, i] < eps_min:
                eps[col, i] = eps_min
                L[col, i] = cde * tke32 / eps_min
