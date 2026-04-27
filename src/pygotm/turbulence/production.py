# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update turbulence production\label{sec:production}
!
! !INTERFACE:
!   subroutine production(nlev,NN,SS,xP, SSCSTK, SSSTK)
!
! !DESCRIPTION:
!  This subroutine calculates the production terms of turbulent kinetic
!  energy as defined in \eq{PandG} and the production of buoyancy
!  variance as defined in \eq{Pbvertical}.
!  The Eulerian shear-production is computed according to
!  \begin{equation}
!    \label{computeP}
!     P = \nu_t (M^2 + \alpha_w N^2) + \nu^S_t S_c^2
!    \comma
!  \end{equation}
!  with the turbulent diffusivity of momentum, $\nu_t$, defined in
!  \eq{nu}. The shear-frequency, $M$, is discretised as described
!  in \sect{sec:shear}.
!   The term multiplied by $\alpha_w$ traces back to
!  a parameterisation of breaking internal waves suggested by
!  \cite{Mellor89}.
!  The turbulent momentum fluxes due to Stokes velocities induce the
!  Stokes-Eulerian cross-shear term
!  $S_c^2 = \frac{\partial u}{\partial z}\frac{\partial u_s}{\partial z} + \frac{\partial v}{\partial z}\frac{\partial v_s}{\partial z}$
!  with corresponding diffusivity $\nu^S_t$, and the additional
!  Stokes shear-production
!  \begin{equation}
!    \label{computePs}
!     P_s = \nu_t S_c^2 + \nu^S_t S_s^2
!  \end{equation}
!  with squared Stokes shear
!  $S_s^2 = \frac{\partial u_s}{\partial z}^2 + \frac{\partial v_s}{\partial z}^2$.
!  $X_P$ is an extra production term, connected for
!  example with turbulence production caused by sea-grass, see
!  \eq{sgProduction} in  \sect{sec:seagrass}. {\tt xP} is an {\tt optional}
!  argument in the FORTRAN code.
!
!  Similarly, according to \eq{PeVertical}, the buoyancy production
!  is computed from the expression
!  \begin{equation}
!   \label{computeG}
!    G=-\nu^B_t N^2 + \tilde{\Gamma}_B
!    \comma
!  \end{equation}
!  with the turbulent diffusivity, $\nu^B_t$, defined in
!  \eq{nu}. The second term in \eq{computeG} represents the non-local
!  buoyancy flux. The buoyancy-frequency, $N$, is discretised as described
!  in \sect{sec:stratification}.
!
!  The production of buoyancy variance by vertical meanflow gradients follows
!  from \eq{PeVertical} and \eq{computeG}
!  \begin{equation}
!   \label{computePb}
!    P_b = -G N^2
!    \point
!  \end{equation}
!  Thus, according to the definition of the potential energy \eq{defkb},
!  the buoyancy production $G$ describes the conversion between turbulent
!  kinetic and potential energy in \eq{tkeA} and \eq{kbeq}, respectively.
!
! !USES:
!   use turbulence, only: P,B,Pb,Px,PSTK
!   use turbulence, only: num,nuh, nucl
!   use turbulence, only: alpha,iw_model
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
!  number of vertical layers
!   integer,  intent(in)                :: nlev
!
!  boyancy frequency squared (1/s^2)
!   REALTYPE, intent(in)                :: NN(0:nlev)
!
!  shear-frequency squared (1/s^2)
!   REALTYPE, intent(in)                :: SS(0:nlev)
!
!  TKE production due to seagrass
!  friction (m^2/s^3)
!   REALTYPE, intent(in), optional      :: xP(0:nlev)
!
!  Stokes-Eulerian cross-shear (1/s^2)
!   REALTYPE, intent(in), optional      :: SSCSTK(0:nlev)
!
!  Stokes shear squared (1/s^2)
!   REALTYPE, intent(in), optional      :: SSSTK (0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding, Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
!
! !LOCAL VARIABLES:
!   REALTYPE                      :: alpha_eff
!   integer                       :: i
!-----------------------------------------------------------------------
!BOC
!  P is -<u'w'>du/dz for q2l production with e1,
!  PSTK is -<u'w'>dus/dz for q2l prodcution with e6,
!  see  (6) in Harcourt2015.
!  -<u'w'> = num*(du/dz) + nucl*(dus/dz), see (7) in Harcourt2015.
!
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = [
    "ProductionWorkspace",
    "step_production",
]


class ProductionWorkspace(TaichiFieldCollection):
    """Taichi fields for the turbulence-production kernel.

    ``Px`` and ``PSTK`` are read-write fields rather than pure outputs because
    the Fortran routine updates them only when the corresponding optional
    arguments are present. Callers that want exact Fortran semantics across
    repeated calls must therefore preserve or reload those arrays.
    """

    NN: ti.Field
    SS: ti.Field
    xP: ti.Field
    SSCSTK: ti.Field
    SSSTK: ti.Field
    num: ti.Field
    nuh: ti.Field
    nucl: ti.Field
    P: ti.Field
    B: ti.Field
    Pb: ti.Field
    Px: ti.Field
    PSTK: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("NN", "SS", "xP", "SSCSTK", "SSSTK"))
        self.allocate_many(("num", "nuh", "nucl"))
        self.allocate_many(("P", "B", "Pb", "Px", "PSTK"))


@ti_kernel
def step_production(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    iw_model: ti.i32,
    alpha: ti.f64,
    has_xP: ti.i32,
    has_sscstk: ti.i32,
    has_ssstk: ti.i32,
    NN: TemplateArg,
    SS: TemplateArg,
    xP: TemplateArg,
    SSCSTK: TemplateArg,
    SSSTK: TemplateArg,
    num: TemplateArg,
    nuh: TemplateArg,
    nucl: TemplateArg,
    P: TemplateArg,
    B: TemplateArg,
    Pb: TemplateArg,
    Px: TemplateArg,
    PSTK: TemplateArg,
):
    r"""Update turbulence production terms for one or more columns.

    Mirrors ``production.F90`` exactly:

    - ``P``, ``B``, and ``Pb`` are always recomputed for every level.
    - ``Px`` is updated only when ``has_xP == 1``.
    - ``PSTK`` is updated only when Stokes optional arguments are present.
    - When ``SSSTK`` is present without ``SSCSTK``, ``PSTK`` is reset to zero
      before adding the Stokes-shear term, matching the Fortran control flow.
    """

    alpha_eff = 0.0
    if iw_model == 1:
        alpha_eff = alpha

    for col in range(n_cols):
        for i in range(nlev + 1):
            P[col, i] = num[col, i] * (SS[col, i] + alpha_eff * NN[col, i])
            B[col, i] = -nuh[col, i] * NN[col, i]
            Pb[col, i] = -B[col, i] * NN[col, i]

            if has_xP != 0:
                Px[col, i] = xP[col, i]

            if has_sscstk != 0:
                P[col, i] = P[col, i] + nucl[col, i] * SSCSTK[col, i]
                PSTK[col, i] = num[col, i] * SSCSTK[col, i]

            if has_ssstk != 0:
                if has_sscstk == 0:
                    PSTK[col, i] = 0.0
                PSTK[col, i] = PSTK[col, i] + nucl[col, i] * SSSTK[col, i]
