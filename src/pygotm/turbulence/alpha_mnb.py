# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Update dimensionless alpha's\label{sec:alpha}
!
! !INTERFACE:
!   subroutine alpha_mnb(nlev,NN,SS, SSCSTK, SSSTK)
!
! !DESCRIPTION:
! This subroutine updates the dimensionless numbers $\alpha_M$, $\alpha_N$,
! and $\alpha_b$ according to \eq{alphaMN}. Note that according to \eq{Nbar}
! and \eq{NbarVertical} the following identities are valid
! \begin{equation}
!  \label{alphaIdentities}
!    \alpha_M = \overline{S}^2 \comma
!    \alpha_N = \overline{N}^2 \comma
!    \alpha_b = \overline{T}   \point
! \end{equation}
!
!
! !USES:
!  use turbulence,  only:     tke,eps,kb
!  use turbulence,  only:     as,an,at
!  use turbulence,  only:     av, aw
!  IMPLICIT NONE
!
! !INPUT PARAMETERS:
!  integer,  intent(in)      :: nlev
!  REALTYPE, intent(in)      :: NN(0:nlev),SS(0:nlev)
!
!  Stokes-Eulerian cross-shear (1/s^2)
!   REALTYPE, intent(in), optional      :: SSCSTK(0:nlev)
!
!  Stokes shear squared (1/s^2)
!   REALTYPE, intent(in), optional      :: SSSTK (0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!  integer              :: i
!  REALTYPE             :: tau2(0:nlev)
!
!-----------------------------------------------------------------------
!BOC
!
!     clip negative values
!
!        clip negative values
!
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection
from pygotm.taichi_typing import TemplateArg, ti_kernel

__all__ = [
    "AlphaMNBWorkspace",
    "step_alpha_mnb",
]

_MIN_NONNEGATIVE_ALPHA: float = 1.0e-10


class AlphaMNBWorkspace(TaichiFieldCollection):
    """Taichi fields for the translated dimensionless-alpha update."""

    tke: ti.Field
    eps: ti.Field
    kb: ti.Field
    NN: ti.Field
    SS: ti.Field
    SSCSTK: ti.Field
    SSSTK: ti.Field
    as_: ti.Field
    an: ti.Field
    at: ti.Field
    av: ti.Field
    aw: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("tke", "eps", "kb", "NN", "SS", "SSCSTK", "SSSTK"))
        self.allocate_many(("as_", "an", "at", "av", "aw"))


@ti_kernel
def step_alpha_mnb(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    has_sscstk: ti.i32,
    has_ssstk: ti.i32,
    tke: TemplateArg,
    eps: TemplateArg,
    kb: TemplateArg,
    NN: TemplateArg,
    SS: TemplateArg,
    SSCSTK: TemplateArg,
    SSSTK: TemplateArg,
    as_: TemplateArg,
    an: TemplateArg,
    at: TemplateArg,
    av: TemplateArg,
    aw: TemplateArg,
):
    r"""Update ``alpha_M``, ``alpha_N``, ``alpha_b``, and optional Stokes terms."""

    for col in range(n_cols):
        for i in range(nlev + 1):
            tau2 = tke[col, i] * tke[col, i] / (eps[col, i] * eps[col, i])
            as_[col, i] = tau2 * SS[col, i]
            an[col, i] = tau2 * NN[col, i]
            at[col, i] = tke[col, i] / eps[col, i] * kb[col, i] / eps[col, i]

            as_[col, i] = ti.max(as_[col, i], _MIN_NONNEGATIVE_ALPHA)
            at[col, i] = ti.max(at[col, i], _MIN_NONNEGATIVE_ALPHA)

        if has_sscstk != 0 and has_ssstk != 0:
            for i in range(nlev + 1):
                tau2 = tke[col, i] * tke[col, i] / (eps[col, i] * eps[col, i])
                av[col, i] = tau2 * SSCSTK[col, i]
                aw[col, i] = tau2 * SSSTK[col, i]
                aw[col, i] = ti.max(aw[col, i], _MIN_NONNEGATIVE_ALPHA)
