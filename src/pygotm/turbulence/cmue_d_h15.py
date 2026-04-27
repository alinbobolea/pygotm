# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!
! !ROUTINE: The Langmuir turbulence quasi-equilibrium stability functions  after Harcourt (2015)\label{sec:cmueDH15}
!
! !INTERFACE:
!   subroutine cmue_d_h15(nlev)
!
! !DESCRIPTION:
!
!  Old Description from GTOM: This subroutine updates the explicit solution of
!  \eq{bijVertical} and \eq{giVertical} under the same assumptions
!  as those discussed in \sect{sec:cmueC}. Now, however, an additional
!  equilibrium assumption is invoked. With the help of \eq{PeVertical},
!  one can write the equilibrium condition for the TKE as
! \begin{equation}
!  \label{quasiEquilibrium}
!     \dfrac{P+G}{\epsilon} =
!    \hat{c}_\mu(\alpha_M,\alpha_N) \alpha_M
!    - \hat{c}'_\mu(\alpha_M,\alpha_N) \alpha_N = 1
!   \comma
! \end{equation}
! where \eq{alphaIdentities} has been used. This is an implicit relation
! to determine $\alpha_M$ as a function of $\alpha_N$.
! With the definitions given in \sect{sec:cmueC}, it turns out that
! $\alpha_M(\alpha_N)$ is a quadratic polynomial that is easily solved.
! The resulting value for $\alpha_M$ is substituted into the stability
! functions described in \sect{sec:cmueC}. For negative $\alpha_N$
! (convection) the shear number $\alpha_M$ computed in this way may
! become negative. The value of $\alpha_N$ is limited such that this
! does not happen, see \cite{UmlaufBurchard2005a}.
! This Langmuir turbulence version includes the CL Vortex force in
! the algebraic models as well as in the vortex production of TKE and L or epsilon
!
! !USES:
!   use turbulence, only: an,as,at
! ! nondimensional forcing functions for Eulerian shear dot Stokes shear, and Stokes shear squared:
! ! Also, surface proximity function SPF=(1-fzs), goes to zero at surface as tanh(0.25*z/l_S) where l_S
! ! the vortex-production-weighted dissipation length scale
!   use turbulence, only: av, aw, SPF
!   use turbulence, only: tke, L
!   use turbulence, only: cmue1,cmue2
!   use turbulence, only: cmue3
!   use turbulence, only: sq, sl, sq_var, sl_var
!   use turbulence, only: length_lim
!
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!
! !  number of vertical layers
!   integer, intent(in)       :: nlev
!
! !DEFINED PARAMETERS:
!   REALTYPE, parameter       :: small       = 1.0D-8
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!  Converted by Ramsey Harcourt, last updated 31 July 2018.  This version uses the Harcourt(2015)
!  stability functions from the quasi-equilibrium Second Moment closure (SMC) with Craik-Leibovich
!  terms, but it has been further modified by replacing the crude limiters applied
!  individually to Gh, Gv and Gs in Harcourt(2015) under unstable/positive production conditions
!  with a combinations of limitations on (L/q)^2 applied consistently across Gm, Gs, Gh, Gv,
!  as a function of Gh and Gv input to the ARSM. This ARSM also applies the Galperin limit to
!  L going into the ARSM (algebraic) diagnosis of Sm, Sh, Ss, regardless of whether it/s
!  being enforced within the dyanamic model.
!
!  Recomend running with e3=5 & length.lim=false, but e3=1.2 & length.lim=true is similar
!  When length.lim=false, length scale or at least L/q is still limited within the ARSM for
!  Stability fcns cmue1,cmue2,cmue3. length.lim=false allows the elevated length scale within
!  the mixed layer to impact the transition zone, while restraining the Stability functions
!  to the stability-limited length scale regime.
!
!EOP
!-----------------------------------------------------------------------
! !LOCAL VARIABLES:
!
!   integer, parameter :: rk = kind(_ONE_)
!     integer                 ::   i
!     REALTYPE            ::   Gv, Gs, Gh, Gm
!     REALTYPE            ::   Sm, Ss, Sh
!
!     REALTYPE, parameter :: my_A1 = 0.92D0
!     REALTYPE, parameter :: my_A2 = 0.74D0
!     REALTYPE, parameter :: my_B1 = 16.6D0
!     REALTYPE, parameter :: my_B2 = 10.1D0
!     REALTYPE, parameter :: my_C1 = 0.08D0
!     REALTYPE, parameter :: my_C2 = 0.7D0
!     REALTYPE, parameter :: my_C3 = 0.2D0
!     REALTYPE, parameter :: h15_Ghmin = -0.28D0
!     REALTYPE, parameter :: h15_Ghoff = 0.003D0
!     REALTYPE, parameter :: h15_Gvoff = 0.006D0
!     REALTYPE, parameter :: h15_Sxmax = 2.12D0
!
!     REALTYPE :: h15_Shn0, h15_Shnh, h15_Shns, h15_Shnv
!     REALTYPE :: h15_Shdah, h15_Shdav, h15_Shdbh
!     REALTYPE :: h15_Shdv, h15_Shdvh, h15_Shdvv
!     REALTYPE :: h15_Ssn0, h15_Ssdh, h15_Ssdv
!     REALTYPE :: h15_Smn0, h15_SmnhSh, h15_SmnsSs
!     REALTYPE :: h15_Smdh, h15_Smdv
!
!     REALTYPE :: tmp0,tmp1,tmp2,tmp3,tmp4
!     REALTYPE :: Ghcrit, Gvcrit
!
!-----------------------------------------------------------------------
! These constants  above & below could all be set or computed elsewhere in advance,
! subject to adjustments in A's, B's & C's. Just sticking them all in here for now.
!-----------------------------------------------------------------------
!BOC
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
    "CmueDH15Workspace",
    "step_cmue_d_h15",
]

_SMALL: float = 1.0e-8
_SQRT2: float = 1.4142135623730951
_H15_VON_KARMAN: float = 0.41

_MY_A1: float = 0.92
_MY_A2: float = 0.74
_MY_B1: float = 16.6
_MY_B2: float = 10.1
_MY_C1: float = 0.08
_MY_C2: float = 0.7
_MY_C3: float = 0.2
_H15_GHMIN: float = -0.28
_H15_GHOFF: float = 0.003
_H15_GVOFF: float = 0.006
_H15_SXMAX: float = 2.12

_H15_SHN0: float = _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
_H15_SHNH: float = -9.0 * _MY_A1 * _MY_A2 * (
    _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
)
_H15_SHNS: float = (
    9.0
    * _MY_A1
    * _MY_A2
    * (1.0 - 6.0 * _MY_A1 / _MY_B1)
    * (2.0 * _MY_A1 + _MY_A2)
)
_H15_SHNV: float = 9.0 * _MY_A1 * _MY_A2 * (
    _MY_A2 * (1.0 - 6.0 * _MY_A1 / _MY_B1 - 3.0 * _MY_C1)
    - 2.0 * _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1 + 3.0 * _MY_C1)
)
_H15_SHDAH: float = -9.0 * _MY_A1 * _MY_A2
_H15_SHDAV: float = -36.0 * _MY_A1 * _MY_A1
_H15_SHDBH: float = -3.0 * _MY_A2 * (6.0 * _MY_A1 + _MY_B2 * (1.0 - _MY_C3))
_H15_SHDV: float = -9.0 * _MY_A2 * _MY_A2 * (1.0 - _MY_C2)
_H15_SHDVH: float = -162.0 * _MY_A1 * _MY_A1 * _MY_A2 * (
    2.0 * _MY_A1 + (2.0 - _MY_C2) * _MY_A2
)
_H15_SHDVV: float = 324.0 * _MY_A1 * _MY_A1 * _MY_A2 * _MY_A2 * (1.0 - _MY_C2)
_H15_SSN0: float = _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1)
_H15_SSDH: float = -9.0 * _MY_A1 * _MY_A2
_H15_SSDV: float = -9.0 * _MY_A1 * _MY_A1
_H15_SMN0: float = _MY_A1 * (1.0 - 6.0 * _MY_A1 / _MY_B1 - 3.0 * _MY_C1)
_H15_SMNHSH: float = 9.0 * _MY_A1 * (
    2.0 * _MY_A1 + _MY_A2 * (1.0 - _MY_C2)
)
_H15_SMNSSS: float = 27.0 * _MY_A1 * _MY_A1
_H15_SMDH: float = -9.0 * _MY_A1 * _MY_A2
_H15_SMDV: float = -36.0 * _MY_A1 * _MY_A1
_H15_SCALE: float = 4.0 / (_MY_B1 * _MY_B1)


class CmueDH15Workspace(TaichiFieldCollection):
    """Taichi fields for the Harcourt (2015) Langmuir stability functions."""

    as_: ti.Field
    an: ti.Field
    av: ti.Field
    aw: ti.Field
    SPF: ti.Field
    cmue1: ti.Field
    cmue2: ti.Field
    cmue3: ti.Field
    sq_var: ti.Field
    sl_var: ti.Field

    def __init__(self, nlev: int, *, n_cols: int = 1) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("as_", "an", "av", "aw", "SPF"))
        self.allocate_many(("cmue1", "cmue2", "cmue3", "sq_var", "sl_var"))


@ti_kernel
def step_cmue_d_h15(  # type: ignore[no-untyped-def]
    n_cols: ti.i32,
    nlev: ti.i32,
    length_lim: ti.i32,
    sq: ti.f64,
    sl: ti.f64,
    as_: TemplateArg,
    an: TemplateArg,
    av: TemplateArg,
    aw: TemplateArg,
    SPF: TemplateArg,
    cmue1: TemplateArg,
    cmue2: TemplateArg,
    cmue3: TemplateArg,
    sq_var: TemplateArg,
    sl_var: TemplateArg,
):
    r"""Update Harcourt (2015) quasi-equilibrium Langmuir stability functions."""

    for col in range(n_cols):
        for i in range(1, nlev):
            gh = -_H15_SCALE * an[col, i]
            gm = _H15_SCALE * as_[col, i]
            gv = _H15_SCALE * av[col, i]
            gs = _H15_SCALE * aw[col, i]
            sh = _SMALL
            ss = _SMALL
            sm = _SMALL

            if length_lim == 0:
                tmp1 = 1.0
                tmp2 = _H15_GHMIN / ti.min(_H15_GHMIN, gh)
                tmp1 = ti.min(tmp1, tmp2)

                if tmp1 < 1.0:
                    gh = gh * tmp1
                    gv = gv * tmp1
                    gs = gs * tmp1

            tmp0 = 2.0

            if gv > 0.0:
                tmp1 = (_H15_SHDAH + _H15_SHDBH) * gh + (_H15_SHDAV + _H15_SHDV) * gv
                tmp1 = tmp1 + (
                    (_H15_SHDAH * _H15_GHOFF + _H15_SHDAV * _H15_GVOFF)
                    * (_H15_SHDBH * gh)
                    + (_H15_SHDVH * _H15_GHOFF + _H15_SHDVV * _H15_GVOFF) * gv
                )
                tmp1 = tmp1 + (
                    (_H15_SHDAH * gh + _H15_SHDAV * gv) * (_H15_SHDBH * _H15_GHOFF)
                    + (_H15_SHDVH * gh + _H15_SHDVV * gv) * _H15_GVOFF
                )

                tmp2 = (
                    (_H15_SHDAH * gh + _H15_SHDAV * gv) * (_H15_SHDBH * gh)
                    + (_H15_SHDVH * gh + _H15_SHDVV * gv) * gv
                )

                tmp4 = (
                    1.0
                    + (_H15_SHDAH + _H15_SHDBH) * _H15_GHOFF
                    + (_H15_SHDAV + _H15_SHDV) * _H15_GVOFF
                    + (_H15_SHDAH * _H15_GHOFF + _H15_SHDAV * _H15_GVOFF)
                    * (_H15_SHDBH * _H15_GHOFF)
                    + (_H15_SHDVH * _H15_GHOFF + _H15_SHDVV * _H15_GVOFF)
                    * _H15_GVOFF
                )

                tmp3 = tmp1 * tmp1 - 4.0 * tmp2 * tmp4

                if tmp3 >= 0.0 and tmp2 < 0.0:
                    tmp3 = (-tmp1 + ti.sqrt(tmp3)) / (2.0 * tmp2)
                elif tmp3 >= 0.0 and tmp3 > 0.0:
                    tmp3 = (-tmp1 - ti.sqrt(tmp3)) / (2.0 * tmp2)
                else:
                    tmp3 = 2.0

                if tmp3 > 0.0 and tmp3 < 1.0:
                    tmp0 = ti.min(tmp0, tmp3)

            gv = gv * SPF[col, i]
            gs = gs * SPF[col, i] * SPF[col, i]

            if gh > 0.0:
                tmp1 = (
                    2.0 * (_H15_SHDAH + _H15_SHDBH) * gh
                    + (_H15_SHDAV + _H15_SHDV) * gv
                )
                tmp2 = (
                    (2.0 * _H15_SHDAH * gh + _H15_SHDAV * gv)
                    * (2.0 * _H15_SHDBH * gh)
                    + (2.0 * _H15_SHDVH * gh + _H15_SHDVV * gv) * gv
                )
                tmp4 = 1.0
                tmp3 = tmp1 * tmp1 - 4.0 * tmp2 * tmp4

                if tmp3 >= 0.0 and tmp2 < 0.0:
                    tmp3 = (-tmp1 + ti.sqrt(tmp3)) / (2.0 * tmp2)
                elif tmp3 >= 0.0 and tmp3 > 0.0:
                    tmp3 = (-tmp1 - ti.sqrt(tmp3)) / (2.0 * tmp2)
                else:
                    tmp3 = 2.0

                if tmp3 > 0.0 and tmp3 < 1.0:
                    tmp0 = ti.min(tmp0, tmp3)

            if tmp0 > 0.0 and tmp0 < 1.0:
                gh = tmp0 * gh
                gm = tmp0 * gm
                gv = tmp0 * gv
                gs = tmp0 * gs

            tmp1 = _H15_SHN0 + _H15_SHNH * gh + _H15_SHNS * gs + _H15_SHNV * gv
            if tmp1 < 0.0:
                sh = _SMALL
            else:
                tmp2 = (
                    (1.0 + _H15_SHDAH * gh + _H15_SHDAV * gv)
                    * (1.0 + _H15_SHDBH * gh)
                    + (_H15_SHDV + _H15_SHDVH * gh + _H15_SHDVV * gv) * gv
                )
                if tmp2 <= 0.0:
                    sh = _H15_SXMAX
                else:
                    sh = ti.min(ti.max(_SMALL, tmp1 / tmp2), _H15_SXMAX)

            tmp2 = 1.0 + _H15_SSDH * gh + _H15_SSDV * gv
            if tmp2 < 0.0:
                ss = _H15_SXMAX
            else:
                ss = ti.min(ti.max(_SMALL, _H15_SSN0 / tmp2), _H15_SXMAX)

            tmp1 = _H15_SMN0 + _H15_SMNHSH * gh * sh + _H15_SMNSSS * gs * ss
            if tmp1 < _SMALL and tmp1 >= 0.0:
                gh = gh + _SMALL
                gv = gv + _SMALL
                tmp1 = _H15_SMN0 + _H15_SMNHSH * gh * sh + _H15_SMNSSS * gs * ss
            elif tmp1 > -_SMALL and tmp1 < 0.0:
                gh = gh - _SMALL
                gv = gv - _SMALL
                tmp1 = _H15_SMN0 + _H15_SMNHSH * gh * sh + _H15_SMNSSS * gs * ss

            if tmp1 < 0.0:
                sm = _SMALL
            else:
                tmp2 = 1.0 + _H15_SMDH * gh + _H15_SMDV * gv
                if tmp2 <= 0.0:
                    sm = _H15_SXMAX
                else:
                    sm = ti.min(ti.max(_SMALL, tmp1 / tmp2), _H15_SXMAX)

            ss = ss * SPF[col, i]

            cmue1[col, i] = _SQRT2 * sm
            cmue2[col, i] = _SQRT2 * sh
            cmue3[col, i] = _SQRT2 * ss

            sq_var[col, i] = ti.sqrt(sq * sq + (_H15_VON_KARMAN * sh) ** 2)
            sl_var[col, i] = ti.sqrt(sl * sl + (_H15_VON_KARMAN * sh) ** 2)
