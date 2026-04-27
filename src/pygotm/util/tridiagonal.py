r"""!-----------------------------------------------------------------------
!BOP
!
! !MODULE: mtridiagonal --- solving the system\label{sec:tridiagonal}
!
! !INTERFACE:
!
! !DESCRIPTION:
!
!  Solves a linear system of equations with a tridiagonal matrix
!  using Gaussian elimination.
!
! !PUBLIC MEMBER FUNCTIONS:
!   public init_tridiagonal, tridiagonal, clean_tridiagonal
!
! !PUBLIC DATA MEMBERS:
!   REALTYPE, dimension(:), allocatable     :: au,bu,cu,du
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
!  private data members
!   REALTYPE, private, dimension(:),allocatable  ::  ru,qu
!
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: Allocate memory
!
! !INTERFACE:
!
! !DESCRIPTION:
!  This routines allocates memory necessary to perform the Gaussian
!  elimination.
!
! !USES:
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: N
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: rc
!
!-----------------------------------------------------------------------
!EOC
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: Simplified Gaussian elimination
!
! !INTERFACE:
!
! !DESCRIPTION:
! A linear equation with tridiagonal matrix structure is solved here. The main
! diagonal is stored on {\tt bu}, the upper diagonal on {\tt au}, and the
! lower diagonal on {\tt cu}, the right hand side is stored on {\tt du}.
! The method used here is the simplified Gauss elimination, also called
! \emph{Thomas algorithm}.
!
! !USES:
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: N,fi,lt
!
! !OUTPUT PARAMETERS:
!   REALTYPE                            :: value(0:N)
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: i
!
!-----------------------------------------------------------------------
!EOC
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: De-allocate memory
!
! !INTERFACE:
!
! !DESCRIPTION:
!  De-allocates memory allocated in init\_tridiagonal.
!
! !USES:
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
!EOC
!-----------------------------------------------------------------------
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import taichi as ti

from pygotm.fields import ColumnLayout, TaichiFieldCollection

__all__ = [
    "TridiagonalWorkspace",
    "clean_tridiagonal",
    "init_tridiagonal",
    "tridiagonal",
    "tridiagonal_column",
]


class TridiagonalWorkspace(TaichiFieldCollection):
    """Allocate GOTM-style tridiagonal coefficients and work arrays."""

    au: ti.Field
    bu: ti.Field
    cu: ti.Field
    du: ti.Field
    ru: ti.Field
    qu: ti.Field

    def __init__(self, nlev: int, *, n_cols: int | None = None) -> None:
        super().__init__(ColumnLayout(nlev=nlev, n_cols=n_cols))
        self.allocate_many(("au", "bu", "cu", "du", "ru", "qu"))

    def clear(self) -> None:
        """Drop references to the allocated Taichi fields."""

        for name in tuple(self._fields):
            delattr(self, name)
        self._fields.clear()


def init_tridiagonal(
    nlev: int,
    *,
    n_cols: int | None = None,
) -> TridiagonalWorkspace:
    """Allocate the tridiagonal coefficients and Thomas work arrays."""

    return TridiagonalWorkspace(nlev=nlev, n_cols=n_cols)


def clean_tridiagonal(workspace: TridiagonalWorkspace) -> None:
    """Release Python references to the allocated Taichi fields."""

    workspace.clear()


@ti.func
def tridiagonal(  # type: ignore[no-untyped-def]
    au,
    bu,
    cu,
    du,
    ru,
    qu,
    value,
    fi,
    lt,
):
    r"""! !IROUTINE: Simplified Gaussian elimination
!
! !DESCRIPTION:
! A linear equation with tridiagonal matrix structure is solved here. The main
! diagonal is stored on {\tt bu}, the upper diagonal on {\tt au}, and the
! lower diagonal on {\tt cu}, the right hand side is stored on {\tt du}.
! The method used here is the simplified Gauss elimination, also called
! \emph{Thomas algorithm}.
    """

    if fi == lt:
        value[fi] = du[fi] / bu[fi]
    else:
        ru[lt] = au[lt] / bu[lt]
        qu[lt] = du[lt] / bu[lt]

        for offset in range(lt - fi - 1):
            i = lt - 1 - offset
            denominator = bu[i] - cu[i] * ru[i + 1]
            ru[i] = au[i] / denominator
            qu[i] = (du[i] - cu[i] * qu[i + 1]) / denominator

        denominator = bu[fi] - cu[fi] * ru[fi + 1]
        qu[fi] = (du[fi] - cu[fi] * qu[fi + 1]) / denominator

        value[fi] = qu[fi]
        for i in range(fi + 1, lt + 1):
            value[i] = qu[i] - ru[i] * value[i - 1]


@ti.func
def tridiagonal_column(  # type: ignore[no-untyped-def]
    col,
    au,
    bu,
    cu,
    du,
    ru,
    qu,
    value,
    fi,
    lt,
):
    r"""! !IROUTINE: Simplified Gaussian elimination
!
! !DESCRIPTION:
! A linear equation with tridiagonal matrix structure is solved here. The main
! diagonal is stored on {\tt bu}, the upper diagonal on {\tt au}, and the
! lower diagonal on {\tt cu}, the right hand side is stored on {\tt du}.
! The method used here is the simplified Gauss elimination, also called
! \emph{Thomas algorithm}.
    """

    if fi == lt:
        value[col, fi] = du[col, fi] / bu[col, fi]
    else:
        ru[col, lt] = au[col, lt] / bu[col, lt]
        qu[col, lt] = du[col, lt] / bu[col, lt]

        for offset in range(lt - fi - 1):
            i = lt - 1 - offset
            denominator = bu[col, i] - cu[col, i] * ru[col, i + 1]
            ru[col, i] = au[col, i] / denominator
            qu[col, i] = (du[col, i] - cu[col, i] * qu[col, i + 1]) / denominator

        denominator = bu[col, fi] - cu[col, fi] * ru[col, fi + 1]
        qu[col, fi] = (du[col, fi] - cu[col, fi] * qu[col, fi + 1]) / denominator

        value[col, fi] = qu[col, fi]
        for i in range(fi + 1, lt + 1):
            value[col, i] = qu[col, i] - ru[col, i] * value[col, i - 1]
