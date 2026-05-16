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

import numba
import numpy as np

from pygotm.arrays import ColumnWorkspace, make_column_array

__all__ = [
    "TridiagonalBatchWorkspace",
    "TridiagonalWorkspace",
    "clean_tridiagonal",
    "init_tridiagonal",
    "tridiagonal",
]


class TridiagonalWorkspace(ColumnWorkspace):
    """Single-column tridiagonal workspace — arrays shape (nlev+1,)."""

    def __init__(self, nlev: int) -> None:
        super().__init__(nlev)
        shape = (nlev + 1,)
        self.au = np.zeros(shape, dtype=np.float64)
        self.bu = np.zeros(shape, dtype=np.float64)
        self.cu = np.zeros(shape, dtype=np.float64)
        self.du = np.zeros(shape, dtype=np.float64)
        self.ru = np.zeros(shape, dtype=np.float64)
        self.qu = np.zeros(shape, dtype=np.float64)


class TridiagonalBatchWorkspace(ColumnWorkspace):
    """Batch tridiagonal workspace — arrays shape (batch_size, nlev+1)."""

    def __init__(self, nlev: int, batch_size: int) -> None:
        super().__init__(nlev, n_cols=batch_size)
        shape = (batch_size, nlev + 1)
        self.au = np.zeros(shape, dtype=np.float64)
        self.bu = np.zeros(shape, dtype=np.float64)
        self.cu = np.zeros(shape, dtype=np.float64)
        self.du = np.zeros(shape, dtype=np.float64)
        self.ru = np.zeros(shape, dtype=np.float64)
        self.qu = np.zeros(shape, dtype=np.float64)


def init_tridiagonal(nlev: int) -> TridiagonalWorkspace:
    """Allocate the tridiagonal coefficients and Thomas work arrays."""
    return TridiagonalWorkspace(nlev=nlev)


def clean_tridiagonal(workspace: TridiagonalWorkspace) -> None:
    """No-op — NumPy workspaces are garbage-collected normally."""


# Suppress the unused import warning: make_column_array is re-exported for
# callers that previously used it via this module.
_ = make_column_array


@numba.njit(cache=True)
def tridiagonal(
    au: np.ndarray,
    bu: np.ndarray,
    cu: np.ndarray,
    du: np.ndarray,
    ru: np.ndarray,
    qu: np.ndarray,
    value: np.ndarray,
    fi: int,
    lt: int,
) -> None:
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
