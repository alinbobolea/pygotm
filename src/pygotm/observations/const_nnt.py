"""
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: const_NNT
!
! !INTERFACE:
!   subroutine const_NNT(nlev,z,zi,T_top,S_const,NN,gravity,T)
!
! !DESCRIPTION:
! This routine creates a vertical profile {\tt prof} with value
! {\tt v1}
!
! !USES:
!   use density, only: get_alpha
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer,  intent(in)                :: nlev
!   REALTYPE, intent(in)                :: z(0:nlev)
!   REALTYPE, intent(in)                :: zi(0:nlev)
!   REALTYPE, intent(in)                :: T_top,S_const,NN
!   REALTYPE, intent(in)                :: gravity
!
! !INOUT PARAMETERS:
!   REALTYPE, intent(inout)             :: T(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
! !LOCAL VARIABLES:
!   integer                             :: i
!   REALTYPE                           :: lalpha
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from pygotm.util.density import DensityState, get_alpha

__all__ = ["const_NNT"]


def const_NNT(
    density_state: DensityState,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    T_top: float,
    S_const: float,
    NN: float,
    gravity: float,
    T: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a temperature profile with constant buoyancy frequency."""

    profile = (
        np.zeros(nlev + 1, dtype=np.float64)
        if T is None
        else np.asarray(T, dtype=np.float64).copy()
    )
    profile[nlev] = T_top
    for i in range(nlev - 1, 0, -1):
        lalpha = get_alpha(density_state, S_const, profile[i + 1], -zi[i])
        profile[i] = profile[i + 1] - (NN * (z[i + 1] - z[i])) / (
            gravity * lalpha
        )
        lalpha = get_alpha(
            density_state,
            S_const,
            0.5 * (profile[i + 1] + profile[i]),
            -zi[i],
        )
        profile[i] = profile[i + 1] - (NN * (z[i + 1] - z[i])) / (
            gravity * lalpha
        )
    return profile
