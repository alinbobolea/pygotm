"""
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: const_NNS
!
! !INTERFACE:
!   subroutine const_NNS(nlev,z,zi,S_top,T_const,NN,gravity,S)
!
! !DESCRIPTION:
! This routine creates a vertical profile {\tt prof} with value
! {\tt v1}
!
! !USES:
!   use density, only: get_beta
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer,  intent(in)                :: nlev
!   REALTYPE, intent(in)                :: z(0:nlev)
!   REALTYPE, intent(in)                :: zi(0:nlev)
!   REALTYPE, intent(in)                :: S_top,T_const,NN
!   REALTYPE, intent(in)                :: gravity
!
! !INOUT PARAMETERS:
!   REALTYPE, intent(inout)             :: S(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Lars Umlauf
!
! !LOCAL VARIABLES:
!   integer                               :: i
!    REALTYPE                             :: lbeta
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from pygotm.util.density import DensityState, get_beta

__all__ = ["const_NNS"]


def const_NNS(
    density_state: DensityState,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    S_top: float,
    T_const: float,
    NN: float,
    gravity: float,
    S: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a salinity profile with constant buoyancy frequency."""

    profile = (
        np.zeros(nlev + 1, dtype=np.float64)
        if S is None
        else np.asarray(S, dtype=np.float64).copy()
    )
    profile[nlev] = S_top
    for i in range(nlev - 1, 0, -1):
        lbeta = get_beta(density_state, profile[i + 1], T_const, -zi[i])
        profile[i] = profile[i + 1] + (NN * (z[i + 1] - z[i])) / (gravity * lbeta)
        lbeta = get_beta(
            density_state,
            0.5 * (profile[i + 1] + profile[i]),
            T_const,
            -zi[i],
        )
        profile[i] = profile[i + 1] + (NN * (z[i + 1] - z[i])) / (gravity * lbeta)
    return profile
