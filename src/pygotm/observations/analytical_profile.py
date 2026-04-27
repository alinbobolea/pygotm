"""
!-----------------------------------------------------------------------
!BOP
!
! !IROUTINE: analytical_profile
!
! !INTERFACE:
!   subroutine analytical_profile(nlev,z,z1,v1,z2,v2,prof)
!
! !DESCRIPTION:
! This routine creates a vertical profile {\tt prof} with value
! {\tt v1} in a surface layer down to depth {\tt z1} and a bottom
! layer of value {\tt v2} reaching from depth {\tt z2} down to the bottom.
! Both layers are connected by an intermediate layer reaching from {\tt z1}
! to {\tt z2} with values linearly varying from {\tt v1} to {\tt v2}.
!
! !USES:
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer,  intent(in)                :: nlev
!   REALTYPE, intent(in)                :: z(0:nlev)
!   REALTYPE, intent(in)                :: z1,v1,z2,v2
!
! !OUTPUT PARAMETERS:
!   REALTYPE, intent(out)               :: prof(0:nlev)
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

__all__ = ["analytical_profile"]


def analytical_profile(
    nlev: int,
    z: np.ndarray,
    z1: float,
    v1: float,
    z2: float,
    v2: float,
) -> np.ndarray:
    """Create the piecewise-linear two-layer profile from ``analytical_profile.F90``."""

    if z2 - z1 <= -1.0e-15:
        msg = "z2 should be larger than z1 in analytical_profile"
        raise ValueError(msg)

    prof = np.zeros(nlev + 1, dtype=np.float64)
    alpha = (v2 - v1) / (z2 - z1 + 2.0e-15)
    for i in range(nlev, 0, -1):
        depth_from_surface = -1.0 * z[i]
        upper_limit = z1 - z[nlev]
        lower_limit = z2 - z[nlev]
        if depth_from_surface <= upper_limit:
            prof[i] = v1
        if alpha <= 1.0e15 and upper_limit < depth_from_surface <= lower_limit:
            prof[i] = v1 + alpha * (depth_from_surface - upper_limit)
        if depth_from_surface > lower_limit:
            prof[i] = v2
    return prof
