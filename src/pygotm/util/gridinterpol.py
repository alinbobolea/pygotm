r"""!-----------------------------------------------------------------------
!BOP
! !ROUTINE: Interpolate from observation space to model grid
!
! !INTERFACE:
!    subroutine gridinterpol(N,cols,obs_z,obs_prof,nlev,model_z,model_prof)
!
! !DESCRIPTION:
!
!  This is a utility subroutine in which observational data, which might
!  be given on an arbitrary, but structured grid, are linearly interpolated and
!  extrapolated to the actual (moving) model grid.
!
! !USES:
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer,  intent(in)                :: N,cols
!   REALTYPE, intent(in)                :: obs_z(0:N),obs_prof(0:N,cols)
!   integer,  intent(in)                :: nlev
!   REALTYPE, intent(in)                :: model_z(0:nlev)
!
! !OUTPUT PARAMETERS:
!   REALTYPE, intent(out)               :: model_prof(0:nlev,cols)
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!
!EOP
!-----------------------------------------------------------------------
!
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

__all__ = ["gridinterpol"]


def gridinterpol(
    obs_z: np.ndarray,
    obs_prof: np.ndarray,
    model_z: np.ndarray,
    nlev: int,
) -> np.ndarray:
    """Linearly interpolate/extrapolate observational data to the model grid.

    Observational data on an arbitrary structured grid (obs_z, obs_prof) are
    mapped to the (moving) model grid (model_z).  Linear extrapolation is used
    outside the observational range: surface levels above obs_z[N] receive the
    topmost observed value, and bottom levels below obs_z[1] receive the lowest
    observed value.

    !BOC
    !  Set surface values to uppermost input value
    !  Set bottom values to lowest input value
    !  Interpolate inner values linearly
    !EOC

    Parameters
    ----------
    obs_z : np.ndarray, shape (N+1,)
        Observation depth levels [m].  Indices 1..N are the actual levels;
        index 0 is a padding element (never used by this routine, following
        GOTM Fortran convention with DIMENSION(0:N)).
    obs_prof : np.ndarray, shape (N+1, cols)
        Observed profile values at each obs level and column.
    model_z : np.ndarray, shape (nlev+1,)
        Model depth levels [m], indexed 0..nlev.
    nlev : int
        Number of model layers.

    Returns
    -------
    model_prof : np.ndarray, shape (nlev+1, cols)
        Interpolated profile on the model grid.  Index 0 is not set by this
        routine (consistent with the Fortran loop bounds 1..nlev).
    """
    N = obs_z.shape[0] - 1  # obs_z is indexed 0:N
    cols = obs_prof.shape[1]

    model_prof = np.zeros((nlev + 1, cols))

    # Set surface values to uppermost input value
    for i in range(nlev, 0, -1):
        if model_z[i] >= obs_z[N]:
            model_prof[i, :] = obs_prof[N, :]

    # Set bottom values to lowest input value
    for i in range(1, nlev + 1):
        if model_z[i] <= obs_z[1]:
            model_prof[i, :] = obs_prof[1, :]

    # Interpolate inner values linearly
    for i in range(1, nlev + 1):
        if obs_z[1] < model_z[i] < obs_z[N]:
            ii = 1
            while obs_z[ii] <= model_z[i]:
                ii += 1
            rat = (model_z[i] - obs_z[ii - 1]) / (obs_z[ii] - obs_z[ii - 1])
            model_prof[i, :] = (1 - rat) * obs_prof[ii - 1, :] + rat * obs_prof[ii, :]

    return model_prof
