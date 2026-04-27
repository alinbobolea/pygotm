r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The external pressure-gradient \label{sec:extpressure}
!
! !INTERFACE:
!   subroutine external_pressure(method,nlev)
!
! !DESCRIPTION:
!
!  This subroutine calculates the external pressure-gradient. Two methods
!  are implemented here, relating either to the velocity vector at a
!  given height above bed prescribed or to the vector for the vertical mean
!  velocity. In the first case, {\tt dpdx} and {\tt dpdy} are $x$-
!  and $y$-components of the prescribed velocity vector at the
!  height {\tt h\_press} above the bed. The velocity profile will in
!  this routive be shifted by a vertically constant vector such that the
!  resulting profile has an (interpolated) velocity at {\tt h\_press}
!  which is identical to the prescribed value. In the second case,
!  {\tt dpdx} and {\tt dpdy} are $x$- and $y$-components of the
!  prescribed vertical mean velocity vector, and {\tt h\_press} is
!  not used. Here the velocity profile is shifted in such a way that
!  the resulting mean velocty vector is identical to {\tt dpdx} and {\tt dpdy}.
!
!  For both cases, this is a recalculation of the external pressure gradient,
!  since at all points the same acceleration has been applied in this
!  operator split method.
!
!  If the external pressure-gradient is prescribed by the
!  surface slope, then it is directly inserted in \eq{uEq} and \eq{vEq}.
!
!  For details of this method, see \cite{Burchard99}.
!
! !USES:
!   use meanflow,     only: u,v,h
!   use observations, only: dpdx_input,dpdy_input,h_press_input
!
! !INPUT PARAMETERS:
!  method to compute external pressure gradient
!   integer, intent(in)                 :: method
!  number of vertical layers
!   integer, intent(in)                 :: nlev
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                             :: i
!   REALTYPE                            :: z(0:nlev)
!   REALTYPE                            :: rat,uint,vint,hint
!
!-----------------------------------------------------------------------
!BOC
!   select case (method)
!      case (1)
!        current measurement at h_press above bed
!         z(1)=0.5*h(1)
!         i   =0
!222      i=i+1
!         z(i+1)=z(i)+0.5*(h(i)+h(i+1))
!         if ((z(i+1).lt.h_press_input%value).and.(i.lt.nlev)) goto 222
!         rat=(h_press_input%value-z(i))/(z(i+1)-z(i))
!         uint=rat*u(i+1)+(1-rat)*u(i)
!         vint=rat*v(i+1)+(1-rat)*v(i)
!         do i=1,nlev
!            u(i)=u(i)+dpdx_input%value-uint
!            v(i)=v(i)+dpdy_input%value-vint
!         end do
!      case (2)
!     vertical mean of current prescribed
!         uint=_ZERO_
!         vint=_ZERO_
!         hint=_ZERO_
!         do i=1,nlev
!            hint=hint+h(i)
!            uint=uint+h(i)*u(i)
!            vint=vint+h(i)*v(i)
!         end do
!         uint=uint/hint
!         vint=vint/hint
!         do i=1,nlev
!            u(i)=u(i)+dpdx_input%value-uint
!            v(i)=v(i)+dpdy_input%value-vint
!         end do
!      case default
!     do nothing if method=0, because then
!     pressure gradient is applied directly
!     in uequation() and vequation()
!   end select
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState

__all__ = [
    "external_pressure",
    "EXT_PRESS_SLOPE",
    "EXT_PRESS_HEIGHT",
    "EXT_PRESS_MEAN",
]

# Method constants mirroring GOTM external_pressure method parameter.
EXT_PRESS_SLOPE: int = 0   # pressure gradient inserted directly in uequation/vequation
EXT_PRESS_HEIGHT: int = 1  # shift profile to match prescribed velocity at h_press
EXT_PRESS_MEAN: int = 2    # shift profile to match prescribed depth-mean velocity


def external_pressure(
    state: MeanflowState,
    nlev: int,
    method: int,
    dpdx: float,
    dpdy: float,
    h_press: float = 0.0,
) -> None:
    """Apply the external (barotropic) pressure-gradient correction.

    Translates external_pressure.F90 verbatim.  Updates ``state.u`` and
    ``state.v`` in-place by adding a depth-uniform velocity shift.

    Method 0 (EXT_PRESS_SLOPE):
        No-op — the surface-slope pressure gradient is applied directly inside
        ``uequation``/``vequation`` (when ``ext_method == 0`` in those routines).

    Method 1 (EXT_PRESS_HEIGHT):
        Shift the velocity profile by a depth-uniform constant so that the
        (linearly interpolated) velocity at height ``h_press`` above the seabed
        equals ``(dpdx, dpdy)``.

    Method 2 (EXT_PRESS_MEAN):
        Shift the velocity profile so that the depth-weighted mean velocity
        equals ``(dpdx, dpdy)``.

    Parameters
    ----------
    state:
        MeanflowState carrying h, u, v (all shape nlev+1, 1-indexed layers 1..nlev).
    nlev:
        Number of vertical model layers.
    method:
        Selector: 0 = no-op, 1 = point-velocity constraint, 2 = mean-velocity constraint.
    dpdx:
        Prescribed x-velocity [m/s] at ``h_press`` (method 1) or depth-mean (method 2).
    dpdy:
        Prescribed y-velocity [m/s] at ``h_press`` (method 1) or depth-mean (method 2).
    h_press:
        Height above the seabed [m] at which the velocity is prescribed (method 1 only).
        Ignored for methods 0 and 2.
    """
    assert state.h is not None
    assert state.u is not None
    assert state.v is not None

    h = state.h
    u = state.u
    v = state.v

    if method == EXT_PRESS_HEIGHT:
        # --- Method 1: current measurement at h_press above bed ---
        # Build cell-centre heights z[1]..z[nlev] above the seabed.
        # z[1] = 0.5*h[1]
        # z[k+1] = z[k] + 0.5*(h[k] + h[k+1])
        # Find the first k such that z[k+1] >= h_press, then interpolate.
        #
        # Allocate z with shape nlev+2 so z[nlev+1] is safe if h_press is
        # above the top cell centre (guards against the Fortran out-of-bounds
        # edge case when h_press > z[nlev]).
        z = np.zeros(nlev + 2)
        z[1] = 0.5 * h[1]
        i = 0
        while True:
            i += 1
            if i < nlev:
                z[i + 1] = z[i] + 0.5 * (h[i] + h[i + 1])
            else:
                # i == nlev: extend with the last layer thickness as a safe sentinel
                z[i + 1] = z[i] + h[nlev]
            if not (z[i + 1] < h_press and i < nlev):
                break

        # Linear interpolation between cell centres i and i+1.
        dz = z[i + 1] - z[i]
        if dz > 0.0:
            rat = (h_press - z[i]) / dz
        else:
            rat = 0.0
        rat = max(0.0, min(1.0, rat))  # clamp to [0, 1] for safety

        uint = rat * u[i + 1] + (1.0 - rat) * u[i]
        vint = rat * v[i + 1] + (1.0 - rat) * v[i]

        shift_u = dpdx - uint
        shift_v = dpdy - vint
        for k in range(1, nlev + 1):
            u[k] += shift_u
            v[k] += shift_v

    elif method == EXT_PRESS_MEAN:
        # --- Method 2: vertical mean of current prescribed ---
        hint = 0.0
        uint = 0.0
        vint = 0.0
        for k in range(1, nlev + 1):
            hint += h[k]
            uint += h[k] * u[k]
            vint += h[k] * v[k]
        uint /= hint
        vint /= hint

        shift_u = dpdx - uint
        shift_v = dpdy - vint
        for k in range(1, nlev + 1):
            u[k] += shift_u
            v[k] += shift_v

    # Method 0: no-op — pressure gradient is applied directly in uequation/vequation
