r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: wequation
!
! !INTERFACE:
!   subroutine wequation(nlev,dt)
!
! !DESCRIPTION:
!  This subroutine calculates vertical velocity profiles, if
!  {\tt w\_adv\_method} is 1 or 2, which has to be chosen in the
!  {\tt w\_advspec} in {\tt gotm.yaml}. The profiles of vertical
!  velocity are determined by two values,
!  the height of maximum absolute value of vertical velocity, {\tt w\_height},
!  and the vertical velocity at this height, {\tt w\_adv}. From {\tt w\_height},
!  the vertical velocity is linearly decreasing towards the surface and
!  the bottom, where its value is zero.
!
! !USES:
!   use meanflow    , only: zi,w
!   use observations, only: w_adv_input,w_height_input
!
! !INPUT PARAMETERS:
!  number of vertical layers
!   integer, intent(in)                 :: nlev
!  time step (s)
!   REALTYPE, intent(in)                :: dt
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                   :: i
!   REALTYPE                  :: z_crit
!-----------------------------------------------------------------------
!BOC
!
!  Vertical velocity calculation:
!
!   select case(w_adv_input%method)
!      case(0)
!         ! no vertical advection
!      case(1,2)
!         ! linearly varying advection velocity with peak at "w_height"
!         z_crit=zi(nlev)-0.01*(zi(nlev)-zi(0))
!         if (w_height_input%value.gt.z_crit) w_height_input%value=z_crit
!         z_crit=zi(0)+0.01*(zi(nlev)-zi(0))
!         if (w_height_input%value.lt.z_crit) w_height_input%value=z_crit
!         do i=1,nlev-1
!            if (zi(i).gt.w_height_input%value) then
!               w(i)=(zi(nlev)-zi(i))/(zi(nlev)-w_height_input%value)*w_adv_input%value
!            else
!               w(i)=(zi(0)-zi(i))/(zi(0)-w_height_input%value)*w_adv_input%value
!            end if
!         end do
!         w(0)    =_ZERO_
!         w(nlev) =_ZERO_
!      case default
!   end select
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from pygotm.meanflow.meanflow import MeanflowState

__all__ = [
    "wequation",
]

# Method constants mirroring GOTM w_adv_input%method values.
W_ADV_NONE: int = 0  # no vertical advection
W_ADV_PROFILE: int = 1  # tent-shaped profile, method 1
W_ADV_PROFILE2: int = 2  # tent-shaped profile, method 2 (same shape)


def wequation(
    state: MeanflowState,
    nlev: int,
    dt: float,
    w_adv_method: int,
    w_adv: float,
    w_height: float,
) -> float:
    """Calculate the vertical velocity profile.

    Implements wequation.F90: builds a tent-shaped (piecewise-linear) vertical
    velocity profile with a peak of ``w_adv`` at height ``w_height`` above the
    seabed.  The velocity decreases linearly to zero at both the surface and
    seabed.

    When ``w_adv_method`` is 0 the function is a no-op (``state.w`` is left
    unchanged, as in the Fortran ``case(0)`` branch).

    Parameters
    ----------
    state:
        MeanflowState; ``state.zi`` and ``state.w`` must be allocated.
        ``state.w`` is updated in-place.
    nlev:
        Number of model layers.
    dt:
        Time step [s].  Passed for interface symmetry; not used in the
        current piecewise-linear profile calculation.
    w_adv_method:
        Vertical advection method selector:
        0 → no advection (w unchanged), 1 or 2 → tent profile.
    w_adv:
        Vertical velocity at the peak height [m/s].
    w_height:
        Height above seabed at which ``w_adv`` is applied [m].
        Clamped to the inner 98 % of the water column (1 % margins at each
        end) following the Fortran source.

    Returns
    -------
    float
        The (possibly clamped) value of ``w_height`` used in the calculation.
        Equals the input ``w_height`` when method is 0 or no clamping occurs.
    """
    assert state.zi is not None
    assert state.w is not None

    if w_adv_method not in (W_ADV_PROFILE, W_ADV_PROFILE2):
        return w_height

    zi = state.zi
    w = state.w

    # Clamp w_height to stay within 1 % of each end of the water column.
    # This prevents the denominator (zi(nlev) - w_height) or
    # (zi(0) - w_height) from reaching zero.
    z_top = float(zi[nlev])
    z_bot = float(zi[0])
    col_depth = z_top - z_bot

    z_crit_top = z_top - 0.01 * col_depth
    z_crit_bot = z_bot + 0.01 * col_depth

    if w_height > z_crit_top:
        w_height = z_crit_top
    if w_height < z_crit_bot:
        w_height = z_crit_bot

    # Build tent-shaped profile at interfaces 1..nlev-1.
    for i in range(1, nlev):
        zi_i = float(zi[i])
        if zi_i > w_height:
            # Above peak: interpolate from peak down to surface (both → 0 at top).
            w[i] = (z_top - zi_i) / (z_top - w_height) * w_adv
        else:
            # Below peak: interpolate from seabed up to peak (both → 0 at bottom).
            w[i] = (z_bot - zi_i) / (z_bot - w_height) * w_adv

    w[0] = 0.0
    w[nlev] = 0.0

    return w_height
