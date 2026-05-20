r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The internal pressure-gradient \label{sec:intpressure}
!
! !INTERFACE:
!   subroutine internal_pressure(nlev)
!
! !DESCRIPTION:
!  With the hydrostatic assumption
!  \begin{equation}\label{HydroStat}
!   \partder{P}{z} + g \mean{\rho} = 0
!   \comma
!  \end{equation}
!  where $P$ denotes the mean pressure, $g=9.81$m\,s$^{-2}$
!  the gravitational acceleration  and $\mean{\rho}$ the mean density,
!  the components of the pressure-gradient may be expressed as
!  \begin{equation}
!   \label{InternalPressurex}
!  - \frac{1}{\rho_0} \partder{P}{x}=
!  -g \partder{\zeta}{x}
!  +\int_z^{\zeta}\partder{B}{x} \, dz'
!  -\frac{1}{\rho_0} \partder{P(\zeta)}{x}
!  \end{equation}
!  and
!  \begin{equation}\label{InternalPressurey}
!  - \frac{1}{\rho_0} \partder{P}{y}=
!  -g \partder{\zeta}{y}
!  +\int_z^{\zeta} \partder{B}{y} \, dz'
!  -\frac{1}{\rho_0} \partder{P(\zeta)}{y}
!   \comma
!  \end{equation}
!  where $\zeta$ is the surface elevation, and $B$ the
!  mean buoyancy as defined in \eq{DefBuoyancy}.
!
!  The first term on the right hand side
!  in \eq{InternalPressurex}
!  and \eq{InternalPressurey} is the external pressure-gradient
!  due to surface slopes,  the second the internal pressure-gradient
!  due to the density gradient and the third term is the
!  atmoshperic pressure gradient at sea surface height.
!  The internal pressure-gradient will only be established by
!  gradients of mean potential temperature $\Theta$ and mean
!  salinity $S$. Sediment concentration is assumed to be
!  horizontally homogeneous.
!
!  In this subroutine there are two ways to calculate the internal pressure
!  gradient.
!
!  {\bf Scenarios with flat bottom:}
!  First, the horizontal buoyancy gradients,
!  $\partial_xB$ and $\partial_yB$,
!  are calculated from the prescribed gradients of salinity, $\partial_xS$
!  and $\partial_yS$, and temperature, $\partial_x\Theta$ and $\partial_y\Theta$,
!  according to the finite-difference expression
!  \begin{equation}
!    \partder{B}{x} \approx
!    \frac{B(S+\Delta_xS,\Theta+\Delta_x\Theta,P)-B(S,\Theta,P)}{\Delta x}
!    \comma
!  \end{equation}
!  \begin{equation}
!    \partder{B}{y} \approx
!    \frac{B(S+\Delta_yS,\Theta+\Delta_y\Theta,P)-B(S,\theta,P)}{\Delta y}
!   \comma
!  \end{equation}
!  where the defintions
!  \begin{equation}
!    \Delta_xS=\Delta x \partial_xS \comma
!    \Delta_yS=\Delta y \partial_yS \comma
!  \end{equation}
!  and
!  \begin{equation}
!   \Delta_x\Theta=\Delta x \partial_x\Theta \comma
!   \Delta_y\Theta=\Delta y \partial_y\Theta \comma
!  \end{equation}
!  have been used. $\Delta x$ and $\Delta y$ are "small enough", but otherwise
!  arbitrary length scales. The buoyancy gradients computed with this method
!  are then vertically integrated according to \eq{InternalPressurex} and
!  \eq{InternalPressurey}.
!
! The horizontal salinity and temperature gradients have to supplied by the
! user, either as constant values or as profiles given in a file (see
! {\tt gotm.yaml}).
!
! {\bf Scenarios for dense bottom and buoyant surface plumes in a sloping frame:}
! Assuming for a {\it sloping water-colum model model}
! that all density gradients
! along the sloping surface or bottom vanish, i.e.,
! \begin{equation}
! \partder{B}{x} = - \partder{\zeta}{x} \partder{B}{z}\comma
! \end{equation}
! \begin{equation}
! \partder{B}{y} = - \partder{\zeta}{y} \partder{B}{z}\comma
! \end{equation}
! we obtain
! \begin{equation}
! -\frac{1}{\rho_0} \partder{P}{x} =
! -\frac{1}{\rho_0} \partder {P(\zeta)}{x}
! +\partder{\zeta}{x} B(z)\comma
! \end{equation}
! \begin{equation}
! -\frac{1}{\rho_0} \partder{P}{y} =
! -\frac{1}{\rho_0} \partder {P(\zeta)}{y}
! +\partder{\zeta}{y} B(z).
! \end{equation}
!
! {\it Buoyant plume under shelf ice.}
! For the ambient water below the plume
! with $z\rightarrow -H$
! with the ambient buoyancy, $B(-H)$, we demand that the pressure
! gradient vanishes, i.e.,
! \begin{equation}
! 0=
! -\frac{1}{\rho_0} \partder{P}{x} =
! -\frac{1}{\rho_0} \partder{P(\zeta)}{x}
! +\partder{\zeta}{x} B(-H)\comma
! \end{equation}
! \begin{equation}
! 0=
! -\frac{1}{\rho_0} \partder{P}{y} =
! -\frac{1}{\rho_0} \partder{P(\zeta)}{y}
! +\partder{\zeta}{y} B(-H)\comma
! \end{equation}
! such that we obtain
! \begin{equation}
! -\frac{1}{\rho_0} \partial_x p =
! \partder{\zeta}{x} \left(B(z)-B(-H)\right)\comma
! \end{equation}
! \begin{equation}
! -\frac{1}{\rho_0} \partial_y p =
! \partder{\zeta}{y} \left(B(z)-B(-H)\right).
! \end{equation}
! Those simulations are only useful in situations with
! a sufficient amount of unstratified ambient water left below the plume,
! i.e., the bottom layer must not be entrained into the plume and must stay
! at ambient buoyancy.
!
! {\it Dense plume over sloping topography.}
! Similar considerations lead to the formulation of a bottom-attached
! dense plume:
! \begin{equation}
! -\frac{1}{\rho_0} \partder{P}{x} =
! \partder{\zeta}{x} \left(B(\zeta)-B(z)\right)\comma
! \end{equation}
! \begin{equation}
! -\frac{1}{\rho_0} \partder{P}{y} =
! \partder{\zeta}{y} \left(B(\zeta)-B(z)\right)\comma
! \end{equation}
! where the ambient water is now assumed to be above the plume.
! In this case, the surface layer must not be entrained
! into the plume.
!
! !USES:
!   use density,       only: get_rho,rho0
!   use meanflow,      only: T,S
!   use meanflow,      only: gravity,h
!   use meanflow,      only: buoy
!   use observations,  only: int_press_type
!   use observations,  only: dsdx_input,dsdy_input,dtdx_input,dtdy_input
!   use observations,  only: plume_type,plume_slope_x,plume_slope_y
!   use observations,  only: idpdx,idpdy
!
! !INPUT PARAMETERS:
!  number of vertical layers
!   integer, intent(in)                 :: nlev
!
! !REVISION HISTORY:
!  Original FORTRAN author(s): Hans Burchard & Karsten Bolding
!
!EOP
!
! !LOCAL VARIABLES:
!   integer                             :: i
!   REALTYPE                            :: z,dx,dy
!   REALTYPE                            :: dSS,dTT,Bl,Br,int
!   REALTYPE                            :: dxB(0:nlev),dyB(0:nlev)
!
!-----------------------------------------------------------------------
!BOC
!   if (int_press_type == 1) then ! T and S gradients
!
!     initialize local depth
!     and pressure gradient
!      z     = _ZERO_
!      idpdx = _ZERO_
!      idpdy = _ZERO_
!
!     the spacing for the finite differences
!      dx    =  10.
!      dy    =  10.
!
!      do i=nlev,1,-1
!         z=z+0.5*h(i)
!
!        buoyancy gradient in x direction
!         dSS    = dx*dsdx_input%data(i)
!         dTT    = dx*dtdx_input%data(i)
!         Bl     = -gravity*(get_rho(S(i)    ,T(i)    ,p=z) - rho0)/rho0
!         Br     = -gravity*(get_rho(S(i)+dSS,T(i)+dTT,p=z) - rho0)/rho0
!         dxB(i) = (Br-Bl)/dx
!
!        buoyancy gradient in y direction
!         dSS    = dy*dsdy_input%data(i)
!         dTT    = dy*dtdy_input%data(i)
!         Bl     = -gravity*(get_rho(S(i)     ,T(i)   ,p=z) - rho0)/rho0
!         Br     = -gravity*(get_rho(S(i)+dSS,T(i)+dTT,p=z) - rho0)/rho0
!         dyB(i) = (Br-Bl)/dy
!
!         z=z+0.5*h(i)
!      end do
!
!     internal pressure gradient in x direction
!      int=0.5*h(nlev)*dxB(nlev)
!      idpdx(nlev)=int
!      do i=nlev-1,1,-1
!         int=int+0.5*h(i+1)*dxB(i+1)+0.5*h(i)*dxB(i)
!         idpdx(i)=int
!      end do
!
!     internal pressure gradient in y direction
!      int=0.5*h(nlev)*dyB(nlev)
!      idpdy(nlev)=int
!      do i=nlev-1,1,-1
!         int=int+0.5*h(i+1)*dyB(i+1)+0.5*h(i)*dyB(i)
!         idpdy(i)=int
!      end do
!
!   endif
!
!   if (int_press_type == 2) then ! plume
!
!     surface plume
!      if (plume_type .eq. 1) then
!         do i=nlev,1,-1
!            idpdx(i) = plume_slope_x*(buoy(i)-buoy(1))
!            idpdy(i) = plume_slope_y*(buoy(i)-buoy(1))
!         end do
!      end if
!
!     bottom plume
!      if (plume_type .eq. 2) then
!         do i=nlev,1,-1
!            idpdx(i) = -plume_slope_x*(buoy(nlev)-buoy(i))
!            idpdy(i) = -plume_slope_y*(buoy(nlev)-buoy(i))
!         end do
!      end if
!
!   endif
!EOC
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.util.density import DensityState, get_rho

__all__ = [
    "INT_PRESS_GRADIENTS",
    "INT_PRESS_NONE",
    "INT_PRESS_PLUME",
    "PLUME_BOTTOM",
    "PLUME_SURFACE",
    "internal_pressure",
]

# int_press_type selector constants (mirror GOTM Fortran observations module)
INT_PRESS_NONE: int = 0  # no internal pressure gradient
INT_PRESS_GRADIENTS: int = 1  # horizontal T/S gradient method
INT_PRESS_PLUME: int = 2  # sloping-frame plume method

# plume_type selector constants
PLUME_SURFACE: int = 1  # buoyant surface plume (under shelf ice)
PLUME_BOTTOM: int = 2  # dense bottom plume over sloping topography


def internal_pressure(
    state: MeanflowState,
    density: DensityState,
    nlev: int,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
    int_press_type: int = INT_PRESS_NONE,
    dsdx: np.ndarray | None = None,
    dsdy: np.ndarray | None = None,
    dtdx: np.ndarray | None = None,
    dtdy: np.ndarray | None = None,
    plume_type: int = 0,
    plume_slope_x: float = 0.0,
    plume_slope_y: float = 0.0,
) -> None:
    """Compute the internal (baroclinic) pressure gradient.

    Translates internal_pressure.F90 verbatim.  Writes the result into the
    caller-supplied output arrays ``idpdx`` and ``idpdy`` (both shape nlev+1,
    1-indexed layers 1..nlev).  Index 0 is never written.

    int_press_type == INT_PRESS_NONE (0):
        No-op — leaves ``idpdx`` and ``idpdy`` unchanged.

    int_press_type == INT_PRESS_GRADIENTS (1):
        T/S gradient method (flat-bottom scenario).  Buoyancy gradients are
        estimated at each layer via a finite-difference perturbation of the
        equation of state, then vertically integrated from the surface
        downward with a trapezoidal scheme.

        Requires: ``dsdx``, ``dsdy``, ``dtdx``, ``dtdy`` — arrays of shape
        (nlev+1,) giving horizontal T/S gradients at each layer centre.

    int_press_type == INT_PRESS_PLUME (2):
        Sloping-frame plume method.  Uses the buoyancy profile already stored
        in ``state.buoy`` together with the prescribed slope parameters.

        plume_type == PLUME_SURFACE (1) — buoyant surface plume:
            idpdx(i) =  plume_slope_x * (buoy(i) - buoy(1))
            idpdy(i) =  plume_slope_y * (buoy(i) - buoy(1))

        plume_type == PLUME_BOTTOM (2) — dense bottom plume:
            idpdx(i) = -plume_slope_x * (buoy(nlev) - buoy(i))
            idpdy(i) = -plume_slope_y * (buoy(nlev) - buoy(i))

    Parameters
    ----------
    state:
        MeanflowState carrying T, S, h, buoy, gravity.
    density:
        DensityState for the equation of state (used by INT_PRESS_GRADIENTS).
    nlev:
        Number of vertical model layers.
    idpdx:
        Output array, shape (nlev+1,).  Layers 1..nlev are written;
        index 0 is left untouched.  Units: m/s².
    idpdy:
        Output array, shape (nlev+1,).  Same convention as idpdx.
    int_press_type:
        Method selector: INT_PRESS_NONE (0), INT_PRESS_GRADIENTS (1),
        INT_PRESS_PLUME (2).
    dsdx, dsdy, dtdx, dtdy:
        Horizontal T/S gradient profiles, shape (nlev+1,), required for
        INT_PRESS_GRADIENTS.  Units: [g/kg/m] for salinity, [K/m] for T.
    plume_type:
        Sub-selector for INT_PRESS_PLUME: PLUME_SURFACE (1) or PLUME_BOTTOM (2).
    plume_slope_x, plume_slope_y:
        Dimensionless along-slope gradient components for the plume method.
    """
    assert state.h is not None
    assert state.T is not None
    assert state.S is not None
    assert state.buoy is not None

    if int_press_type == INT_PRESS_GRADIENTS:
        _internal_pressure_gradients(
            state=state,
            density=density,
            nlev=nlev,
            idpdx=idpdx,
            idpdy=idpdy,
            dsdx=dsdx if dsdx is not None else np.zeros(nlev + 1),
            dsdy=dsdy if dsdy is not None else np.zeros(nlev + 1),
            dtdx=dtdx if dtdx is not None else np.zeros(nlev + 1),
            dtdy=dtdy if dtdy is not None else np.zeros(nlev + 1),
        )

    elif int_press_type == INT_PRESS_PLUME:
        _internal_pressure_plume(
            state=state,
            nlev=nlev,
            idpdx=idpdx,
            idpdy=idpdy,
            plume_type=plume_type,
            plume_slope_x=plume_slope_x,
            plume_slope_y=plume_slope_y,
        )

    # INT_PRESS_NONE: no-op


def _internal_pressure_gradients(
    state: MeanflowState,
    density: DensityState,
    nlev: int,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
    dsdx: np.ndarray,
    dsdy: np.ndarray,
    dtdx: np.ndarray,
    dtdy: np.ndarray,
) -> None:
    """T/S gradient method — translates int_press_type==1 branch of Fortran."""
    h = state.h
    T = state.T
    S = state.S
    assert h is not None
    assert T is not None
    assert S is not None
    gravity = state.gravity
    rho0 = density.rho0

    # Reset output arrays (Fortran: idpdx = _ZERO_; idpdy = _ZERO_)
    idpdx[1 : nlev + 1] = 0.0
    idpdy[1 : nlev + 1] = 0.0

    # Finite-difference step for buoyancy gradient estimation (arbitrary but
    # "small enough" per the Fortran comment; cancels in the ratio dB/dx).
    dx = 10.0
    dy = 10.0

    dxB = np.zeros(nlev + 1)
    dyB = np.zeros(nlev + 1)

    # Step from the surface downward, tracking the pressure depth z.
    z = 0.0
    for i in range(nlev, 0, -1):
        z += 0.5 * h[i]

        # buoyancy gradient in x direction
        dSS = dx * dsdx[i]
        dTT = dx * dtdx[i]
        Bl = -gravity * (get_rho(density, S[i], T[i], z) - rho0) / rho0
        Br = -gravity * (get_rho(density, S[i] + dSS, T[i] + dTT, z) - rho0) / rho0
        dxB[i] = (Br - Bl) / dx

        # buoyancy gradient in y direction
        dSS = dy * dsdy[i]
        dTT = dy * dtdy[i]
        Bl = -gravity * (get_rho(density, S[i], T[i], z) - rho0) / rho0
        Br = -gravity * (get_rho(density, S[i] + dSS, T[i] + dTT, z) - rho0) / rho0
        dyB[i] = (Br - Bl) / dy

        z += 0.5 * h[i]

    # Internal pressure gradient in x direction — trapezoidal integration from
    # the surface down.
    # idpdx(nlev) = 0.5*h(nlev)*dxB(nlev)
    # idpdx(i)    = idpdx(i+1) + 0.5*h(i+1)*dxB(i+1) + 0.5*h(i)*dxB(i)
    acc = 0.5 * h[nlev] * dxB[nlev]
    idpdx[nlev] = acc
    for i in range(nlev - 1, 0, -1):
        acc += 0.5 * h[i + 1] * dxB[i + 1] + 0.5 * h[i] * dxB[i]
        idpdx[i] = acc

    # Internal pressure gradient in y direction
    acc = 0.5 * h[nlev] * dyB[nlev]
    idpdy[nlev] = acc
    for i in range(nlev - 1, 0, -1):
        acc += 0.5 * h[i + 1] * dyB[i + 1] + 0.5 * h[i] * dyB[i]
        idpdy[i] = acc


def _internal_pressure_plume(
    state: MeanflowState,
    nlev: int,
    idpdx: np.ndarray,
    idpdy: np.ndarray,
    plume_type: int,
    plume_slope_x: float,
    plume_slope_y: float,
) -> None:
    """Plume method — translates int_press_type==2 branch of Fortran."""
    buoy = state.buoy
    assert buoy is not None

    if plume_type == PLUME_SURFACE:
        # Buoyant surface plume: vanishes at the bottom (k=1), maximum at surface
        for i in range(1, nlev + 1):
            idpdx[i] = plume_slope_x * (buoy[i] - buoy[1])
            idpdy[i] = plume_slope_y * (buoy[i] - buoy[1])

    elif plume_type == PLUME_BOTTOM:
        # Dense bottom plume: vanishes at the surface (k=nlev), maximum at bottom
        for i in range(1, nlev + 1):
            idpdx[i] = -plume_slope_x * (buoy[nlev] - buoy[i])
            idpdy[i] = -plume_slope_y * (buoy[nlev] - buoy[i])
