r"""!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Lagrangian particle random walk \label{sec:lagrange}
!
! !INTERFACE:
!    subroutine lagrange(nlev,dt,zlev,nuh,w,npar,active,zi,zp)
!
! !DESCRIPTION:
!
! Here a Lagrangian particle random walk for spatially
! inhomogeneous turbulence according to \cite{Visser1997} is implemented.
! With the random walk, the particle $i$ is moved from the vertical
! position $z_i^n$ to $z_i^{n+1}$ according to the following algorithm:
! \begin{equation}
! \begin{array}{rcl}
! z_i^{n+1} &=&
! z^n_i + \partial_z \nu_t (z^n_i)\Delta t \\ \\
! &+&
! R \left\{2 r^{-1} \nu_t (z^n_i + \frac12  \partial_z \nu_t (z^n_i)\Delta t)
! \Delta t\right\}^{1/2},
! \end{array}
! \end{equation}
! where $R$ is a random process with $\langle R \rangle =0$ (zero mean) and
! and the variance $\langle R^2 \rangle=r$.
! Set {\tt visc\_corr=.true.} for
! evaluating eddy viscosity in a semi-implicit way. A background viscosity
! ({\tt visc\_back}) may be set. The variance $r$ of the random walk scheme
! ({\tt rnd\_var}) has to be set manually as well here.
!
! !USES:
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: nlev
!   REALTYPE, intent(in)                :: dt
!   REALTYPE, intent(in)                :: zlev(0:nlev)
!   REALTYPE, intent(in)                :: nuh(0:nlev)
!   REALTYPE, intent(in)                :: w
!   integer, intent(in)                 :: npar
!   logical, intent(in)                 :: active(npar)
!
! !INPUT/OUTPUT PARAMETERS:
!   integer, intent(inout)              :: zi(npar)
!   REALTYPE, intent(inout)             :: zp(npar)
!
! !REVISION HISTORY:
!  Original author(s): Hans Burchard & Karsten Bolding
!
! !LOCAL VARIABLES:
!   integer            :: i,n
!   REALTYPE           :: rnd(npar),rnd_var_inv
!   REALTYPE,parameter :: visc_back=0.e-6,rnd_var=0.333333333
!   REALTYPE           :: depth,dz(nlev),dzn(nlev),step,zp_old
!   REALTYPE           :: visc,rat,dt_inv,zloc
!   logical,parameter  :: visc_corr=.false.
!
!EOP
!-----------------------------------------------------------------------
!
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["lagrange", "VISC_BACK", "RND_VAR"]

# Fortran PARAMETER constants (visc_back, rnd_var from lagrange.F90)
VISC_BACK: float = 0.0e-6
RND_VAR: float = 0.333333333
_VISC_CORR: bool = False


def lagrange(
    nlev: int,
    dt: float,
    zlev: np.ndarray,
    nuh: np.ndarray,
    w: float,
    npar: int,
    active: np.ndarray,
    zi: np.ndarray,
    zp: np.ndarray,
    rng: np.random.Generator | None = None,
) -> None:
    """Lagrangian particle random walk for spatially inhomogeneous turbulence.

    Implements the Visser (1997) random-walk scheme. Each particle position
    zp[n] and its enclosing layer index zi[n] are updated in-place.

    The step formula is:
        z^{n+1} = z^n + (dzn[i] + w) * dt
                + sqrt(2 * rnd_var_inv * dt_inv * visc) * rnd * dt

    Reflective boundary conditions are applied at the surface (z=0) and
    bottom (z=-depth). Particle indices zi are 1-based (matching Fortran
    DIMENSION(0:nlev) convention).

    Parameters
    ----------
    nlev : int
        Number of model layers.
    dt : float
        Time step [s].
    zlev : np.ndarray, shape (nlev+1,)
        Layer interface depths [m], indexed 0..nlev. zlev[0] is the bottom.
    nuh : np.ndarray, shape (nlev+1,)
        Eddy diffusivity [m²/s] at layer interfaces, indexed 0..nlev.
    w : float
        Vertical velocity [m/s], positive upward.
    npar : int
        Number of particles.
    active : np.ndarray of bool, shape (npar,)
        Active flag per particle (passed for interface compatibility; not used
        inside the walk loop, matching GOTM Fortran behaviour).
    zi : np.ndarray of int, shape (npar,)
        Layer index (1-based) enclosing each particle. Modified in-place.
    zp : np.ndarray of float, shape (npar,)
        Particle vertical position [m]. Modified in-place.
    rng : np.random.Generator, optional
        Random number generator. If None, a default generator is used.
        Pass a seeded generator for reproducible results.

    !BOC
    !  dt_inv=1./dt
    !  rnd_var_inv=1./rnd_var
    !  call random_number(rnd)
    !  rnd=(2.*rnd-1.)
    !  do i=1,nlev
    !     dz(i)=zlev(i)-zlev(i-1)
    !     dzn(i)=(nuh(i)-nuh(i-1))/dz(i)
    !  end do
    !  depth=-zlev(0)
    !EOC
    """
    if rng is None:
        rng = np.random.default_rng()

    dt_inv = 1.0 / dt
    rnd_var_inv = 1.0 / RND_VAR

    # Uniform [-1, 1) random values matching Fortran: rnd=(2.*rnd-1.)
    rnd = rng.uniform(-1.0, 1.0, npar)

    dz = np.empty(nlev + 1)
    dzn = np.empty(nlev + 1)
    for i in range(1, nlev + 1):
        dz[i] = zlev[i] - zlev[i - 1]
        dzn[i] = (nuh[i] - nuh[i - 1]) / dz[i]

    depth = -zlev[0]

    for n in range(npar):
        # visc_corr=.false. → use particle's current position and level directly
        i = zi[n]
        zloc = zp[n]

        rat = (zloc - zlev[i - 1]) / dz[i]
        visc = rat * nuh[i] + (1.0 - rat) * nuh[i - 1]
        if visc < VISC_BACK:
            visc = VISC_BACK

        zp_old = zp[n]
        step = dt * (math.sqrt(2.0 * rnd_var_inv * dt_inv * visc) * rnd[n] + w + dzn[i])
        zp[n] = zp[n] + step

        # Reflective boundary conditions at surface (0) and bottom (-depth)
        while zp[n] < -depth or zp[n] > 0.0:
            if zp[n] < -depth:
                zp[n] = -depth + (-depth - zp[n])
            else:
                zp[n] = -zp[n]

        step = zp[n] - zp_old

        # Update layer index: search from current position in direction of motion
        if step > 0:
            idx = zi[n]
            while idx < nlev and zlev[idx] <= zp[n]:
                idx += 1
            zi[n] = idx
        else:
            idx = zi[n]
            while idx > 1 and zlev[idx - 1] >= zp[n]:
                idx -= 1
            zi[n] = idx
