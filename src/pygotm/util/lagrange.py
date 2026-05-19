"""
Lagrangian particle random walk — translation of ``lagrange.F90``.

Implements the Visser (1997) random-walk scheme for spatially inhomogeneous
turbulence.  Each active particle is advanced from :math:`z^n` to
:math:`z^{n+1}` by:

.. math::

   z^{n+1} = z^n + \\partial_z \\nu_t(z^n)\\,\\Delta t
            + R\\left[2\\,r^{-1}\\,\\nu_t\\!\\left(z^n
              + \\tfrac{1}{2}\\partial_z\\nu_t(z^n)\\,\\Delta t\\right)
              \\Delta t\\right]^{1/2}

where :math:`R` is a zero-mean random variable with variance
:math:`\\langle R^2 \\rangle = r`.

Fixed parameters: background viscosity ``VISC_BACK = 0.0e-6`` m²/s,
random-walk variance ``RND_VAR = 1/3``.  Semi-implicit viscosity correction
(``visc_corr``) is disabled, matching the Fortran default.  Reflective
boundary conditions are applied at the surface and bottom.

Original authors: Hans Burchard, Karsten Bolding.

Public interface: :func:`lagrange`, :data:`VISC_BACK`, :data:`RND_VAR`.
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
