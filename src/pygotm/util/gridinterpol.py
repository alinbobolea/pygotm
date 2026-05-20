"""
Grid interpolation from observation space to model grid — translation of
``gridinterpol.F90``.

Linearly interpolates (and extrapolates) observational data defined on an
arbitrary structured depth grid to the (potentially moving) GOTM model grid.
Extrapolation clamps values: model levels above the topmost observation receive
the topmost observed value; model levels below the deepest observation receive
the deepest observed value.

Original FORTRAN authors: Karsten Bolding, Hans Burchard.

Public interface: :func:`gridinterpol`.
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
