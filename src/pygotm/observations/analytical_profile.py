"""
Analytical two-layer vertical profile — translation of ``analytical_profile.F90``.

Creates a piecewise-linear vertical profile ``prof`` with:

* value ``v1`` in a surface layer from the surface down to depth ``z1``;
* value ``v2`` in a bottom layer from depth ``z2`` to the sea floor;
* values linearly interpolated between ``v1`` and ``v2`` in the intermediate
  layer from ``z1`` to ``z2``.

Used to initialise temperature or salinity when the GOTM YAML method is set to
``two_layer``.

Original FORTRAN author: Karsten Bolding.
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
