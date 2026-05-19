"""
Constant-buoyancy-frequency temperature profile — translation of ``const_NNT.F90``.

Constructs a temperature profile such that the squared buoyancy frequency
:math:`N^2` equals a prescribed constant value ``NN`` [s⁻²] throughout the
water column, given a uniform background salinity ``S_const``.  The thermal
expansion coefficient :math:`\\alpha` is evaluated at each grid interface via
:func:`~pygotm.util.density.get_alpha`, and iterated once per level for
accuracy.

Used to initialise temperature when the GOTM YAML method is set to
``buoyancy``.

Original author: Lars Umlauf.
"""

from __future__ import annotations

import numpy as np

from pygotm.util.density import DensityState, get_alpha

__all__ = ["const_NNT"]


def const_NNT(
    density_state: DensityState,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    T_top: float,
    S_const: float,
    NN: float,
    gravity: float,
    T: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a temperature profile with constant buoyancy frequency."""

    profile = (
        np.zeros(nlev + 1, dtype=np.float64)
        if T is None
        else np.asarray(T, dtype=np.float64).copy()
    )
    profile[nlev] = T_top
    for i in range(nlev - 1, 0, -1):
        lalpha = get_alpha(density_state, S_const, profile[i + 1], -zi[i])
        profile[i] = profile[i + 1] - (NN * (z[i + 1] - z[i])) / (gravity * lalpha)
        lalpha = get_alpha(
            density_state,
            S_const,
            0.5 * (profile[i + 1] + profile[i]),
            -zi[i],
        )
        profile[i] = profile[i + 1] - (NN * (z[i + 1] - z[i])) / (gravity * lalpha)
    return profile
