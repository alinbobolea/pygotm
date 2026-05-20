"""
Constant-buoyancy-frequency salinity profile — translation of ``const_NNS.F90``.

Constructs a salinity profile such that the squared buoyancy frequency
:math:`N^2` equals a prescribed constant value ``NN`` [s⁻²] throughout the
water column, given a uniform background temperature ``T_const``.  The haline
contraction coefficient :math:`\\beta` is evaluated at each grid interface via
:func:`~pygotm.util.density.get_beta`, and iterated once per level for
accuracy.

Used to initialise salinity when the GOTM YAML method is set to ``buoyancy``.

Original FORTRAN author: Lars Umlauf.
"""

from __future__ import annotations

import numpy as np

from pygotm.util.density import DensityState, get_beta

__all__ = ["const_NNS"]


def const_NNS(
    density_state: DensityState,
    nlev: int,
    z: np.ndarray,
    zi: np.ndarray,
    S_top: float,
    T_const: float,
    NN: float,
    gravity: float,
    S: np.ndarray | None = None,
) -> np.ndarray:
    """Construct a salinity profile with constant buoyancy frequency."""

    profile = (
        np.zeros(nlev + 1, dtype=np.float64)
        if S is None
        else np.asarray(S, dtype=np.float64).copy()
    )
    profile[nlev] = S_top
    for i in range(nlev - 1, 0, -1):
        lbeta = get_beta(density_state, profile[i + 1], T_const, -zi[i])
        profile[i] = profile[i + 1] + (NN * (z[i + 1] - z[i])) / (gravity * lbeta)
        lbeta = get_beta(
            density_state,
            0.5 * (profile[i + 1] + profile[i]),
            T_const,
            -zi[i],
        )
        profile[i] = profile[i + 1] + (NN * (z[i + 1] - z[i])) / (gravity * lbeta)
    return profile
