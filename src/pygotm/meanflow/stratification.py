r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculation of the stratification\label{sec:stratification}
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

import numpy as np

from pygotm.meanflow.meanflow import MeanflowState
from pygotm.util.density import DensityState

__all__ = [
    "stratification",
]


def stratification(
    state: MeanflowState,
    density_state: DensityState,
    nlev: int,
) -> None:
    r"""Compute buoyancy frequency squared (N²) at layer interfaces.

    Updates ``state.NN``, ``state.NNT``, and ``state.NNS`` in-place.
    """
    assert state.h is not None
    assert state.T is not None
    assert state.S is not None
    assert state.NN is not None
    assert state.NNT is not None
    assert state.NNS is not None
    assert density_state.alpha is not None
    assert density_state.beta is not None

    h = state.h
    T = state.T
    S = state.S
    alpha = density_state.alpha
    beta = density_state.beta
    g = state.gravity

    i = np.arange(1, nlev)

    idz = 2.0 / (h[i] + h[i + 1])
    dT = T[i + 1] - T[i]
    dS = S[i + 1] - S[i]

    state.NNT[i] = alpha[i] * g * dT * idz
    state.NNS[i] = -beta[i] * g * dS * idz
    state.NN[i] = state.NNT[i] + state.NNS[i]

    state.NNT[0] = 0.0
    state.NNT[nlev] = 0.0
    state.NNS[0] = 0.0
    state.NNS[nlev] = 0.0
    state.NN[0] = 0.0
    state.NN[nlev] = 0.0
