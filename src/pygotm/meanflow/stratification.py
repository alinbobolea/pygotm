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
from pygotm.util.density import DensityState, _is_legacy_lake_method, _legacy_density

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

    if _is_legacy_lake_method(density_state.density_method):
        assert state.buoy is not None
        assert density_state.rho is not None
        assert density_state.rho_p is not None

        z_face = 0.0
        z_center = 0.5 * h[nlev]
        rho = _legacy_density(
            density_state.density_method,
            float(S[nlev]),
            float(T[nlev]),
            z_center / 10.0,
        )
        state.buoy[nlev] = -g * (rho - density_state.rho0) / density_state.rho0
        density_state.rho[nlev] = rho
        density_state.rho_p[nlev] = rho

        for i_lake in range(nlev - 1, 0, -1):
            dz_lake = 0.5 * (h[i_lake] + h[i_lake + 1])
            z_face += h[i_lake + 1]
            z_center += dz_lake
            p_face = z_face / 10.0

            denom = h[i_lake + 1] + h[i_lake]
            Sface = (S[i_lake + 1] * h[i_lake] + S[i_lake] * h[i_lake + 1]) / denom
            Tface = (T[i_lake + 1] * h[i_lake] + T[i_lake] * h[i_lake + 1]) / denom

            rho_p = _legacy_density(
                density_state.density_method,
                float(Sface),
                float(T[i_lake + 1]),
                p_face,
            )
            rho_m = _legacy_density(
                density_state.density_method,
                float(Sface),
                float(T[i_lake]),
                p_face,
            )
            buoy_p = -g * (rho_p - density_state.rho0) / density_state.rho0
            buoy_m = -g * (rho_m - density_state.rho0) / density_state.rho0
            state.NNT[i_lake] = (buoy_p - buoy_m) / dz_lake

            rho_p = _legacy_density(
                density_state.density_method,
                float(S[i_lake + 1]),
                float(Tface),
                p_face,
            )
            rho_m = _legacy_density(
                density_state.density_method,
                float(S[i_lake]),
                float(Tface),
                p_face,
            )
            buoy_p = -g * (rho_p - density_state.rho0) / density_state.rho0
            buoy_m = -g * (rho_m - density_state.rho0) / density_state.rho0
            state.NNS[i_lake] = (buoy_p - buoy_m) / dz_lake
            state.NN[i_lake] = state.NNT[i_lake] + state.NNS[i_lake]

            rho = _legacy_density(
                density_state.density_method,
                float(S[i_lake]),
                float(T[i_lake]),
                z_center / 10.0,
            )
            state.buoy[i_lake] = -g * (rho - density_state.rho0) / density_state.rho0
            density_state.rho[i_lake] = rho
            density_state.rho_p[i_lake] = rho

        state.NNT[0] = 0.0
        state.NNT[nlev] = 0.0
        state.NNS[0] = 0.0
        state.NNS[nlev] = 0.0
        state.NN[0] = 0.0
        state.NN[nlev] = 0.0
        return

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
