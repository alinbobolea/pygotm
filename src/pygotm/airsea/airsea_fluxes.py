# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Wrapper for air-sea fluxes calculations \label{sec:airsea-fluxes}
!
! !INTERFACE:
!   subroutine airsea_fluxes(method,sst,airt,u10,v10,precip, &
!                            evap,taux,tauy,qe,qh)
!
! !DESCRIPTION:
!  A wrapper around the different methods for calculating momentum
!  fluxes and sensible and latent heat fluxes at the air-sea interface.
!  To have a complete air-sea exchange also the short wave radiation
!  and longwave-wave radiation must be calculated.
!EOP
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

from pygotm.airsea.airsea_variables import AirSeaState
from pygotm.airsea.fairall import fairall
from pygotm.airsea.kondo import kondo

__all__ = ["FAIRALL", "KONDO", "airsea_fluxes"]

KONDO = 1
FAIRALL = 2


def airsea_fluxes(
    method: int,
    state: AirSeaState,
    sst: float,
    airt: float,
    u10: float,
    v10: float,
    precip: float,
) -> tuple[float, float, float, float, float]:
    """Dispatch to the selected GOTM bulk flux routine."""

    if method == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if method == KONDO:
        return kondo(state, sst, airt, u10, v10, precip)
    if method == FAIRALL:
        return fairall(state, sst, airt, u10, v10, precip)

    msg = f"invalid airsea flux method={method}"
    raise ValueError(msg)
