# ruff: noqa: E501
"""
Dispatcher for air-sea bulk flux calculations — translation of ``airsea_fluxes.F90``.

Routes to either the Kondo (1975) or Fairall et al. (1996) bulk-flux routine
based on the integer ``method`` selector.  Returns evaporation rate, wind
stress components ``(taux, tauy)``, latent heat flux ``qe``, and sensible heat
flux ``qh``.  Shortwave and longwave radiation are handled separately by their
own modules.
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
