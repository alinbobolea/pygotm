"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: The total water balance \\label{sec:water_balance}
!
! !DESCRIPTION:
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding
!
!EOP
!-----------------------------------------------------------------------
"""

from typing import cast

import numba
import numpy as np

from pygotm.meanflow.hypsograph import HypsographState, vc2zi
from pygotm.meanflow.meanflow import MeanflowState
from pygotm.observations.streams import StreamsState

__all__ = [
    "WATER_BALANCE_ALLLAYERS",
    "WATER_BALANCE_NONE",
    "WATER_BALANCE_SURFACE",
    "WATER_BALANCE_ZETA",
    "step_water_balance_single",
    "water_balance",
]

WATER_BALANCE_NONE = 0
WATER_BALANCE_SURFACE = 1
WATER_BALANCE_ALLLAYERS = 2
WATER_BALANCE_ZETA = 3


@numba.njit(cache=True, fastmath=False)
def step_water_balance_single(
    nlev: int,
    dt: float,
    lake: int,
    water_balance_method: int,
    precip: float,
    evap: float,
    int_inflow: float,
    int_outflow: float,
    Af: np.ndarray,
    Vc: np.ndarray,
    Qlayer: np.ndarray,
    Qres: np.ndarray,
    scalars: np.ndarray,
) -> None:
    """Update residual stream forcing and integrated lake water balance."""

    for i in range(nlev + 1):
        Qres[i] = 0.0

    if lake != 0:
        int_flows = (int_inflow + int_outflow) / Af[nlev]
        scalars[2] += int_flows

        sum_vc = 0.0
        sum_q = 0.0
        for i in range(1, nlev + 1):
            sum_vc += Vc[i]
            sum_q += Qlayer[i]
        net_water_balance = sum_q + Af[nlev] * (evap + precip)

        if water_balance_method == WATER_BALANCE_SURFACE:
            Qres[nlev] = -net_water_balance
        elif water_balance_method == WATER_BALANCE_ALLLAYERS:
            for i in range(1, nlev + 1):
                Qres[i] = -net_water_balance * Vc[i] / sum_vc
        elif water_balance_method == WATER_BALANCE_ZETA:
            # The zeta branch needs the hypsograph inverse and is handled by
            # the Python wrapper where the HypsographState object is available.
            pass

        scalars[0] = net_water_balance
        scalars[1] += dt * net_water_balance
        scalars[3] = int_flows
    else:
        net_water_balance = evap + precip
        scalars[0] = net_water_balance
        scalars[1] += dt * net_water_balance


def water_balance(
    meanflow: MeanflowState,
    streams: StreamsState,
    nlev: int,
    dt: float,
    *,
    precip: float,
    evap: float,
    Qlayer: np.ndarray,
    Qres: np.ndarray,
    zeta_method: int = 0,
) -> float | None:
    """Calculate the lake or ocean water balance for the current step."""

    assert meanflow.Af is not None
    assert meanflow.Vc is not None

    scalars = np.asarray(
        [
            meanflow.net_water_balance,
            meanflow.int_water_balance,
            meanflow.int_fwf,
            meanflow.int_flows,
        ],
        dtype=np.float64,
    )
    step_water_balance_single(
        nlev,
        dt,
        1 if meanflow.lake else 0,
        meanflow.water_balance_method,
        precip,
        evap,
        streams.int_inflow,
        streams.int_outflow,
        meanflow.Af,
        meanflow.Vc,
        Qlayer,
        Qres,
        scalars,
    )

    new_zeta: float | None = None
    if meanflow.lake and meanflow.water_balance_method == WATER_BALANCE_ZETA:
        if zeta_method != 3:
            raise ValueError("WATER_BALANCE_ZETA requires zeta_method=3")
        if meanflow.hypsograph is None:
            raise ValueError("WATER_BALANCE_ZETA requires a hypsograph")
        hypsograph = cast(HypsographState, meanflow.hypsograph)
        sum_vc = float(np.sum(meanflow.Vc[1 : nlev + 1]))
        sum_q = float(np.sum(Qlayer[1 : nlev + 1]))
        net_water_balance = sum_q + float(meanflow.Af[nlev]) * (evap + precip)
        vc1 = np.asarray([0.0, sum_vc + dt * net_water_balance], dtype=np.float64)
        zi1 = np.zeros(2, dtype=np.float64)
        vc2zi(hypsograph, 1, meanflow.depth0, vc1, zi1)
        new_zeta = float(zi1[1])
        scalars[0] = net_water_balance
        scalars[1] = meanflow.int_water_balance + dt * net_water_balance

    meanflow.net_water_balance = float(scalars[0])
    meanflow.int_water_balance = float(scalars[1])
    meanflow.int_fwf = float(scalars[2])
    meanflow.int_flows = float(scalars[3])
    return new_zeta
