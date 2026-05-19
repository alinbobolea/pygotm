# ruff: noqa: E501
"""
Kondo (1975) bulk flux parameterisation — translation of ``kondo.F90``.

Computes transfer coefficients for the surface momentum flux vector
:math:`(\\tau_x, \\tau_y)` (:math:`c_{dd}`), latent heat flux :math:`Q_e`
(:math:`c_{ed}`), and sensible heat flux :math:`Q_h` (:math:`c_{hd}`)
following the Kondo (1975) bulk formulae.

Transfer coefficients are selected from five wind-speed regimes
(0–2.2, 2.2–5, 5–8, 8–25, and >25 m s⁻¹) and then modified by a bulk
Richardson-number stability correction.  Rain impact on sensible heat and wind
stress is applied optionally when ``state.rain_impact`` is ``True``.
"""

from __future__ import annotations

import math

from pygotm.airsea.airsea_variables import (
    AirSeaState,
    const06,
    cpa,
    cpw,
    kelvin,
    rgas,
    rho_0,
)

__all__ = ["kondo"]

_AE_D = (0.0, 0.771, 0.867, 1.2, 0.0)
_AE_H = (0.0, 0.927, 1.15, 1.17, 1.652)
_AE_E = (0.0, 0.969, 1.18, 1.196, 1.68)
_BE_D = (1.08, 0.0858, 0.0667, 0.025, 0.073)
_BE_H = (1.185, 0.0546, 0.01, 0.0075, -0.017)
_BE_E = (1.23, 0.0521, 0.01, 0.008, -0.016)
_CE_H = (0.0, 0.0, 0.0, -0.00045, 0.0)
_CE_E = (0.0, 0.0, 0.0, -0.0004, 0.0)
_PE_D = (-0.15, 1.0, 1.0, 1.0, 1.0)
_PE_H = (-0.157, 1.0, 1.0, 1.0, 1.0)
_PE_E = (-0.16, 1.0, 1.0, 1.0, 1.0)
_EPS = 1.0e-12


def kondo(
    state: AirSeaState,
    sst: float,
    airt: float,
    u10: float,
    v10: float,
    precip: float,
) -> tuple[float, float, float, float, float]:
    """Return ``(evap, taux, tauy, qe, qh)`` from GOTM's Kondo bulk flux code."""

    evap = 0.0
    w = math.hypot(u10, v10)
    latent_heat = (2.5 - 0.00234 * sst) * 1.0e6

    if sst < 100.0:
        tw = sst
    else:
        tw = sst - kelvin

    if airt < 100.0:
        ta = airt
        ta_k = airt + kelvin
    else:
        ta = airt - kelvin
        ta_k = airt

    s0 = 0.25 * (sst - airt) / (w + 1.0e-10) ** 2
    s = s0 * abs(s0) / (abs(s0) + 0.01)

    if w < 2.2:
        x = math.log(w + _EPS)
        cdd = (_BE_D[0] * math.exp(_PE_D[0] * x)) * 1.0e-3
        chd = (_BE_H[0] * math.exp(_PE_H[0] * x)) * 1.0e-3
        ced = (_BE_E[0] * math.exp(_PE_E[0] * x)) * 1.0e-3
    elif w < 5.0:
        x = math.exp(math.log(w + _EPS))
        cdd = (_AE_D[1] + _BE_D[1] * x) * 1.0e-3
        chd = (_AE_H[1] + _BE_H[1] * x) * 1.0e-3
        ced = (_AE_E[1] + _BE_E[1] * x) * 1.0e-3
    elif w < 8.0:
        x = math.exp(math.log(w + _EPS))
        cdd = (_AE_D[2] + _BE_D[2] * x) * 1.0e-3
        chd = (_AE_H[2] + _BE_H[2] * x) * 1.0e-3
        ced = (_AE_E[2] + _BE_E[2] * x) * 1.0e-3
    elif w < 25.0:
        x = math.exp(math.log(w + _EPS))
        cdd = (_AE_D[3] + _BE_D[3] * x) * 1.0e-3
        chd = (_AE_H[3] + _BE_H[3] * x + _CE_H[3] * (w - 8.0) ** 2) * 1.0e-3
        ced = (_AE_E[3] + _BE_E[3] * x + _CE_E[3] * (w - 8.0) ** 2) * 1.0e-3
    else:
        x = math.exp(math.log(w + _EPS))
        cdd = (_AE_D[4] + _BE_D[4] * x) * 1.0e-3
        chd = (_AE_H[4] + _BE_H[4] * x) * 1.0e-3
        ced = (_AE_E[4] + _BE_E[4] * x) * 1.0e-3

    if s < 0.0:
        if s > -3.3:
            x = 0.1 + 0.03 * s + 0.9 * math.exp(4.8 * s)
        else:
            x = 0.0
        cdd = x * cdd
        chd = x * chd
        ced = x * ced
    else:
        cdd = cdd * (1.0 + 0.47 * math.sqrt(s))
        chd = chd * (1.0 + 0.63 * math.sqrt(s))
        ced = ced * (1.0 + 0.63 * math.sqrt(s))

    qh = -chd * cpa * state.rhoa * w * (sst - airt)
    qe = -ced * latent_heat * state.rhoa * w * (state.qs - state.qa)

    if state.rain_impact:
        rainfall = precip * 1000.0
        x1 = 2.11e-5 * (ta_k / kelvin) ** 1.94
        x2 = 0.02411 * (1.0 + ta * (3.309e-3 - 1.44e-6 * ta)) / (state.rhoa * cpa)
        x3 = state.qa * latent_heat / (rgas * ta_k * ta_k)
        cd_rain = 1.0 / (1.0 + const06 * (x3 * latent_heat * x1) / (cpa * x2))
        cd_rain = (
            cd_rain * cpw * ((tw - ta) + (state.qs - state.qa) * latent_heat / cpa)
        )
        qh = qh - rainfall * cd_rain

        if state.calc_evaporation:
            evap = state.rhoa / rho_0 * ced * w * (state.qa - state.qs)

    tmp = cdd * state.rhoa * w
    taux = tmp * u10
    tauy = tmp * v10

    if state.rain_impact:
        rainfall = precip * 1000.0
        tmp = 0.85 * rainfall
        taux = taux + tmp * u10
        tauy = tauy + tmp * v10

    return evap, taux, tauy, qe, qh
