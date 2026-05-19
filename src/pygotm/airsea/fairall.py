# ruff: noqa: E501
"""
Fairall et al. (1996) COARE bulk fluxes — translation of ``fairall.F90``.

Computes the surface momentum flux vector :math:`(\\tau_x, \\tau_y)` [N m⁻²],
latent heat flux :math:`Q_e` [W m⁻²], and sensible heat flux :math:`Q_h`
[W m⁻²] according to Fairall et al. (1996a), built on the
Liu–Katsaros–Businger (Liu et al. 1979) method.  Cool-skin and warm-layer
effects follow Fairall et al. (1996b).

Temperature inputs (``airt``, ``sst``) may be in Celsius or Kelvin; values
above 100 are treated as Kelvin.

Adapted from the COARE code originally written by David Rutgers and Frank
Bradley.  Original GOTM Python port by Adolf Stips.
"""

from __future__ import annotations

import math

from pygotm.airsea.airsea_variables import (
    AirSeaState,
    const06,
    cpa,
    cpw,
    g,
    kappa,
    kelvin,
    rgas,
    rho_0,
)

__all__ = ["fairall", "psi"]

_FDG = 1.0
_BETA = 1.2
_ZABL = 600.0
_R3 = 1.0 / 3.0
_ZT = 2.0
_ZQ = 2.0
_ZW = 10.0
_ITERMAX = 20
_WGUST = 0.0

_LIU_A = (
    (0.177, 0.292),
    (1.376, 1.808),
    (1.026, 1.393),
    (1.625, 1.956),
    (4.661, 4.994),
    (34.904, 30.709),
    (1667.190, 1448.680),
    (588000.0, 298000.0),
)
_LIU_B = (
    (0.0, 0.0),
    (0.929, 0.826),
    (-0.599, -0.528),
    (-1.018, -0.870),
    (-1.475, -1.297),
    (-2.067, -1.845),
    (-2.907, -2.682),
    (-3.935, -3.616),
)
_LIU_RR = (0.0, 0.11, 0.825, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def fairall(
    state: AirSeaState,
    sst: float,
    airt: float,
    u10: float,
    v10: float,
    precip: float,
) -> tuple[float, float, float, float, float]:
    """Return ``(evap, taux, tauy, qe, qh)`` from GOTM's Fairall bulk flux code."""

    evap = 0.0
    w = math.hypot(u10, v10)

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

    qe = 0.0
    qh = 0.0
    taux = 0.0
    tauy = 0.0
    delw = math.sqrt(w * w + _WGUST * _WGUST)

    if delw != 0.0:
        vis_air = 1.326e-5 * (1.0 + ta * (6.542e-3 + ta * (8.301e-6 - 4.84e-9 * ta)))
        latent_heat = (2.501 - 0.00237 * tw) * 1.0e6
        ier = 0
        delq = state.qa - state.qs
        delt = ta - tw
        zwol = 0.0
        zow = 0.0005
        wstar = 0.04 * delw
        tstar = 0.04 * delt
        qstar = 0.04 * delq
        tvstar = tstar * (1.0 + 0.61 * state.qa) + 0.61 * ta_k * qstar
        ri = g * _ZW * (delt + 0.61 * ta_k * delq) / (ta_k * delw * delw)
        wgus = 0.0

        if ri <= 0.25:
            for _ in range(_ITERMAX):
                if ier >= 0:
                    ol = (
                        g
                        * kappa
                        * tvstar
                        / (ta_k * (1.0 + 0.61 * state.qa) * wstar * wstar)
                    )
                    zwol = _ZW * ol
                    ztol = _ZT * ol
                    zqol = _ZQ * ol

                    wpsi = psi(1, zwol)
                    tpsi = psi(2, ztol)
                    qpsi = psi(2, zqol)

                    zow = 0.011 * wstar * wstar / g + 0.11 * vis_air / wstar
                    wstar = delw * kappa / (math.log(_ZW / zow) - wpsi)

                    rr = zow * wstar / vis_air
                    if 0.0 <= rr < 1000.0:
                        rt = 0.0
                        rq = 0.0
                        for k in range(8):
                            if _LIU_RR[k] <= rr < _LIU_RR[k + 1]:
                                rt = _LIU_A[k][0] * rr ** _LIU_B[k][0]
                                rq = _LIU_A[k][1] * rr ** _LIU_B[k][1]
                                break

                        cff = vis_air / wstar
                        zot = rt * cff
                        zoq = rq * cff
                        cff = kappa * _FDG
                        tstar = delt * cff / (math.log(_ZT / zot) - tpsi)
                        qstar = delq * cff / (math.log(_ZQ / zoq) - qpsi)

                        tvstar = tstar * (1.0 + 0.61 * state.qa) + 0.61 * ta_k * qstar
                        bf = -g / ta_k * wstar * tvstar
                        if bf > 0.0:
                            wgus = _BETA * (bf * _ZABL) ** _R3
                        else:
                            wgus = 0.0
                        delw = math.sqrt(w * w + wgus * wgus)
                    else:
                        ier = -2

            if ier >= 0:
                wspeed = math.sqrt(w * w + wgus * wgus)
                cd = wstar * wstar / (wspeed * wspeed)
                qh = cpa * state.rhoa * wstar * tstar

                if state.rain_impact:
                    rainfall = precip * 1000.0
                    x1 = 2.11e-5 * (ta_k / kelvin) ** 1.94
                    x2 = (
                        0.02411
                        * (1.0 + ta * (3.309e-3 - 1.44e-6 * ta))
                        / (state.rhoa * cpa)
                    )
                    x3 = state.qa * latent_heat / (rgas * ta_k * ta_k)
                    cd_rain = 1.0 / (
                        1.0 + const06 * (x3 * latent_heat * x1) / (cpa * x2)
                    )
                    cd_rain = (
                        cd_rain
                        * cpw
                        * ((tw - ta) + (state.qs - state.qa) * latent_heat / cpa)
                    )
                    qh = qh - rainfall * cd_rain

                qe = latent_heat * state.rhoa * wstar * qstar
                upvel = (
                    -1.61 * wstar * qstar
                    - (1.0 + 1.61 * state.qa) * wstar * tstar / ta_k
                )
                qe = qe - state.rhoa * latent_heat * upvel * state.qa

                if state.rain_impact and state.calc_evaporation:
                    evap = state.rhoa / rho_0 * wstar * qstar
                else:
                    evap = 0.0

                cff = state.rhoa * cd * wspeed
                taux = cff * u10
                tauy = cff * v10

                if state.rain_impact:
                    rainfall = precip * 1000.0
                    tmp = 0.85 * rainfall
                    taux = taux + tmp * u10
                    tauy = tauy + tmp * v10

    return evap, taux, tauy, qe, qh


def psi(iflag: int, zol: float) -> float:
    """Evaluate the Fairall/Liu stability function ``psi``."""

    psi_value = 0.0

    if zol < 0.0:
        chik = (1.0 - 16.0 * zol) ** 0.25
        if iflag == 1:
            psik = (
                2.0 * math.log(0.5 * (1.0 + chik))
                + math.log(0.5 * (1.0 + chik * chik))
                - 2.0 * math.atan(chik)
                + 0.5 * math.pi
            )
        elif iflag == 2:
            psik = 2.0 * math.log(0.5 * (1.0 + chik * chik))
        else:
            msg = f"invalid iflag={iflag}"
            raise ValueError(msg)

        sqr3 = 1.7320508
        chic = (1.0 - 12.87 * zol) ** _R3
        psic = (
            1.5 * math.log(_R3 * (1.0 + chic + chic * chic))
            - sqr3 * math.atan((1.0 + 2.0 * chic) / sqr3)
            + math.pi / sqr3
        )
        fw = 1.0 / (1.0 + zol * zol)
        psi_value = fw * psik + (1.0 - fw) * psic
    elif zol > 0.0:
        psi_value = -4.7 * zol

    return psi_value
