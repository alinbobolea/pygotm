# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculate the solar zenith angle \label{sec:swr}
!
! !INTERFACE:
!   REALTYPE function solar_zenith_angle(yday,hh,dlon,dlat)
!
! !DESCRIPTION:
!  This subroutine calculates the solar zenith angle as being used both
!  in albedo_water() and shortwave_radiation(). The result is in degrees.
!
! !USES:
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   integer, intent(in)                 :: yday
!   REALTYPE, intent(in)                :: hh
!   REALTYPE, intent(in)                :: dlon,dlat
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding
!
! !LOCAL VARIABLES:
!   REALTYPE, parameter       :: pi=3.14159265358979323846
!   REALTYPE, parameter       :: deg2rad=pi/180.
!   REALTYPE, parameter       :: rad2deg=180./pi
!
!   REALTYPE                  :: rlon,rlat
!   REALTYPE                  :: yrdays
!   REALTYPE                  :: th0,th02,th03,sundec
!   REALTYPE                  :: thsun,coszen
!EOP
!-----------------------------------------------------------------------
!BOC
!  from now on everything in radians
!   rlon = deg2rad*dlon
!   rlat = deg2rad*dlat
!
!   yrdays=365.25
!
!   th0 = 2.*pi*yday/yrdays
!   th02 = 2.*th0
!   th03 = 3.*th0
!  sun declination :
!   sundec = 0.006918 - 0.399912*cos(th0) + 0.070257*sin(th0)         &
!           - 0.006758*cos(th02) + 0.000907*sin(th02)                 &
!           - 0.002697*cos(th03) + 0.001480*sin(th03)
!  sun hour angle :
!   thsun = (hh-12.)*15.*deg2rad + rlon
!
!  cosine of the solar zenith angle :
!   coszen =sin(rlat)*sin(sundec)+cos(rlat)*cos(sundec)*cos(thsun)
!   if (coszen .lt. _ZERO_) coszen = _ZERO_
!
!   solar_zenith_angle = rad2deg*acos(coszen)
!
!   return
!   end function solar_zenith_angle
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math

from pygotm.constants import DEG_TO_RAD, PI, RAD_TO_DEG

__all__ = ["solar_zenith_angle"]


def solar_zenith_angle(yday: int, hh: float, dlon: float, dlat: float) -> float:
    """Return the solar zenith angle in degrees."""

    rlon = DEG_TO_RAD * dlon
    rlat = DEG_TO_RAD * dlat

    yrdays = 365.25
    th0 = 2.0 * PI * yday / yrdays
    th02 = 2.0 * th0
    th03 = 3.0 * th0

    sundec = (
        0.006918
        - 0.399912 * math.cos(th0)
        + 0.070257 * math.sin(th0)
        - 0.006758 * math.cos(th02)
        + 0.000907 * math.sin(th02)
        - 0.002697 * math.cos(th03)
        + 0.001480 * math.sin(th03)
    )
    thsun = (hh - 12.0) * 15.0 * DEG_TO_RAD + rlon

    coszen = (
        math.sin(rlat) * math.sin(sundec)
        + math.cos(rlat) * math.cos(sundec) * math.cos(thsun)
    )
    if coszen < 0.0:
        coszen = 0.0

    return RAD_TO_DEG * math.acos(coszen)
