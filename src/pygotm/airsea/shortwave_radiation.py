# ruff: noqa: E501
r"""
!-----------------------------------------------------------------------
!BOP
!
! !ROUTINE: Calculate the short--wave radiation \label{sec:swr}
!
! !INTERFACE:
!   REALTYPE function shortwave_radiation(zenith_angle,yday,dlon,dlat,cloud)
!
! !DESCRIPTION:
!  This subroutine calculates the short--wave net radiation based on
!  solar zenith angle, year day, longitude, latitude, and fractional cloud cover.
!  No corrections for albedo - must be done by calls to {\tt albedo\_water()} and
!  if ice is included {\tt albedo\_ice()}.
!  The basic formula for the short-wave radiation at the surface, $Q_s$,
!  has been taken from \cite{RosatiMiyacoda88}, who adapted the work
!  of \cite{Reed77} and \cite{SimpsonPaulson99}:
!
!  \begin{equation}
!  Q_s=Q_{tot} (1-0.62 C + 0.0019 \beta) (1-\alpha),
!  \end{equation}
!
!  with the total radiation reaching the surface under clear skies,
!  $Q_{tot}$, the fractional cloud cover, $C$, the solar noon altitude,
!  $\beta$, and the albedo, $\alpha$.
!  This piece of code has been taken the MOM-I (Modular Ocean Model)
!  version at the INGV (Istituto Nazionale di Geofisica e Vulcanologia,
!  see {\tt http://www.bo.ingv.it/}).
!
! !USES:
!   IMPLICIT NONE
!
! !INPUT PARAMETERS:
!   REALTYPE, intent(in)                :: zenith_angle
!   integer, intent(in)                 :: yday
!   REALTYPE, intent(in)                :: dlon,dlat
!   REALTYPE, intent(in)                :: cloud
!
! !REVISION HISTORY:
!  Original author(s): Karsten Bolding & Hans Burchard
!EOP
!-----------------------------------------------------------------------
!BOC
!   coszen = cos(deg2rad*zenith_angle)
!   if (coszen .le. 0.0) then
!      coszen = 0.0
!      qatten = 0.0
!   else
!      qatten = tau**(_ONE_/coszen)
!   end if
!
!   qzer  = coszen * solar
!   qdir  = qzer * qatten
!   qdiff = ((_ONE_-aozone)*qzer - qdir) * 0.5
!   qtot  =  qdir + qdiff
!
!  from now on everything in radians
!   rlon = deg2rad*dlon
!   rlat = deg2rad*dlat
!
!   yrdays=365.
!   eqnx = (yday-81.)/yrdays*2.*pi
!  sin of the solar noon altitude in radians :
!   sunbet=sin(rlat)*sin(eclips*sin(eqnx))+cos(rlat)*cos(eclips*sin(eqnx))
!  solar noon altitude in degrees :
!   sunbet = asin(sunbet)*rad2deg
!
!  radiation as from Reed(1977), Simpson and Paulson(1979)
!  calculates SHORT WAVE FLUX ( watt/m*m )
!  Rosati,Miyakoda 1988 ; eq. 3.8
!  clouds from COADS perpetual data set
!   qshort  = qtot*(1-0.62*cloud + .0019*sunbet)
!   if(qshort .gt. qtot ) then
!      qshort  = qtot
!   end if
!   shortwave_radiation = qshort
!
!   return
!   end function shortwave_radiation
!EOC
!
!-----------------------------------------------------------------------
! Copyright by the GOTM-team under the GNU Public License - www.gnu.org
!-----------------------------------------------------------------------
"""

from __future__ import annotations

import math

from pygotm.constants import DEG_TO_RAD, PI, RAD_TO_DEG, SOLAR_CONSTANT_W_M2

__all__ = ["shortwave_radiation"]

_ECLIPTIC_OBLIQUITY = 23.439 * DEG_TO_RAD
_TAU = 0.7
_AOZONE = 0.09


def shortwave_radiation(
    zenith_angle: float,
    yday: int,
    dlon: float,
    dlat: float,
    cloud: float,
) -> float:
    """Return short-wave radiation at the sea surface in W/m²."""

    coszen = math.cos(DEG_TO_RAD * zenith_angle)
    if coszen <= 0.0:
        coszen = 0.0
        qatten = 0.0
    else:
        qatten = _TAU ** (1.0 / coszen)

    qzer = coszen * SOLAR_CONSTANT_W_M2
    qdir = qzer * qatten
    qdiff = ((1.0 - _AOZONE) * qzer - qdir) * 0.5
    qtot = qdir + qdiff

    rlat = DEG_TO_RAD * dlat
    eqnx = (yday - 81.0) / 365.0 * 2.0 * PI
    sunbet = math.sin(rlat) * math.sin(_ECLIPTIC_OBLIQUITY * math.sin(eqnx)) + (
        math.cos(rlat) * math.cos(_ECLIPTIC_OBLIQUITY * math.sin(eqnx))
    )
    sunbet = math.asin(sunbet) * RAD_TO_DEG

    qshort = qtot * (1.0 - 0.62 * cloud + 0.0019 * sunbet)
    if qshort > qtot:
        qshort = qtot

    return qshort
