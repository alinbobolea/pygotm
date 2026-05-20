# ruff: noqa: E501
"""
Humidity and air-density calculation — translation of ``humidity.F90``.

Updates an :class:`~pygotm.airsea.airsea_variables.AirSeaState` object with:

* ``es`` — saturation vapour pressure at SST [Pa], corrected for seawater
  salinity (factor 0.98, Kraus 1972).
* ``qs`` — saturation specific humidity at SST [kg kg⁻¹].
* ``ea`` — actual vapour pressure at air temperature [Pa].
* ``qa`` — actual specific humidity [kg kg⁻¹].
* ``rhoa`` — moist-air density [kg m⁻³].

Saturation vapour pressure uses the Lowe (1977, J. Appl. Met.) seven-term
polynomial, converted from millibar to Pascal.

Four humidity input methods are supported via ``hum_method`` (``gotm.yaml``
key ``hum_method``):

1. Relative humidity in percent.
2. Wet-bulb temperature [°C or K]; psychrometer formula from Smithsonian
   Meteorological Tables, 6th edition, p. 366, eq. 3.
3. Dew-point temperature [°C or K].
4. Specific humidity in kg kg⁻¹ (direct input).

Temperature inputs > 100 are treated as Kelvin; values ≤ 100 as Celsius.

Original FORTRAN authors: Adolf Stips, Hans Burchard, Karsten Bolding.
"""

from __future__ import annotations

from pygotm.airsea.airsea_variables import AirSeaState, const06, kelvin, rgas

__all__ = ["humidity"]

_A1 = 6.107799961
_A2 = 4.436518521e-1
_A3 = 1.428945805e-2
_A4 = 2.650648471e-4
_A5 = 3.031240396e-6
_A6 = 2.034080948e-8
_A7 = 6.136820929e-11


def _saturation_vapor_pressure(temp_c: float) -> float:
    """Return saturation vapour pressure in Pascal for a temperature in Celsius."""

    pressure_mb = _A1 + temp_c * (
        _A2
        + temp_c
        * (_A3 + temp_c * (_A4 + temp_c * (_A5 + temp_c * (_A6 + temp_c * _A7))))
    )
    return pressure_mb * 100.0


def _specific_humidity(airp: float, vapour_pressure: float) -> float:
    """Return specific humidity in kg/kg from total and vapour pressure in Pascal."""

    return const06 * vapour_pressure / (airp - 0.377 * vapour_pressure)


def humidity(
    state: AirSeaState,
    hum_method: int,
    hum: float,
    airp: float,
    tw: float,
    ta: float,
) -> None:
    """Update ``state`` with humidity variables following ``humidity.F90``.

    Parameters
    ----------
    state:
        Shared air-sea module state updated in place.
    hum_method:
        GOTM humidity selector.
        `1`: relative humidity in percent.
        `2`: wet-bulb temperature.
        `3`: dew-point temperature.
        `4`: specific humidity in kg/kg.
    hum:
        Humidity input associated with ``hum_method``.
    airp:
        Air pressure [Pa].
    tw:
        Sea-surface temperature [Celsius].
    ta:
        Air temperature [Celsius].
    """

    state.ta = ta

    state.es = 0.98 * _saturation_vapor_pressure(tw)
    state.qs = _specific_humidity(airp, state.es)

    if hum_method == 1:
        rh = 0.01 * hum
        state.ea = rh * _saturation_vapor_pressure(ta)
        state.qa = _specific_humidity(airp, state.ea)
    elif hum_method == 2:
        twet = hum if hum < 100.0 else hum - kelvin
        state.ea = _saturation_vapor_pressure(twet)
        state.ea = state.ea - 6.6e-4 * (1.0 + 1.15e-3 * twet) * airp * (ta - twet)
        state.qa = _specific_humidity(airp, state.ea)
    elif hum_method == 3:
        dew = hum if hum < 100.0 else hum - kelvin
        state.ea = _saturation_vapor_pressure(dew)
        state.qa = _specific_humidity(airp, state.ea)
    elif hum_method == 4:
        state.qa = hum
        state.ea = state.qa * airp / (const06 + 0.378 * state.qa)
    else:
        msg = "not a valid hum_method"
        raise ValueError(msg)

    state.rhoa = airp / (rgas * (ta + kelvin) * (1.0 + const06 * state.qa))
