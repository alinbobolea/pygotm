"""Shared physical constants used across translated GOTM modules.

These constants intentionally record the numeric defaults present in the
authoritative Fortran sources instead of silently substituting generic textbook
values. When GOTM uses different defaults in different modules, both values are
kept here with their source noted explicitly.
"""

from __future__ import annotations

import math

__all__ = [
    "AIRSEA_EMISSIVITY",
    "AIRSEA_REFERENCE_DENSITY_KG_M3",
    "AIRSEA_SPECIFIC_HEAT_AIR_J_KG_K",
    "AIRSEA_SPECIFIC_HEAT_WATER_J_KG_K",
    "AIRSEA_VON_KARMAN",
    "DRY_AIR_GAS_CONSTANT_J_KG_K",
    "DENSITY_MODULE_REFERENCE_DENSITY_KG_M3",
    "HUMIDITY_MOLAR_MASS_RATIO",
    "KELVIN_OFFSET_C",
    "PI",
    "RAD_TO_DEG",
    "SOLAR_CONSTANT_W_M2",
    "STANDARD_GRAVITY_M_S2",
    "STEFAN_BOLTZMANN_CONSTANT_W_M2_K4",
    "DEG_TO_RAD",
]


# Source: gotm-model/code/src/airsea/airsea_variables.F90
AIRSEA_SPECIFIC_HEAT_AIR_J_KG_K = 1008.0
# Source: gotm-model/code/src/airsea/airsea_variables.F90
AIRSEA_SPECIFIC_HEAT_WATER_J_KG_K = 3985.0
# Source: gotm-model/code/src/airsea/airsea_variables.F90
AIRSEA_EMISSIVITY = 0.97
# Source: gotm-model/code/src/airsea/airsea_variables.F90
STEFAN_BOLTZMANN_CONSTANT_W_M2_K4 = 5.670374419e-8
# Source: gotm-model/code/src/airsea/airsea_variables.F90
KELVIN_OFFSET_C = 273.15
# Source: gotm-model/code/src/airsea/airsea_variables.F90
HUMIDITY_MOLAR_MASS_RATIO = 0.62198
# Source: gotm-model/code/src/airsea/airsea_variables.F90
DRY_AIR_GAS_CONSTANT_J_KG_K = 287.1
# Source: gotm-model/code/src/airsea/airsea_variables.F90
STANDARD_GRAVITY_M_S2 = 9.81
# Source: gotm-model/code/src/airsea/airsea_variables.F90
AIRSEA_REFERENCE_DENSITY_KG_M3 = 1025.0
# Source: gotm-model/code/src/util/density.F90
DENSITY_MODULE_REFERENCE_DENSITY_KG_M3 = 1027.0
# Source: gotm-model/code/src/airsea/airsea_variables.F90
AIRSEA_VON_KARMAN = 0.41

# Source: gotm-model/code/src/airsea/solar_zenith_angle.F90
PI = math.pi
# Source: gotm-model/code/src/airsea/solar_zenith_angle.F90
DEG_TO_RAD = PI / 180.0
# Source: gotm-model/code/src/airsea/solar_zenith_angle.F90
RAD_TO_DEG = 180.0 / PI
# Source: gotm-model/code/src/airsea/shortwave_radiation.F90
SOLAR_CONSTANT_W_M2 = 1350.0
