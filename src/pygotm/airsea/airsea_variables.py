"""
Shared constants and state for the air-sea module — translation of ``airsea_variables.F90``.

Declares the public physical constants (``cpa``, ``cpw``, ``emiss``, ``bolz``,
``kelvin``, ``const06``, ``rgas``, ``g``, ``rho_0``, ``kappa``), integer
method selectors for albedo (``CONST``, ``PAYNE``, ``COGLEY``) and longwave
radiation (``CLARK``, ``HASTENRATH_LAMB``, ``BIGNAMI``, ``BERLIAND_BERLIAND``,
``JOSEY1``, ``JOSEY2``), and the mutable :class:`AirSeaState` object that
carries intermediate humidity and vapour-pressure values between routines.

Original authors: Karsten Bolding, Hans Burchard.
"""

from __future__ import annotations

from pygotm.constants import (
    AIRSEA_EMISSIVITY,
    AIRSEA_REFERENCE_DENSITY_KG_M3,
    AIRSEA_SPECIFIC_HEAT_AIR_J_KG_K,
    AIRSEA_SPECIFIC_HEAT_WATER_J_KG_K,
    AIRSEA_VON_KARMAN,
    DRY_AIR_GAS_CONSTANT_J_KG_K,
    HUMIDITY_MOLAR_MASS_RATIO,
    KELVIN_OFFSET_C,
    STANDARD_GRAVITY_M_S2,
    STEFAN_BOLTZMANN_CONSTANT_W_M2_K4,
)

__all__ = [
    "AirSeaState",
    "BIGNAMI",
    "BERLIAND_BERLIAND",
    "CLARK",
    "COGLEY",
    "CONST",
    "HASTENRATH_LAMB",
    "JOSEY1",
    "JOSEY2",
    "PAYNE",
    "bolz",
    "const06",
    "cpa",
    "cpw",
    "emiss",
    "g",
    "kappa",
    "kelvin",
    "rgas",
    "rho_0",
]


# Source: gotm-model/code/src/airsea/airsea_variables.F90
cpa = AIRSEA_SPECIFIC_HEAT_AIR_J_KG_K
# Source: gotm-model/code/src/airsea/airsea_variables.F90
cpw = AIRSEA_SPECIFIC_HEAT_WATER_J_KG_K
# Source: gotm-model/code/src/airsea/airsea_variables.F90
emiss = AIRSEA_EMISSIVITY
# Source: gotm-model/code/src/airsea/airsea_variables.F90
bolz = STEFAN_BOLTZMANN_CONSTANT_W_M2_K4
# Source: gotm-model/code/src/airsea/airsea_variables.F90
kelvin = KELVIN_OFFSET_C
# Source: gotm-model/code/src/airsea/airsea_variables.F90
const06 = HUMIDITY_MOLAR_MASS_RATIO
# Source: gotm-model/code/src/airsea/airsea_variables.F90
rgas = DRY_AIR_GAS_CONSTANT_J_KG_K
# Source: gotm-model/code/src/airsea/airsea_variables.F90
g = STANDARD_GRAVITY_M_S2
# Source: gotm-model/code/src/airsea/airsea_variables.F90
rho_0 = AIRSEA_REFERENCE_DENSITY_KG_M3
# Source: gotm-model/code/src/airsea/airsea_variables.F90
kappa = AIRSEA_VON_KARMAN

# Albedo selector constants from airsea_variables.F90
CONST = 0
PAYNE = 1
COGLEY = 2

# Longwave-radiation selector constants from airsea_variables.F90
CLARK = 3
HASTENRATH_LAMB = 4
BIGNAMI = 5
BERLIAND_BERLIAND = 6
JOSEY1 = 7
JOSEY2 = 8


class AirSeaState:
    """Mutable public state declared by ``airsea_variables.F90``.

    This class stores the public air-sea scalars that are shared across the
    translated air-sea routines. Parameter constants remain module-level names
    to preserve the Fortran import style used throughout GOTM.
    """

    def __init__(self) -> None:
        self.es: float = 0.0
        self.ea: float = 0.0
        self.qs: float = 0.0
        self.qa: float = 0.0
        self.L: float = 0.0
        self.rhoa: float = 0.0
        self.ta: float = 0.0
        self.rain_impact: bool = False
        self.calc_evaporation: bool = False
