"""Constants for pyGOTM ice thermodynamics kernels.

Values in this module come from the model papers kept outside the package tree
under the local ``papers/`` directory:

* Winton (2000), "A Reformulated Three-Layer Sea Ice Model", for the three-layer
  sea-ice heat-capacity and optics constants.
* Holland and Jenkins (1999), "Modeling Thermodynamic Ice-Ocean Interactions at
  the Base of an Ice Shelf", for ice-ocean exchange and basal-melt constants.
* McDougall and Jackett (2002), "Accurate and Computationally Efficient
  Algorithms for Potential Temperature and Density of Seawater", for the
  potential-temperature freezing polynomial used by the basal-melt closure.

Sign convention: positive atmospheric heat flux means heat leaves the ocean;
the ice modules report positive ``ocean_ice_heat_flux`` when ice extracts heat
from the ocean.  See :doc:`/physics/ice_thermodynamics` for the full sign
convention description.
"""

RHO_ICE: float = 910.0
RHO_WATER: float = 1025.0
RHO_SNOW: float = 330.0

L_ICE: float = 333_500.0
K_ICE: float = 2.03
K_SNOW: float = 0.31
C_ICE: float = 2100.0
C_WATER: float = 4200.0
C_WATER_VOL: float = 4.18e6

FREEZE_SLOPE: float = 0.0575
MU_TS: float = 0.054

ALB_OCEAN: float = 0.06
ALB_SNOW: float = 0.85
ALB_ICE: float = 0.5826
PEN_ICE: float = 0.3
OPT_DEP_ICE: float = 0.67
T_RANGE_MELT: float = 1.0
WINTON_ICE_SALINITY: float = 1.0
WINTON_FREEZE_TEMP: float = -1.8

LEBEDEV_FAC: float = 1.33
LEBEDEV_EXP: float = 0.58
LEBEDEV_ALBEDO: float = 0.545
LEBEDEV_ATTN: float = -1.6

MAX_FRAZIL: float = 0.03
MYLAKE_ATTN: float = 5.0
MIN_ICE_THICKNESS: float = 1.0e-6

# McDougall-Jackett potential-temperature freezing polynomial written as
# T_b = LAMBDA1 * S_b + LAMBDA2 + LAMBDA3 * depth_or_ice_draft.
LAMBDA1: float = -5.6705121472405570e-2
LAMBDA2: float = 7.5436448744204881e-2
LAMBDA3: float = 7.6828512903539831e-4

C_ICE_BASAL: float = 1995.0
C_WATER_BASAL: float = 4180.0
T_ICE_CORE: float = -20.0
RHO_ICE_BASAL: float = 910.0
RHO_WATER_BASAL: float = 1030.0

# Holland-Jenkins turbulent-exchange factors. Multiplying by ustar [m/s] gives
# exchange velocities [m/s]. DEFAULT_BASAL_USTAR reproduces the common
# gamma_T ~= 1e-4 m/s and gamma_S ~= 5e-7 m/s scale.
GAMMA_T: float = 1.0e-2
GAMMA_S: float = 5.05e-5
DEFAULT_BASAL_USTAR: float = 1.0e-2
