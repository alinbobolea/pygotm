Air–Sea Interaction
===================

The ``pygotm.airsea`` package computes the surface boundary conditions for every
physics timestep: wind stress, sensible heat, latent heat, shortwave and longwave
radiation, freshwater flux (precipitation and evaporation), and bottom drag.  It is
a direct Python translation of the GOTM Fortran module hierarchy under
``gotm-model/code/src/airsea/``.

All functions are pure Python (no Numba) and operate on scalar ``float64`` values.
They are called from within the compiled Numba time loop by first extracting the
per-step forcing scalars, computing fluxes, and then passing the results back as
pre-computed arrays.

Modules at a glance
-------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Module
     - Purpose
   * - :mod:`pygotm.airsea.airsea`
     - Integration driver: ``do_airsea``, ``surface_fluxes``, ``AirSeaDriverState``
   * - :mod:`pygotm.airsea.airsea_variables`
     - Shared state object ``AirSeaState``, method selector constants, physical constants
   * - :mod:`pygotm.airsea.airsea_fluxes`
     - Dispatcher that routes to Kondo or Fairall bulk flux routines
   * - :mod:`pygotm.airsea.kondo`
     - Kondo (1975) bulk momentum, sensible, and latent heat fluxes
   * - :mod:`pygotm.airsea.fairall`
     - Fairall et al. (1996) COARE bulk fluxes with cool-skin / warm-layer corrections
   * - :mod:`pygotm.airsea.humidity`
     - Saturation vapour pressure, specific humidity, and air density
   * - :mod:`pygotm.airsea.longwave_radiation`
     - Net longwave (back-radiation) by six parameterisation methods
   * - :mod:`pygotm.airsea.shortwave_radiation`
     - Shortwave radiation at the surface (Reed 1977 / Rosati–Miyakoda 1988)
   * - :mod:`pygotm.airsea.solar_zenith_angle`
     - Solar zenith angle from time, longitude, and latitude
   * - :mod:`pygotm.airsea.albedo_water`
     - Sea-surface albedo (constant, Payne 1972, or Cogley 1979)

.. _airsea-driver:

Integration Driver — ``airsea``
--------------------------------

``airsea.py`` is the entry point called by the GOTM time loop.  It owns
``AirSeaDriverState``, the full air-sea configuration and runtime state, and
exposes the high-level functions ``do_airsea`` and ``surface_fluxes`` that are
called once per timestep.

Per-timestep sequence executed by ``do_airsea``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On each physics timestep the driver performs the following operations in order:

1. **Humidity** — calls :func:`~pygotm.airsea.humidity.humidity` to update
   saturation vapour pressure at SST (``es``, ``qs``), actual vapour pressure
   at air temperature (``ea``, ``qa``), and moist-air density (``rhoa``).
2. **Shortwave radiation** — if ``shortwave_method`` is not 0, calls
   :func:`~pygotm.airsea.solar_zenith_angle.solar_zenith_angle` and
   :func:`~pygotm.airsea.shortwave_radiation.shortwave_radiation`, then
   subtracts the reflected fraction via
   :func:`~pygotm.airsea.albedo_water.albedo_water`.
3. **Longwave radiation** — if ``longwave_method`` is not 0, calls
   :func:`~pygotm.airsea.longwave_radiation.longwave_radiation` to compute the
   net back-radiation ``ql``.
4. **Bulk fluxes** — calls :func:`~pygotm.airsea.airsea_fluxes.airsea_fluxes`
   (which dispatches to Kondo or Fairall) to compute evaporation, wind stress
   :math:`(\tau_x, \tau_y)`, latent heat ``qe``, and sensible heat ``qh``.
5. **Net heat flux** — assembles ``heat = shortwave + longwave + qe + qh``.
6. **SST observation nudge** — optionally overwrites ``sst`` from an
   observational forcing series.

.. automodule:: pygotm.airsea.airsea
   :members:
   :undoc-members:
   :show-inheritance:

.. _airsea-variables:

Shared State and Constants — ``airsea_variables``
--------------------------------------------------

``airsea_variables.py`` translates ``airsea_variables.F90``.  It declares
module-level physical constants and the mutable ``AirSeaState`` object that
carries intermediate humidity and vapour-pressure values between the humidity,
longwave, and bulk-flux routines.

Physical constants
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 45 15 25

   * - Name
     - Description
     - Units
     - Value source
   * - ``cpa``
     - Specific heat of air at constant pressure
     - J kg⁻¹ K⁻¹
     - ``AIRSEA_SPECIFIC_HEAT_AIR_J_KG_K``
   * - ``cpw``
     - Specific heat of seawater
     - J kg⁻¹ K⁻¹
     - ``AIRSEA_SPECIFIC_HEAT_WATER_J_KG_K``
   * - ``emiss``
     - Ocean surface emissivity
     - —
     - ``AIRSEA_EMISSIVITY``
   * - ``bolz``
     - Stefan–Boltzmann constant
     - W m⁻² K⁻⁴
     - ``STEFAN_BOLTZMANN_CONSTANT_W_M2_K4``
   * - ``kelvin``
     - Celsius-to-Kelvin offset (273.15)
     - K
     - ``KELVIN_OFFSET_C``
   * - ``const06``
     - Molar mass ratio water/dry air (≈ 0.622)
     - —
     - ``HUMIDITY_MOLAR_MASS_RATIO``
   * - ``rgas``
     - Specific gas constant for dry air
     - J kg⁻¹ K⁻¹
     - ``DRY_AIR_GAS_CONSTANT_J_KG_K``
   * - ``g``
     - Standard gravity
     - m s⁻²
     - ``STANDARD_GRAVITY_M_S2``
   * - ``rho_0``
     - Reference seawater density
     - kg m⁻³
     - ``AIRSEA_REFERENCE_DENSITY_KG_M3``
   * - ``kappa``
     - von Kármán constant
     - —
     - ``AIRSEA_VON_KARMAN``

Method selector constants
~~~~~~~~~~~~~~~~~~~~~~~~~~

Albedo selectors (set via ``gotm.yaml`` key ``albedo_method``):

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Constant
     - Value
     - Meaning
   * - ``CONST``
     - 0
     - Fixed zero albedo (no reflection correction)
   * - ``PAYNE``
     - 1
     - Payne (1972) — interpolation in tabulated zenith-angle values
   * - ``COGLEY``
     - 2
     - Cogley (1979) — bilinear interpolation in zenith angle and month

Longwave radiation selectors (set via ``gotm.yaml`` key ``longwave_method``):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Constant
     - Value
     - Reference
   * - ``CLARK``
     - 3
     - Clark et al. (1974)
   * - ``HASTENRATH_LAMB``
     - 4
     - Hastenrath & Lamb (1978)
   * - ``BIGNAMI``
     - 5
     - Bignami et al. (1995)
   * - ``BERLIAND_BERLIAND``
     - 6
     - Berliand & Berliand (1952)
   * - ``JOSEY1``
     - 7
     - Josey et al. (2003) equation J1 (eq. 9)
   * - ``JOSEY2``
     - 8
     - Josey et al. (2003) equation J2 (eq. 14)

.. automodule:: pygotm.airsea.airsea_variables
   :members:
   :undoc-members:
   :show-inheritance:

.. _airsea-bulk-fluxes:

Bulk Momentum and Heat Flux Schemes
-------------------------------------

Bulk flux dispatcher — ``airsea_fluxes``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``airsea_fluxes.py`` is the thin dispatcher that routes to the selected bulk
scheme.  It translates ``airsea_fluxes.F90`` and its two selector constants:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Constant
     - Value
     - Scheme
   * - ``KONDO``
     - 1
     - Kondo (1975) — five wind-speed regimes, stability correction
   * - ``FAIRALL``
     - 2
     - Fairall et al. (1996) COARE — Liu–Katsaros–Businger iteration

All bulk schemes return the tuple ``(evap, taux, tauy, qe, qh)``:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Output
     - Units
     - Description
   * - ``evap``
     - m s⁻¹
     - Evaporation rate (positive upward); non-zero only when ``rain_impact`` and ``calc_evaporation`` are both enabled
   * - ``taux``
     - N m⁻²
     - Eastward wind stress
   * - ``tauy``
     - N m⁻²
     - Northward wind stress
   * - ``qe``
     - W m⁻²
     - Latent heat flux (negative = ocean loses heat)
   * - ``qh``
     - W m⁻²
     - Sensible heat flux (negative = ocean loses heat)

.. automodule:: pygotm.airsea.airsea_fluxes
   :members:
   :undoc-members:
   :show-inheritance:

Kondo (1975) bulk fluxes — ``kondo``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates ``kondo.F90``.  Transfer coefficients :math:`C_{dd}`, :math:`C_{hd}`,
and :math:`C_{ed}` are selected from five wind-speed regimes (0–2.2, 2.2–5,
5–8, 8–25, and >25 m s⁻¹) and then modified by a bulk Richardson-number
stability correction.  Rain impact on sensible heat and wind stress is applied
optionally when ``state.rain_impact`` is ``True``.

Latent heat of vaporisation depends on SST:

.. math::

   L = (2.5 - 0.00234 \cdot T_w) \times 10^6 \;\text{J kg}^{-1}

Stability parameter:

.. math::

   s = \frac{s_0 |s_0|}{|s_0| + 0.01}, \quad
   s_0 = \frac{T_w - T_a}{(|\mathbf{u}_{10}| + \varepsilon)^2}

For stable conditions (:math:`s < 0`), transfer coefficients are multiplied by
:math:`0.1 + 0.03s + 0.9 e^{4.8s}` (clamped to zero for :math:`s < -3.3`).
For unstable conditions (:math:`s > 0`), they are multiplied by
:math:`1 + 0.47\sqrt{s}` (drag) and :math:`1 + 0.63\sqrt{s}` (heat/moisture).

.. automodule:: pygotm.airsea.kondo
   :members:
   :undoc-members:
   :show-inheritance:

Fairall et al. (1996) COARE bulk fluxes — ``fairall``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates ``fairall.F90``.  Implements the COARE 2.0 algorithm of
Fairall et al. (1996a, b), adapted from the original COARE code by
David Rutgers and Frank Bradley.  The algorithm is built on the
Liu–Katsaros–Businger (Liu et al. 1979) roughness Reynolds number method
with up to 20 iterations (``_ITERMAX = 20``) to converge the Obukhov-length
stability correction.

Key internal parameters:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Value
     - Description
   * - ``_ZW``
     - 10.0 m
     - Wind reference height
   * - ``_ZT``, ``_ZQ``
     - 2.0 m
     - Temperature and humidity reference heights
   * - ``_ZABL``
     - 600.0 m
     - Atmospheric boundary layer height for gustiness
   * - ``_BETA``
     - 1.2
     - Gustiness coefficient
   * - ``_WGUST``
     - 0.0 m s⁻¹
     - Minimum gustiness wind speed (disabled)
   * - ``_ITERMAX``
     - 20
     - Maximum Obukhov-length iterations

The stability function :func:`~pygotm.airsea.fairall.psi` handles both
unstable (``zol < 0``, Paulson / convective blending) and stable (``zol > 0``,
linear :math:`-4.7\,\zeta`) regimes.  For ``zol >= 0.25`` (strongly stable),
bulk fluxes are set to zero and no iteration is performed.

.. automodule:: pygotm.airsea.fairall
   :members:
   :undoc-members:
   :show-inheritance:

.. _airsea-humidity:

Humidity — ``humidity``
------------------------

Translates ``humidity.F90``.  Updates ``AirSeaState`` with:

* ``es`` — saturation vapour pressure at SST [Pa] (corrected by factor 0.98 for
  seawater following Kraus 1972)
* ``qs`` — saturation specific humidity at SST [kg kg⁻¹]
* ``ea`` — actual vapour pressure at air temperature [Pa]
* ``qa`` — actual specific humidity [kg kg⁻¹]
* ``rhoa`` — moist-air density [kg m⁻³]

Saturation vapour pressure is computed from the Lowe (1977) polynomial
(coefficients ``_A1`` … ``_A7``) converted from millibar to Pascal.

Four humidity input methods are supported (``gotm.yaml`` key ``hum_method``):

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - Method
     - Input ``hum``
     - Conversion
   * - 1
     - Relative humidity [%]
     - :math:`e_a = (hum/100) \cdot e_\mathrm{sat}(T_a)`
   * - 2
     - Wet-bulb temperature [°C or K]
     - Smithsonian psychrometer formula (Met. Tables 6th ed., p. 366)
   * - 3
     - Dew-point temperature [°C or K]
     - :math:`e_a = e_\mathrm{sat}(T_\mathrm{dew})`
   * - 4
     - Specific humidity [kg kg⁻¹]
     - :math:`e_a = q_a p / (\varepsilon + 0.378 q_a)`

Moist-air density:

.. math::

   \rho_a = \frac{p}{R_\mathrm{air}\,(T_a + 273.15)\,(1 + \varepsilon\,q_a)}

where :math:`\varepsilon = 0.622` (``const06``) and :math:`R_\mathrm{air}`
is the specific gas constant for dry air (``rgas``).

.. automodule:: pygotm.airsea.humidity
   :members:
   :undoc-members:
   :show-inheritance:

.. _airsea-radiation:

Radiation
----------

Longwave radiation — ``longwave_radiation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates ``longwave_radiation.F90``.  Returns the **net** longwave (back)
radiation in W m⁻² (sign convention: negative when the ocean loses heat).
Six methods are available, selected by ``longwave_method`` (see
:ref:`airsea-variables` for the integer constants):

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Formula summary
   * - ``CLARK`` (3)
     - :math:`Q_l = -\varepsilon\sigma[(1-f_{ccf}C^2)T_w^4(0.39-0.05\sqrt{e_a}) + 4T_w^3(T_w-T_a)]`
   * - ``HASTENRATH_LAMB`` (4)
     - As Clark but uses :math:`0.056\sqrt{1000\,q_a}` instead of :math:`0.05\sqrt{e_a}`
   * - ``BIGNAMI`` (5)
     - :math:`Q_l = -\sigma[-\varepsilon_B T_a^4(0.653 + 0.00535\,e_a)(1+0.1762C^2) + \varepsilon T_w^4]`
   * - ``BERLIAND_BERLIAND`` (6)
     - :math:`Q_l = -\varepsilon\sigma[(1-0.6823C^2)T_a^4(0.39-0.05\sqrt{0.01e_a}) + 4T_a^3(T_w-T_a)]`
   * - ``JOSEY1`` (7)
     - Josey et al. (2003) eq. 9; effective sky temperature from cloud cover only
   * - ``JOSEY2`` (8)
     - Josey et al. (2003) eq. 14; effective sky temperature from cloud cover and dew-point

The cloud correction factor ``ccf`` in Clark/Hastenrath_Lamb/Berliand_Berliand
is read from a 91-element look-up table indexed by integer latitude (degrees).

All methods require ``state.ea`` and/or ``state.qa`` to be set by a prior call
to :func:`~pygotm.airsea.humidity.humidity`.

.. automodule:: pygotm.airsea.longwave_radiation
   :members:
   :undoc-members:
   :show-inheritance:

Shortwave radiation — ``shortwave_radiation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates ``shortwave_radiation.F90``.  Implements the Reed (1977) /
Rosati–Miyakoda (1988) formula for the net shortwave flux at the surface:

.. math::

   Q_s = Q_\mathrm{tot}(1 - 0.62\,C + 0.0019\,\beta)

where :math:`Q_\mathrm{tot} = Q_\mathrm{dir} + Q_\mathrm{diff}` is the clear-sky
total (direct + diffuse) radiation, :math:`C` is fractional cloud cover (0–1),
and :math:`\beta` is the solar noon altitude in degrees.

Internal constants: atmospheric transmittance :math:`\tau = 0.7`, ozone
absorption :math:`A_\mathrm{oz} = 0.09`, solar constant
:math:`S_0` from ``SOLAR_CONSTANT_W_M2``.

The result does **not** include the albedo correction; the caller subtracts
the reflected fraction using :func:`~pygotm.airsea.albedo_water.albedo_water`.

.. automodule:: pygotm.airsea.shortwave_radiation
   :members:
   :undoc-members:
   :show-inheritance:

Solar zenith angle — ``solar_zenith_angle``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates ``solar_zenith_angle.F90``.  Returns the solar zenith angle in
degrees from day-of-year, decimal hour UTC, longitude, and latitude.  Called
by both the shortwave and albedo routines.

Sun declination (in radians) from Spencer (1971) four-term Fourier series:

.. math::

   \delta = 0.006918 - 0.399912\cos\theta_0 + 0.070257\sin\theta_0
           - 0.006758\cos 2\theta_0 + 0.000907\sin 2\theta_0
           - 0.002697\cos 3\theta_0 + 0.001480\sin 3\theta_0

where :math:`\theta_0 = 2\pi\,d/365.25` and :math:`d` is day-of-year.
The hour-angle is :math:`h = (hh - 12)\cdot 15° + \lambda` (longitude in
degrees).  Cosine of zenith angle:

.. math::

   \cos\zeta = \sin\phi\sin\delta + \cos\phi\cos\delta\cos h

Clamped to zero (below horizon) before taking the arc-cosine.

.. automodule:: pygotm.airsea.solar_zenith_angle
   :members:
   :undoc-members:
   :show-inheritance:

.. _airsea-albedo:

Water-Surface Albedo — ``albedo_water``
----------------------------------------

Translates ``albedo_water.F90``.  Returns the fractional sea-surface albedo for
the three methods selected by the ``albedo_method`` integer constant (see
:ref:`airsea-variables`):

``CONST`` (0)
   Always returns 0.0.  Used when shortwave forcing is supplied as net radiation
   (albedo already removed in the observational product).

``PAYNE`` (1)
   Payne (1972) table of albedo values for 30°–40° N Atlantic.  The table has
   20 entries covering zenith angles from 90° down to 0° with non-uniform spacing
   (2° bins near the horizon, 10° bins near zenith).  Linear interpolation is
   performed within each bin.  This is the default for most GOTM test cases.

``COGLEY`` (2)
   Cogley (1979) table with bilinear interpolation in both solar zenith angle
   (10 entries, 10°-bin spacing) and month-of-year (12 monthly midpoints).
   Provides more accurate latitudinal and seasonal variation than Payne.

.. automodule:: pygotm.airsea.albedo_water
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

* :doc:`/physics/airsea` — physics derivation and sign conventions
* :doc:`/physics/biogeochemistry` — how ``shortwave`` and ``wind_speed`` are
  passed to pyfabm
* :doc:`/api/gotm` — ``AirSeaDriverState`` is owned by ``RuntimeState`` and
  its fields are written to NetCDF output
