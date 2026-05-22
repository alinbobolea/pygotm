Ice Thermodynamics
==================

pyGOTM provides five distinct sea-ice and lake-ice thermodynamics models (plus
an option to disable ice entirely) through the ``pygotm.icethm`` package.  All
models share a common :class:`~pygotm.icethm.IceState` container and a unified
dispatch function :func:`~pygotm.icethm.step_ice` compiled with Numba.  The
model is selected at run time via the ``surface.ice.model`` key in the GOTM
YAML configuration.

Overview
--------

The ice module sits between the atmospheric flux calculation and the ocean
temperature/salinity update.  At each timestep the dispatcher:

1. Receives the surface water temperature :math:`T_w` and salinity
   :math:`S_w`, the air temperature :math:`T_\mathrm{air}`, all four
   atmospheric heat-flux components
   (:math:`Q_\mathrm{sw}`, :math:`Q_l`, :math:`Q_h`, :math:`Q_e`),
   precipitation, and friction velocity :math:`u_*`.
2. Updates the internal ice state (thicknesses, temperatures, cover flag,
   albedo, transmissivity).
3. Returns a modified upward temperature diffusion flux
   ``diff_t_up`` that incorporates the latent heat exchanged at the
   ice-ocean interface.

The returned flux modifies the implicit vertical diffusion boundary condition
in the temperature equation, so ice thermodynamics is coupled to the ocean
column through the upper boundary condition rather than as a volumetric source
term.

Sign convention
~~~~~~~~~~~~~~~

pyGOTM uses the GOTM sign convention: a positive atmospheric heat flux means
heat is *leaving* the ocean.  The ice modules report ``ocean_ice_heat_flux``
as *positive when ice extracts heat from the ocean* (basal freezing or cooling
of supercooled water).

State variables
~~~~~~~~~~~~~~~

All models use the same :class:`~pygotm.icethm.IceState` dataclass whose
fields are length-one NumPy arrays for Numba compatibility:

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Field
     - Unit
     - Description
   * - ``Hice``
     - m
     - Ice thickness (total)
   * - ``Hsnow``
     - m
     - Snow thickness (Winton only)
   * - ``Hfrazil``
     - m
     - Frazil ice thickness (MyLake only)
   * - ``T1``
     - °C
     - Upper ice layer temperature (Winton only)
   * - ``T2``
     - °C
     - Lower ice layer temperature (Winton only)
   * - ``Tice_surface``
     - °C
     - Ice surface temperature
   * - ``fdd``
     - °C·day
     - Accumulated freezing degree days (Lebedev only)
   * - ``ice_cover``
     - —
     - Cover flag: 0 = open water, 2 = ice-covered
   * - ``Tf``
     - °C
     - Local freezing temperature
   * - ``albedo_ice``
     - —
     - Shortwave albedo (ocean or ice value)
   * - ``transmissivity``
     - —
     - Fraction of shortwave transmitted through ice
   * - ``ocean_ice_flux``
     - W m⁻²
     - Prescribed ocean-to-ice heat flux (Winton input)
   * - ``ocean_ice_heat_flux``
     - W m⁻²
     - Diagnosed heat flux from ocean to ice
   * - ``ocean_ice_salt_flux``
     - psu·m s⁻¹
     - Diagnosed salt flux from ocean to ice
   * - ``surface_ice_energy``
     - J m⁻²
     - Accumulated surface melt energy (Winton)
   * - ``bottom_ice_energy``
     - J m⁻²
     - Accumulated basal melt energy (Winton)
   * - ``melt_rate``
     - m s⁻¹
     - Basal melt rate (basal-melt model only)
   * - ``T_melt``
     - °C
     - Interface temperature at basal melt (basal-melt)
   * - ``S_melt``
     - psu
     - Interface salinity at basal melt (basal-melt)

Freezing Point Parameterisations
---------------------------------

Three distinct expressions for the seawater freezing temperature are used
across the model suite.

Linear GOTM freezing point
~~~~~~~~~~~~~~~~~~~~~~~~~~

The simple and Lebedev models use the linear approximation

.. math::

   T_f = -\mu_1 \, S

where :math:`\mu_1 = 0.0575\,{}^\circ\text{C psu}^{-1}` (``FREEZE_SLOPE`` in
the constants module).  This is the standard GOTM convention and applies to the
salinity range :math:`0 \text{–} 40\,\text{psu}`.

Winton freezing point
~~~~~~~~~~~~~~~~~~~~~

The Winton model uses a slightly different slope

.. math::

   T_f = -\mu_{TS} \, S, \qquad \mu_{TS} = 0.054\,{}^\circ\text{C psu}^{-1},

matching the value tabulated in Winton (2000) (Table 1, symbol
:math:`\mu`).

McDougall–Jackett basal freezing temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The basal-melt model uses a pressure-adjusted freezing point derived from
McDougall et al. (2003):

.. math::

   T_b = \lambda_1 S_b + \lambda_2 + \lambda_3 H,

where :math:`H` is the ice draft (m below sea level) used as a proxy for
pressure, and the empirical coefficients are

.. math::

   \lambda_1 &= -5.6705\times10^{-2}\,{}^\circ\text{C psu}^{-1}, \\
   \lambda_2 &= 7.5436\times10^{-2}\,{}^\circ\text{C}, \\
   \lambda_3 &= 7.6829\times10^{-4}\,{}^\circ\text{C m}^{-1}.

This linear form is valid for :math:`S_b \in [4, 40]\,\text{psu}` and
represents the freezing-point depression due to both salinity and hydrostatic
pressure.  McDougall et al. (2003) fit this expression to the Feistel–Hagen
(1995) Gibbs function, providing accuracies of order :math:`3\times10^{-3}` kg
m⁻³ in density and :math:`4\times10^{-7}` °C⁻¹ in thermal expansion
coefficient.

Ice Models
----------

NO_ICE (model = 0)
~~~~~~~~~~~~~~~~~~

When ``model = 0`` the ice module updates the local freezing temperature
``Tf`` using the linear GOTM expression but applies no modification to the
temperature flux or any state variable.  This is equivalent to running the
original GOTM configuration with no ice parameterisation.

SIMPLE (model = 1)
~~~~~~~~~~~~~~~~~~

The simple model is a *boundary-condition limiter* rather than a prognostic
ice model.  It prevents the ocean from warming above the local freezing
temperature by suppressing upward temperature diffusivity whenever
:math:`T_w \leq T_f` and the unmodified flux is positive (heat flowing into
the ocean):

.. math::

   \widetilde{D}_T^{\mathrm{up}} =
   \begin{cases}
   0 & \text{if } T_w \leq T_f \text{ and } D_T^{\mathrm{up}} > 0,\\
   D_T^{\mathrm{up}} & \text{otherwise.}
   \end{cases}

No ice thickness or surface temperature is tracked.  This model reproduces the
behaviour of the original GOTM ``ice_model = 1``.

LEBEDEV (model = 3)
~~~~~~~~~~~~~~~~~~~~

The Lebedev model applies the empirical freezing-degree-day (FDD) relation
from Lebedev (1938), widely used in lake-ice and sea-ice climatology:

.. math::

   H_\mathrm{ice} = 0.01 \times f_L \times \mathrm{FDD}^{e_L},

where the accumulated FDD is integrated over time as

.. math::

   \frac{d\,\mathrm{FDD}}{dt} =
   \begin{cases}
   T_f - T_\mathrm{air} & \text{if } T_\mathrm{air} < T_f, \\
   -(T_\mathrm{air} - T_f) & \text{if } T_\mathrm{air} > T_f,\; \text{floored at 0}.
   \end{cases}

The constants :math:`f_L = 1.33` and :math:`e_L = 0.58` are the standard
Lebedev parameters (``LEBEDEV_FAC``, ``LEBEDEV_EXP``).  Once
:math:`\mathrm{FDD} > 1` day·°C and :math:`T_w \leq T_f`, ice forms;
otherwise the column is open water.

Shortwave optics follow an exponential Beer-Lambert law

.. math::

   \tau = \exp\!\left(\frac{H_\mathrm{ice}}{l_a}\right),
   \quad l_a = -1.6\,\text{m},

with a fixed ice albedo :math:`\alpha_\mathrm{ice} = 0.545`
(``LEBEDEV_ALBEDO``).  The model does not diagnose ocean–ice heat or salt
fluxes; it modifies only the albedo and shortwave transmissivity seen by the
pyGOTM surface forcing.

MyLake Slab Ice (model = 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MyLake-style model (following Saloranta and Andersen, 2007) represents a
vertically uniform ice slab with prognostic thickness.  It advances through
four sequential processes per timestep:

**Frazil formation** — supercooling of the surface water layer drives frazil
ice accumulation:

.. math::

   \Delta H_\mathrm{frazil} =
   \frac{\rho_w c_{w,v} h_\mathrm{sfc} (T_f - T_w)}{\rho_i L_i},

where :math:`c_{w,v} = 4.18\times10^6` J m⁻³ K⁻¹ is the volumetric heat
capacity of seawater, :math:`\rho_i = 910` kg m⁻³ is ice density, and
:math:`L_i = 333\,500` J kg⁻¹ is the latent heat of fusion.  When
:math:`H_\mathrm{frazil} \geq 0.03` m and no solid ice exists, frazil
consolidates into the solid ice slab.

**Conductive surface growth** — when :math:`T_\mathrm{air} < T_f`, Stefan's
law drives ice growth from the top:

.. math::

   \Delta H_\mathrm{ice} = \frac{K_i (T_f - T_\mathrm{air})\,\Delta t}
   {\rho_i L_i \, \max(H_\mathrm{ice},\, 0.05)},

where :math:`K_i = 2.03` W m⁻¹ K⁻¹ is the thermal conductivity of ice.

**Surface melt** — net positive surface energy flux melts ice from above:

.. math::

   \Delta H_\mathrm{melt} = \frac{(Q_\mathrm{sw} + Q_h + Q_e + Q_l)\,\Delta t}
   {\rho_i L_i}.

**Basal melt** — heat stored in a warm surface water layer melts ice from
below, and the resulting heat flux is reported as ``ocean_ice_heat_flux``:

.. math::

   Q_\mathrm{oi} = \frac{\rho_w c_{w,v} h_\mathrm{sfc}(T_w - T_f)}{\Delta t}
   \quad [\text{W m}^{-2}].

Shortwave transmissivity follows Beer-Lambert with an attenuation coefficient
of :math:`5.0` m⁻¹ (``MYLAKE_ATTN``).  Ice albedo is set to
:math:`\alpha_\mathrm{ice} = 0.5826`.

Ice-Shelf Basal Melt (model = 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model implements the *three-equation ice-shelf basal-melt closure* of
Holland and Jenkins (1999).  It is appropriate for simulating the thermodynamic
exchange at the base of a floating ice shelf where the ice-ocean interface
temperature is constrained to the pressure-dependent freezing point.

.. rubric:: Physical framework

The interface temperature :math:`T_b` and salinity :math:`S_b` must
simultaneously satisfy:

1. **Freezing point constraint** (McDougall–Jackett):

   .. math::

      T_b = \lambda_1 S_b + \lambda_2 + \lambda_3 H.

2. **Heat conservation** — the turbulent heat flux from ocean to interface
   equals the latent heat of melting plus the conductive heat flux into the
   ice interior.  In the simplified form used here (treating the ice as
   conducting only to the core at a fixed temperature :math:`T_i = -20` °C),
   the thermal balance is

   .. math::

      \gamma_T (T_w - T_b) =
      M \!\left[\frac{L_i}{c_w} + \frac{c_i}{c_w}(T_b - T_i)\right],

   where :math:`M` [m s⁻¹] is the melt rate (positive for melting),
   :math:`c_w = 4180` J kg⁻¹ K⁻¹, and :math:`c_i = 1995` J kg⁻¹ K⁻¹.

3. **Salt conservation** — the turbulent salt flux from ocean to interface
   equals the salt rejected by melting ice (treating ice as salt-free):

   .. math::

      \gamma_S (S_w - S_b) = M \, S_b.

.. rubric:: Turbulent exchange velocities

Following Holland and Jenkins (1999) and Jenkins (2016), the turbulent
thermal and haline exchange velocities are parameterised as proportional to
the local friction velocity :math:`u_*`:

.. math::

   \gamma_T = \Gamma_T \, u_*, \quad \gamma_S = \Gamma_S \, u_*,

with :math:`\Gamma_T = 10^{-2}` (``GAMMA_T``) and
:math:`\Gamma_S = 5.05\times10^{-5}` (``GAMMA_S``).  These reproduce the
canonical values :math:`\gamma_T \approx 10^{-4}` m s⁻¹ and
:math:`\gamma_S \approx 5\times10^{-7}` m s⁻¹ at
:math:`u_* = 10^{-2}` m s⁻¹.  When :math:`u_*` is unavailable, the default
value :math:`u_* = 10^{-2}` m s⁻¹ (``DEFAULT_BASAL_USTAR``) is applied.

.. rubric:: Solution procedure

Eliminating :math:`T_b` and :math:`M` from equations 1–3 yields a quadratic
in :math:`S_b`:

.. math::

   q_a S_b^2 + q_b S_b + q_c = 0,

where the coefficients depend on :math:`T_w`, :math:`S_w`, :math:`H`,
:math:`\gamma_T`, and :math:`\gamma_S`.  The physically admissible root
(:math:`0 < S_b < S_w` for melting, :math:`S_b > 0` for freezing) is
selected, from which :math:`T_b` and :math:`M` follow directly.  The
diagnosed fluxes are

.. math::

   Q_\mathrm{oi}^T &= \rho_i L_i \, M \quad [\text{W m}^{-2}], \\
   Q_\mathrm{oi}^S &= M (S_b - S_w) \quad [\text{psu m s}^{-1}].

These are stored in ``ocean_ice_heat_flux`` and ``ocean_ice_salt_flux``
respectively and subtracted from the ocean column temperature diffusion
boundary condition.

Winton Three-Layer Sea Ice (model = 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Winton model (Winton, 2000) provides the most physically complete
sea-ice thermodynamics available in pyGOTM.  It tracks prognostic snow and
ice thicknesses and two internal ice temperatures, solves for ice formation and
melt, and uses a fully implicit time-stepping scheme.

.. rubric:: Layer structure

The model consists of three thermodynamically distinct layers (Fig. 1 of
Winton, 2000):

* **Snow layer** (zero heat capacity): thickness :math:`h_s`, zero-heat-capacity
  insulating layer above the ice.
* **Upper ice layer** (variable heat capacity): thickness :math:`h_i/2`,
  temperature :math:`T_1` at depth :math:`h_i/4` below the snow–ice interface.
  Brine inclusions give this layer a variable effective heat capacity.
* **Lower ice layer** (fixed heat capacity): thickness :math:`h_i/2`,
  temperature :math:`T_2` at depth :math:`3h_i/4` below the interface.
  The bottom of the ice is fixed at the freezing temperature :math:`T_f`.

.. rubric:: Variable heat capacity (brine content)

Sea ice retains brine in pockets whose salinity equals the local freezing
temperature.  The enthalpy per unit mass is therefore

.. math::

   E(T, S) \equiv C(T + \mu S) - L\!\left(1 + \frac{\mu S}{T}\right),

where :math:`C = 2100` J kg⁻¹ K⁻¹ is the ice heat capacity,
:math:`L = 333\,500` J kg⁻¹ is the latent heat of fusion, and
:math:`\mu = 0.054` °C psu⁻¹ relates freezing temperature to salinity
(Eq. 1 in Winton, 2000).  Conservation of this enthalpy drives the
upper-layer temperature evolution and eliminates the need for a separate brine
variable.

.. rubric:: Conductive coupling

The effective conductivity between the surface and the upper ice layer,
accounting for the insulating snow layer, is

.. math::

   K_{1/2} = \frac{4 K_i K_s}{K_s h_i + 4 K_i h_s},

where :math:`K_i = 2.03` W m⁻¹ K⁻¹ and :math:`K_s = 0.31` W m⁻¹ K⁻¹
are the thermal conductivities of ice and snow (Eq. 5 in Winton, 2000).
When :math:`h_s = 0`, this reduces to
:math:`K_{1/2} = 4K_i/h_i` (the factor 4 arises from the layer midpoint
location at :math:`h_i/4`).  The inter-layer conductivity between the two ice
layers is :math:`K_{3/2} = 2K_i/h_i`.

.. rubric:: Surface temperature

The surface heat flux is linearised about the previous surface temperature
:math:`\hat T_s`:

.. math::

   F_s(T_s) \approx A + B\,T_s, \qquad B = \frac{\partial F_s}{\partial T_s}.

pyGOTM computes :math:`A` and :math:`B` from the pre-computed air-sea flux
components (Qsw absorbed, Ql, Qh, Qe) and uses a conservative fixed slope
:math:`B = -4` W m⁻² K⁻¹ in the absence of a flux-recalculation callback.
The surface temperature is then found diagnostically from the heat-balance
between the linearised surface flux and the conductive flux into the upper ice
layer (Eq. 6 in Winton, 2000):

.. math::

   T_s = \frac{K_{1/2} T_1 - A}{K_{1/2} + B}.

If :math:`T_s > 0` (melting conditions), the surface is clamped to
:math:`T_s = 0` and the excess energy is accumulated in ``surface_ice_energy``.

.. rubric:: Ice temperature time stepping

Equations (12) and (13) of Winton (2000) advance :math:`T_1` and
:math:`T_2` implicitly.  In pyGOTM's implementation, :math:`T_2` is solved
first (Eq. 15 of the paper) and :math:`T_1` follows from the resulting
quadratic (Eqs. 16–21).  Temperatures are clamped to zero and excess energy
routed to the surface or basal energy budgets.

.. rubric:: Snow and ice thickness changes

Four mass-change processes are applied sequentially after the temperature step:

1. **Snow accumulation** from precipitation when ice exists.
2. **Surface melt** — snow melts first, then upper ice, from the accumulated
   ``surface_ice_energy``.
3. **Basal growth or melt** from the basal energy budget
   ``bottom_ice_energy + ocean_ice_flux · Δt``.
4. **Snow flooding** — when freeboard is negative (snow weight pushes the
   ice surface below sea level), excess snow is converted to upper-layer ice.

The layer-equalization step ``even_up`` ensures that the upper and lower layers
remain equal in thickness by transferring enthalpy when mass shifts between
layers (Eqs. 37–40 in Winton, 2000).

.. rubric:: Shortwave optics

Ice albedo depends on snow cover, ice thickness, and surface temperature
following the parameterisation of Winton (2000):

.. math::

   \alpha =
   \begin{cases}
   \alpha_\mathrm{snow} = 0.85 & h_s > 0, \\
   f_h \, \alpha_\mathrm{ice} + (1-f_h)\,\alpha_\mathrm{ocean}
   - 0.075 \, \delta_\mathrm{melt} & h_s = 0,
   \end{cases}

where :math:`f_h = \min(h_i/0.5, 1)` is a thickness-based blend factor
(:math:`\alpha_\mathrm{ice} = 0.5826`, :math:`\alpha_\mathrm{ocean} = 0.06`)
and :math:`\delta_\mathrm{melt}` is a linear reduction applied when the
surface temperature exceeds :math:`-1` °C.  The penetrating shortwave
fraction follows

.. math::

   \tau = P_0 \exp\!\left(-\frac{h_i}{d_0}\right),

with penetrating fraction :math:`P_0 = 0.3` (``PEN_ICE``) and optical depth
:math:`d_0 = 0.67` m (``OPT_DEP_ICE``).

.. rubric:: Ocean–ice coupling

The basal heat flux diagnosed from ice thickness changes is reported as
``ocean_ice_heat_flux``.  This is used by the calling layer in
:func:`~pygotm.icethm.compute_diff_t_up_from_ice` to modify the upward
temperature diffusion boundary condition:

.. math::

   \widetilde{D}_T^\mathrm{up} = D_T^\mathrm{up}
   - \frac{Q_\mathrm{oi}^T}{\rho_0 c_p},

where :math:`\rho_0` and :math:`c_p` are the ocean reference density and heat
capacity.

Constants Summary
------------------

.. list-table:: Physical constants used by the ice module
   :header-rows: 1
   :widths: 22 12 10 56

   * - Symbol
     - Python name
     - Value
     - Description
   * - :math:`\rho_i`
     - ``RHO_ICE``
     - 910 kg m⁻³
     - Density of sea ice
   * - :math:`\rho_w`
     - ``RHO_WATER``
     - 1025 kg m⁻³
     - Density of seawater
   * - :math:`\rho_s`
     - ``RHO_SNOW``
     - 330 kg m⁻³
     - Density of snow
   * - :math:`L_i`
     - ``L_ICE``
     - 333 500 J kg⁻¹
     - Latent heat of fusion
   * - :math:`K_i`
     - ``K_ICE``
     - 2.03 W m⁻¹ K⁻¹
     - Thermal conductivity of sea ice
   * - :math:`K_s`
     - ``K_SNOW``
     - 0.31 W m⁻¹ K⁻¹
     - Thermal conductivity of snow
   * - :math:`C`
     - ``C_ICE``
     - 2100 J kg⁻¹ K⁻¹
     - Specific heat capacity of ice
   * - :math:`\mu_1`
     - ``FREEZE_SLOPE``
     - 0.0575 °C psu⁻¹
     - GOTM simple freezing slope
   * - :math:`\mu_{TS}`
     - ``MU_TS``
     - 0.054 °C psu⁻¹
     - Winton freezing slope
   * - :math:`\Gamma_T`
     - ``GAMMA_T``
     - 1.0 × 10⁻²
     - Thermal exchange coefficient (Holland–Jenkins)
   * - :math:`\Gamma_S`
     - ``GAMMA_S``
     - 5.05 × 10⁻⁵
     - Haline exchange coefficient (Holland–Jenkins)

References
----------

* **Winton (2000)** — M. Winton, "A Reformulated Three-Layer Sea Ice Model,"
  *J. Atmos. Oceanic Technol.* **17**, 525–531.
  DOI: `10.1175/1520-0426(2000)017<0525:ARTLSI>2.0.CO;2 <https://doi.org/10.1175/1520-0426(2000)017%3C0525:ARTLSI%3E2.0.CO;2>`__

* **Holland and Jenkins (1999)** — D. M. Holland and A. Jenkins,
  "Modeling Thermodynamic Ice–Ocean Interactions at the Base of an Ice Shelf,"
  *J. Phys. Oceanogr.* **29**, 1787–1800.
  DOI: `10.1175/1520-0485(1999)029<1787:MTIOIA>2.0.CO;2 <https://doi.org/10.1175/1520-0485(1999)029%3C1787:MTIOIA%3E2.0.CO;2>`__

* **McDougall et al. (2003)** — T. J. McDougall, D. R. Jackett, D. G. Wright,
  and R. Feistel, "Accurate and Computationally Efficient Algorithms for
  Potential Temperature and Density of Seawater,"
  *J. Atmos. Oceanic Technol.* **20**, 730–741.
  DOI: `10.1175/1520-0426(2003)20<730:AACEFA>2.0.CO;2 <https://doi.org/10.1175/1520-0426(2003)20%3C730:AACEFA%3E2.0.CO;2>`__

* **Jenkins (2016)** — A. Jenkins, "A Simple Model of the Ice Shelf–Ocean
  Boundary Layer and Current," *J. Phys. Oceanogr.* **46**, 1785–1803.
  DOI: `10.1175/JPO-D-15-0194.1 <https://doi.org/10.1175/JPO-D-15-0194.1>`__

* **Saloranta and Andersen (2007)** — T. M. Saloranta and T. Andersen,
  "MyLake — A multi-year lake simulation model code suitable for uncertainty
  and sensitivity analysis simulations," *Ecol. Modell.* **207** (1), 45–60.
  DOI: `10.1016/j.ecolmodel.2007.03.018 <https://doi.org/10.1016/j.ecolmodel.2007.03.018>`__

See Also
--------

* :doc:`/api/icethm` — ice thermodynamics API reference
* :doc:`airsea` — surface flux calculation that feeds heat components into the ice models
* :doc:`overview` — solution sequence showing where ice is called
