Equation of State and Bottom Boundary
======================================

.. _yaml-eos:

``equation_of_state``
---------------------

Controls the seawater density formulation used to compute buoyancy,
stratification, and the baroclinic pressure gradient.

The density calculation feeds directly into the stratification module
:mod:`pygotm.meanflow.stratification` and all buoyancy-dependent turbulence
terms.

.. code-block:: yaml

   equation_of_state:
     method: full_teos-10
     rho0: 1027.0
     linear:
       T0: 10.0
       S0: 35.0
       p0: 0.0
       alpha: 2.0e-4
       beta: 7.5e-4
       cp: 3991.87

``equation_of_state.method``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``full_teos-10``, ``linear_teos-10``, ``linear_custom``
   * - **Default**
     - ``"full_teos-10"``

Selects the density formulation.

``full_teos-10``
   Full nonlinear TEOS-10 equation of state (IOC et al. 2010).  Uses the
   GSW toolbox to compute density, thermal expansion coefficient
   :math:`\alpha`, and saline contraction coefficient :math:`\beta` at every
   level and every time step.  Most accurate option; recommended for
   open-ocean and lake simulations.

``linear_teos-10``
   Linearised equation of state using TEOS-10 to compute :math:`\alpha`,
   :math:`\beta`, and :math:`c_p` at a reference state (:math:`T_0`, :math:`S_0`,
   :math:`p_0`); these coefficients are then held constant:

   .. math::

      \rho = \rho_0 \left[1 - \alpha(T - T_0) + \beta(S - S_0)\right]

   Faster than ``full_teos-10`` and appropriate when T/S variations are small.

``linear_custom``
   Same linearised formula as above, but :math:`\alpha`, :math:`\beta`,
   :math:`\rho_0`, and :math:`c_p` are all specified directly by the user in
   the ``linear`` sub-block rather than computed from TEOS-10.  Use when
   you want full control over the equation of state (e.g., for idealised
   freshwater lake studies).

``equation_of_state.rho0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - kg m\ :sup:`−3`
   * - **Default**
     - ``1027.0``

Reference density.  Used throughout the model to non-dimensionalise fluxes.
Specifically:

- Surface wind stress is divided by ``rho0`` to give kinematic stress
  :math:`\tau/\rho_0` [m² s⁻²].
- Heat flux is converted to temperature flux via :math:`Q / (\rho_0 c_p)`.

``equation_of_state.linear``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sub-block for the linearised EOS.  Only used when ``method: linear_teos-10``
or ``method: linear_custom``.

``T0``
   Reference temperature (°C).  *Default*: ``15.0``; *range*: −2 to 40 °C.

``S0``
   Reference salinity (g kg\ :sup:`−1`).  *Default*: ``35.0``; *range*: ≥ 0.

``p0``
   Reference pressure (dbar).  *Default*: ``0.0``; *range*: ≥ 0.

``alpha``
   Thermal expansion coefficient :math:`\alpha = -\partial\rho/(\rho_0\partial T)`
   (K\ :sup:`−1`).  *Default*: ``2.0e-4``.  Used when ``method: linear_custom``.

``beta``
   Saline contraction coefficient :math:`\beta = \partial\rho/(\rho_0\partial S)`
   (kg g\ :sup:`−1`).  *Default*: ``7.5e-4``.  Used when ``method: linear_custom``.

``cp``
   Specific heat capacity at constant pressure (J kg\ :sup:`−1` K\ :sup:`−1`).
   *Default*: ``3991.87``.  Used when ``method: linear_custom`` to convert
   heat fluxes to temperature tendencies.

.. _yaml-meanflow:

``meanflow``
------------

Physical constants and switches for the mean-flow equations.  These keys
appear directly under a ``meanflow`` block in full GOTM YAML files.

.. code-block:: yaml

   meanflow:
     gravity: 9.81
     rho0: 1027.0
     cori: 0.0
     avmolu: 1.3e-6
     avmolT: 1.4e-7
     avmolS: 1.1e-9
     cp: 3991.87
     h0b: 0.05
     z0s_min: 0.02
     calc_bottom_stress: true
     charnock: false
     MaxItz0b: 10

``meanflow.gravity``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m s\ :sup:`−2`
   * - **Default**
     - ``9.81``

Gravitational acceleration.  Used in buoyancy frequency and pressure
gradient calculations.

``meanflow.rho0``
~~~~~~~~~~~~~~~~~~

Reference density (see ``equation_of_state.rho0``).  *Default*: ``1027.0``.

``meanflow.cori``
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - s\ :sup:`−1`
   * - **Default**
     - computed from latitude

Coriolis parameter :math:`f = 2\Omega\sin\phi`.  By default pyGOTM computes
this from ``location.latitude``.  Setting an explicit value here overrides
the geographic computation.

``meanflow.avmolu``
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−1`
   * - **Default**
     - ``1.3e-6``

Molecular kinematic viscosity of seawater.  Added to the turbulent eddy
viscosity as a background value.

``meanflow.avmolT``
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−1`
   * - **Default**
     - ``1.4e-7``

Molecular thermal diffusivity.

``meanflow.avmolS``
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−1`
   * - **Default**
     - ``1.1e-9``

Molecular salinity (haline) diffusivity.

``meanflow.cp``
~~~~~~~~~~~~~~~~

Specific heat capacity of seawater (J kg\ :sup:`−1` K\ :sup:`−1`).
*Default*: ``3991.87``.

.. _yaml-bottom:

``bottom``
----------

Bottom boundary condition parameters for the mean-flow equations.

.. code-block:: yaml

   bottom:
     calc_bottom_stress: true
     h0b: 0.05

``bottom.calc_bottom_stress``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``true``

If ``true``, the bottom stress is computed from the law of the wall using the
bottom roughness length ``h0b`` and the velocity at the first grid level.  If
``false``, the bottom boundary condition is a no-stress (free-slip) condition.

Implemented in :func:`pygotm.meanflow.friction.friction`.

``bottom.h0b``
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``0.05``

Physical bottom roughness length :math:`z_{0b}`.  Used in the logarithmic
law of the wall to compute bottom friction velocity:

.. math::

   u_{\tau b} = \frac{\kappa \, u(z_1)}{\ln(z_1 / z_{0b})}

where :math:`z_1` is the height of the first grid level and :math:`u(z_1)`
is the velocity magnitude at that level.

Larger values of ``h0b`` represent rougher bottoms (gravel, boulders) while
smaller values represent smoother bottoms (fine sediment, mud).

``meanflow.z0s_min``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m
   * - **Default**
     - ``0.02``

Minimum surface roughness length :math:`z_{0s}`.  When the Charnock
parameterisation is inactive, the surface roughness is held at this constant
value.

``meanflow.charnock``
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``false``

If ``true``, use the Charnock (1955) relation to derive the surface roughness
length from the surface friction velocity:
:math:`z_{0s} = \alpha_c u_{\tau s}^2 / g`.
**[legacy only]** in the current compiled runtime.

``meanflow.MaxItz0b``
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - integer
   * - **Default**
     - ``10``

Maximum number of iterations for the implicit roughness-velocity coupling
at the bottom boundary.

.. rubric:: References

IOC, SCOR, IAPSO (2010). *The International Thermodynamic Equation of
Seawater — 2010*. UNESCO.

Charnock, H. (1955). Wind stress over a water surface.
*Q. J. R. Meteorol. Soc.*, 81, 639–640.
