Dynamics: Pressure Gradients, Velocities, Waves, and Vertical Advection
========================================================================

These sections provide 3-D-effect forcing for the inherently 1-D column
model: external and internal pressure gradients, prescribed velocity profiles,
wave–current interactions, and vertical advection.

All scalar forcing fields in this section that use ``method: file`` read
the :ref:`fmt-timeseries` format.  Velocity profiles (``velocities.u``,
``velocities.v``) use the :ref:`fmt-profile` format.  See
:ref:`input-file-formats` for complete specifications.

.. _yaml-mimic3d:

``mimic_3d``
------------

Horizontal gradients that drive mean flow within the single column.
Because pyGOTM is a 1-D model, horizontal advection and pressure gradients
must be prescribed externally.  The ``mimic_3d`` block allows these
boundary-condition-like inputs to be provided as time series.

Parsed by :class:`pygotm.config.settings.Mimic3DSettings`.

.. code-block:: yaml

   mimic_3d:
     ext_pressure:
       type: elevation
       dpdx:
         method: file
         file: ext_press_file.dat
         column: 2
       dpdy:
         method: file
         file: ext_press_file.dat
         column: 3
     int_pressure:
       type: none
     zeta:
       method: constant
       constant_value: 0.0

.. _yaml-ext-pressure:

``mimic_3d.ext_pressure``
~~~~~~~~~~~~~~~~~~~~~~~~~

External (barotropic) pressure gradient.

Parsed by :class:`pygotm.config.settings.ExtPressureSettings`.

``mimic_3d.ext_pressure.type``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``elevation``, ``velocity``, ``average_velocity``
   * - **Default**
     - ``"elevation"``

How the external pressure gradient is specified.

``elevation``
   The gradient is expressed as a horizontal gradient in sea-surface elevation
   :math:`\partial\zeta/\partial x` (dimensionless).  The values provided in
   ``dpdx`` and ``dpdy`` are directly :math:`\partial\zeta/\partial x` and
   :math:`\partial\zeta/\partial y`.

``velocity``
   A reference velocity at a given height above the bottom is specified;
   the pressure gradient is inferred to maintain that velocity.

``average_velocity``
   Vertically averaged (depth-mean) horizontal velocities are prescribed.

``mimic_3d.ext_pressure.dpdx``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

External pressure gradient in the West–East direction.

Follows the extended ``InputSetting`` pattern (``method: constant | tidal | file``).
When ``method: file``: timeseries file (see :ref:`fmt-timeseries`).  The
conventional layout stores ``dpdx`` and ``dpdy`` as columns 2 and 3 of a
shared three-column file (column 1 is often a placeholder or a third
gradient quantity):

.. code-block:: text

   # Columns: placeholder   dpdx [m/m]   dpdy [m/m]
   1976/04/06 06:00:00    0.00  -3.2010771E-06  -1.6297247E-06
   1976/04/06 06:15:00    0.00  -2.9533401E-06  -2.1274509E-06

.. list-table::
   :widths: 20 80

   * - **Default**
     - ``0.0``

The tidal extension adds a ``tidal`` sub-block for composing two harmonic
constituents:

.. code-block:: yaml

   dpdx:
     method: tidal
     tidal:
       amp_1: 0.0        # amplitude of 1st harmonic
       phase_1: 0.0      # phase of 1st harmonic [s]
       amp_2: 0.0        # amplitude of 2nd harmonic
       phase_2: 0.0      # phase of 2nd harmonic [s]
     period_1: 44714.0   # period of 1st harmonic [s]  (≈ M2 tide)
     period_2: 43200.0   # period of 2nd harmonic [s]  (≈ S2 tide)

``period_1`` and ``period_2`` are the tidal periods (seconds).  Default values
correspond to the M2 (44714 s ≈ 12.42 h) and S2 (43200 s = 12 h) tidal
constituents.

``mimic_3d.ext_pressure.dpdy``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

External pressure gradient in the South–North direction.  Same structure as
``dpdx``.

``mimic_3d.ext_pressure.h``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Reference height above the bottom (m) used when ``type: velocity``.

Follows the ``InputSetting`` pattern.

.. _yaml-int-pressure:

``mimic_3d.int_pressure``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Internal (baroclinic) pressure gradient arising from horizontal density
gradients.

Parsed by :class:`pygotm.config.settings.IntPressureSettings`.

``mimic_3d.int_pressure.type``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``none``, ``prescribed``, ``plume``
   * - **Default**
     - ``"none"``

``none``
   No internal pressure gradient.

``prescribed``
   Horizontal T/S gradients are prescribed as time series and used to compute
   the baroclinic pressure gradient.

``plume``
   Dense water inflow (bottom plume) configuration.

``mimic_3d.int_pressure.gradients``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Temperature and salinity gradient inputs.  Only used when
``int_pressure.type: prescribed``.

Each follows the ``InputSetting`` pattern:

``dtdx`` / ``dtdy``
   Horizontal temperature gradients (°C m\ :sup:`−1`).

``dsdx`` / ``dsdy``
   Horizontal salinity gradients (psu m\ :sup:`−1`).

``mimic_3d.int_pressure.plume``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dense bottom-plume configuration.  Only used when ``int_pressure.type: plume``.

``type``
   Plume type: ``bottom`` or ``surface``.  *Default*: ``"bottom"``.

``x_slope`` / ``y_slope``
   Bottom slope in the West–East and South–North directions.  Dimensionless.
   *Default*: ``0.0``.

``mimic_3d.int_pressure.t_adv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``false``

Include temperature horizontal advection computed from the prescribed gradient.

``mimic_3d.int_pressure.s_adv``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``false``

Include salinity horizontal advection.

``mimic_3d.zeta``
~~~~~~~~~~~~~~~~~~

Sea-surface elevation :math:`\zeta` (m).

Follows the extended ``InputSetting`` pattern (``method: constant | tidal | file``).
Used to set the free-surface height for pressure gradient calculations.

.. _yaml-velocities:

``velocities``
--------------

Prescribes and/or relaxes the horizontal velocity profiles.

Parsed by :class:`pygotm.config.settings.VelocitySettings`.

.. code-block:: yaml

   velocities:
     u:
       method: off
     v:
       method: off
     relax:
       tau: 1.0e15
       ramp: 1.0e15

``velocities.u`` / ``velocities.v``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Observed or prescribed horizontal velocity profiles (m s\ :sup:`−1`) in the
West–East and South–North directions.

Follows the ``InputSetting`` pattern (``method: off | constant | file``).

When ``method: file``, a time series of full-depth profiles is read.  These
are used to nudge the model velocities via ``relax.tau``, or to initialise
the velocity field.  The file uses the :ref:`fmt-profile` format.  Both
components can share one file using ``column: 1`` for ``u`` and
``column: 2`` for ``v``:

.. code-block:: text

   # Horizontal velocity profiles [m/s]: col1=u (W-E), col2=v (S-N)
   # Header: timestamp   N_levels   up_down
   2001/08/30 07:00:00   22   2
      -6.00000   0.00000   0.12000
      -8.00000   0.01000   0.11000
     -10.00000   0.02000   0.09000
     ...
     -50.00000  -0.14000  -0.14000

YAML reference:

.. code-block:: yaml

   velocities:
     u: { method: file, file: velprof.dat, column: 1 }
     v: { method: file, file: velprof.dat, column: 2 }

``velocities.relax.tau``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - s
   * - **Default**
     - ``1.0e15`` (no relaxation)

Relaxation time scale for nudging model velocities toward the prescribed
profile.

``velocities.relax.ramp``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - s
   * - **Default**
     - ``1.0e15``

Duration of a linear ramp-up period at the start of the simulation over
which the relaxation strength increases from 0 to its full value.

.. _yaml-waves:

``waves``
---------

Surface wave parameters for Stokes drift and wave–turbulence interaction.

Parsed by :class:`pygotm.config.settings.WaveSettings`.

.. note::
   Full Stokes drift integration (Craig–Banner TKE injection) is
   **[legacy only]** in the current compiled Numba runtime.  The
   ``stokes_active`` flag in :class:`pygotm.gotm.runtime_params.RuntimeParams`
   must be non-zero to activate this path.

.. code-block:: yaml

   waves:
     Hs:
       method: constant
       constant_value: 0.0
     Tz:
       method: constant
       constant_value: 1.0
     phiw:
       method: constant
       constant_value: 0.0

``waves.Hs``
~~~~~~~~~~~~~

Significant wave height.

Follows the ``InputSetting`` pattern.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m
   * - **Default**
     - ``0.0``

Used to compute the Stokes drift profile and wave-breaking TKE injection.

``waves.Tz``
~~~~~~~~~~~~~

Zero-crossing wave period (peak period).

Follows the ``InputSetting`` pattern.

.. list-table::
   :widths: 20 80

   * - **Units**
     - s
   * - **Default**
     - ``0.0``

Together with ``Hs``, determines the amplitude and depth penetration of the
Stokes drift velocity profile.

``waves.phiw``
~~~~~~~~~~~~~~

Wave propagation direction (meteorological convention: direction *from*).

Follows the ``InputSetting`` pattern.

.. list-table::
   :widths: 20 80

   * - **Units**
     - degrees
   * - **Default**
     - ``0.0``

Used to project Stokes drift into U and V components.

.. _yaml-w:

``w``
-----

Vertical (upwelling/downwelling) velocity prescribed at a given height.

Parsed by :class:`pygotm.config.settings.VerticalVelocitySettings`.

.. code-block:: yaml

   w:
     max:
       method: off
     height:
       method: constant
       constant_value: 0.0
     adv_discr: p2_pdm

``w.max``
~~~~~~~~~~

Maximum vertical velocity magnitude.

Follows the ``InputSetting`` pattern (``method: off | constant | file``).

.. list-table::
   :widths: 20 80

   * - **Units**
     - m s\ :sup:`−1`
   * - **Default**
     - ``0.0`` (off)

When active, a vertical velocity profile is imposed following a Gaussian
distribution centred at ``w.height``.  The vertical velocity drives
temperature and salinity advection via the scheme selected by ``adv_discr``.

When ``method: file``, ``w.max`` and ``w.height`` can share a single
two-column timeseries file (see :ref:`fmt-timeseries`):

.. code-block:: text

   # Columns: w_max [m/s]   height_above_bottom [m]
   2001/08/30 16:24:09   -0.268241e+02   -0.277951e-04
   2001/08/30 16:34:09   -0.268404e+02   -0.265529e-04

YAML reference:

.. code-block:: yaml

   w:
     max:    { method: file, file: vertvel.dat, column: 1 }
     height: { method: file, file: vertvel.dat, column: 2 }

``w.height``
~~~~~~~~~~~~~

Height above the bottom where the prescribed vertical velocity is maximum.

Follows the ``InputSetting`` pattern.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m
   * - **Default**
     - ``0.0``

``w.adv_discr``
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``p2_pdm``, ``upstream``, ``p2``, ``superbee``, ``muscl``
   * - **Default**
     - ``"p2_pdm"``

Advection discretisation scheme for the vertical tracer advection terms.

``p2_pdm``
   Second-order piecewise parabolic scheme with the Positive-Definite Monotone
   (PDM) limiter.  Recommended default; balances accuracy and monotonicity.

``upstream``
   First-order upwind scheme.  Diffusive but robust.

``p2``
   Second-order parabolic scheme without limiter.

``superbee`` / ``muscl``
   TVD (Total Variation Diminishing) schemes for steep gradients.

Implemented in :mod:`pygotm.util.adv_center`.
