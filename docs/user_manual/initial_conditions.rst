Initial Conditions: Temperature and Salinity
============================================

The ``temperature`` and ``salinity`` sections define how the water column is
initialised and — optionally — how observed profiles are used to relax the
model state throughout the simulation.

Both sections share the same structure (an extended ``InputSetting`` with
additional sub-blocks) and are parsed by
:class:`pygotm.config.settings.TemperatureSettings` and
:class:`pygotm.config.settings.SalinitySettings` respectively.  The
underlying physics is implemented in
:mod:`pygotm.meanflow.temperature` and :mod:`pygotm.meanflow.salinity`.

.. _yaml-temperature:

``temperature``
---------------

Specifies the initial temperature profile and optional relaxation forcing.

.. code-block:: yaml

   temperature:
     method: file
     constant_value: 20.0
     file: t_prof_file.dat
     column: 1
     type: in-situ
     two_layer:
       z_s: 30.0
       t_s: 20.0
       z_b: 40.0
       t_b: 15.0
     NN: 2.56e-4
     relax:
       tau: 1.0e15
       h_s: 0.0
       tau_s: 1.0e15
       h_b: 0.0
       tau_b: 1.0e15

``temperature.method``
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``off``, ``constant``, ``file``, ``two_layer``, ``buoyancy``
   * - **Default**
     - ``"off"``

Selects the source of the initial temperature profile.

``off``
   Temperature is not initialised from external data; arrays start at zero
   (or the restart state if ``restart.load: true``).

``constant``
   All levels are set to ``temperature.constant_value`` at initialisation.
   No relaxation file is needed.

``file``
   A time series of full-depth temperature profiles is read from
   ``temperature.file``.  The first profile in the file initialises the
   column.  If ``relax.tau`` is finite the model state is nudged toward the
   observed profiles at each subsequent time step.

``two_layer``
   A simple two-layer structure with a linear gradient between the layers.
   Defined by the ``temperature.two_layer`` sub-block.

``buoyancy``
   Temperature profile is reconstructed from the specified buoyancy frequency
   squared ``temperature.NN`` together with the salinity profile (which must
   be separately specified).

``temperature.constant_value``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - °C
   * - **Default**
     - ``0.0``

Temperature value used when ``method: constant``.  Ignored for all other
methods.

``temperature.file``
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string (file path)
   * - **Default**
     - ``""``

Path to an ASCII profile file.  The file format is a time series of depth–
value columns: each block begins with a date-time header line followed by
``nlev + 1`` lines of ``depth [m]   temperature [°C]``.  Relative paths are
resolved from the directory containing ``gotm.yaml``.

Required when ``method: file``; ignored otherwise.

``temperature.column``
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - integer
   * - **Default**
     - ``1``

Column index (1-based) to read from the profile file.  Use this when the data
file contains multiple variables (e.g., temperature in column 1, salinity in
column 2).

``temperature.type``
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``in-situ``, ``potential``, ``conservative``
   * - **Default**
     - ``"in-situ"``

Temperature measure of the data in the profile file.  pyGOTM uses the GSW
toolbox (``gsw`` Python package) to convert the input values to Conservative
Temperature before storing them in the model state.

``in-situ``
   In-situ temperature :math:`T` (°C).  No conversion applied.

``potential``
   Potential temperature :math:`\theta` (°C).  Converted to Conservative
   Temperature via :func:`gsw.CT_from_pt`.

``conservative``
   Conservative Temperature :math:`\Theta` (°C).  Stored directly.

``temperature.two_layer``
~~~~~~~~~~~~~~~~~~~~~~~~~

Sub-block used only when ``method: two_layer``.

``temperature.two_layer.z_s`` / ``z_b``
   Depths (m, positive) where the upper (``z_s``) and lower (``z_b``) layers
   end / begin respectively.  A linear gradient is imposed in the transition
   zone ``[z_s, z_b]``.  Below ``z_b`` the temperature is uniform at
   ``t_b``.

   *Default*: ``0.0``; *range*: ≥ 0.0.

``temperature.two_layer.t_s``
   Upper-layer temperature (°C).  Applied from the surface down to ``z_s``.

   *Default*: ``0.0``; *range*: −2 to 40 °C.

``temperature.two_layer.t_b``
   Lower-layer temperature (°C).  Applied from ``z_b`` to the bottom.

   *Default*: ``0.0``; *range*: −2 to 40 °C.

``temperature.NN``
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - s\ :sup:`−2`
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``0.0``

Buoyancy frequency squared :math:`N^2` used when ``method: buoyancy``.  The
temperature profile is reconstructed from the relationship
:math:`N^2 = -\frac{g}{\rho_0}\frac{\partial\rho}{\partial z}`.

``temperature.relax``
~~~~~~~~~~~~~~~~~~~~~

Nudging parameters that pull the model temperature toward the observed profile
at every time step.  Relaxation is only active when ``method: file``.

``temperature.relax.tau``
   Interior relaxation time scale (s).  The model temperature is nudged with
   rate :math:`1/\tau` toward the observed profile in layers deeper than
   ``h_s`` from the surface and ``h_b`` from the bottom.

   *Default*: ``1.0e15`` (no relaxation).

``temperature.relax.h_s``
   Surface relaxation layer thickness (m).  Layers shallower than ``h_s``
   below the surface use ``tau_s`` instead of ``tau``.

   *Default*: ``0.0`` (no distinct surface layer).

``temperature.relax.tau_s``
   Relaxation time scale (s) for the surface layer (depth < ``h_s``).

   *Default*: ``1.0e15``.

``temperature.relax.h_b``
   Bottom relaxation layer thickness (m).

   *Default*: ``0.0``.

``temperature.relax.tau_b``
   Relaxation time scale (s) for the bottom layer (depth > total depth − ``h_b``).

   *Default*: ``1.0e15``.

.. _yaml-salinity:

``salinity``
------------

Identical structure to ``temperature``.  Parameters have the same meaning
unless noted below.

.. code-block:: yaml

   salinity:
     method: file
     constant_value: 35.0
     file: s_prof_file.dat
     column: 1
     type: practical
     two_layer:
       z_s: 30.0
       s_s: 35.0
       z_b: 40.0
       s_b: 34.5
     NN: 2.56e-4
     relax:
       tau: 172800.0
       tau_s: 172800.0
       tau_b: 172800.0

``salinity.method``
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``off``, ``constant``, ``file``, ``two_layer``, ``buoyancy``
   * - **Default**
     - ``"off"``

Same as ``temperature.method``.  The ``buoyancy`` option reconstructs
salinity from a specified :math:`N^2` together with the temperature profile.

``salinity.constant_value``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - psu (practical salinity units) or g kg\ :sup:`−1`
   * - **Default**
     - ``0.0``

``salinity.type``
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``practical``, ``absolute``
   * - **Default**
     - ``"practical"``

Salinity measure of the input data.

``practical``
   Practical Salinity Scale 1978 (PSS-78) values.  No conversion.

``absolute``
   Absolute Salinity :math:`S_A` (g kg\ :sup:`−1`).  Converted to Absolute
   Salinity using :func:`gsw.SA_from_SP`.

``salinity.two_layer.s_s`` / ``s_b``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Upper- and lower-layer salinities (g kg\ :sup:`−1` or psu).

*Default*: ``0.0``; *range*: 0 to 40 g kg\ :sup:`−1`.

``salinity.NN``
~~~~~~~~~~~~~~~

Same as ``temperature.NN``.  Used only when ``salinity.method: buoyancy``.
