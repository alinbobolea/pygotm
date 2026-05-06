Output and Restart
==================

.. _yaml-output:

``output``
----------

Controls how and when pyGOTM writes results to NetCDF files.  The ``output``
block is a mapping of *output stream names* to schedule configurations.
Multiple named streams can be defined simultaneously, each producing a
separate NetCDF file.

.. code-block:: yaml

   output:
     my_results:                  # name of this output stream (= output filename stem)
       time_unit: dt              # time unit for the interval
       time_step: 30              # write every 30 time units
       time_method: point         # instantaneous snapshot
       variables:
         - source: /*             # include all registered variables

   output:
     hourly:
       time_unit: hour
       time_step: 1
       variables:
         - source: /temp          # temperature only

Output is implemented in :mod:`pygotm.gotm.runtime_output` and scheduled
via :class:`pygotm.gotm.runtime_params.RuntimeParams`.

Stream Name
~~~~~~~~~~~

The key under ``output`` (e.g., ``my_results``) is both the name of the
output stream and the stem of the output NetCDF filename.  The file is
written to the working directory as ``<name>.nc``.

``output.<name>.time_unit``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``second``, ``hour``, ``day``, ``month``, ``year``, ``dt``
   * - **Default**
     - ``"day"``

The time unit for the output interval.

``dt``
   Model time steps.  A ``time_step`` of ``30`` with ``time_unit: dt`` means
   write output every 30 integration time steps.  This is the most flexible
   option and is independent of the actual time-step length.

``second`` / ``hour`` / ``day`` / ``month`` / ``year``
   Calendar-based intervals.  A ``time_step`` of ``1`` with
   ``time_unit: hour`` writes every hour regardless of ``dt``.

``output.<name>.time_step``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - integer
   * - **Range**
     - ≥ 1
   * - **Default**
     - ``1``

Number of ``time_unit`` intervals between output writes.  For example,
``time_step: 6`` with ``time_unit: hour`` writes every 6 hours.

``output.<name>.time_method``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``point``, ``mean``, ``integrated``
   * - **Default**
     - ``"point"``

How the output quantity is temporally aggregated over the output interval.

``point``
   Instantaneous snapshot at the output time.  This is the only method
   currently supported by the compiled Numba runtime.

``mean``
   Time-average of the variable over the output interval.
   **[legacy only]**.

``integrated``
   Time-integral of the variable.  **[legacy only]**.

``output.<name>.variables``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A list of variable selector records that controls which model fields are
written.

.. code-block:: yaml

   variables:
     - source: /*              # all registered variables
     - source: /temp           # temperature profile
     - source: /salt           # salinity profile
     - source: /u              # U-velocity
     - source: /tke            # turbulent kinetic energy

Each entry has a ``source`` key that is a path into the field registry.  Use
``/*`` to select all registered variables.  Individual variable names use the
GOTM field-registry convention (see
:func:`pygotm.gotm.register_all_variables.do_register_all_variables` for the
full list of available names).

.. rubric:: Registered output variables

The following variables are registered in the compiled runtime and available
for output:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Name
     - Units
     - Description
   * - ``temp``
     - °C
     - Conservative temperature (cell centres)
   * - ``salt``
     - g kg\ :sup:`−1`
     - Absolute salinity (cell centres)
   * - ``u``
     - m s\ :sup:`−1`
     - West–East velocity (cell centres)
   * - ``v``
     - m s\ :sup:`−1`
     - South–North velocity (cell centres)
   * - ``tke``
     - m\ :sup:`2` s\ :sup:`−2`
     - Turbulent kinetic energy (cell interfaces)
   * - ``eps``
     - m\ :sup:`2` s\ :sup:`−3`
     - Dissipation rate ε (cell interfaces)
   * - ``num``
     - m\ :sup:`2` s\ :sup:`−1`
     - Eddy viscosity (cell interfaces)
   * - ``nuh``
     - m\ :sup:`2` s\ :sup:`−1`
     - Heat diffusivity (cell interfaces)
   * - ``nus``
     - m\ :sup:`2` s\ :sup:`−1`
     - Salinity diffusivity (cell interfaces)
   * - ``L``
     - m
     - Turbulent length scale (cell interfaces)
   * - ``P``
     - m\ :sup:`2` s\ :sup:`−3`
     - Shear production (cell interfaces)
   * - ``G``
     - m\ :sup:`2` s\ :sup:`−3`
     - Buoyancy production/destruction (cell interfaces)
   * - ``cmue1`` / ``cmue2``
     - dimensionless
     - Stability functions (cell interfaces)
   * - ``as`` / ``an``
     - dimensionless
     - Dimensionless shear/stratification (cell interfaces)
   * - ``h``
     - m
     - Layer thickness (cell centres)
   * - ``z``
     - m
     - Layer centre depth (negative below surface)
   * - ``SS``
     - s\ :sup:`−2`
     - Shear frequency squared (cell interfaces)
   * - ``xP``
     - W m\ :sup:`−2`
     - Internal absorbed shortwave radiation (cell centres)
   * - ``avh``
     - m\ :sup:`2` s\ :sup:`−1`
     - Biological light attenuation (cell centres)

.. _yaml-restart:

``restart``
-----------

Controls restart/checkpoint behaviour.

.. code-block:: yaml

   restart:
     load: false

``restart.load``
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``false``

If ``true``, initialise the model state from a restart file ``restart.nc``
located in the same directory as ``gotm.yaml``.  The restart file must have
been produced by a previous pyGOTM run with identical grid dimensions.

When ``load: false`` (default), the model is initialised from the
``temperature`` and ``salinity`` profile specifications.

.. note::
   The restart file format is compatible with Fortran GOTM's ``restart.nc``
   convention.  A restart can therefore be used to continue a Fortran GOTM
   run in pyGOTM or vice versa, provided the grid is identical.

.. _yaml-seagrass:

``seagrass``
------------

Seagrass canopy drag parameterisation.

.. code-block:: yaml

   seagrass:
     method: 0
     file: seagrass.dat
     alpha: 0.0

``seagrass.method``
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - integer
   * - **Valid values**
     - ``0`` (off), ``1`` (from file)
   * - **Default**
     - ``0``

``0``
   No seagrass parameterisation.

``1``
   Seagrass canopy geometry and density are read from ``seagrass.file``.
   The canopy drag is added to the bottom boundary layer of the turbulence
   model.  Implemented in :mod:`pygotm.extras.seagrass.seagrass`.
   **[legacy only]** in the current compiled runtime.

``seagrass.file``
~~~~~~~~~~~~~~~~~

Path to the seagrass specification file.  *Default*: ``"seagrass.dat"``.

``seagrass.alpha``
~~~~~~~~~~~~~~~~~~

Efficiency of leaf-level turbulence production.  *Default*: ``0.0``.
