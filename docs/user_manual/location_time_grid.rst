Location, Time, and Grid
========================

These three top-level sections define where the simulation is located on
Earth, the time span to integrate, and the vertical grid structure.

.. _yaml-location:

``location``
------------

Geographic metadata for the simulation column.  These values affect Coriolis
forcing, solar angle calculations, and are written to NetCDF output attributes.

Parsed by :class:`pygotm.config.settings.LocationSettings`.

.. code-block:: yaml

   location:
     name: FLEX Experiment 1976
     latitude: 58.92
     longitude: 0.32
     depth: 145.0

``location.name``
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Default**
     - ``"GOTM site"``

Station or site name.  Used as a label in output files and reports only; has
no effect on the numerical solution.

``location.latitude``
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - degrees North
   * - **Range**
     - −90.0 to 90.0
   * - **Default**
     - ``0.0``

Geographic latitude of the simulated water column.  Controls the Coriolis
parameter via :math:`f = 2\Omega\sin(\phi)` (implemented in
:func:`pygotm.meanflow.coriolis.coriolis`) and solar zenith angle for
shortwave radiation computations.

``location.longitude``
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - degrees East
   * - **Range**
     - −360.0 to 360.0
   * - **Default**
     - ``0.0``

Geographic longitude.  Used for solar zenith angle computations and is
written to the output NetCDF as a coordinate attribute.

``location.depth``
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - metres (positive downward)
   * - **Range**
     - > 0.0
   * - **Default**
     - ``100.0``

Total water depth.  The vertical grid spans from the surface (z = 0) to
z = −depth.  Layer thicknesses are derived from this value when
``grid.method: analytical``.

.. _yaml-time:

``time``
--------

Controls the temporal span and numerical time-stepping parameters.

Parsed by :class:`pygotm.config.settings.TimeSettings`.

.. code-block:: yaml

   time:
     start: 1976-04-06 06:00:00
     stop:  1976-06-07 00:00:00
     dt: 360.0
     cnpar: 1.0

``time.start``
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Format**
     - ``yyyy-mm-dd HH:MM:SS``
   * - **Default**
     - ``"2017-01-01 00:00:00"``

Simulation start date and time.  pyGOTM accepts ISO-8601 datetime strings;
the YAML parser coerces Python ``datetime`` objects to this format
automatically.

``time.stop``
~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Format**
     - ``yyyy-mm-dd HH:MM:SS``
   * - **Default**
     - ``"2018-01-01 00:00:00"``

Simulation end date and time.  The total number of time steps is computed as
:math:`N = \lfloor (t_{stop} - t_{start}) / \Delta t \rfloor`.

``time.dt``
~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - seconds
   * - **Range**
     - > 0.0
   * - **Default**
     - ``3600.0``

Integration time step :math:`\Delta t`.  Typical values range from 10 s
(idealized turbulence cases) to 3600 s (long climate simulations).

.. warning::

   Stability of the implicit Crank–Nicolson solver degrades for very large
   ``dt`` combined with sharp stratification.  Use ``cnpar: 1.0`` (fully
   implicit) for coarse time steps; reduce ``cnpar`` only after verifying
   stability.

``time.cnpar``
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Range**
     - 0.5 to 1.0
   * - **Default**
     - ``1.0``

Crank–Nicolson weighting parameter :math:`\theta`.  A value of ``1.0`` gives
a fully implicit (backward-Euler) scheme, which is unconditionally stable but
first-order accurate in time.  The value ``0.5`` gives second-order (Crank–
Nicolson) accuracy but may be oscillatory near sharp gradients.  GOTM default
is ``1.0`` for robustness.

This parameter is passed directly to the diffusion solvers in
:mod:`pygotm.util.diff_center` and :mod:`pygotm.util.diff_face`.

.. _yaml-grid:

``grid``
--------

Defines the number and distribution of vertical layers.

Parsed by :class:`pygotm.config.settings.GridSettings`.

.. code-block:: yaml

   grid:
     nlev: 145
     method: analytical
     ddu: 0.0
     ddl: 0.0
     file: ""

``grid.nlev``
~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - integer
   * - **Range**
     - ≥ 1
   * - **Default**
     - ``100``

Number of vertical layers :math:`N`.  The model uses :math:`N+1` cell
interfaces (including surface and bottom) and :math:`N` cell centres.  Arrays
inside pyGOTM are allocated with shape ``(nlev+1,)``; index 0 is the bottom
interface and index ``nlev`` is the surface interface.

Practical guidance: use one layer per metre of water depth (``nlev == depth``)
as a starting point.  Increase for fine-scale surface-layer dynamics.

``grid.method``
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``analytical``, ``file_sigma``, ``file_h``
   * - **Default**
     - ``"analytical"``

How the layer thickness profile is constructed.

``analytical``
   Layers are equally spaced by default, with optional zooming toward the
   surface and/or bottom controlled by ``ddu`` and ``ddl``.  Implemented in
   :func:`pygotm.meanflow.updategrid.updategrid`.

``file_sigma``
   Layer thicknesses are read as normalised fractions (sigma coordinates)
   from the file specified in ``grid.file``.  The fractions must sum to 1.

``file_h``
   Layer thicknesses (in metres) are read directly from ``grid.file``.

``grid.ddu``
~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - dimensionless
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``0.0``

Surface zooming coefficient.  Larger values concentrate layers near the
surface.  Only used when ``grid.method: analytical``.  A value of ``0.0``
gives equal spacing.

``grid.ddl``
~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - float
   * - **Units**
     - dimensionless
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``0.0``

Bottom zooming coefficient.  Larger values concentrate layers near the
bottom.  Only used when ``grid.method: analytical``.

``grid.file``
~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string (file path)
   * - **Default**
     - ``""`` (empty — not used)

Path to the file containing prescribed layer thicknesses or sigma fractions.
Required when ``grid.method`` is ``file_sigma`` or ``file_h``; ignored
otherwise.  Relative paths are resolved relative to the directory containing
``gotm.yaml``.

.. seealso:: :ref:`fmt-grid` for the complete file format specification,
   including exact column layout, layer ordering, and sum-constraint
   requirements.
