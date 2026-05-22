.. _input-file-formats:

Input Data File Formats
=======================

This section is the definitive reference for every ASCII data file that
pyGOTM reads when ``method: file`` is specified in ``gotm.yaml``.  Four
distinct file formats exist.  Use the table below to find the right format
for any given parameter.

.. list-table:: File format quick reference
   :header-rows: 1
   :widths: 30 20 50

   * - Format
     - Section
     - Used for
   * - :ref:`fmt-timeseries`
     - All
     - Scalar forcing fields: heat flux, wind, air temperature, humidity,
       cloud, precipitation, shortwave radiation, longwave radiation, SST, SSS,
       pressure gradients, wave parameters, vertical velocity,
       light extinction coefficients
   * - :ref:`fmt-profile`
     - Initial conditions, Dynamics
     - Full-depth time series of temperature, salinity, and velocity profiles
   * - :ref:`fmt-grid`
     - Location / Grid
     - Custom vertical layer thicknesses or sigma fractions
   * - :ref:`fmt-seagrass`
     - Output / Extras
     - Seagrass canopy geometry (static throughout the simulation)

.. note::

   All files are plain UTF-8 text.  Blank lines and lines whose first
   non-whitespace character is ``#`` or ``!`` are treated as comments and
   ignored everywhere.  Both ``YYYY-MM-DD`` and ``YYYY/MM/DD`` date
   separators are accepted in timestamps (the parser reads by character
   position, not by separator character).

----

.. _fmt-timeseries:

Timeseries (Scalar) File
------------------------

**Used by:** every ``InputSetting`` field where ``method: file`` is set —
including ``surface.fluxes.heat``, ``surface.fluxes.tx``/``ty``,
``surface.u10``, ``surface.v10``, ``surface.airp``, ``surface.airt``,
``surface.hum``, ``surface.cloud``, ``surface.precip``, ``surface.swr``,
``surface.longwave``, ``surface.sst``, ``surface.sss``,
``mimic_3d.ext_pressure.dpdx``/``dpdy``/``h``,
``mimic_3d.int_pressure.gradients.*``, ``mimic_3d.zeta``,
``waves.Hs``/``Tz``/``phiw``, ``w.max``/``w.height``, and
``light_extinction.A``/``g1``/``g2``.

**Implemented in:** :func:`pygotm.input.read_obs` and
:class:`pygotm.input._TimeseriesFile`.

Description
~~~~~~~~~~~

Each non-comment line is one *observation record* consisting of a
date-time timestamp followed by one or more space-separated floating-point
values.  Multiple variables can be stored in a single file with each
variable occupying its own column; the YAML ``column:`` key (1-based)
selects which column to read.

Line format
~~~~~~~~~~~

.. code-block:: text

   YYYY-MM-DD HH:MM:SS   val_1   val_2   ...   val_N

* **Timestamp** — exactly the first 19 characters of the line
  (``YYYY-MM-DD HH:MM:SS``).  The date separator may be ``-`` or ``/``;
  the time separator must be ``:``.  Seconds must be present (``SS``).

  Examples of valid timestamps:

  .. code-block:: text

     1976-04-06 06:00:00
     2013/01/15 00:00:00

* **Values** — whitespace-separated (spaces or tabs) floating-point numbers
  starting at character position 19.  Scientific notation (``1.23e-04``)
  is accepted.  There must be at least as many columns as the highest
  ``column:`` index used by any variable reading this file.

* **Column selection** — The YAML key ``column: N`` (default ``1``) selects
  the *N*-th whitespace-delimited value after the timestamp.  Multiple
  variables can share a file by pointing to different columns, as in the
  ``meteo.dat`` example below.

Requirements and behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Records must be in **strictly chronological order**.
* The first record's timestamp must be **≤ simulation start time**; the
  last record must be **≥ simulation end time**.  If end-of-file is reached
  before the simulation ends, pyGOTM raises a ``RuntimeError``.
* Values are **linearly interpolated** to each model time step (not clamped
  to the nearest record).  The interpolated value is additionally clamped to
  the bracket ``[min(obs1, obs2), max(obs1, obs2)]``.
* ``scale_factor`` and ``add_offset`` from the YAML are applied after
  reading: ``final_value = scale_factor × file_value + add_offset``.
* ``minimum`` and ``maximum`` bounds are checked after scaling; a violation
  raises a ``ValueError``.

Column count
~~~~~~~~~~~~

There is no fixed column count.  The file must contain at least as many
value columns as the maximum ``column:`` index referencing it.  Extra
columns are silently ignored.

Examples
~~~~~~~~

**Single-column file** — one variable, one column:

.. code-block:: text

   # Net heat flux [W/m^2]
   1976-04-06 06:00:00  -212.2162
   1976-04-06 07:00:00  -202.4025
   1976-04-06 08:00:00  -191.4594

YAML reference:

.. code-block:: yaml

   surface:
     fluxes:
       heat:
         method: file
         file: heatflux.dat    # 1 column, column defaults to 1

**Two-column file** — two variables (tx and ty) sharing a single file:

.. code-block:: text

   # Wind stress [Pa]: column 1 = tx (West-East), column 2 = ty (South-North)
   1976-04-06 06:00:00   3.953950e-01  -6.470100e-01
   1976-04-06 07:00:00   6.952790e-01  -5.391750e-01
   1976-04-06 08:00:00   8.113300e-01  -6.182540e-01

YAML reference:

.. code-block:: yaml

   surface:
     fluxes:
       tx: { method: file, file: momentumflux.dat, column: 1 }
       ty: { method: file, file: momentumflux.dat, column: 2 }

**Six-column meteorological file** — multiple variables in one file:

.. code-block:: text

   # Columns: u10[m/s]  v10[m/s]  airp[Pa]  airt[°C]  hum[°C]  cloud[-]
   1979-01-01 00:00:00   4.5087   10.2034   1002.02   13.9150   8.8157   1.0000
   1979-01-01 06:00:00   6.9826   11.0999    997.53   14.4349   9.0317   0.6770

YAML reference:

.. code-block:: yaml

   surface:
     u10:   { method: file, file: meteo.dat, column: 1 }
     v10:   { method: file, file: meteo.dat, column: 2 }
     airp:  { method: file, file: meteo.dat, column: 3 }
     airt:  { method: file, file: meteo.dat, column: 4 }
     hum:   { method: file, file: meteo.dat, column: 5 }
     cloud: { method: file, file: meteo.dat, column: 6 }

----

.. _fmt-profile:

Profile (Depth-Series) File
----------------------------

**Used by:** ``temperature.file``, ``salinity.file``,
``velocities.u`` (``method: file``), ``velocities.v`` (``method: file``).

**Implemented in:** :func:`pygotm.input.read_profiles` and
:class:`pygotm.input._ProfileFile`.

Description
~~~~~~~~~~~

A profile file contains a **time series of full-depth vertical profiles**.
Each profile is a self-contained block consisting of a header line followed
by a fixed number of depth–value rows.  Unlike the timeseries format, the
depth coordinate is explicit in each data row, and pyGOTM interpolates the
raw data onto the model grid at runtime.

Block header line
~~~~~~~~~~~~~~~~~

.. code-block:: text

   YYYY-MM-DD HH:MM:SS   N   up_down

* **Timestamp** — same 19-character format as :ref:`fmt-timeseries`.
* **N** (integer) — number of depth-value rows that follow in this block.
* **up_down** (integer) — ordering of the depth rows:

  * ``1`` — **bottom-first**: the first data row is the **deepest** level;
    rows progress from the bottom toward the surface.
  * ``2`` — **surface-first**: the first data row is the **shallowest** level;
    rows progress from the surface toward the bottom.

Data rows
~~~~~~~~~

.. code-block:: text

   depth_m   val_1   [val_2   ...]

* **depth_m** — depth in metres, **negative below the surface**
  (e.g., ``-10.0`` is 10 m below the surface).  Values must be in
  ascending or descending order consistent with ``up_down``.
* **val_1 … val_N** — one or more space-separated floats.  The
  ``column:`` YAML key (1-based, default ``1``) selects which column to
  use.  Multiple profiles (e.g., temperature and salinity) can share a file.

Requirements and behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Profile blocks must be in **strictly chronological order**.
* The simulation must start no earlier than the first profile's timestamp.
* If only **one profile block** is present in the file, that profile is used
  as a static initial condition without temporal interpolation — no further
  blocks are required.
* For **multiple profiles**: pyGOTM reads ahead to bracket the current
  simulation time and linearly interpolates between the two bounding
  profiles in time.
* Each block is **vertically interpolated** onto the model grid using the
  linear scheme in :func:`pygotm.util.gridinterpol.gridinterpol`.  The
  raw data points do **not** need to align with the model layers.
* ``scale_factor`` and ``add_offset`` are applied after reading (same as
  timeseries).
* End-of-file before the simulation end raises a ``RuntimeError`` (unless
  only one profile is in the file).

Column count
~~~~~~~~~~~~

At least ``max(column)`` value columns must follow the depth column.
Extra columns are ignored.  A single-variable file has 2 columns total
(depth + one value); a shared two-variable file has 3 columns
(depth + two values).

Examples
~~~~~~~~

**Single-variable temperature profile, surface-first (up_down=2)**:

.. code-block:: text

   # Temperature profiles [°C]
   # Header: timestamp   N_levels   up_down(2=surface-first)
   1958-01-16 00:00:00   30   2
   -5.022      8.3823
   -15.079     8.3834
   -25.161     8.3843
   ...
   -270.534   10.0303

   1958-02-15 00:00:00   30   2
   -5.022      8.4100
   ...

*Depths go from shallowest to deepest because up_down=2 (surface-first).*

YAML reference:

.. code-block:: yaml

   temperature:
     method: file
     file: t_prof_file.dat
     column: 1

**Single-variable salinity profile, bottom-first (up_down=1)**:

.. code-block:: text

   # Salinity profiles [psu]
   # Header: timestamp   N_levels   up_down(1=bottom-first)
   1976/04/06 00:00:00   50   1
      -143.55    35.105
      -140.65    35.104
      -137.75    35.103
      ...
        -0.90    35.170

*Depths go from deepest to shallowest because up_down=1 (bottom-first).*

YAML reference:

.. code-block:: yaml

   salinity:
     method: file
     file: sprof.dat
     column: 1

**Two-variable profile file** — temperature (col 1) and salinity (col 2), surface-first:

.. code-block:: text

   # T [°C] and S [psu] profiles, 56 levels, surface-first (up_down=2)
   2013/01/15 00:00:00   56   2
   -0.506    13.1615   38.4442
   -1.556    13.1615   38.4442
   -2.668    13.1615   38.4442
   ...
   -53.851   13.1672   38.4442

YAML reference:

.. code-block:: yaml

   temperature:
     method: file
     file: init_ts.dat
     column: 1   # temperature column

   salinity:
     method: file
     file: init_ts.dat
     column: 2   # salinity column

**Velocity profile file** — u (col 1) and v (col 2), surface-first (up_down=2):

.. code-block:: text

   # Horizontal velocity profiles [m/s]
   # Header: timestamp   N_levels   up_down(2=surface-first)
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

----

.. _fmt-grid:

Grid Layer File
---------------

**Used by:** ``grid.file`` when ``grid.method`` is ``file_sigma`` or
``file_h``.

**Implemented in:** :func:`pygotm.meanflow.updategrid._read_sigma_grid_file`
and :func:`pygotm.meanflow.updategrid._read_cartesian_grid_file`.

Description
~~~~~~~~~~~

Defines a custom vertical layer distribution.  The file contains no
timestamps — the grid is static throughout the simulation.  This format
does **not** use the comment-skipping logic of the other formats; every
line must be valid data.

Structure
~~~~~~~~~

.. code-block:: text

   N
   value_1
   value_2
   ...
   value_N

* **Line 1** — a single integer ``N`` equal to ``grid.nlev``.  If ``N``
  does not match the YAML ``grid.nlev``, pyGOTM raises a ``ValueError``.
* **Lines 2 … N+1** — one layer per line.  The **first whitespace-separated
  token** on each line is used; any additional columns are silently ignored.
* **Layer ordering** — the surface layer (model index ``i = nlev``) is
  listed **first** in the file; the bottom layer (``i = 1``) is listed
  **last**.

Values by method
~~~~~~~~~~~~~~~~

``file_sigma`` (normalised sigma fractions)
   Each value is the fractional thickness of one layer expressed as a
   fraction of the total water depth (dimensionless, 0–1).  All ``N``
   values must sum to exactly ``1.0`` within a tolerance of ``1e-8``.
   A violation raises a ``ValueError``.

``file_h`` (absolute layer thicknesses)
   Each value is the layer thickness in **metres** (positive).  The sum of
   all values must equal ``location.depth`` within ``1e-5 m``.

.. note::

   Because files may contain extra columns (e.g., from data-processing
   tools that append additional diagnostics), only the first token per line
   is read.  The ``grid_z.dat`` file from the *asics_med* reference
   case illustrates this — it stores three columns per row but only the
   first is consumed.

Example (``file_h``, 5 layers, surface first)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   5
   2.0
   5.0
   10.0
   20.0
   63.0

Total = 100 m; must equal ``location.depth: 100.0``.

YAML reference:

.. code-block:: yaml

   location:
     depth: 100.0

   grid:
     nlev: 5
     method: file_h
     file: my_layers.dat

----

.. _fmt-seagrass:

Seagrass Canopy File
---------------------

**Used by:** ``seagrass.file`` (default filename ``seagrass.dat``) when
``seagrass.method: 1``.

**Implemented in:** :func:`pygotm.extras.seagrass.seagrass._read_grass_file`.

Description
~~~~~~~~~~~

Specifies the geometric and dynamic properties of a seagrass canopy as a
vertical profile of discrete canopy elements.  This is a **static** file
— values do not change during the simulation.

Structure
~~~~~~~~~

.. code-block:: text

   # comment lines (optional)
   N
   z_1   excursion_1   friction_1
   z_2   excursion_2   friction_2
   ...
   z_N   excursion_N   friction_N

* **Comments** — blank lines and lines beginning with ``#`` or ``!`` are
  skipped.
* **Line 1** (first non-comment line) — a single integer ``N`` giving the
  number of canopy height levels.
* **Lines 2 … N+1** — one canopy element per line with **exactly three**
  whitespace-separated values:

  1. **z** (column 1) — height above the bottom (metres, positive upward).
     Defines the vertical position of the canopy element within the water
     column.
  2. **excursion** (column 2) — maximum lateral excursion distance of the
     Lagrangian leaf tip (metres).  When the displacement exceeds this
     limit, the leaf exerts drag on the current.  Set to ``0.0`` for an
     element with no excursion (rigid canopy).
  3. **friction** (column 3) — canopy friction coefficient applied to the
     horizontal velocity when the excursion limit is reached (m s\ :sup:`-1`
     equivalent, used as a drag scaling in Verduin & Backhaus (2000)).

* If the file declares ``N`` rows but fewer are present, pyGOTM raises a
  ``ValueError``.
* Extra columns beyond column 3 on any data line are silently ignored.

.. note::
   Due to a known bug in the original ``seagrass.F90`` (see the docstring
   of :func:`pygotm.extras.seagrass.seagrass.init_seagrass`), the seagrass
   module is **never activated** regardless of the ``seagrass.method``
   setting.  The file is not read during normal simulation runs.  The
   format is documented here for completeness and future use.

Example
~~~~~~~

.. code-block:: text

   # Seagrass canopy: z [m], excursion [m], friction [-]
   87
       0.0500   0.0000   0.0500
       0.0550   0.0040   0.0550
       0.0600   0.0080   0.0600
       0.0650   0.0120   0.0650
       0.0700   0.0160   0.0700

(Excerpt from the *seagrass* reference case ``seagrass.dat``, 87 levels.)

YAML reference:

.. code-block:: yaml

   seagrass:
     method: 1
     file: seagrass.dat
     alpha: 0.0

----

.. _input-file-path-resolution:

Path Resolution
---------------

All ``file:`` paths in ``gotm.yaml`` are interpreted **relative to the
directory that contains gotm.yaml** (the case directory).  Absolute paths
are also accepted.  To use a shared file from a parent directory, prefix
with ``../``, e.g.:

.. code-block:: yaml

   temperature:
     method: file
     file: ../shared/t_prof.dat

.. _input-file-common-errors:

Common Errors
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error message (excerpt)
     - Cause and fix
   * - ``simulation starts before the first observation``
     - The first timestamp in the file is later than ``time.start``.
       Prepend a record at or before the start date.
   * - ``end of file reached while updating time series``
     - The last timestamp in the file is earlier than ``time.stop``.
       Append a record at or beyond the stop date.
   * - ``expected N values after timestamp, found M``
     - A timeseries row has fewer columns than the ``column:`` index
       requires.  Check for truncated or malformed lines.
   * - ``profile block header must contain N and up_down``
     - A profile header line has fewer than two values after the timestamp.
       Verify the header format ``YYYY-MM-DD HH:MM:SS   N   up_down``.
   * - ``profile row must contain depth plus N values``
     - A data row in a profile block is missing the depth column or value
       columns.  Check for blank or short lines inside a profile block.
   * - ``Number of layers specified in file != number of model layers``
     - The integer on line 1 of a grid file does not match ``grid.nlev``.
       Ensure the header integer and ``nlev`` are consistent.
   * - ``Sum of all sigma fractions in grid_file should be 1.0``
     - For ``file_sigma``: the fractions do not sum to 1.  Re-normalise.
   * - ``seagrass file declared N rows, found M``
     - The header integer does not match the number of data lines in the
       seagrass file.  Update the header count or add/remove rows.
