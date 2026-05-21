Surface Forcing
===============

The ``surface`` section controls all atmospheric and surface boundary
conditions: heat and momentum fluxes, meteorological inputs, and sea-surface
nudging.  It maps directly to the air-sea interaction layer implemented in
:mod:`pygotm.airsea`.

When any surface forcing field uses ``method: file``, the data file must
follow the :ref:`fmt-timeseries` format.  Multiple variables can share a
single file using different ``column:`` indices — this is the standard
pattern for combined meteorological files (``meteo.dat``).

.. seealso:: :ref:`input-file-formats` for complete specifications and
   examples of all supported input file formats.

.. code-block:: yaml

   surface:
     fluxes:
       method: fairall
       heat:   { method: file, file: heat_flux.dat }
       tx:     { method: file, file: momentum_flux.dat, column: 1 }
       ty:     { method: file, file: momentum_flux.dat, column: 2 }
     u10:    { method: file, file: meteo.dat, column: 1 }
     v10:    { method: file, file: meteo.dat, column: 2 }
     ssuv_method: absolute
     airp:   { method: file, file: meteo.dat, column: 3 }
     airt:   { method: file, file: meteo.dat, column: 4 }
     hum:
       method: file
       file: meteo.dat
       column: 5
       type: wet_bulb
     cloud:  { method: file, file: meteo.dat, column: 6 }
     precip: { method: constant, constant_value: 0.0 }
     swr:    { method: file, file: swr.dat }
     longwave: { method: file, file: lw.dat }
     sst:    { method: off }
     sss:    { method: off }

.. _yaml-surface-fluxes:

``surface.fluxes``
------------------

``surface.fluxes.method``
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``off``, ``kondo``, ``fairall``
   * - **Default**
     - ``"off"``

Selects how heat and momentum fluxes at the sea surface are computed.

``off``
   Prescribed fluxes are read directly from ``heat``, ``tx``, and ``ty``
   inputs.  No bulk formula is applied.  Use this when you have observed
   flux values.

``kondo``
   Heat and momentum fluxes are computed from meteorological variables
   (wind speed, air temperature, humidity) using the bulk formula of
   Kondo (1975).  Implemented in :mod:`pygotm.airsea.kondo`.  Requires
   ``u10``, ``v10``, ``airp``, ``airt``, and ``hum`` to be provided.

``fairall``
   COARE 3.0 bulk algorithm (Fairall et al. 1996/2003).  More accurate than
   Kondo for open-ocean conditions with high wind speeds.  Implemented in
   :mod:`pygotm.airsea.fairall`.  Requires the same meteorological inputs as
   ``kondo``.

   .. note::
      The compiled Numba runtime currently supports ``fairall`` for the
      profile path.  The ``kondo`` method requires the legacy Python loop.

``surface.fluxes.heat``
~~~~~~~~~~~~~~~~~~~~~~~

Prescribed non-solar surface heat flux (sensible + latent + net longwave).

Follows the ``InputSetting`` pattern (``method: constant | file``).
When ``method: file``: single-column timeseries file; see :ref:`fmt-timeseries`.

.. list-table::
   :widths: 20 80

   * - **Units**
     - W m\ :sup:`−2`
   * - **Default**
     - ``0.0``
   * - **Sign convention**
     - **Atmospheric convention**: positive = ocean loses heat (upward).

.. warning::

   GOTM uses the atmospheric sign convention for ``hflux``: a positive value
   means the ocean is **losing** heat to the atmosphere.  This is the
   **opposite** of the physical-oceanography convention.  In
   :func:`pygotm.meanflow.temperature.temperature`, the flux enters as
   ``DiffTup = -hflux / (rho0 * cp)``, so a positive heat loss produces a
   negative (cooling) Neumann boundary condition.

``surface.fluxes.tx``
~~~~~~~~~~~~~~~~~~~~~

Prescribed wind stress in the West–East direction.

.. list-table::
   :widths: 20 80

   * - **Units**
     - Pa
   * - **Default**
     - ``0.0``

Used directly as the upper boundary condition for the U-momentum equation
when ``fluxes.method: off`` or to provide a fallback when bulk methods are
active.

When ``method: file``: ``tx`` and ``ty`` can share a single two-column
timeseries file using ``column: 1`` and ``column: 2`` respectively.
See :ref:`fmt-timeseries` for format details.

``surface.fluxes.ty``
~~~~~~~~~~~~~~~~~~~~~

Prescribed wind stress in the South–North direction.

.. list-table::
   :widths: 20 80

   * - **Units**
     - Pa
   * - **Default**
     - ``0.0``

.. _yaml-surface-wind:

Wind Speed
----------

``surface.u10``
~~~~~~~~~~~~~~~

West–East wind speed at 10 m height.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m s\ :sup:`−1`
   * - **Default**
     - ``0.0``

Required when ``fluxes.method: kondo`` or ``fairall``.

``surface.v10``
~~~~~~~~~~~~~~~

South–North wind speed at 10 m height.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m s\ :sup:`−1`
   * - **Default**
     - ``0.0``

``surface.ssuv_method``
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``absolute``, ``relative``
   * - **Default**
     - ``"relative"``

Treatment of the wind vector used in bulk formula.

``absolute``
   Wind components ``u10`` and ``v10`` are interpreted as absolute wind speed
   (relative to the ground).

``relative``
   Wind speed used in the bulk formula is the wind velocity relative to the
   surface ocean current: :math:`(u_{10} - u_s,\; v_{10} - v_s)`.  This is
   more physically correct but requires surface currents.

.. _yaml-surface-met:

Meteorological Inputs
---------------------

These inputs are required by the Kondo and Fairall bulk algorithms.
All follow the ``InputSetting`` pattern (``method: constant | file``).

When ``method: file``, these inputs are read from a timeseries file
(see :ref:`fmt-timeseries`).  The standard pattern is to store all
meteorological variables in a single multi-column file and select each
variable by its ``column:`` index.  The conventional six-column layout
expected by most reference cases is:

.. list-table:: Conventional ``meteo.dat`` column layout
   :header-rows: 1
   :widths: 15 25 20 40

   * - Column
     - Variable
     - Units
     - YAML key
   * - 1
     - ``u10``
     - m s\ :sup:`−1`
     - ``surface.u10``
   * - 2
     - ``v10``
     - m s\ :sup:`−1`
     - ``surface.v10``
   * - 3
     - ``airp``
     - Pa (or hPa — see note below)
     - ``surface.airp``
   * - 4
     - ``airt``
     - °C
     - ``surface.airt``
   * - 5
     - ``hum``
     - type-dependent (see ``hum.type``)
     - ``surface.hum``
   * - 6
     - ``cloud``
     - fraction (0–1)
     - ``surface.cloud``

.. note::

   The reference cases store air pressure in **hPa** (hectopascals).  The
   GOTM bulk formulae expect Pa.  If your ``meteo.dat`` is in hPa, set
   ``airp: { scale_factor: 100.0 }`` in ``gotm.yaml`` to convert
   automatically.

``surface.airp``
~~~~~~~~~~~~~~~~

Air pressure at the surface.

.. list-table::
   :widths: 20 80

   * - **Units**
     - Pa
   * - **Default**
     - ``0.0``

Used in humidity calculations and to correct sea-level pressure.

``surface.airt``
~~~~~~~~~~~~~~~~

Air temperature at 2 m height.

.. list-table::
   :widths: 20 80

   * - **Units**
     - °C or K (detected automatically)
   * - **Default**
     - ``0.0``

``surface.hum``
~~~~~~~~~~~~~~~

Humidity at 2 m height.  Follows the ``InputSetting`` pattern plus an
additional ``type`` key.

``surface.hum.type``
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``relative``, ``wet_bulb``, ``dew_point``, ``specific``
   * - **Default**
     - ``"relative"``

Humidity measurement type.  The conversion to specific humidity is
implemented in :mod:`pygotm.airsea.humidity`.

``relative``
   Relative humidity (%).

``wet_bulb``
   Wet-bulb temperature (°C).

``dew_point``
   Dew-point temperature (°C).

``specific``
   Specific humidity (kg kg\ :sup:`−1`).

``surface.cloud``
~~~~~~~~~~~~~~~~~

Cloud cover fraction.

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless (0–1)
   * - **Range**
     - 0.0 to 1.0
   * - **Default**
     - ``0.0``

Used in longwave radiation bulk formulae.

``surface.precip``
~~~~~~~~~~~~~~~~~~

Precipitation rate.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m s\ :sup:`−1`
   * - **Default**
     - ``0.0``

Applied as a freshwater flux at the surface (affects salinity) and as a
latent-heat equivalent in bulk schemes.

.. _yaml-surface-radiation:

Radiation
---------

``surface.swr``
~~~~~~~~~~~~~~~

Incident shortwave radiation at the sea surface.

.. list-table::
   :widths: 20 80

   * - **Units**
     - W m\ :sup:`−2`
   * - **Default**
     - ``0.0``

Used when ``fluxes.method: off``.  When a bulk algorithm (``kondo``,
``fairall``) is selected, shortwave radiation can still be prescribed here
and added to the computed heat budget.  Penetration into the water column is
controlled by the ``light_extinction`` section.

When ``method: file``: single-column timeseries file (see :ref:`fmt-timeseries`).
Example: ``1976-04-06 06:00:00   13.6887``.

``surface.longwave``
~~~~~~~~~~~~~~~~~~~~

Incident or net longwave (infrared) radiation at the surface.

.. list-table::
   :widths: 20 80

   * - **Units**
     - W m\ :sup:`−2`
   * - **Default**
     - ``0.0``

Whether this represents downward or net flux depends on the longwave radiation
method configured in the ``surface`` block (``longwave_method`` in the
underlying runtime, see :attr:`pygotm.gotm.runtime_params.RuntimeParams.airsea_longwave_method`).

.. _yaml-surface-nudging:

Sea-Surface Nudging
-------------------

``surface.sst``
~~~~~~~~~~~~~~~

Sea-surface temperature nudging target.

Follows the ``InputSetting`` pattern.  When ``method: file``, model SST
is nudged toward the observed values on a configurable time scale.

When ``method: file``: single-column timeseries file (see :ref:`fmt-timeseries`).
Each record is one SST observation: ``YYYY-MM-DD HH:MM:SS   temperature_celsius``.

``surface.sss``
~~~~~~~~~~~~~~~

Sea-surface salinity nudging target.

Follows the ``InputSetting`` pattern.  Used to represent evaporation/
precipitation corrections or constrain surface salinity in long simulations.

When ``method: file``: single-column timeseries file; one SSS observation
per record: ``YYYY-MM-DD HH:MM:SS   salinity_psu``.

.. _yaml-surface-stokes:

Wave Forcing (Stokes Drift)
---------------------------

These inputs are under the ``surface`` section but also interact with the
``waves`` top-level block (see :ref:`yaml-waves`).

``surface.tx_method`` / ``ty_method``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternative wave-state-dependent momentum flux methods (Stokes drift
contribution).  These are passed through to the Stokes drift module
:mod:`pygotm.stokes_drift`.

.. note::
   Full Stokes drift integration (Craig–Banner wave-breaking BC) is
   **[legacy only]** in the current release.
