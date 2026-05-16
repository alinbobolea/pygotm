Test Cases
==========

pyGOTM is validated against the 22 official GOTM 6.0.7 test cases.  The table
below summarizes the latest checked-in ``validation/results.json`` snapshot,
generated at ``2026-05-14T01:16:40Z``.

Case status is aggregated from Frechet variable statuses:

* ``PASS`` means every compared numeric variable has status ``PASS``.
* ``FAIL`` means at least one compared variable is ``MARGINAL``,
  ``DISCREPANT``, or ``BROKEN``.
* ``ERROR`` means the case failed during setup, execution, or comparison before
  a complete variable table could be produced.

The snapshot verdict is ``PARTIAL PARITY``: 3 cases pass and 19 cases fail.
Across all cases, the variable totals are 1702 ``PASS``, 81 ``MARGINAL``, 120
``DISCREPANT``, and 511 ``BROKEN``.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 15 20

   * - Case
     - Case status
     - PASS
     - MARGINAL
     - DISCREPANT
     - BROKEN
     - Notes
   * - ``couette``
     - PASS
     - 100
     - 0
     - 0
     - 0
     - Simple Couette flow.
   * - ``blacksea``
     - FAIL
     - 106
     - 8
     - 3
     - 4
     - Black Sea seasonal cycle.
   * - ``channel``
     - PASS
     - 100
     - 0
     - 0
     - 0
     - Open-channel flow.
   * - ``entrainment``
     - PASS
     - 100
     - 0
     - 0
     - 0
     - Convective entrainment.
   * - ``estuary``
     - FAIL
     - 99
     - 1
     - 0
     - 0
     - Estuarine circulation.
   * - ``flex``
     - FAIL
     - 62
     - 2
     - 14
     - 26
     - FLEX experiment.
   * - ``gotland``
     - FAIL
     - 67
     - 5
     - 17
     - 15
     - Baltic Sea Gotland Deep.
   * - ``lago_maggiore``
     - FAIL
     - 95
     - 3
     - 2
     - 5
     - Alpine lake.
   * - ``langmuir``
     - FAIL
     - 68
     - 5
     - 9
     - 31
     - Langmuir turbulence with Stokes drift.
   * - ``liverpool_bay``
     - FAIL
     - 57
     - 0
     - 10
     - 38
     - Tidal mixing in Liverpool Bay.
   * - ``medsea_east``
     - FAIL
     - 105
     - 9
     - 12
     - 14
     - Eastern Mediterranean.
   * - ``medsea_west``
     - FAIL
     - 93
     - 22
     - 12
     - 13
     - Western Mediterranean.
   * - ``nns_annual``
     - FAIL
     - 35
     - 0
     - 2
     - 78
     - North Sea annual cycle.
   * - ``nns_seasonal``
     - FAIL
     - 51
     - 1
     - 4
     - 49
     - North Sea seasonal cycle.
   * - ``ows_papa``
     - FAIL
     - 57
     - 4
     - 16
     - 36
     - Ocean Weather Station Papa.
   * - ``plume``
     - FAIL
     - 60
     - 0
     - 0
     - 53
     - Freshwater plume.
   * - ``resolute``
     - FAIL
     - 43
     - 1
     - 6
     - 65
     - Arctic mixing.
   * - ``reynolds``
     - FAIL
     - 57
     - 6
     - 1
     - 41
     - Reynolds number scaling.
   * - ``rouse``
     - FAIL
     - 102
     - 1
     - 2
     - 3
     - Rouse sediment profile.
   * - ``seagrass``
     - FAIL
     - 75
     - 0
     - 0
     - 29
     - Seagrass canopy dynamics.
   * - ``wave_breaking``
     - FAIL
     - 82
     - 6
     - 5
     - 7
     - Wave-breaking enhanced mixing.
   * - ``asics_med``
     - FAIL
     - 88
     - 7
     - 5
     - 4
     - Mediterranean deep convection.

Indicator Summary
-----------------

The current validation suite uses Frechet-distance indicators from
``src/pygotm/validation``:

``d_raw``
   Discrete Frechet distance on aligned original values.

``d_norm``
   Discrete Frechet distance after robust dynamic linear/log normalization.

``d_rel``
   ``d_raw / signal_scale`` for variables whose signal magnitude is below the
   configured variable floor.

``score``
   The status-driving value.  This is normally ``d_norm`` and switches to
   ``d_rel`` for below-floor signals.  The selected indicator is recorded in
   ``metric_mode``.

Variable status bands are:

* ``PASS`` - ``score < 0.01``
* ``MARGINAL`` - ``0.01 <= score < 0.05``
* ``DISCREPANT`` - ``0.05 <= score < 0.20``
* ``BROKEN`` - ``score >= 0.20`` or a structural comparison failure

Variables listed in ``PYGOTM_VARIABLES`` are reported in the PyGOTM section.
Other numeric variables are treated as FABM biogeochemical variables.

The previous per-variable tolerance parameters ``atol``, ``rtol``, and
``scale_floor`` are not used by the current Frechet suite.
