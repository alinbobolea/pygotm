Test Cases
==========

pyGOTM is validated against the 22 official GOTM 6.0.7 test cases.  The table
below summarizes the latest checked-in ``validation/results.json`` snapshot,
generated at ``2026-05-19T17:50:23Z``.

Case status is aggregated from Frechet variable statuses:

* ``PASS`` means every compared numeric variable has status ``PASS``.
* ``FAIL`` means at least one compared variable is ``MARGINAL``,
  ``DISCREPANT``, or ``BROKEN``.
* ``ERROR`` means the case failed during setup, execution, or comparison before
  a complete variable table could be produced.

The snapshot verdict is ``PARTIAL PARITY``: 14 cases pass and 8 cases fail.
Across all cases, the variable totals are 2184 ``PASS``, 93 ``MARGINAL``, 78
``DISCREPANT``, and 59 ``BROKEN``.

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
     - PASS
     - 121
     - 0
     - 0
     - 0
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
     - PASS
     - 100
     - 0
     - 0
     - 0
     - Estuarine circulation.
   * - ``flex``
     - PASS
     - 104
     - 0
     - 0
     - 0
     - FLEX experiment.
   * - ``gotland``
     - FAIL
     - 73
     - 5
     - 20
     - 6
     - Baltic Sea Gotland Deep.
   * - ``lago_maggiore``
     - PASS
     - 105
     - 0
     - 0
     - 0
     - Alpine lake.
   * - ``langmuir``
     - FAIL
     - 88
     - 8
     - 15
     - 2
     - Langmuir turbulence with Stokes drift.
   * - ``liverpool_bay``
     - PASS
     - 105
     - 0
     - 0
     - 0
     - Tidal mixing in Liverpool Bay.
   * - ``medsea_east``
     - FAIL
     - 114
     - 24
     - 2
     - 0
     - Eastern Mediterranean.
   * - ``medsea_west``
     - FAIL
     - 118
     - 21
     - 1
     - 0
     - Western Mediterranean.
   * - ``nns_annual``
     - FAIL
     - 90
     - 17
     - 7
     - 1
     - North Sea annual cycle.
   * - ``nns_seasonal``
     - PASS
     - 105
     - 0
     - 0
     - 0
     - North Sea seasonal cycle.
   * - ``ows_papa``
     - FAIL
     - 58
     - 5
     - 14
     - 36
     - Ocean Weather Station Papa.
   * - ``plume``
     - PASS
     - 113
     - 0
     - 0
     - 0
     - Freshwater plume.
   * - ``resolute``
     - FAIL
     - 88
     - 7
     - 14
     - 6
     - Arctic mixing.
   * - ``reynolds``
     - PASS
     - 105
     - 0
     - 0
     - 0
     - Reynolds number scaling.
   * - ``rouse``
     - PASS
     - 108
     - 0
     - 0
     - 0
     - Rouse sediment profile.
   * - ``seagrass``
     - PASS
     - 104
     - 0
     - 0
     - 0
     - Seagrass canopy dynamics. See :ref:`fortran-parity-deviations`.
   * - ``wave_breaking``
     - FAIL
     - 81
     - 6
     - 5
     - 8
     - Wave-breaking enhanced mixing.
   * - ``asics_med``
     - PASS
     - 104
     - 0
     - 0
     - 0
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

.. _fortran-parity-deviations:

Fortran Parity Deviations
--------------------------

Some pyGOTM behaviours intentionally mirror bugs or quirks in the reference
Fortran GOTM 6.0.7 code.  These deviations are required for bit-for-bit parity
with the reference NetCDF files and must not be "fixed" without also regenerating
the reference outputs.

seagrass: ``init_seagrass`` activation bug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affected case:** ``seagrass``

**Source file:** ``src/pygotm/extras/seagrass/seagrass.py`` — :func:`~pygotm.extras.seagrass.seagrass.init_seagrass`

Fortran ``seagrass.F90``'s ``init_seagrass`` reads the ``method`` namelist
variable into a local integer but then checks the activation flag against an
*uninitialized* local variable ``i`` instead of ``method``::

    read(namlst, nml=seagrass, err=99)
    i = method                       ! i is set correctly here …
    …
    if (i .ne. 0) seagrass_calc = .true.   ! … but this line uses the
                                            !   wrong variable in practice

Under typical gfortran stack-frame conditions ``i`` is zero on entry, so
``seagrass_calc`` is never set to ``.true.`` regardless of the YAML
configuration.  The seagrass drag kernel therefore becomes a no-op for all
reference runs.  pyGOTM mirrors this by leaving ``state.seagrass_calc`` at its
default ``False`` value in :func:`~pygotm.extras.seagrass.seagrass.init_seagrass`
and never executing the drag kernel during the timestep loop.

first_order turbulence: step-0 ``cmue1``/``cmue2`` initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affected cases:** any case using ``turb_method = first_order`` (including ``seagrass``)

**Source file:** ``src/pygotm/gotm/time_loop.py`` — :func:`~pygotm.gotm.time_loop.time_loop_compiled`

Fortran GOTM's ``init_turbulence`` seeds stability-function arrays only at
level 1 via an internal ``compute_cpsi3`` probe; all other levels remain zero
in the reference NetCDF at timestep 0.  An earlier pyGOTM version pre-populated
``cmue1``/``cmue2`` across all levels before the step-0 write by re-running the
stability-function kernel (Munk-Anderson, Schumann-Gerz, or Constant), which
overwrote those zeros with ``cm0``-derived values and broke parity.  The
pre-population block has been removed.

first_order turbulence: ``kb`` forwarded to ``alpha_mnb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affected cases:** any case using ``turb_method = first_order`` (including ``seagrass``)

**Source file:** ``src/pygotm/gotm/time_loop.py`` — :func:`~pygotm.gotm.time_loop.step_turbulence_first_order_single`

The buoyancy-variance array ``kb`` is read-only in first_order mode (it is
initialised to ``kb_min`` and never updated by the first-order closure), but it
must still be passed to :func:`~pygotm.gotm.time_loop.step_turbulence_first_order_single`
so that ``alpha_mnb`` can compute the correct ``at`` stability parameter.  A
previous version passed ``tke`` as a placeholder for ``kb``, producing a wrong
``at`` value whenever ``kb > kb_min``.
