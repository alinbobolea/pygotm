Test Cases
==========

pyGOTM is validated against the 22 official GOTM 6.0.7 test cases.  The table
below summarises the current validation status.  **PASS** means all output
fields satisfy the combined tolerance criterion.  **FAIL** means one or more
fields exceed the tolerance.  **UNSUPPORTED** means the compiled runtime does
not yet implement the physics required by that case; an
``UnsupportedConfigurationError`` is raised at setup.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Case
     - Status
     - Notes
   * - ``couette``
     - FAIL (29/30)
     - Simple Couette flow.  29 of 30 fields pass; ``avh`` diffusivity
       field fails the tolerance by a small margin.
   * - ``channel``
     - FAIL (26/30)
     - Open-channel flow.  26 of 30 fields pass; stability-function
       fields (``avh``, ``cmue1``, ``as``) exceed tolerance.
   * - ``blacksea``
     - FAIL (7/30)
     - Black Sea seasonal cycle.  Core fields (temp, salt, u) fail;
       internal pressure gradient implementation under investigation.
   * - ``medsea_west``
     - FAIL (8/30)
     - Western Mediterranean.  Similar failure pattern to ``blacksea``.
   * - ``medsea_east``
     - UNSUPPORTED
     - Eastern Mediterranean — requires features not yet compiled.
   * - ``entrainment``
     - FAIL (19/30)
     - Convective entrainment.  Turbulence fields (num, tke, eps) fail.
   * - ``flex``
     - FAIL (7/30)
     - FLEX experiment.  Core fields fail.
   * - ``gotland``
     - FAIL (8/30)
     - Baltic Sea (Gotland Deep).  Core fields fail.
   * - ``lago_maggiore``
     - FAIL (8/30)
     - Alpine lake.  Core fields fail.
   * - ``asics_med``
     - UNSUPPORTED
     - Mediterranean deep convection.
   * - ``estuary``
     - UNSUPPORTED
     - Estuarine circulation.
   * - ``langmuir``
     - UNSUPPORTED
     - Langmuir turbulence with Stokes drift.
   * - ``liverpool_bay``
     - UNSUPPORTED
     - Tidal mixing in Liverpool Bay.
   * - ``nns_annual``
     - UNSUPPORTED
     - North Sea annual cycle.
   * - ``nns_seasonal``
     - UNSUPPORTED
     - North Sea seasonal cycle.
   * - ``ows_papa``
     - UNSUPPORTED
     - Ocean Weather Station Papa.
   * - ``plume``
     - UNSUPPORTED
     - Freshwater plume.
   * - ``resolute``
     - UNSUPPORTED
     - Arctic mixing.
   * - ``reynolds``
     - UNSUPPORTED
     - Reynolds number scaling.
   * - ``rouse``
     - UNSUPPORTED
     - Rouse sediment profile.
   * - ``seagrass``
     - UNSUPPORTED
     - Seagrass canopy dynamics.
   * - ``wave_breaking``
     - UNSUPPORTED
     - Wave-breaking enhanced mixing.

.. note::
   Validation status is updated with each development milestone.  The
   results above correspond to the ``validation/results.json`` generated on
   2026-04-30.

Tolerance Parameters
--------------------

The tolerance parameters used in the validation criterion are:

* Relative tolerance (field range): :math:`10^{-7}`
* Absolute tolerance floor: :math:`10^{-12}`
* Relative tolerance (:math:`|b|`): :math:`10^{-6}`

These match the criterion defined in :doc:`overview`.
