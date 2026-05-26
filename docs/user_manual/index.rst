User Manual
===========

This section documents the GOTM-compatible ``gotm.yaml`` keys recognized by
pyGOTM. Each section below corresponds to a top-level YAML block and records
the parameter type, units, valid values, default value, and the physics it
controls where that information is available.

Every parameter listed here is implemented in the pyGOTM source.
Parameters recognized by the source but not supported by the current compiled
Numba runtime are marked **[unsupported in compiled runtime]**. Unsupported
compiled-runtime configurations fail during setup; pyGOTM does not silently
fall back to a legacy Python timestep loop during parity runs.

.. note::

   pyGOTM reads GOTM-compatible ``gotm.yaml`` files (format version 7).
   Any official GOTM 6.0.7 test-case YAML can be loaded directly.  The 22
   reference cases live under ``validation/reference/``.

.. rubric:: Input source pattern

Many scalar and profile parameters follow the same ``InputSetting`` pattern:

.. code-block:: yaml

   some_field:
     method: constant         # constant | file
     constant_value: 0.0      # value when method=constant
     file: path/to/file.dat   # path when method=file
     column: 1                # column index in the file
     scale_factor: 1.0        # multiplier applied to file values
     offset: 0.0              # additive offset applied to file values

When ``method: constant`` the ``constant_value`` is used for the entire
simulation.  When ``method: file`` values are read from a column-separated
ASCII data file and interpolated to each model time step.

For a complete specification of every file format accepted by pyGOTM —
including exact column layouts, timestamp syntax, ordering requirements,
and annotated examples — see :ref:`input-file-formats`.

.. toctree::
   :maxdepth: 2
   :caption: Reference

   global
   location_time_grid
   initial_conditions
   surface_forcing
   light_extinction
   turbulence
   dynamics
   equation_of_state
   output
   input_file_formats
