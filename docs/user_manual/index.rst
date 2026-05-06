User Manual
===========

This section is the complete reference for every key in the ``gotm.yaml``
configuration file understood by pyGOTM.  Each section below corresponds to
a top-level YAML block and documents every parameter: its type, units, valid
values, default value, and what physics it controls.

Every parameter listed here is implemented in the pyGOTM source.
Parameters that are currently supported only by the legacy Python time-step
loop (and not yet by the compiled Numba runtime) are marked with
**[legacy only]**.  Parameters not yet implemented at all are
marked **[not implemented]**.

.. note::

   pyGOTM reads GOTM-compatible ``gotm.yaml`` files (format version 7).
   Any official GOTM 6.0.7 test-case YAML can be loaded directly.  The 22
   reference cases live under ``gotm-model/cases-runs/``.

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
