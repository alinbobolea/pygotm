FABM Coupling
=============

Framework for Aquatic Biogeochemical Models (FABM) coupling interface.
Provides state containers, input-variable adapters, and the chunked time loop
for linking GOTM with FABM-compatible biogeochemical models.

See :doc:`/physics/biogeochemistry` for the full scientific description of the
coupling architecture.

Configuration
-------------

.. automodule:: pygotm.fabm.config
   :members:
   :undoc-members:
   :show-inheritance:

Engine
------

``FABMEngine.output_variable_specs()`` exposes the model-discovered NetCDF
surface: interior state and output-enabled diagnostics are reported as
z-profiles, while surface, bottom, and horizontal variables are reported as
scalars.  Each spec carries the normalized output name, units, and long name
from pyfabm.

.. automodule:: pygotm.fabm.engine
   :members:
   :undoc-members:
   :show-inheritance:

Coupling
--------

.. automodule:: pygotm.fabm.coupling
   :members:
   :undoc-members:
   :show-inheritance:

FABM Loop
---------

.. automodule:: pygotm.fabm.fabm_loop
   :members:
   :undoc-members:
   :show-inheritance:

State Containers
----------------

.. automodule:: pygotm.fabm.gotm_fabm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pygotm.fabm.state
   :members:
   :undoc-members:
   :show-inheritance:

Input Variables
---------------

.. automodule:: pygotm.fabm.gotm_fabm_input
   :members:
   :undoc-members:
   :show-inheritance:
