Driver
======

The :class:`~pygotm.driver.GotmDriver` class is the primary entry point for
running pyGOTM simulations.  It loads a GOTM-compatible YAML configuration,
constructs the runtime containers, runs the compiled timestep loop, and
returns the output as an :class:`xarray.Dataset`.

.. automodule:: pygotm.driver
   :members:
   :undoc-members:
   :show-inheritance:
