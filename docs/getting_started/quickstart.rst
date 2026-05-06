Quickstart
==========

Run a Single-Column Simulation
-------------------------------

The simplest way to run pyGOTM is with a GOTM-compatible YAML configuration
file. The official GOTM test cases are included in the repository:

.. code-block:: python

   from pygotm.driver import GotmDriver

   driver = GotmDriver("gotm-model/cases-runs/couette/gotm.yaml")
   ds = driver.run()

   print(ds)          # xarray Dataset with all model output
   ds.to_netcdf("couette_out.nc")

The returned ``ds`` is a CF-conventions NetCDF-compatible
:class:`xarray.Dataset` with time, depth, and all model fields.

Run from the Command Line
--------------------------

.. code-block:: bash

   # Run the built-in validation suite for a specific case
   pygotm validate --case couette

   # Benchmark the compiled runtime (records timing to JSON)
   pygotm benchmark --cases couette,channel

Suppress Output (no-output integration)
-----------------------------------------

For benchmarking or performance measurement, suppress NetCDF output:

.. code-block:: python

   from pygotm.driver import GotmDriver

   driver = GotmDriver("gotm-model/cases-runs/couette/gotm.yaml")
   ds = driver.run(output=False)   # empty Dataset — compiled loop still runs

Inspect Output
--------------

.. code-block:: python

   import xarray as xr
   import matplotlib.pyplot as plt

   ds = xr.open_dataset("couette_out.nc")
   ds["u"].isel(time=-1).plot()
   plt.show()

Run the Validation Suite
-------------------------

Compare pyGOTM output against the official Fortran GOTM reference cases:

.. code-block:: bash

   uv run python -m pygotm.validation.run_validation \
       --cases couette,channel,entrainment

Output: ``validation/report.html`` and ``validation/results.json``

Python API Summary
------------------

.. code-block:: python

   from pygotm.driver import GotmDriver

   # Load config and run
   driver = GotmDriver("path/to/gotm.yaml")
   ds = driver.run()                  # returns xarray.Dataset

   # Inspect available fields
   print(list(ds.data_vars))

   # Access a specific field
   tke = ds["tke"]                    # turbulent kinetic energy [m²/s²]
