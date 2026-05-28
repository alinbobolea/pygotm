Quickstart
==========

Setup
-----

Create the conda environment and register this checkout as the importable
``pygotm`` package:

.. code-block:: bash

   conda env create -f pygotm-conda-env.yml
   conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .

Before running official validation cases, provide a local reference-data tree
so ``validation/reference/couette/gotm.yaml`` exists. Top-level
``validation/reference/`` data is intentionally not tracked in normal Git
history. The source repository does track ``validation/report/`` as the current
documentation report snapshot, but it does not vendor the full reference-data
tree.

Run a Single-Column Simulation
-------------------------------

The simplest way to run pyGOTM is with a GOTM-compatible YAML configuration
file. The official GOTM test cases are available in the external validation
reference-data tree when it is present locally:

.. code-block:: python

   from pygotm.driver import GotmDriver

   driver = GotmDriver("validation/reference/couette/gotm.yaml")
   ds = driver.run()

   print(ds)          # xarray Dataset with all model output
   ds.to_netcdf("couette_out.nc")

The returned ``ds`` is a CF-conventions NetCDF-compatible
:class:`xarray.Dataset` with time, depth, and all model fields.

Run from the Command Line
--------------------------

.. code-block:: bash

   # Run one YAML configuration and write NetCDF output
   conda run -n pygotm pygotm run validation/reference/couette/gotm.yaml \
       --output couette_out.nc

   # Run the Frechet validation suite for specific cases
   conda run -n pygotm pygotm validate --cases couette,channel

Suppress Output (no-output integration)
-----------------------------------------

For benchmarking or performance measurement, suppress NetCDF output:

.. code-block:: python

   from pygotm.driver import GotmDriver

   driver = GotmDriver("validation/reference/couette/gotm.yaml")
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

   conda run -n pygotm python -m pygotm.validation.run_validation \
       --cases couette,channel,entrainment

Output: ``validation/report/report.html``, ``validation/report/report.json``,
``validation/report/results.json``, and one per-case HTML report under
``validation/report/``. Generated NetCDF outputs remain under
``validation/runs/``.

The default validation set is ``couette,channel,entrainment`` and should exit
successfully.  Running all 22 reference cases currently completes every case
but exits nonzero because the generated full-suite snapshot is still
``PARTIAL PARITY``: 15 cases pass and 7 cases fail.  See
:doc:`../validation/test_cases` for the current case summary.

See :doc:`interfaces` for the full user and developer command surfaces.

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
