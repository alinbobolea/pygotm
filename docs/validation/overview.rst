Validation Overview
===================

pyGOTM validates its output against the official GOTM 6.0.7 Fortran NetCDF
reference files.  The validation suite is the primary guarantee of scientific
parity with the Fortran reference implementation.

Validation Criterion
--------------------

Each scalar output field :math:`a` (from pyGOTM) is compared against the
Fortran reference :math:`b` using a range-aware combined tolerance:

.. math::

   |a - b| \leq \max\!\bigl(10^{-7} \times r_\mathrm{ref},\; 10^{-12}\bigr)
            + 10^{-6} \times |b|

where :math:`r_\mathrm{ref} = \max(b) - \min(b)` is the range of the
reference field across all time steps and depth levels.

This criterion adapts to the magnitude of the field: fields that vary by
many orders of magnitude are compared with a relative tolerance, while
fields near machine epsilon are compared with an absolute tolerance.  The
formula is intentionally tighter than standard relative tolerance tests to
detect systematic bias.

Running the Validation Suite
-----------------------------

.. code-block:: bash

   # Run specific cases
   uv run python -m pygotm.validation.run_validation \
       --cases couette,channel,entrainment

   # Run all supported cases
   uv run python -m pygotm.validation.run_validation

Output files:

* ``validation/results.json`` — machine-readable per-variable results
* ``validation/report.html`` — human-readable HTML report

Benchmark Mode
--------------

To measure wall-clock performance without writing validation artifacts:

.. code-block:: bash

   pygotm benchmark --cases couette,channel --no-validate

The first run pays Numba JIT compilation cost.  Subsequent runs use the
cached compiled code.  Benchmark results are printed to the terminal by default.
Use ``--output-dir`` only when an explicit aggregate JSON timing artifact is
needed.

Validation API
--------------

.. code-block:: python

   from pygotm.validation.runner import run_case
   from pygotm.validation.compare import compare_netcdf

   result = run_case("couette")
   comparison = compare_netcdf(result.py_nc_path, result.ref_nc_path)

See :doc:`../api/validation` for the full API reference.

Test Cases
----------

See :doc:`test_cases` for the per-case validation status.
