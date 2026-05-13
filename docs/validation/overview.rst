Validation Overview
===================

pyGOTM validates its output against the official GOTM 6.0.7 Fortran NetCDF
reference files using a **three-indicator strict validation system**.

Validation Indicators
---------------------

Three indicators are computed per variable.  Status is determined exclusively
by the **primary tolerance-normalized score**.

Pointwise Normalized Error
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   E_i = \frac{|\text{calc}_i - \text{ref}_i|}
              {a_\text{tol} + r_\text{tol} \cdot \max(|\text{ref}_i|,\, s_\text{floor})}

where :math:`a_\text{tol}`, :math:`r_\text{tol}`, and :math:`s_\text{floor}`
are variable-specific tolerance parameters defined in
``src/pygotm/validation/tolerances.py``.

Primary Score (p99)
~~~~~~~~~~~~~~~~~~~

.. math::

   P_{99} = \text{percentile}(E_i,\; 99)

The 99th percentile of the pointwise normalized error.  **This is the sole
basis for status classification.**  Using p99 instead of max avoids false
alarms from single outliers while still catching systematic problems.

Birge Ratio
~~~~~~~~~~~

.. math::

   B = \sqrt{\text{mean}(E_i^2)}

Root-mean-square of the normalized error.  Reported as a secondary diagnostic
only.  Not used for status classification.

Normalized Signed Bias
~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{NSB} = \frac{\overline{\text{calc} - \text{ref}}}
                     {a_\text{tol} + r_\text{tol} \cdot \max(\overline{|\text{ref}|},\; s_\text{floor})}

Detects systematic offset, persistent drift, wrong sign conventions, or
biased formulas.  Reported as a secondary diagnostic only.

Status Bands
------------

Status is assigned from the primary score only:

.. list-table::
   :header-rows: 1
   :widths: 15 10 20 55

   * - Status
     - Color
     - Primary score range
     - Meaning
   * - PASS
     - green
     - :math:`P_{99} \leq 1`
     - Within tolerance.
   * - MARGINAL
     - yellow
     - :math:`1 < P_{99} \leq 3`
     - Slightly outside tolerance.  Diagnostic plots are generated.
   * - DISCREPANT
     - orange
     - :math:`3 < P_{99} \leq 10`
     - Deterministic implementation difference.  Diagnostic plots generated.
   * - BROKEN
     - red
     - :math:`P_{99} > 10`
     - Severe mismatch.  Debug before inspecting plots.

Report Structure
----------------

For each case the report contains two separate sections:

1. **PyGOTM variables** — core GOTM physics outputs (temperature, salinity,
   velocity, turbulence quantities, etc.)
2. **PyFABM variables** — biogeochemical model outputs (species-specific, model-dependent)

This separation allows immediate identification of whether a non-conformance
originates in the physics solver or the biogeochemical model.

Comparison plots (Plotly time-series of reference vs. calculated) are embedded
only for **MARGINAL** and **DISCREPANT** variables.  PASS variables do not
receive plots.  BROKEN variables are listed without plots as a signal to start
with deeper debugging before visualization.

Tolerance Configuration
-----------------------

Per-variable tolerance parameters are defined in
``src/pygotm/validation/tolerances.py``::

   VariableTolerance(atol=..., rtol=..., scale_floor=..., section="pygotm")

- **atol** — absolute tolerance floor (prevents failures from floating-point
  noise in near-zero fields)
- **rtol** — relative tolerance (scales with the reference field magnitude)
- **scale_floor** — physical scale floor (applied when the reference field is
  near zero, preventing ``atol`` from being the only guard)
- **section** — ``"pygotm"`` for GOTM physics variables, ``"pyfabm"`` for
  FABM biogeochemical variables

Variables not in the registry are treated as FABM variables and use the
project-approved ``DEFAULT_PYFABM_TOLERANCE``.

Running the Validation Suite
-----------------------------

.. code-block:: bash

   # Activate the project conda environment first
   conda activate pygotm

   # Run specific cases
   python -m pygotm.validation.run_validation \
       --cases couette,channel,entrainment

   # Run all supported cases
   python -m pygotm.validation.run_validation --all

   # Re-render report from saved results without re-running
   python -m pygotm.validation.render_report validation/results.json

Output files:

* ``validation/results.json`` — machine-readable per-variable results
* ``validation/report.html`` — human-readable HTML report with embedded Plotly plots

Interpreting the Report
-----------------------

Open ``validation/report.html`` in a browser.  Each case expands to show:

- A **PyGOTM variables** table with columns: Status, Variable, Reference (full
  precision), Calculated (full precision), Primary score, Birge ratio,
  Normalized signed bias, Parameter plot.
- A **PyFABM variables** table with the same columns.
- Embedded Plotly comparison plots for MARGINAL and DISCREPANT variables.

Investigate **BROKEN** variables first — they typically indicate missing
physics, wrong units, or structural output mismatches.  **DISCREPANT** variables
indicate deterministic differences in the algorithm.  **MARGINAL** variables
may reflect float32 forcing data sensitivity.

Benchmark Mode
--------------

.. code-block:: bash

   pygotm benchmark --cases couette,channel --no-validate

Validation API
--------------

.. code-block:: python

   from pygotm.validation.runner import run_case, validate_case

   result = validate_case("couette", Path("validation/runs"), skip_run=True)

See :doc:`../api/validation` for the full API reference.

Test Cases
----------

See :doc:`test_cases` for the per-case validation status.
