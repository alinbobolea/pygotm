Validation Overview
===================

pyGOTM validates compiled-runtime output against the official GOTM 6.0.7
Fortran NetCDF reference files with a Frechet-distance comparison pipeline.
The validation suite compares every numeric reference variable, records
structural mismatches as failed variables, and writes both machine-readable
JSON and an HTML report.

The current parity gate is the Frechet suite in
``src/pygotm/validation/run_validation.py`` and
``src/pygotm/validation/compare.py``.  The older pointwise tolerance metrics
(``P99``, Birge ratio, normalized signed bias, ``rtol``/``atol`` pass bands,
RMSE, NRMSE, and mean/max absolute or relative errors) are not status-driving
validation indicators for the current suite.

Reference Data Distribution
---------------------------

Top-level ``validation/`` data is intentionally excluded from normal Git
history. Generated reports and ``validation/runs`` outputs are reproducible, and
the official Fortran reference NetCDF files are large enough that they should be
distributed as release/data artifacts rather than committed to the source repo.

Download the reference-data release asset and unpack it so the working tree has
``validation/reference/<case>/`` directories before running validation. The
planned public asset path is:

.. code-block:: text

   https://github.com/<org>/pygotm/releases/download/reference-data-v0.1.0/pygotm-validation-reference.tar.zst

Validation Pipeline
-------------------

Each validation run follows the same implementation path:

1. Case metadata is resolved from ``validation/reference``.
2. The case YAML is selected explicitly, or by trying ``gotm.yaml``,
   ``gotm.yml``, then a single non-``fabm.yaml``/``output.yaml`` YAML file.
3. The reference NetCDF is selected as the largest ``*.nc`` file in the case
   directory, excluding ``restart.nc``.
4. ``GotmDriver(case.yaml_path).run(output_path=...)`` produces the pyGOTM
   NetCDF output under ``validation/runs/<run_name>/<run_name>.nc``.
5. The output dataset must report ``runtime == "compiled"``.  Validation does
   not silently fall back to legacy Python timestep logic.
6. ``compare_nc(py_path, ref_path, case_name)`` opens both NetCDF files and
   compares all numeric reference variables.
7. Results are written to ``validation/results.json`` and rendered to
   ``validation/report.html``.

The comparison is reference-driven.  A numeric variable present in the
reference but missing from pyGOTM is ``BROKEN``.  Extra numeric variables in the
pyGOTM output are also reported as ``BROKEN`` so the report captures output
schema drift in either direction.

Alignment Before Comparison
---------------------------

The Frechet calculation is applied after structural and time alignment:

* Numeric variables with no time dimension must have identical shapes.
* If one side has a time dimension and the other does not, the variable is
  structurally ``BROKEN``.
* Recognized time dimension names are ``time`` and ``t``.
* Non-time dimensions must match exactly.
* Time coordinates come from the dataset coordinate when present, otherwise
  from ``np.arange(size)``.
* ``datetime64`` and ``timedelta64`` coordinates are converted to nanosecond
  integer coordinates before interpolation.
* Both arrays are interpolated onto the union of finite reference and pyGOTM
  time coordinates with ``np.interp(..., left=np.nan, right=np.nan)``.
* Paired non-finite samples are dropped before Frechet distances are computed.

After alignment, the time axis is moved first and the aligned values are
flattened.  The Frechet curve points are value-only points of shape
``(-1, 1)``; time itself is not a coordinate in the Frechet point.  For large
variables, the paired aligned samples are evenly downsampled to at most
``frechet_k`` points using ``np.linspace(0, n - 1, frechet_k)``.  The default
``frechet_k`` is ``400``.

Frechet Theory
--------------

For aligned reference values :math:`p_0, \ldots, p_{n-1}` and calculated values
:math:`q_0, \ldots, q_{m-1}`, pyGOTM computes the discrete Frechet distance.
The dynamic-programming table :math:`C` is initialized with Euclidean distance
between value-only points:

.. math::

   C_{0,0} = \lVert p_0 - q_0 \rVert

The first row and column accumulate the largest distance needed to traverse
one curve while staying on the start of the other:

.. math::

   C_{i,0} = \max(C_{i-1,0}, \lVert p_i - q_0 \rVert)

.. math::

   C_{0,j} = \max(C_{0,j-1}, \lVert p_0 - q_j \rVert)

Interior cells use the standard discrete Frechet recurrence:

.. math::

   C_{i,j} =
   \max\left(
     \min(C_{i-1,j}, C_{i-1,j-1}, C_{i,j-1}),
     \lVert p_i - q_j \rVert
   \right)

The raw Frechet distance is:

.. math::

   d_\mathrm{raw} = C_{n-1,m-1}

This distance answers the "leash length" question for two ordered value
sequences: how far apart the two trajectories must be, at worst, when both are
walked forward without reordering samples.  That makes it stricter than a mean
error while remaining less sensitive to a single isolated point than a pure
maximum pointwise error.

The implementation uses the Numba-accelerated iterative version when available
and falls back to the pure NumPy/Python iterative implementation if needed.

Normalization Theory
--------------------

``d_raw`` is reported in the native units of each variable.  To classify many
variables on a common scale, the suite also computes a normalized Frechet
distance:

.. math::

   d_\mathrm{norm} = F(N(\mathrm{ref}), N(\mathrm{calc}))

where :math:`F` is the discrete Frechet operator and :math:`N` is a dynamic
range normalization based on absolute magnitudes from both arrays.

The normalization uses robust range estimates by default:

.. math::

   l = P_1(|x|), \qquad h = P_{99}(|x|)

where :math:`x` is the combined finite set of reference and calculated values.
If the robust range is invalid, the implementation falls back to the finite
minimum and maximum.  If the high estimate is non-positive, both arrays
normalize to zero and the normalization mode is ``degenerate``.

For variables spanning fewer than ``switch_oom`` orders of magnitude, linear
normalization is used:

.. math::

   N(x) = \operatorname{clip}
   \left(
     \frac{|x| - l}{h - l},
     0,
     1
   \right)

For variables spanning at least ``switch_oom`` orders of magnitude, log
normalization is used:

.. math::

   \epsilon_\mathrm{dyn} =
   \max(\epsilon_\mathrm{floor}, 0.1 \cdot \min(|x|_{>0}))

.. math::

   N(x) = \operatorname{clip}
   \left(
     \frac{\log_{10}(|x| + \epsilon_\mathrm{dyn}) - l_\log}
          {h_\log - l_\log},
     0,
     1
   \right)

The default switch threshold is ``2.0`` orders of magnitude, so log
normalization is used for fields whose meaningful magnitudes span at least two
decades.

Frechet Indicators
------------------

Each variable result reports these current indicators:

``d_raw``
   Discrete Frechet distance on aligned original values.  If ``d_raw`` is below
   ``frechet_abs_tol`` (default ``1e-12``), both ``d_raw`` and ``d_norm`` are
   reported as zero.

``d_norm``
   Discrete Frechet distance after dynamic linear/log normalization.  If
   ``d_raw`` is below ``frechet_rel_tol * signal_scale`` (default relative
   tolerance ``1e-6``), ``d_norm`` is reported as zero.

``d_rel``
   Relative raw Frechet score used only for below-floor signals:
   ``d_raw / signal_scale``.  Here ``signal_scale`` is the maximum finite
   absolute magnitude across the aligned reference and calculated arrays, with
   an ``eps_floor`` lower bound.

``score``
   The status-driving value stored in report rows.  For normal signals this is
   ``d_norm``.  For variables whose ``signal_scale`` is positive but below the
   variable magnitude floor, this is ``d_rel``.  The ``metric_mode`` field is
   ``"d_norm"`` or ``"d_rel"`` accordingly.

The report displays full-precision reference and calculated values at the
aligned sample with the largest absolute finite difference.  Structural
failures use infinite distances internally; JSON sanitization writes those
non-finite values as ``null``.

Status Bands
------------

Variable status is assigned from ``score``:

.. list-table::
   :header-rows: 1
   :widths: 15 10 25 50

   * - Status
     - Color
     - Score range
     - Meaning
   * - PASS
     - green
     - :math:`score < 0.01`
     - The variable is within the current Frechet parity threshold.
   * - MARGINAL
     - yellow
     - :math:`0.01 \leq score < 0.05`
     - The variable is close to parity but outside the pass threshold.
   * - DISCREPANT
     - orange
     - :math:`0.05 \leq score < 0.20`
     - The variable has a deterministic implementation difference.
   * - BROKEN
     - red
     - :math:`score \geq 0.20` or structural failure
     - The variable is severely mismatched, missing, extra, non-finite, or
       structurally incompatible.

A case has status ``PASS`` only when all compared variables pass.  It has
status ``FAIL`` when one or more variables are ``MARGINAL``, ``DISCREPANT``, or
``BROKEN``.  Runtime or validation exceptions are reported as ``ERROR``.

Tolerance Configuration
-----------------------

Frechet thresholds and normalization settings are defined by ``FrechetConfig``
in ``src/pygotm/validation/tolerances.py``.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Setting
     - Default
     - Purpose
   * - ``pass_tol``
     - ``0.01``
     - Upper score bound for ``PASS``.
   * - ``marginal_tol``
     - ``0.05``
     - Upper score bound for ``MARGINAL``.
   * - ``discrepant_tol``
     - ``0.20``
     - Upper score bound for ``DISCREPANT``.
   * - ``frechet_abs_tol``
     - ``1e-12``
     - Absolute raw-distance cutoff reported as exact zero.
   * - ``frechet_rel_tol``
     - ``1e-6``
     - Relative raw-distance cutoff for zero normalized distance.
   * - ``frechet_k``
     - ``400``
     - Maximum paired samples used for each Frechet calculation.
   * - ``robust``
     - ``True``
     - Use percentile-based range estimates.
   * - ``q_low`` / ``q_high``
     - ``1.0`` / ``99.0``
     - Robust normalization percentiles.
   * - ``switch_oom``
     - ``2.0``
     - Order-of-magnitude span that switches from linear to log
       normalization.
   * - ``eps_floor``
     - ``1e-12``
     - Lower bound used in signal and log normalization.
   * - ``default_magnitude_floor``
     - ``1e-6``
     - Default tiny-signal floor for ``d_rel`` score mode.

Variable-specific magnitude floors are used only to choose ``d_rel`` instead
of ``d_norm`` for tiny signals:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Floor
     - Variables
   * - ``1e-2``
     - ``temp``, ``salt``, ``rho``, ``I_0``
   * - ``1e-3``
     - ``h``
   * - ``1e-4``
     - ``u``, ``v``, ``bioshade``, ``NN``, ``SS``, ``L``
   * - ``1e-5``
     - ``buoy``
   * - ``1e-7``
     - ``num``, ``nuh``, ``nus``, ``nucl``, ``avh``
   * - ``1e-8``
     - ``ga``, ``tke``, ``kb``, ``PSTK``, ``idpdy``, ``idpdx``, ``w``
   * - ``1e-12``
     - ``eps``, ``P``, ``G``, ``Pb``, ``epsb``, ``xP``
   * - ``1e-6``
     - ``an``, ``cmue1``, ``cmue2``, ``as``, ``at``, ``fric``, ``drag``,
       ``taub`` and unlisted variables

Report Structure
----------------

For each case the report contains separate sections:

* **PyGOTM variables** - core GOTM physics outputs, including temperature,
  salinity, velocity, turbulence quantities, and related diagnostics.
* **PyFABM variables** - biogeochemical outputs and any other numeric
  variables not listed in ``PYGOTM_VARIABLES``.

This split shows whether a non-conformance originates in the physics solver or
in biogeochemical coupling.

Per-variable tables use these columns:

* Status
* Variable
* Reference (full precision)
* Calculated (full precision)
* Raw Frechet
* Score (Normalized Frechet / d_rel)
* Parameter plot

Comparison plots are embedded only for ``MARGINAL`` and ``DISCREPANT``
variables.  ``PASS`` variables do not receive plots.  ``BROKEN`` variables are
listed without plots because structural mismatches should be debugged before
time-series visualization.

Running the Validation Suite
----------------------------

All project Python commands must run inside the ``pygotm`` conda environment
with ``conda run -n pygotm``.

.. code-block:: bash

   # Run the default Frechet cases: couette, channel, entrainment
   conda run -n pygotm python -m pygotm.validation.run_validation

   # Run specific cases
   conda run -n pygotm python -m pygotm.validation.run_validation \
       --cases couette,channel,entrainment

   # Run all reference cases
   conda run -n pygotm python -m pygotm.validation.run_validation --all

   # Run a named group
   conda run -n pygotm python -m pygotm.validation.run_validation \
       --group non-stim

   # Re-render report from saved results without re-running cases
   conda run -n pygotm python -m pygotm.validation.render_report \
       validation/results.json --output validation/report.html

Useful suite options include ``--exclude``, ``--device``, ``--workers``,
``--dashboard-port``, ``--output-dir``, ``--no-run``, ``--no-warmup``, and
``--debug-turbulence``.

Output files:

* ``validation/results.json`` - machine-readable per-case and per-variable
  results.
* ``validation/report.html`` - human-readable HTML report with embedded Plotly
  plots for marginal and discrepant variables.
* ``validation/runs/<run_name>/<run_name>.nc`` - generated pyGOTM NetCDF files.

Benchmark Mode
--------------

Benchmarking is separate from Frechet parity validation but can optionally run
validation after timed case execution.

.. code-block:: bash

   conda run -n pygotm pygotm benchmark --cases couette,channel --no-validate

Benchmark options include ``--cases``, ``--max-steps``, ``--output-dir``,
``--no-output``, ``--no-warmup``, and ``--no-validate``.

Validation API
--------------

.. code-block:: python

   from pathlib import Path

   from pygotm.validation.runner import validate_case

   result = validate_case("couette", Path("validation/runs"), skip_run=True)

See :doc:`../api/validation` for the full API reference.

Test Cases
----------

See :doc:`test_cases` for the current generated case status summary and the
case-level aggregation rules.
