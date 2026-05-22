Validation Overview
===================

pyGOTM validates compiled-runtime output against the official GOTM 6.0.7
Fortran NetCDF reference files with a Frechet-distance comparison pipeline.
The validation suite compares every numeric reference variable, records
structural mismatches as failed variables, and writes an HTML index plus one
HTML report page per case.

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
7. Each case report is written directly from the reference and pyGOTM NetCDF
   files. The final index is written to ``validation/report.html``.

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
``frechet_k`` is ``200``. This value intentionally balances shape sensitivity
against localized peak-timing sensitivity in chaotic turbulence regimes.

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

Normalization is section-aware. Core PyGOTM variables use the full finite
minimum and maximum magnitude range by default. This preserves active
turbulence tails for fields such as ``tke``, ``eps``, ``kb``, ``G`` and
``Pb``; clipping those variables with ordinary 1st/99th robust percentiles can
make floor-dominated profiles look worse by collapsing the useful dynamic
range.

Non-PyGOTM variables, primarily PyFABM outputs, use a wide robust range by
default:

.. math::

   l = P_{0.1}(|x|), \qquad h = P_{99.9}(|x|)

where :math:`x` is the combined finite set of reference and calculated values.
The wide robust range suppresses isolated biogeochemical outliers while keeping
nearly all physically meaningful active tails. If the selected range is
invalid, the implementation falls back to the finite minimum and maximum. If
the high estimate is non-positive, both arrays normalize to zero and the
normalization mode is ``degenerate``.

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
     \frac{\log_{10}(|x| + \epsilon_\mathrm{dyn}) - l_{\log}}
          {h_{\log} - l_{\log}},
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

``peak_d_norm``
   Non-classifying diagnostic normalized Frechet distance computed with the
   stricter peak-sensitive configuration: full-range normalization and
   ``frechet_k = 400``. It preserves a debugging signal for localized peak
   misalignment without letting that signal decide case status.

The report displays full-precision reference and calculated values at the
aligned sample with the largest absolute finite difference.  Structural
failures use infinite distances internally and are displayed without plots.

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
     - ``200``
     - Maximum paired samples used for the status-driving Frechet calculation.
   * - ``robust``
     - ``False``
     - Use full finite range estimates for PyGOTM variables.
   * - ``q_low`` / ``q_high``
     - ``0.1`` / ``99.9``
     - Robust percentiles used when robust PyGOTM normalization is explicitly
       enabled.
   * - ``pyfabm_robust``
     - ``True``
     - Use percentile-based range estimates for non-PyGOTM variables.
   * - ``pyfabm_q_low`` / ``pyfabm_q_high``
     - ``0.1`` / ``99.9``
     - Robust normalization percentiles for non-PyGOTM variables.
   * - ``peak_frechet_k``
     - ``400``
     - Sample count for the non-classifying peak-sensitive diagnostic score.
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
* Peak-sensitive d_norm
* Parameter plot

Comparison plots are embedded only for ``MARGINAL`` and ``DISCREPANT``
variables.  ``PASS`` variables do not receive plots.  ``BROKEN`` variables are
listed without plots because structural mismatches should be debugged before
time-series visualization.

Interpreting Remaining Differences
----------------------------------

The remaining non-PASS reference cases are dominated by two different classes
of difference: chaotic amplification in the physical turbulence trajectory and
localized active-tail differences in biogeochemical or diagnostic variables.

``gotland``
   The first 20 simulation days match the Fortran reference to near machine
   precision. A small turbulence difference appears around day 21, grows by
   roughly an order of magnitude by day 22, and reaches order-one decorrelation
   by about day 27. The inferred amplification rate is consistent with
   stratified turbulence near the mixed-layer base. Once the trajectory has
   decorrelated, localized peak timing in ``tke``, ``G``, ``NNT`` and related
   turbulence diagnostics can differ even when means and 1st/99th percentile
   distributions remain close.

``ows_papa`` and ``resolute``
   These cases show the same practical issue in a weaker form: broad
   distributions and budgets are close, but localized peaks in turbulence,
   mixed-layer-depth and integrated diagnostic variables are not exactly
   time-aligned after nonlinear amplification. The status-driving
   ``frechet_k = 200`` metric keeps these differences mostly marginal while
   the peak-sensitive diagnostic still exposes them for debugging.

``langmuir``
   Most core variables are close. Remaining differences are concentrated in a
   small number of Langmuir/Stokes or diagnostic outputs where localized
   structure differs more than the broad trajectory.

``medsea_east``, ``medsea_west`` and ``nns_annual``
   These are mostly PyFABM or biogeochemical active-tail differences, such as
   ``jrc_med_ergom_OFL`` and ``npzd_*`` variables. The wide robust PyFABM
   normalization is intended to avoid classifying isolated outliers as
   structural failure while still reporting persistent shape differences.

Known numerical sources that can seed these differences include compiler
reassociation and fused-multiply-add choices in gfortran versus Numba/LLVM,
Fortran unsuffixed literals stored through GOTM's precision model, Python
``float`` constants that are naturally double precision, and small differences
between the GSW C-library path and Fortran-library calculations used by the
reference executable. Air-sea bulk formula constants, including Kondo and
Fairall coefficients, are especially sensitive because small forcing
differences can seed turbulence divergence that later grows chaotically.

Two precision-matching iterations have already been tested in the turbulence
equations and Kondo bulk coefficients. They did not materially change the
remaining Frechet scores for ``gotland``, ``ows_papa`` or ``resolute``. That
indicates the current residuals are not isolated to one obvious constant or
single equation. Achieving bit-identical trajectories would likely require a
broader project to align Fortran literal precision, library calls and compiler
operation ordering across the model.

The validation policy therefore keeps strict structural checks and a
peak-sensitive diagnostic, but uses a status-driving metric that is less
dominated by post-decorrelation peak timing. This is intended to identify real
schema and trajectory errors without labeling physically plausible chaotic
decorrelation as a broken translation.

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

   # Re-render report pages directly from existing NetCDF outputs
   conda run -n pygotm python -m pygotm.validation.render_report \
       --all --output-dir validation

Useful suite options include ``--exclude``, ``--device``, ``--workers``,
``--dashboard-port``, ``--output-dir``, ``--no-run``, ``--no-warmup``, and
``--debug-turbulence``.

Output files:

* ``validation/report.html`` - human-readable HTML index with one frame per
  case.
* ``validation/<run_name>.html`` - per-case report generated directly from the
  reference and pyGOTM NetCDF files, with embedded Plotly plots for marginal
  and discrepant variables.
* ``validation/runs/<run_name>/<run_name>.nc`` - generated pyGOTM NetCDF files.

Generated Reports
-----------------

The documentation build copies the HTML reports currently present in
``validation/`` (produced by ``conda run -n pygotm python -m
pygotm.validation.run_validation``) into the built docs at
``validation/``. The index page links to each per-case report:

* `Validation report (all cases) <report.html>`_

If you are reading this page after a fresh clone and no validation run has
been executed yet, the link above will return a 404. Run the validation suite
locally and rebuild the documentation to populate the reports.

Developer Benchmarking
----------------------

Benchmarking is separate from Frechet parity validation. The developer-only
benchmark command measures execution timing and does not classify scientific
parity:

.. code-block:: bash

   conda run -n pygotm python -m pygotm.execution.benchmark \
       --cases couette,channel

Run ``pygotm validate`` or ``python -m pygotm.validation.run_validation`` as a
separate acceptance gate after any benchmarked code change. See
:doc:`../getting_started/interfaces` for every user and developer command
option.

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
