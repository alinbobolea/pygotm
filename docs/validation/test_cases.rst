Test Cases
==========

.. include:: _generated/test_cases_summary.inc

Indicator Summary
-----------------

The current validation suite uses Frechet-distance indicators from
``src/pygotm/validation``:

``d_raw``
   Discrete Frechet distance on aligned original values.

``d_norm``
   Discrete Frechet distance after section-aware dynamic linear/log
   normalization. Core PyGOTM variables use full finite ranges by default;
   non-PyGOTM variables use a wide robust range.

``d_rel``
   ``d_raw / signal_scale`` for variables whose signal magnitude is below the
   configured variable floor.

``score``
   The status-driving value.  This is normally ``d_norm`` and switches to
   ``d_rel`` for below-floor signals.  The selected indicator is recorded in
   ``metric_mode``.

``peak_d_norm``
   Non-classifying diagnostic ``d_norm`` computed with full-range
   normalization and ``frechet_k = 400`` to retain a peak-sensitive debugging
   signal.

Variable status bands are:

* ``PASS`` - ``score < 0.01``
* ``MARGINAL`` - ``0.01 <= score < 0.05``
* ``DISCREPANT`` - ``0.05 <= score < 0.20``
* ``BROKEN`` - ``score >= 0.20`` or a structural comparison failure

Variables listed in ``PYGOTM_VARIABLES`` are reported in the PyGOTM section.
Other numeric variables are treated as FABM biogeochemical variables.

.. _fortran-parity-deviations:

Fortran Parity Deviations
--------------------------

Some pyGOTM behaviours intentionally preserve quirks in the GOTM 6.0.7
reference path. These are validation-contract choices for the current reference
NetCDF set; changing them changes the validation target and requires new
reference outputs.

seagrass: ``init_seagrass`` activation bug
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affected case:** ``seagrass``

**Source file:** ``src/pygotm/extras/seagrass/seagrass.py`` —
:func:`~pygotm.extras.seagrass.seagrass.init_seagrass`

Fortran ``seagrass.F90`` declares module variable ``method`` and a separate
local variable ``i`` inside ``init_seagrass``. The YAML value is read into
``method``, but the activation check uses local ``i``. There is no assignment to
``i`` in the subroutine::

    call branch%get(method, 'method', ..., default=0)
    ...
    if (i .ne. 0) seagrass_calc = .true.

pyGOTM mirrors the current reference behaviour by storing ``method`` but leaving
``state.seagrass_calc`` at its default ``False`` value in
:func:`~pygotm.extras.seagrass.seagrass.init_seagrass`. Runtime construction
then emits ``seagrass_active = 0`` and the timestep loop does not run the
seagrass drag path. In the current validation report, ``seagrass`` is ``PASS``
with 104 passing variables and no ``MARGINAL``, ``DISCREPANT``, or ``BROKEN``
variables.

first_order turbulence: step-0 ``cmue1``/``cmue2`` initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affected reference cases:** current ``turb_method = first_order`` cases:
``seagrass`` and ``wave_breaking``.

**Source files:** ``src/pygotm/gotm/time_loop.py`` —
:func:`~pygotm.gotm.time_loop.time_loop_compiled`; and
``src/pygotm/turbulence/compute_cpsi3.py``.

Fortran ``turbulence.F90`` allocates ``cmue1`` and ``cmue2`` as zero-filled
arrays. During model-parameter setup, ``compute_cpsi3`` can write
stability-function probe values before the first output. In pyGOTM this
initialisation side effect lives in ``compute_cpsi3.py``: the ``Constant``
first-order stability path fills the full arrays, while the other first-order
stability paths update only the probe level. ``time_loop_compiled`` therefore
does not run the regular first-order stability-function update before the
step-0 output.

first_order turbulence: ``kb`` forwarded to ``alpha_mnb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Affected reference cases:** current ``turb_method = first_order`` cases:
``seagrass`` and ``wave_breaking``.

**Source file:** ``src/pygotm/gotm/time_loop.py`` —
:func:`~pygotm.gotm.time_loop.step_turbulence_first_order_single`

Fortran ``alpha_mnb.F90`` computes ``at`` from ``tke``, ``eps``, and ``kb``:
``at(i) = tke(i) / eps(i) * kb(i) / eps(i)``. pyGOTM implements the same
calculation in ``src/pygotm/turbulence/alpha_mnb.py``. In the first-order
compiled path, ``kb`` is initialised to ``kb_min`` and is not advanced by
``step_turbulence_first_order_single``, but it is still passed to
``step_alpha_mnb_single`` so ``at`` is computed from the real ``kb`` array, not
from a placeholder.


.. toctree::
   :hidden:
   :glob:

   cases/*
