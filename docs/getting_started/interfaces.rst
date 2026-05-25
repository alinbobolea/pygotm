Command and Developer Surfaces
==============================

pyGOTM keeps the user surface intentionally small. User commands run models or
run the official Frechet validation suite. Developer module commands support
report regeneration and execution optimization work.

User Surface
------------

Python API
~~~~~~~~~~

Use :class:`pygotm.driver.GotmDriver` for programmatic single-column runs:

.. code-block:: python

   from pygotm.driver import GotmDriver

   ds = GotmDriver("path/to/gotm.yaml").run()
   ds.to_netcdf("result.nc")

``GotmDriver(config).run(...)`` accepts:

``max_steps``
   Optional integer limit on integration steps. ``None`` runs the full
   configured simulation.

``output_path``
   Optional NetCDF path. When set, pyGOTM writes the returned dataset to this
   file.

``output``
   Boolean. ``True`` writes the normal dense output dataset. ``False`` runs the
   compiled loop and returns an empty dataset with coordinates only. This mode
   is mainly useful for internal performance checks.

Command Line
~~~~~~~~~~~~

``pygotm run`` runs one GOTM YAML file and writes one NetCDF output:

.. code-block:: bash

   conda run -n pygotm pygotm run path/to/gotm.yaml --output result.nc

Options:

``CONFIG_PATH``
   Required positional path to a GOTM-compatible YAML file.

``--output, -o PATH``
   Required NetCDF output path.

``--max-steps N``
   Optional integer limit on integration steps.

``pygotm validate`` runs the official Frechet parity suite and writes HTML
reports:

.. code-block:: bash

   conda run -n pygotm pygotm validate --cases couette,channel

The default case set is ``couette,channel,entrainment``.  The current default
set exits successfully.  ``--all`` completes all 22 cases but exits nonzero
until the known full-suite ``PARTIAL PARITY`` cases are resolved; see
:doc:`../validation/test_cases`.

Options:

``--cases NAMES``
   Comma-separated case names or case/input-yaml-base specs. Defaults to
   ``couette,channel,entrainment``. Use this option for a single case too,
   for example ``--cases couette``.

``--all``
   Run all 22 GOTM reference cases.  Current full-suite validation is expected
   to complete but return a failure exit status because 7 of 22 cases remain
   non-PASS.

``--group {all,default,non-stim}``
   Run a named case group.

``--exclude NAMES``
   Comma-separated case names to remove from the selected set.

``--device ARCH``
   Execution backend label. The current Numba validation backend is ``cpu``.

``--workers N``
   Dask worker count for multi-case validation. Defaults to detected CPU count.

``--dashboard-port PORT``
   Dask dashboard port. Default: ``8787``.

``--output-dir DIR``
   Directory for generated NetCDF runs and HTML reports. Default:
   ``validation``.

``--no-run``
   Skip simulation and compare existing NetCDF files under ``DIR/runs``.

``--no-warmup``
   Skip Numba warm-up before simulation runs.

``--debug-turbulence``
   Write per-time turbulence comparison dumps under each run directory.

Developer Surface
-----------------

Frechet validation module command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``python -m pygotm.validation.run_validation`` is the developer equivalent of
``pygotm validate``. It accepts the same options listed above and is the
canonical command used by quality gates and optimization acceptance checks:

.. code-block:: bash

   conda run -n pygotm python -m pygotm.validation.run_validation \
       --cases couette,channel,entrainment

Report regeneration
~~~~~~~~~~~~~~~~~~~

``python -m pygotm.validation.render_report`` rebuilds HTML reports from
existing NetCDF outputs without rerunning simulations:

.. code-block:: bash

   conda run -n pygotm python -m pygotm.validation.render_report --all

Options:

``--cases NAMES``
   Comma-separated cases or case/input-yaml-base specs. Defaults to
   ``couette,channel,entrainment``.

``--all``
   Render all reference cases.

``--group {all,default,non-stim}``
   Render a named case group.

``--exclude NAMES``
   Comma-separated case names to omit.

``--workers N``
   Dask worker count for multi-case report rendering.

``--dashboard-port PORT``
   Dask dashboard port. Default: ``8787``.

``--output-dir DIR``
   Directory containing ``runs/`` and receiving report HTML. Default:
   ``validation``.

Execution benchmark
~~~~~~~~~~~~~~~~~~~

``python -m pygotm.execution.benchmark`` is developer-only timing tooling for
optimization work. It does not validate scientific parity; run Frechet
validation separately for acceptance.

.. code-block:: bash

   conda run -n pygotm python -m pygotm.execution.benchmark \
       --cases couette,channel --output-dir validation/benchmark-runs/dev

Options:

``--cases NAMES``
   Comma-separated reference case names to benchmark. Default:
   ``couette,channel``.

``--max-steps N``
   Optional integer limit on integration steps.

``--output-dir DIR``
   Optional directory for one aggregate ``results.json`` benchmark artifact.

``--no-output``
   Disable dense output buffer conversion for hydro-only timing.

``--no-warmup``
   Skip Numba warm-up before the benchmark run.

Surface Boundary
----------------

The public CLI intentionally does not expose benchmarking. Benchmarks are
developer execution tooling, while ``pygotm validate`` is the only user-facing
validation path and always uses the Frechet validation suite.
