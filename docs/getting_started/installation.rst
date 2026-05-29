Installation
============

Requirements
------------

* Linux or macOS (Windows: supported in CPU mode)
* `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or
  `Anaconda <https://www.anaconda.com/>`_

pyGOTM uses `Numba <https://numba.pydata.org>`_ for CPU acceleration and
requires no GPU.

Create the Conda Environment
-----------------------------

pyGOTM is managed through the ``pygotm`` conda environment. All third-party
dependencies are declared in ``pygotm-conda-env.yml``. After creating or
updating the environment, install the local checkout in editable mode with
``--no-deps`` so conda remains the dependency manager.

.. code-block:: bash

   # From the repository root:
   conda env create -f pygotm-conda-env.yml
   conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .

Run commands through the environment:

.. code-block:: bash

   conda run -n pygotm pygotm --help

Environment Contents and Boundaries
-----------------------------------

``pygotm-conda-env.yml`` is the authoritative conda environment definition for
local development, validation, documentation, and integration work.  It is a
deliberate superset covering both pyGOTM kernel and pyGOTM Studio runtime
dependencies. It currently includes:

* Python 3.12 and the core scientific stack: Numba, NumPy, SciPy, xarray,
  NetCDF4, PyYAML, pandas, Pydantic, Click, GSW, and pyfabm.
* Dask and distributed for multi-case validation orchestration.
* pybtex for citation parsing, Matplotlib for documentation figures, and
  Plotly for embedded Frechet validation report plots.
* pyGOTM Studio runtime: FastAPI, Uvicorn, WebSockets, python-dotenv,
  NiceGUI, Jinja2, markdown-it-py, httpx, rich, cmocean, and WeasyPrint.
  Studio deps are declared here so the single ``pygotm`` conda environment
  covers both the kernel and the studio application layer.
* Developer and documentation tools: pytest, anyio, Ruff, mypy, PyYAML type
  stubs, pre-commit, hatch, reuse (REUSE/SPDX licensing compliance),
  import-linter (kernel–studio boundary contract), Sphinx 8.x, Furo,
  MyST, MyST-NB, copybutton, and Mermaid.

FABM support depends on ``pyfabm`` from conda-forge. pyGOTM does not publish a
``pygotm[fabm]`` pip extra because current FABM-capable ``pyfabm`` builds are
distributed through conda, not PyPI. Use this conda environment when running
configurations with ``fabm.use: true``.

The broader environment is a development and runtime convenience.  It does not
change the project boundary: physics, execution, validation, schemas, and
citations stay in ``src/pygotm/``.

Verify the Correct Python Interpreter
--------------------------------------

Confirm that the Python executable belongs to the ``pygotm`` environment and
that the package resolves from this checkout:

.. code-block:: bash

   conda run -n pygotm python -c "import sys; print(sys.executable)"
   # Expected output: .../envs/pygotm/bin/python

   conda run -n pygotm python -c "import pygotm; print(pygotm.__file__)"

Editable Install Policy
-----------------------

Use ``pip`` only for the no-dependency editable install:

.. code-block:: bash

   conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .

Do not use ``pip`` to install third-party packages into this project
environment. Add dependencies to ``pygotm-conda-env.yml`` and update the conda
environment instead.

Validation Reference Data
-------------------------

pyGOTM ships with a small set of canonical reference cases vendored under
``tests/fixtures/cases/`` so the test suite runs on a fresh checkout without
any external download. The bundled cases (couette, channel, asics_med, rouse,
seagrass, wave_breaking, entrainment) cover the distinct physics regimes
exercised by pyGOTM and are used by ``tests.fixtures.bundled_case``.

The top-level ``validation/`` directory is mostly local validation state.
``validation/report/`` is tracked as the current documentation report snapshot,
while ``validation/reference/`` and ``validation/runs/`` remain gitignored. The
``pygotm validate`` CLI uses those ignored directories to drive the full 22-case
validation sweep against Fortran GOTM reference output. Maintainers may
distribute that larger reference-data archive separately. To run the full
validation sweep locally, unpack it into the repository root so the
``validation/reference/<case>/`` directories exist locally:

.. code-block:: text

   validation/reference/couette/gotm.yaml
   validation/reference/couette/couette.nc

The ``pygotm validate`` CLI reads cases from ``validation/reference/`` so
it requires the external archive. For an in-tree regression gate on a clean
checkout, run ``python -m pytest`` — every test resolves its case configs and
reference NetCDFs through ``tests/fixtures/cases/``.

Numba JIT Compilation
---------------------

pyGOTM compiles its physics kernels with Numba at first run. Subsequent runs
reuse the cached compiled code (``cache=True``). Developers can force
compilation before internal timing work with the execution benchmark module:

.. code-block:: bash

   conda run -n pygotm python -m pygotm.execution.benchmark --cases couette

The first run pays the compilation cost; subsequent runs are fast.

Running Tests
-------------

.. code-block:: bash

   conda run -n pygotm python -m pytest -W error::RuntimeWarning

Building the Documentation
--------------------------

.. code-block:: bash

   conda run -n pygotm sphinx-build -W -b html docs docs/build

Output: ``docs/build/index.html``

Updating the Environment
------------------------

If ``pygotm-conda-env.yml`` changes (new dependencies added), update the
installed environment with:

.. code-block:: bash

   conda env update -f pygotm-conda-env.yml --prune
   conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .

.. note::

   pyGOTM does **not** use ``uv`` or ``.venv``. Use conda with the ``pygotm``
   environment for dependencies, and use ``pyproject.toml`` for package and tool
   metadata.
