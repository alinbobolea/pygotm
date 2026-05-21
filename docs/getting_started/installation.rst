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

   git clone https://github.com/<org>/pygotm.git
   cd pygotm
   conda env create -f pygotm-conda-env.yml
   conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .

Run commands through the environment:

.. code-block:: bash

   conda run -n pygotm pygotm --help

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

The source repository does not track top-level ``validation/`` data. Generated
reports and runs are reproducible outputs, and the official Fortran reference
NetCDF files are too large for normal Git hosting. Download the reference-data
release asset and unpack it so the tree contains:

.. code-block:: text

   validation/reference/couette/gotm.yaml
   validation/reference/couette/couette.nc

The planned release asset path is:

.. code-block:: text

   https://github.com/<org>/pygotm/releases/download/reference-data-v0.1.0/pygotm-validation-reference.tar.zst

Numba JIT Compilation
---------------------

pyGOTM compiles its physics kernels with Numba at first run. Subsequent runs
reuse the cached compiled code (``cache=True``). To force compilation before
benchmarking:

.. code-block:: bash

   conda run -n pygotm pygotm benchmark --cases couette

The first run pays the compilation cost; subsequent runs are fast.

Running Tests
-------------

.. code-block:: bash

   conda run -n pygotm python -m pytest tests/ -v

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
