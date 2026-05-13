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

pyGOTM is managed through the ``pygotm`` conda environment.  All dependencies
are declared in ``pygotm-conda-env.yml``.

.. code-block:: bash

   git clone https://github.com/<org>/pygotm.git
   cd pygotm
   conda env create -f pygotm-conda-env.yml

Activate the environment before every session:

.. code-block:: bash

   conda activate pygotm

Verify the Correct Python Interpreter
--------------------------------------

After activation, confirm that the Python executable belongs to the
``pygotm`` environment:

.. code-block:: bash

   which python
   # Expected output: .../envs/pygotm/bin/python

   python -c "import pygotm; print(pygotm.__version__)"

Numba JIT Compilation
---------------------

pyGOTM compiles its physics kernels with Numba at first run. Subsequent runs
reuse the cached compiled code (``cache=True``). To force compilation before
benchmarking:

.. code-block:: bash

   conda activate pygotm
   pygotm benchmark --cases couette

The first run pays the compilation cost; subsequent runs are fast.

Running Tests
-------------

.. code-block:: bash

   conda activate pygotm
   python -m pytest tests/ -v

Building the Documentation
--------------------------

.. code-block:: bash

   conda activate pygotm
   sphinx-build -W -b html docs docs/build

Output: ``docs/build/index.html``

Updating the Environment
------------------------

If ``pygotm-conda-env.yml`` changes (new dependencies added), update the
installed environment with:

.. code-block:: bash

   conda env update -f pygotm-conda-env.yml --prune

.. note::

   pyGOTM does **not** use `uv`, `pip`, `.venv`, or ``pyproject.toml`` for
   environment management.  Use only ``conda`` with the ``pygotm`` environment.
