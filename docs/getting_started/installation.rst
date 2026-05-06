Installation
============

Requirements
------------

* Linux or macOS (Windows: supported in CPU mode)
* Python 3.12 or later
* `uv <https://docs.astral.sh/uv/>`_ (recommended)

pyGOTM uses `Numba <https://numba.pydata.org>`_ for CPU acceleration and
requires no GPU.

Install
-------

.. code-block:: bash

   uv add pygotm

Or install from source:

.. code-block:: bash

   git clone https://github.com/<org>/pygotm.git
   cd pygotm
   uv sync

Verify the installation:

.. code-block:: bash

   uv run python -c "import pygotm; print(pygotm.__version__)"

Numba JIT Compilation
---------------------

pyGOTM compiles its physics kernels with Numba at first run. Subsequent runs
reuse the cached compiled code (``cache=True``). To force compilation before
benchmarking:

.. code-block:: bash

   pygotm benchmark --cases couette

The first run pays the compilation cost; subsequent runs are fast.

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/<org>/pygotm.git
   cd pygotm
   uv sync --all-extras
   uv run pytest tests/ -v

Building the Documentation
--------------------------

.. code-block:: bash

   uv run --group docs sphinx-build -W -b html docs docs/build

Output: ``docs/build/html/index.html``
