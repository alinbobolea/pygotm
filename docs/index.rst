pyGOTM Documentation
====================

**pyGOTM** is a Python + Numba reimplementation of GOTM (General Ocean
Turbulence Model), providing compiled single-column 1D ocean and lake turbulence
modelling with no Fortran compiler required.

.. note::
   pyGOTM is a reimplementation of GOTM, which was created by
   **Lars Umlauf**, **Hans Burchard**, and **Karsten Bolding**.
   All physics equations and algorithms originate from the GOTM
   Fortran codebase. See the `GOTM project <https://gotm.net>`__ for
   the authoritative scientific documentation.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/introduction
   getting_started/installation
   getting_started/quickstart
   getting_started/configuration

.. toctree::
   :maxdepth: 2
   :caption: User Manual

   user_manual/index

.. toctree::
   :maxdepth: 2
   :caption: Physics & Methods

   physics/overview
   physics/meanflow
   physics/turbulence
   physics/airsea

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/driver
   api/config
   api/gotm
   api/meanflow
   api/turbulence
   api/airsea
   api/observations
   api/util
   api/input
   api/validation
   api/stokes_drift
   api/fabm
   api/extras
   api/cvmix

.. toctree::
   :maxdepth: 2
   :caption: Validation

   validation/overview
   validation/test_cases

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
