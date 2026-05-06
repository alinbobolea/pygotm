Introduction
============

About GOTM
----------

GOTM — the **General Ocean Turbulence Model** — is a one-dimensional water
column model for the most important hydrodynamic and thermodynamic processes
related to vertical mixing in natural waters. It has been designed so that it
can easily be coupled to 3-D circulation models, and used as a module for the
computation of vertical turbulent mixing.

The core of the model computes solutions for the one-dimensional versions of
the transport equations of momentum, salt, and heat. The key component is the
model for the turbulent fluxes of these quantities. GOTM's strength is the
vast number of well-tested turbulence models it implements — spanning simple
prescribed diffusivities through complex Reynolds-stress models with several
differential transport equations (empirical models, energy models, two-equation
models, Algebraic Stress Models, K-profile parameterisations, and more).

GOTM has grown considerably since its first public release. Sediment transport,
seagrass dynamics, atmosphere–ocean interaction modules, and biogeochemical
models have all been added by the community. In that sense GOTM is an
integrated, community-based software environment for an almost unlimited range
of applications in geophysical turbulence modelling.

**Original GOTM authors:**
`Lars Umlauf <https://www.io-warnemuende.de/lars-umlauf.html>`_,
`Hans Burchard <https://www.io-warnemuende.de/hans-burchard.html>`_, and
`Karsten Bolding <https://bolding-bruggeman.com>`_.
Full documentation and the original Fortran source are available at
`gotm.net <https://gotm.net>`__.

About pyGOTM
------------

**pyGOTM** is a faithful Python + `Numba <https://numba.pydata.org>`_
reimplementation of GOTM 6.0.7, targeting Fortran-comparable single-column
wall-clock performance on standard CPU hardware with no Fortran compiler or
compiled binary required.

Key differences from the Fortran GOTM:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Fortran GOTM 6.0.7
     - pyGOTM
   * - Language
     - Fortran 90/95
     - Python 3.12 + Numba
   * - Timestep loop
     - Compiled Fortran
     - Compiled Numba JIT (``@numba.njit``)
   * - Multi-column parallel
     - No
     - Numba ``prange`` within batch + Dask across batches
   * - Configuration
     - YAML (gotm.yaml)
     - GOTM-compatible YAML (same files)
   * - Output
     - NetCDF
     - NetCDF (CF conventions, xarray-compatible)
   * - Deployment
     - Compile from source
     - ``uv add pygotm``; browser SaaS planned

**Scientific parity:** pyGOTM must pass every official GOTM test case with
the range-aware combined tolerance

.. math::

   |a - b| \leq \max(10^{-7} \times \mathrm{ref\_range},\, 10^{-12})
                + 10^{-6} \times |b|

against the Fortran GOTM 6.0.7 reference output.

Execution Model
---------------

pyGOTM uses a compiled single-column execution model. The Python/Numba boundary
is crossed exactly **once** per full simulation run:

.. mermaid::

   flowchart TD
       config["Load YAML config\n(Pydantic validation)"]
       alloc["Allocate runtime containers\nRuntimeState · RuntimeWork · RuntimeOutput · RuntimeForcing"]
       preproc["Preprocess forcing arrays"]
       compiled["time_loop_*_compiled()  @numba.njit\nall physics · turbulence closure · output buffering"]
       xarray["Convert output buffers to xarray.Dataset"]
       netcdf["Write NetCDF\n(CF conventions)"]

       config  --> alloc
       alloc   --> preproc
       preproc -. "Python / Numba boundary — crossed ONCE per run" .-> compiled
       compiled -. "return to Python" .-> xarray
       xarray  --> netcdf

       style compiled fill:#fef3cd,stroke:#c8960c,color:#7a5c00
       style config   fill:#dce8f7,stroke:#3a78b5,color:#1a4a7a
       style alloc    fill:#dce8f7,stroke:#3a78b5,color:#1a4a7a
       style preproc  fill:#dce8f7,stroke:#3a78b5,color:#1a4a7a
       style xarray   fill:#dce8f7,stroke:#3a78b5,color:#1a4a7a
       style netcdf   fill:#dce8f7,stroke:#3a78b5,color:#1a4a7a

This eliminates per-timestep Python overhead and targets within 1.5–2× of
Fortran wall-clock time on a warm (cached JIT) run.

Design Principles
-----------------

1. **Scientific accuracy is non-negotiable.** Every kernel is validated against
   Fortran GOTM. No accuracy tradeoffs before parity is established.
2. **Reproducibility is a feature.** Every simulation result is fully
   reproducible from its YAML config. No hidden state.
3. **The kernel is the product.** The API and UI are wrappers. The physics
   lives in ``src/pygotm/``.
4. **Double precision everywhere.** ``np.float64`` throughout — GOTM uses ``REAL(8)``.
5. **Parallelism is horizontal, not vertical.** Each vertical column
   (:math:`N_\mathrm{lev} \approx 100`) is solved serially (Thomas algorithm).
   Parallelism comes from Numba ``prange`` across columns in a batch, and Dask
   across batches.
6. **Fail loudly on science errors.** NaN or physical-bounds violations
   raise immediately with a diagnostic message.
7. **Unsupported configurations fail at setup.** No silent fallback to a
   Python timestep loop. If a compiled loop is not yet implemented for a given
   configuration, ``UnsupportedConfigurationError`` is raised during setup.

Acknowledgements
----------------

pyGOTM is built on the scientific foundation of GOTM. We are deeply grateful to
**Lars Umlauf**, **Hans Burchard**, and **Karsten Bolding** for creating,
maintaining, and freely distributing GOTM. We also acknowledge the broader
GOTM community, the CARTUM project (Comparative Analysis and Rationalisation of
Second-Moment Turbulence Models, MAS3-CT98-0172), and all contributors whose
work is cited in the GOTM manual.

The original GOTM Fortran source is used here as the authoritative scientific
reference. All physics equations, constants, and algorithms in pyGOTM are
derived from or verified against that source.
