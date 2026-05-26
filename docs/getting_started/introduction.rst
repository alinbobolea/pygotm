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
   * - Configuration
     - YAML (gotm.yaml)
     - GOTM-compatible YAML (same files)
   * - Output
     - NetCDF
     - NetCDF (CF conventions, xarray-compatible)
   * - Deployment
     - Compile from source
     - Conda environment plus editable local install

**Scientific parity:** pyGOTM must pass every official GOTM test case using a
discrete Fréchet distance comparison against Fortran GOTM 6.0.7 reference
NetCDF output.  After time-aligning both datasets, each variable is scored by
the **normalized discrete Fréchet distance** :math:`d_\mathrm{norm}` — the
worst-case leash length between the two value trajectories after dynamic
linear-or-log normalization to :math:`[0, 1]`.  For variables whose signal
magnitude is below a per-variable floor, the relative raw Fréchet score
:math:`d_\mathrm{rel} = d_\mathrm{raw} / \text{signal\_scale}` is used instead
(``metric_mode = "d_rel"``).

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Status
     - Score range
     - Meaning
   * - **PASS**
     - :math:`score < 0.01`
     - Variable is within the Fréchet parity threshold.
   * - **MARGINAL**
     - :math:`0.01 \leq score < 0.05`
     - Variable is close to parity but outside the pass threshold.
   * - **DISCREPANT**
     - :math:`0.05 \leq score < 0.20`
     - Variable has a deterministic implementation difference.
   * - **BROKEN**
     - :math:`score \geq 0.20` or structural failure
     - Variable is severely mismatched, missing, extra, or structurally
       incompatible.

A case has status **PASS** only when every compared variable passes.  Fréchet
thresholds, normalization settings, and per-variable magnitude floors are
configured in ``src/pygotm/validation/tolerances.py``; see
:doc:`../validation/overview` for the full pipeline and scoring details.
The latest generated full-suite snapshot is **PARTIAL PARITY**: 15 of 22
official reference cases pass, and the seven remaining failures are documented
in :doc:`../validation/test_cases`.

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
3. **The kernel is the product.** The physics and validation runtime live in
   ``src/pygotm/``.
4. **Double precision everywhere.** ``np.float64`` throughout — GOTM uses ``REAL(8)``.
5. **The compiled runtime is single-column.** Each vertical column
   (:math:`N_\mathrm{lev} \approx 100`) is solved serially through the
   parity-tested Numba timestep loop.
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
