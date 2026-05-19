Utilities
=========

Shared numerical utilities used across all pyGOTM physics kernels.  Each
submodule is a direct Python translation of the corresponding GOTM Fortran
source file under ``src/``.

.. list-table:: Submodules at a glance
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - :mod:`~pygotm.util.tridiagonal`
     - Thomas algorithm (tridiagonal) solver
   * - :mod:`~pygotm.util.adv_center`
     - Vertical advection — cell-centred tracers
   * - :mod:`~pygotm.util.diff_center`
     - Vertical diffusion — cell-centred variables
   * - :mod:`~pygotm.util.diff_face`
     - Vertical diffusion — face-centred variables
   * - :mod:`~pygotm.util.density`
     - Equation of state (density, α, β)
   * - :mod:`~pygotm.util.gsw`
     - Numba-compiled TEOS-10 GSW library
   * - :mod:`~pygotm.util.time`
     - Time management (Julian day + seconds)
   * - :mod:`~pygotm.util.convert_fluxes`
     - Flux conversion for the KPP turbulence model
   * - :mod:`~pygotm.util.gridinterpol`
     - Grid interpolation from observation to model grid
   * - :mod:`~pygotm.util.lagrange`
     - Lagrangian particle random walk
   * - :mod:`~pygotm.util.ode_solvers`
     - ODE solvers for biogeochemical models (11 methods)
   * - :mod:`~pygotm.util.ode_solvers_template`
     - Alternate RK2/RK4 implementations (template variant)
   * - :mod:`~pygotm.util.gotm_version`
     - GOTM version constants
   * - :mod:`~pygotm.util.compilation`
     - Compilation metadata stub


Selector Constants
------------------

:mod:`~pygotm.util.util` defines the integer selector constants shared by the
advection, diffusion, and advection-boundary-condition routines.

**Advection schemes** (``method`` argument to :func:`~pygotm.util.adv_center.adv_center`):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Constant
     - Value
     - Scheme
   * - ``CENTRAL``
     - ``-1``
     - Central differencing
   * - ``UPSTREAM``
     - ``1``
     - First-order upwind
   * - ``P1``
     - ``2``
     - First-order upwind (P1)
   * - ``P2``
     - ``3``
     - Second-order unbounded
   * - ``Superbee``
     - ``4``
     - Superbee limiter (Roe 1986)
   * - ``MUSCL``
     - ``5``
     - MUSCL (van Leer 1979)
   * - ``P2_PDM``
     - ``6``
     - P2 with Positive Definite Method (Pietrzak 1998)
   * - ``SPLMAX13``
     - ``13``
     - SPLMAX13 (Pietrzak 1998)

**Diffusion boundary conditions** (``bc_up`` / ``bc_down``):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Constant
     - Value
     - Meaning
   * - ``Dirichlet``
     - ``0``
     - Prescribe value at boundary
   * - ``Neumann``
     - ``1``
     - Prescribe flux at boundary

**Advection boundary conditions** (``bc_up`` / ``bc_down``):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Constant
     - Value
     - Meaning
   * - ``flux``
     - ``1``
     - Prescribed flux [tracer·m/s]
   * - ``value``
     - ``2``
     - Prescribed tracer value
   * - ``oneSided``
     - ``3``
     - One-sided upwind (outflow only)
   * - ``zeroDivergence``
     - ``4``
     - Zero-divergence extrapolation


Tridiagonal Solver
------------------

Translation of ``mtridiagonal.F90``.  Provides a Numba-JIT Thomas algorithm
(simplified Gaussian elimination) for tridiagonal linear systems of the form:

.. math::

   b_i\,y_i + c_i\,y_{i+1} + a_i\,y_{i-1} = d_i,
   \quad i = f_i,\ldots,l_t

The main diagonal is stored in ``bu``, upper diagonal in ``au``, lower diagonal
in ``cu``, and the right-hand side in ``du``.  Work arrays ``ru`` and ``qu``
hold intermediate values.

All diffusion, momentum, and turbulence-closure routines in pyGOTM call
:func:`~pygotm.util.tridiagonal.tridiagonal` as their final linear-system step.

Workspace classes pre-allocate all six arrays (shape ``(nlev+1,)`` for a single
column, shape ``(batch_size, nlev+1)`` for batch runs):

* :class:`~pygotm.util.tridiagonal.TridiagonalWorkspace`
* :class:`~pygotm.util.tridiagonal.TridiagonalBatchWorkspace`

.. automodule:: pygotm.util.tridiagonal
   :members:
   :undoc-members:


Vertical Advection
------------------

Translation of ``adv_center.F90``.  Solves the one-dimensional advection
equation for tracers defined at cell centres.

Two conservation modes are supported:

* **Conservative** (``CONSERVATIVE = 1``): :math:`\partial_t Y = -\partial_z(wY)`.
  Used for settling or rising tracers such as sediment or phytoplankton.
* **Non-conservative** (``NON_CONSERVATIVE = 0``): :math:`\partial_t Y = -w\,\partial_z Y`.
  Used when the water column has a prescribed net vertical velocity.

The face flux is reconstructed with one of the eight slope-limiter schemes
listed in the selector-constants table above (``UPSTREAM`` through
``SPLMAX13``).  Sub-stepping is applied automatically when
:math:`\max(|w|\Delta t / \Delta z) > 1`, up to 100 sub-steps per time step.

Workspace classes:

* :class:`~pygotm.util.adv_center.AdvectionWorkspace` — single column
* :class:`~pygotm.util.adv_center.AdvectionBatchWorkspace` — batch

.. automodule:: pygotm.util.adv_center
   :members:
   :undoc-members:


Vertical Diffusion (Cell Centres)
----------------------------------

Translation of ``diff_center.F90``.  Solves the one-dimensional diffusion
equation with optional source terms and relaxation for variables defined at
cell centres:

.. math::

   \frac{\partial Y}{\partial t}
   = \frac{\partial}{\partial z}\!\left(\nu_Y \frac{\partial Y}{\partial z}\right)
   - \frac{Y - Y_{\mathrm{obs}}}{\tau_R}
   + Y\,L_{\mathrm{sour}} + Q_{\mathrm{sour}}

The diffusivity :math:`\nu_Y` is defined at cell faces.  The diffusion term,
linear source :math:`L_{\mathrm{sour}}`, and relaxation are treated implicitly
with Crank–Nicolson implicitness ``cnpar``; the constant source
:math:`Q_{\mathrm{sour}}` is explicit.  Relaxation is only applied where
``tau_r[i] < 1e10``.

For non-negative concentrations (``posconc = 1``), negative Neumann boundary
fluxes are linearised following Patankar (1980) to preserve positivity.

.. automodule:: pygotm.util.diff_center
   :members:
   :undoc-members:


Vertical Diffusion (Cell Faces)
---------------------------------

Translation of ``diff_face.F90``.  Solves the same diffusion equation as
:mod:`~pygotm.util.diff_center` but for variables defined at grid *faces*
(interfaces) rather than cell centres.  The solved range spans interior
face indices ``1`` to ``nlev − 1``.

A bug fix attributed to Georg Umgiesser handles the ``nlev == 2`` edge case:
when only two layers are present, boundary diffusivities and values are
replicated from the single interior face to stabilise the tridiagonal system.

.. automodule:: pygotm.util.diff_face
   :members:
   :undoc-members:


Equation of State
-----------------

Translation of ``density.F90``.  Computes in-situ density
:math:`\bar{\rho}(S,\Theta,P)`, potential density, and buoyancy expansion
coefficients from salinity, temperature, and pressure.

Three methods are selected via ``DensityState.density_method``:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Code
     - Constant
     - Description
   * - ``1``
     - ``METHOD_TEOS10``
     - Full TEOS-10 EOS via the ``gsw`` package: ``gsw_rho``, ``gsw_sigma0``,
       ``gsw_alpha``, ``gsw_beta``.
   * - ``2``
     - ``METHOD_LINEAR_TEOS10``
     - Linearised EOS; :math:`\alpha_0` and :math:`\beta_0` computed from
       TEOS-10 at the user reference point :math:`(S_0, T_0, p_0)`.
   * - ``3``
     - ``METHOD_LINEAR_USER``
     - Linearised EOS with user-supplied :math:`\rho_0`, :math:`\alpha_0`,
       :math:`\beta_0`.

The TEOS-10 standard specific heat of seawater at :math:`S_A = 0`,
:math:`C_T = 0`, :math:`p = 0` is ``CP0 = 3991.86795711963`` J/(kg·K),
matching ``gsw_mod_teos10_constants::gsw_cp0`` in the Fortran source.

The linear EOS (methods 2 and 3) computes density as:

.. math::

   \rho = \rho_b\,(1 - \alpha_0\,(T - T_0) + \beta_0\,(S - S_0))

No pressure dependence is applied in the linear methods (matching the Fortran
comment from Lars Umlauf).

:class:`~pygotm.util.density.DensityState` holds all module-level state.  Call
:func:`~pygotm.util.density.init_density` once to allocate arrays, then
:func:`~pygotm.util.density.do_density` each time step.

.. automodule:: pygotm.util.density
   :members:
   :undoc-members:


TEOS-10 GSW Library
-------------------

A Numba-compiled subset of the TEOS-10 Gibbs SeaWater toolbox, mirroring the
Fortran files in ``extern/gsw/``.  The equation-of-state functions used from
the time loop are scalar ``@numba.njit`` functions; the salinity conversion
helpers are vectorised Python wrappers around GOTM's bundled SAAR data.

.. list-table:: Exported functions
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``gsw_rho``
     - In-situ density [kg/m³] from :math:`(S_A, C_T, p)`
   * - ``gsw_sigma0``
     - Potential density anomaly [kg/m³] at :math:`p = 0`
   * - ``gsw_alpha``
     - Thermal expansion coefficient :math:`\alpha` [1/K]
   * - ``gsw_beta``
     - Haline contraction coefficient :math:`\beta` [kg/g]
   * - ``gsw_specvol``
     - Specific volume [m³/kg]
   * - ``gsw_sa_from_sp``
     - Absolute salinity from practical salinity
   * - ``gsw_sp_from_sa``
     - Practical salinity from absolute salinity
   * - ``gsw_saar``
     - Salinity anomaly ratio (SAAR) from the GOTM lookup table

.. automodule:: pygotm.util.gsw
   :members:
   :undoc-members:


Time Management
---------------

Translation of ``time.F90``.  Time is represented as a pair of integers — true
Julian day and integer seconds since midnight — so all arithmetic is simple
integer operations.

**timefmt modes** (configured on :class:`~pygotm.util.time.GotmTime`):

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Mode
     - Behaviour
   * - ``1``
     - ``MaxN`` given directly; fake start date 2000-01-01 00:00:00 is used.
   * - ``2``
     - ``start`` and ``stop`` strings given; ``MaxN`` computed from total
       duration divided by ``timestep``.
   * - ``3``
     - ``start`` string and ``MaxN`` given; ``stop`` is computed.

Time strings use the format ``'yyyy-mm-dd hh:mm:ss'``.

:class:`~pygotm.util.time.GotmTime` is a Python dataclass that mirrors the
Fortran module-level variables.  Call :meth:`~pygotm.util.time.GotmTime.init_time`
once, then :meth:`~pygotm.util.time.GotmTime.update_time` each step.

.. automodule:: pygotm.util.time
   :members:
   :undoc-members:


Flux Conversion
---------------

Translation of ``convert_fluxes.F90``.  Converts surface heat, salinity, and
shortwave radiative fluxes to the temperature and buoyancy flux forms required
by the KPP turbulence closure.

:func:`~pygotm.util.convert_fluxes.convert_fluxes` returns a 6-tuple:

.. code-block:: python

   t_flux, s_flux, bt_flux, bs_flux, t_rad, b_rad = convert_fluxes(
       state, nlev, gravity, shf, ssf, rad, T_srf, S_srf
   )

.. list-table:: Return values
   :header-rows: 1
   :widths: 15 20 65

   * - Name
     - Units
     - Description
   * - ``t_flux``
     - K·m/s
     - Temperature flux from surface heat flux
   * - ``s_flux``
     - psu·m/s
     - Salinity flux from P − E
   * - ``bt_flux``
     - m²/s³
     - Buoyancy flux from surface heat flux
   * - ``bs_flux``
     - m²/s³
     - Buoyancy flux from surface salinity flux
   * - ``t_rad``
     - K·m/s, shape ``(nlev+1,)``
     - Temperature flux profile from shortwave radiation
   * - ``b_rad``
     - m²/s³, shape ``(nlev+1,)``
     - Buoyancy flux profile from shortwave radiation

Only called when the KPP turbulence model is active.

.. automodule:: pygotm.util.convert_fluxes
   :members:
   :undoc-members:


Grid Interpolation
------------------

Translation of ``gridinterpol.F90``.
:func:`~pygotm.util.gridinterpol.gridinterpol` linearly interpolates (and
extrapolates) observational data defined on an arbitrary structured depth grid
to the (potentially moving) GOTM model grid:

* Model levels above the topmost observation receive the topmost observed value.
* Model levels below the deepest observation receive the deepest observed value.
* Interior levels are interpolated linearly between the two nearest
  observations.

Original authors: Karsten Bolding, Hans Burchard.

.. automodule:: pygotm.util.gridinterpol
   :members:
   :undoc-members:


Lagrangian Particle Tracking
-----------------------------

Translation of ``lagrange.F90``.  Implements the Visser (1997) random-walk
scheme for Lagrangian particles in spatially inhomogeneous turbulence.  Each
particle is advanced by:

.. math::

   z^{n+1} = z^n + \partial_z \nu_t(z^n)\,\Delta t
            + R\left[2\,r^{-1}\,\nu_t\!\left(z^n
              + \tfrac{1}{2}\partial_z\nu_t(z^n)\,\Delta t\right)
              \Delta t\right]^{1/2}

where :math:`R` is a zero-mean random variable with variance
:math:`\langle R^2 \rangle = r`.

Fixed parameters:

* ``VISC_BACK = 0.0e-6`` m²/s — background viscosity floor.
* ``RND_VAR = 0.333333333`` — variance :math:`r` of the random walk.

Reflective boundary conditions are applied at the surface (:math:`z = 0`) and
the bottom (:math:`z = -\mathrm{depth}`).  Pass a seeded
``np.random.Generator`` for reproducible results.

Original authors: Hans Burchard, Karsten Bolding.

.. automodule:: pygotm.util.lagrange
   :members:
   :undoc-members:


ODE Solvers
-----------

Translation of ``ode_solvers.F90``.  Provides 11 numerical solvers for the
biogeochemical reaction-term ODE:

.. math::

   \partial_t c_i = P_i(\mathbf{c}) - D_i(\mathbf{c}), \quad i = 1, \ldots, I

where :math:`c_i` are species concentrations and :math:`P_i`, :math:`D_i` are
the production and destruction terms (Burchard et al. 2003).

.. list-table:: Available solvers
   :header-rows: 1
   :widths: 8 52 20 20

   * - Code
     - Method
     - Conservative
     - Positive
   * - 1
     - First-order explicit (Euler)
     - No
     - No
   * - 2
     - Second-order explicit Runge-Kutta
     - No
     - No
   * - 3
     - Fourth-order explicit Runge-Kutta
     - No
     - No
   * - 4
     - First-order Patankar
     - No
     - Yes
   * - 5
     - Second-order Patankar-Runge-Kutta
     - No
     - Yes
   * - 6
     - Fourth-order Patankar-Runge-Kutta (**non-functional**)
     - —
     - —
   * - 7
     - First-order Modified Patankar
     - Yes
     - Yes
   * - 8
     - Second-order Modified Patankar-Runge-Kutta
     - Yes
     - Yes
   * - 9
     - Fourth-order Modified Patankar-Runge-Kutta (**non-functional**)
     - —
     - —
   * - 10
     - First-order Extended Modified Patankar (EMP)
     - Stoichiometric
     - Yes
   * - 11
     - Second-order EMP-Runge-Kutta
     - Stoichiometric
     - Yes

Solvers 6 and 9 are not yet developed (matching the Fortran status).  EMP
schemes 10–11 were developed by Bruggeman et al. (2005) to extend Modified
Patankar to full stoichiometric conservation with multiple limiting nutrients.

.. automodule:: pygotm.util.ode_solvers
   :members:
   :undoc-members:


ODE Solver Template
-------------------

Translation of ``ode_solvers_template.F90``.  Provides alternate
implementations of the RK2 and RK4 solvers that differ from
:mod:`~pygotm.util.ode_solvers`:

* **RK2** (code 2) — explicit midpoint method:

  .. math::

     c^{\mathrm{mid}} = c^n + \tfrac{\Delta t}{2}\,f(c^n),\quad
     c^{n+1} = c^n + \Delta t\,f(c^{\mathrm{mid}})

* **RK4** (code 3) — standard four-stage RK with half-step intermediates:

  .. math::

     k_1 = f(c^n),\quad k_2 = f\!\left(c^n + \tfrac{\Delta t}{2}k_1\right),\quad
     k_3 = f\!\left(c^n + \tfrac{\Delta t}{2}k_2\right),\quad
     k_4 = f(c^n + \Delta t\,k_3)

  .. math::

     c^{n+1} = c^n + \tfrac{\Delta t}{3}\!\left(\tfrac{k_1}{2} + k_2 + k_3 + \tfrac{k_4}{2}\right)

All other solvers are re-exported unchanged from
:mod:`~pygotm.util.ode_solvers`.

.. automodule:: pygotm.util.ode_solvers_template
   :members:
   :undoc-members:


Version and Compilation
-----------------------

Two stub modules expose build-time metadata from the original Fortran GOTM:

* :mod:`~pygotm.util.gotm_version` — ``git_commit_id = "4.1.0"`` and
  ``git_branch_name = "master"``.
* :mod:`~pygotm.util.compilation` — ``compiler``, ``compiler_id``,
  ``compiler_version`` (all empty strings; injected by CMake in the Fortran
  build but not applicable in Python).

.. automodule:: pygotm.util.gotm_version
   :members:
   :undoc-members:

.. automodule:: pygotm.util.compilation
   :members:
   :undoc-members:
