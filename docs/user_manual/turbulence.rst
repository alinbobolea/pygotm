Turbulence Closure
==================

The ``turbulence`` section is the most complex part of the GOTM configuration.
It selects the turbulence closure model and sets all associated equation
coefficients.

The turbulence scheme is the heart of pyGOTM's physics.  The closure chain
is: (1) select a transport equation for TKE and the length scale; (2) select
stability functions; (3) supply boundary conditions.  Each choice narrows the
relevant coefficient sub-blocks.

The code is distributed across :mod:`pygotm.turbulence`.

.. code-block:: yaml

   turbulence:
     turb_method: second_order
     tke_method: tke
     len_scale_method: dissipation
     stab_method: constant
     bc:
       k_ubc: neumann
       k_lbc: neumann
       psi_ubc: neumann
       psi_lbc: neumann
       ubc_type: logarithmic
       lbc_type: logarithmic
     turb_param:
       cm0_fix: 0.5477
       kappa: 0.4
       k_min: 1.0e-10
       eps_min: 1.0e-12
     keps:
       ce1: 1.44
       ce2: 1.92

.. _yaml-turb-method:

``turbulence.turb_method``
--------------------------

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``no_model``, ``first_order``, ``second_order``, ``cvmix``
   * - **Default**
     - ``"second_order"``

Selects the overall turbulence closure level.

``no_model``
   No turbulence model.  The eddy viscosity ``num`` and diffusivity ``nuh``
   are held constant at ``turb_param.const_num`` and ``turb_param.const_nuh``
   respectively.  Useful for laminar flow tests or to isolate mean-flow
   dynamics.

``first_order``
   Algebraic (zero-equation) models.  The length scale is prescribed
   analytically from the vertical grid and boundary layer depth; TKE and
   dissipation are diagnosed algebraically.  Appropriate for simple
   idealized cases.

``second_order``
   Two-equation models solved with differential transport equations: one for
   TKE and one for the length-scale variable (ε, ω, or ψ depending on
   ``len_scale_method``).  This is the standard production choice.

``cvmix``
   CVMix community parameterisation library interface
   (:mod:`pygotm.cvmix.gotm_cvmix`).  **[legacy only]** in the
   current compiled runtime.

.. _yaml-turb-tke:

``turbulence.tke_method``
-------------------------

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``local_eq``, ``tke``, ``mellor_yamada``
   * - **Default**
     - ``"tke"``

How the turbulent kinetic energy (TKE) is obtained.  Only relevant when
``turb_method: second_order``.

``local_eq``
   TKE is in local equilibrium with shear and buoyancy production:
   :math:`k = (P + G) / \varepsilon`.  No transport equation is solved.
   Appropriate only for steady-state or weakly unsteady flows.

``tke``
   A differential transport equation for :math:`k` (k–ε or k–ω style) is
   solved at each time step.  Implemented in :mod:`pygotm.turbulence.tkeeq`.

``mellor_yamada``
   A differential equation for :math:`q^2/2` (twice the TKE as used in the
   Mellor–Yamada formulation) is solved.  Implemented in
   :mod:`pygotm.turbulence.q2over2eq`.

.. _yaml-turb-len:

``turbulence.len_scale_method``
--------------------------------

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``parabolic``, ``triangular``, ``xing_davies``,
       ``robert_ouellet``, ``blackadar``, ``bougeault_andre``,
       ``dissipation``, ``mellor_yamada``, ``gls``, ``omega``
   * - **Default**
     - ``"dissipation"``

Determines how the turbulent length scale (or dissipation rate) is computed.

**Algebraic length-scale prescriptions** (``tke_method: local_eq`` or
``first_order``):

``parabolic``
   Parabolic profile between boundaries.

``triangular``
   Triangular profile, maximum in the interior.

``xing_davies``
   Xing and Davies (1995) formulation.

``robert_ouellet``
   Robert and Ouellet (1987) formulation.

``blackadar``
   Blackadar (1962) two-boundary mixing length.

``bougeault_andre``
   Bougeault and André (1986) eddy-size formulation.

**Two-equation models** (``tke_method: tke``):

``dissipation``
   Solves a differential transport equation for the dissipation rate
   :math:`\varepsilon` (k–ε model).  Implemented in
   :mod:`pygotm.turbulence.dissipationeq`.  Coefficients are in ``keps``.

``omega``
   Solves a differential equation for the specific dissipation rate
   :math:`\omega` (k–ω model).  Implemented in
   :mod:`pygotm.turbulence.omegaeq`.  Coefficients are in ``kw``.

``gls``
   Generic Length Scale (GLS) formulation of Umlauf and Burchard (2003).
   Generalises k–ε, k–ω, and Mellor–Yamada into one framework via the
   ``turbulence.generic`` coefficient block.  Implemented in
   :mod:`pygotm.turbulence.genericeq`.

``mellor_yamada``
   Mellor–Yamada q²l length-scale equation (two-equation MY 2.5 closure).
   Coefficients are in ``turbulence.my``.

.. _yaml-turb-stab:

``turbulence.stab_method``
--------------------------

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``constant``, ``munk_anderson``, ``schumann_gerz``
   * - **Default**
     - ``"constant"``

Selects the stability functions that relate TKE and length scale to eddy
viscosity :math:`\nu_t` and diffusivity :math:`\nu_h`.

``constant``
   Stability functions are fixed at their neutral values.  The eddy
   viscosity is :math:`\nu_t = c_\mu k^{1/2} l` with :math:`c_\mu` fixed at
   ``cm0_fix``.  Appropriate for the simple couette/channel test cases.

``munk_anderson``
   Stability functions from Munk and Anderson (1954).  Depend on the
   gradient Richardson number.

``schumann_gerz``
   Stability functions from Schumann and Gerz (1995).  Recommended for
   stably stratified conditions.  Implemented in :mod:`pygotm.turbulence.cmue_sg`.

.. _yaml-turb-bc:

``turbulence.bc``
-----------------

Boundary conditions for the two-equation turbulence model.

``turbulence.bc.k_ubc`` / ``k_lbc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Valid values**
     - ``dirichlet``, ``neumann``
   * - **Default**
     - ``"neumann"``

Upper (``k_ubc``) and lower (``k_lbc``) boundary condition type for the TKE
transport equation.

``dirichlet``
   TKE is fixed at the boundary value implied by the log-law.

``neumann``
   TKE flux is prescribed at the boundary (wave-breaking flux for the upper
   boundary).

``turbulence.bc.psi_ubc`` / ``psi_lbc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same as ``k_ubc``/``k_lbc`` but for the length-scale equation (ε, ω, or ψ).

*Default*: ``"neumann"``.

``turbulence.bc.ubc_type``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Valid values**
     - ``logarithmic``, ``tke_injection``
   * - **Default**
     - ``"logarithmic"``

Upper boundary layer parameterisation.

``logarithmic``
   Surface boundary conditions are derived from the logarithmic law of the
   wall: :math:`k_s = u_{\tau s}^2 / \sqrt{c_\mu}`, where :math:`u_{\tau s}`
   is the friction velocity at the surface.

``tke_injection``
   Wave-breaking TKE injection following Craig and Banner (1994).  Requires
   wave height input via the ``waves`` section.
   **[legacy only]** in the current compiled runtime.

``turbulence.bc.lbc_type``
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Valid values**
     - ``logarithmic``
   * - **Default**
     - ``"logarithmic"``

Lower (bottom) boundary layer parameterisation.  Currently only the
logarithmic law of the wall is supported.

.. _yaml-turb-param:

``turbulence.turb_param``
--------------------------

Universal turbulence parameters not specific to any particular length-scale
equation.

``turbulence.turb_param.cm0_fix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless
   * - **Range**
     - > 0.0
   * - **Default**
     - ``0.5477``

Value of the stability function :math:`c_{\mu 0}` in the log-law limit.
Relates the friction velocity to TKE: :math:`k = u_\tau^2 / \sqrt{c_{\mu 0}}`.
This value sets ``cm0`` in :attr:`pygotm.gotm.runtime_params.RuntimeParams`.

``turbulence.turb_param.Prandtl0_fix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless
   * - **Range**
     - > 0.0
   * - **Default**
     - ``0.74``

Turbulent Prandtl number in the log-law limit.  Used to compute the neutral
diffusivity stability function from ``cm0_fix``.

``turbulence.turb_param.cw``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless
   * - **Range**
     - > 0.0
   * - **Default**
     - ``100.0``

Constant of the wave-breaking model (Craig and Banner 1994).  Controls the
fraction of wave energy injected as TKE at the surface.

``turbulence.turb_param.compute_kappa``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``true``

If ``true``, the von Kármán constant :math:`\kappa` is computed consistently
from the model parameters (``cm0_fix``, ``Prandtl0_fix``, and ``sig_k`` or
``sig_w``).  Set to ``false`` to use the explicit value given in ``kappa``.

``turbulence.turb_param.kappa``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless
   * - **Range**
     - > 0.0
   * - **Default**
     - ``0.4``

Von Kármán constant.  Only used when ``compute_kappa: false``.

``turbulence.turb_param.compute_c3``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``true``

If ``true``, the buoyancy production coefficient *c₃* (or *E3* in
Mellor–Yamada notation) is diagnosed from the steady-state Richardson number
``Ri_st`` via the method of Burchard (2002).  Set to ``false`` to use the
coefficient provided directly in the ``keps`` or ``kw`` blocks.

``turbulence.turb_param.Ri_st``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``0.25``

Desired steady-state gradient Richardson number used to compute *c₃*.  Only
used when ``compute_c3: true``.

``turbulence.turb_param.length_lim``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Type**
     - boolean
   * - **Default**
     - ``true``

Apply the Galperin et al. (1988) length-scale limitation to prevent
excessive mixing in stable stratification:
:math:`l \leq \text{galp} \cdot \sqrt{2k / N^2}`.

``turbulence.turb_param.galp``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``0.27``

Galperin coefficient (Galperin et al. 1988).  Used only when
``length_lim: true``.

``turbulence.turb_param.const_num``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−1`
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``5.0e-4``

Background (molecular + background turbulent) eddy viscosity.  Used when
``turb_method: no_model`` or as a floor value in other methods.

``turbulence.turb_param.const_nuh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−1`
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``5.0e-4``

Background heat diffusivity.

``turbulence.turb_param.k_min``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−2`
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``1.0e-10``

Minimum allowed TKE value (numerical floor).  Prevents division by zero in
length-scale diagnostics.

``turbulence.turb_param.eps_min``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−3`
   * - **Range**
     - ≥ 0.0
   * - **Default**
     - ``1.0e-12``

Minimum allowed dissipation rate ε.

``turbulence.turb_param.kb_min``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−4`
   * - **Default**
     - ``1.0e-10``

Minimum buoyancy variance :math:`k_b`.

``turbulence.turb_param.epsb_min``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80

   * - **Units**
     - m\ :sup:`2` s\ :sup:`−5`
   * - **Default**
     - ``1.0e-14``

Minimum buoyancy variance destruction rate.

.. _yaml-turb-keps:

``turbulence.keps`` — k–ε Coefficients
---------------------------------------

Used when ``len_scale_method: dissipation``.

Implemented in :mod:`pygotm.turbulence.dissipationeq`.

``ce1``
   Coefficient in the production term of the ε equation.  *Default*: ``1.44``.

``ce2``
   Coefficient in the destruction term.  *Default*: ``1.92``.

``ce3minus``
   Buoyancy coefficient for stable stratification (:math:`G < 0`).
   *Default*: ``0.0``.

``ce3plus``
   Buoyancy coefficient for unstable stratification (:math:`G > 0`).
   *Default*: ``1.5``.

``cex``
   Cross-diffusion coefficient.  *Default*: ``1.44``.

``ce4``
   Additional empirical coefficient.  *Default*: ``0.0``.

``sig_k``
   Schmidt number for TKE diffusivity in the k-equation.  *Default*: ``1.0``.

``sig_e``
   Schmidt number for ε diffusivity.  *Default*: ``1.3``.

``sig_peps``
   Use Burchard (2001) wave-breaking parameterisation.  *Type*: boolean.
   *Default*: ``false``.

.. _yaml-turb-kw:

``turbulence.kw`` — k–ω Coefficients
--------------------------------------

Used when ``len_scale_method: omega``.

Implemented in :mod:`pygotm.turbulence.omegaeq`.

``cw1``
   Production coefficient in the ω equation.  *Default*: ``0.555``.

``cw2``
   Destruction coefficient.  *Default*: ``0.833``.

``cw3minus``
   Buoyancy coefficient, stable.  *Default*: ``0.0``.

``cw3plus``
   Buoyancy coefficient, unstable.  *Default*: ``0.5``.

``cwx``
   Cross-diffusion coefficient.  *Default*: ``0.555``.

``cw4``
   Additional coefficient.  *Default*: ``0.15``.

``sig_kw``
   Schmidt number for TKE diffusivity.  *Default*: ``2.0``.

``sig_w``
   Schmidt number for ω diffusivity.  *Default*: ``2.0``.

.. _yaml-turb-generic:

``turbulence.generic`` — GLS Coefficients
------------------------------------------

Used when ``len_scale_method: gls``.

The Generic Length Scale (GLS) model of Umlauf and Burchard (2003)
unifies k–ε, k–ω, and Mellor–Yamada into one framework via the power-law
exponents *m*, *n*, *p*.

Implemented in :mod:`pygotm.turbulence.genericeq`.

``compute_param``
   If ``true``, all GLS coefficients are derived from ``gen_m``, ``gen_n``,
   ``gen_p``.  *Default*: ``false``.

``gen_m``
   Power-law exponent for TKE (:math:`m`).  *Default*: ``1.5``.

``gen_n``
   Power-law exponent for length scale (:math:`n`).  *Default*: ``-1.0``.

``gen_p``
   Power-law exponent for :math:`c_{\mu 0}` (:math:`p`).  *Default*: ``3.0``.

``cpsi1`` / ``cpsi2``
   Production and destruction coefficients in the ψ equation.
   *Defaults*: ``1.44`` / ``1.92``.

``cpsi3minus`` / ``cpsi3plus``
   Buoyancy coefficients for stable/unstable stratification.
   *Defaults*: ``0.0`` / ``1.0``.

``cpsix`` / ``cpsi4``
   Cross-diffusion and additional coefficients.  *Defaults*: ``1.44`` / ``0.0``.

``sig_kpsi``
   Schmidt number for TKE diffusivity in the ψ-equation context.
   *Default*: ``1.0``.

``sig_psi``
   Schmidt number for ψ diffusivity.  *Default*: ``1.3``.

``gen_d``
   Temporal decay exponent in homogeneous turbulence.  *Default*: ``-1.2``.

``gen_alpha``
   Decay exponent α.  *Default*: ``-2.0``.

``gen_l``
   Slope L of the length scale in shear-free turbulence.  *Default*: ``0.2``.

.. _yaml-turb-my:

``turbulence.my`` — Mellor–Yamada 2.5 Coefficients
----------------------------------------------------

Used when ``tke_method: mellor_yamada`` and ``len_scale_method: mellor_yamada``.

``e1`` / ``e2`` / ``e3`` / ``ex`` / ``e6``
   Empirical coefficients in the :math:`q^2 l` equation.
   *Defaults*: 1.8, 1.33, 1.8, 1.8, 4.0.

``sq``
   Turbulent diffusivity of :math:`q^2`.  *Default*: ``0.2``.

.. _yaml-turb-scnd:

``turbulence.scnd`` — Second-Order Model Coefficients
------------------------------------------------------

Coefficients for the Reynolds-stress second-order closure.

``cc1``, ``ct1``, ``ctt``
   Pressure–strain and pressure–scrambling constants.
   Values depend on the specific second-order model variant.

``a1``–``a5``, ``at1``–``at5``
   Additional empirical constants for the pressure–scalar interaction terms.

.. _yaml-turb-iw:

``turbulence.iw`` — Internal-Wave Mixing
-----------------------------------------

Parameters for the internal-wave parameterisation of Mellor (1989) and
Large et al. (1994).

``iw_model``
   Selection (integer). ``0`` = off; ``1`` = Mellor (1989); ``2`` = Large
   et al. (1994).  Implemented in :mod:`pygotm.turbulence.internal_wave`.

``klimiw``
   Minimum TKE for internal-wave mixing.  *Default*: ``1.0e-6``.

``rich_cr``
   Critical Richardson number.  *Default*: ``0.7``.

``numiw``
   Background viscosity from internal-wave shear.  *Default*: ``1.0e-4`` m² s⁻¹.

``nuhiw``
   Background diffusivity from internal-wave mixing.  *Default*: ``5.0e-5`` m² s⁻¹.

``numshear``
   Background viscosity from shear instability.  *Default*: ``5.0e-3`` m² s⁻¹.

``alpha``
   Empirical coefficient for internal-wave energy injection.
   *Default*: ``0.0``.

.. _yaml-turb-epsprof:

``turbulence.epsprof``
-----------------------

Observed dissipation rate profile (prescribed ε).

Follows the ``InputSetting`` pattern (``method: off | file``).

When ``method: file``, observed dissipation profiles are read and can be used
to diagnose turbulence quantities without integrating the full closure.

Parsed by :class:`pygotm.config.settings.ObservationTurbulenceSettings`.
This is the only turbulence key that lives in the typed Pydantic model;
all other ``turbulence.*`` keys are parsed at runtime by the GOTM state
initialisation code.

.. rubric:: References

Burchard, H. (2002). *Applied Turbulence Modelling in Marine Waters*.
Springer.

Galperin, B., L. H. Kantha, S. Hassid, and A. Rosati (1988). A
quasi-equilibrium turbulent energy model for geophysical flows.
*J. Atmos. Sci.*, 45, 55–62.

Mellor, G. L., and T. Yamada (1982). Development of a turbulence closure
model for geophysical fluid problems. *Rev. Geophys.*, 20, 851–875.

Umlauf, L., and H. Burchard (2003). A generic length-scale equation for
geophysical turbulence models. *J. Mar. Res.*, 61, 235–265.
