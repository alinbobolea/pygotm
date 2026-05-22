Turbulence Closures
===================

pyGOTM implements the two-equation turbulence closures from GOTM manual
Chapter 4 (Umlauf, Burchard & Bolding).

Two-Equation Models
-------------------

All two-equation models solve a transport equation for the turbulent kinetic
energy :math:`k` and a second quantity that determines the turbulent length
scale.  The **TKE equation** (Section 4.2) is common to all models:

.. math::

   \dot{k} = \mathcal{D}_k + P + G + P_x + P_s - \varepsilon \comma

where :math:`P = \nu_t M^2` is shear production, :math:`G = -\kappa_t N^2`
is buoyancy production (negative in stable stratification),
:math:`P_s` is Stokes shear production, :math:`P_x` is extra production,
and :math:`\varepsilon` is the dissipation rate.

The diffusive transport is

.. math::

   \mathcal{D}_k = \frstder{z}\left(\frac{\nu_t}{\sigma_k} \partder{k}{z}\right) \comma

with Schmidt number :math:`\sigma_k`.

k–ε Model
~~~~~~~~~

The :math:`k`–:math:`\varepsilon` model (Section 4.4) solves a transport
equation for the dissipation rate :math:`\varepsilon`:

.. math::

   \dot{\varepsilon} = \mathcal{D}_\varepsilon
   + \frac{\varepsilon}{k}\bigl(
       c_{\varepsilon 1} P + c_{\varepsilon 3} G
       - c_{\varepsilon 2} \varepsilon
     \bigr) \comma

from which the turbulent length scale follows as
:math:`l = (c_\mu^0)^3 k^{3/2} / \varepsilon`.

   Default model constants (Rodi 1987): :math:`c_\mu^0 = 0.5477`,
   :math:`\sigma_k = 1.0`, :math:`\sigma_\varepsilon = 1.3`,
   :math:`c_{\varepsilon 1} = 1.44`, :math:`c_{\varepsilon 2} = 1.92`.

k–ω Model
~~~~~~~~~

The :math:`k`–:math:`\omega` model of Umlauf et al. (2003) (Section 4.5)
solves a transport equation for the inverse time scale
:math:`\omega = (c_\mu^0)^4 \varepsilon / k`:

.. math::

   \dot{\omega} = \mathcal{D}_\omega
   + \frac{\omega}{k}\bigl(
       c_{\omega 1} P + c_{\omega 3} G - c_{\omega 2} \varepsilon
     \bigr) \comma

Stability Functions
-------------------

The eddy viscosity and diffusivity are related to :math:`k` and
:math:`\varepsilon` by

.. math::

   \nu_t = c_\mu \frac{k^2}{\varepsilon}, \quad
   \kappa_t = c_\mu' \frac{k^2}{\varepsilon}

The **stability functions** :math:`c_\mu` and :math:`c_\mu'` depend on the
dimensionless parameters :math:`\alpha_M = k^2 M^2 / \varepsilon^2` and
:math:`\alpha_N = k^2 N^2 / \varepsilon^2`.

pyGOTM implements two variants (Section 4.7):

* **Weak-equilibrium** (``cmue_c``): local balance of Reynolds-stress
  production and dissipation; computationally cheaper.
* **Quasi-equilibrium** (``cmue_d``): retains slow-manifold corrections;
  more accurate in strongly stratified flows.

Algebraic Models
----------------

For simpler configurations, algebraic (zero-equation) closures are available:

* **Buoyancy-variance** (:math:`k_b`): algebraic balance for
  :math:`\langle b'^2 \rangle / 2` (§4.7.30).
* **Buoyancy dissipation** (:math:`\varepsilon_b`): algebraic
  :math:`\varepsilon_b` from local production–dissipation balance (§4.7.32).
* **Velocity variances** (:math:`\langle u'^2 \rangle`, etc.): diagonal
  Reynolds-stress components from algebraic expressions (§4.7.33).

Internal-Wave Background Mixing
--------------------------------

A prescribed background mixing rate :math:`\nu_{IW}` accounts for mixing
driven by internal waves not resolved by the turbulence closure (§4.7.45).
It is added to :math:`\nu_t` and :math:`\kappa_t` after the closure update.

Boundary Conditions
-------------------

At solid boundaries (surface and bottom), boundary conditions for :math:`k`
and :math:`\varepsilon` follow logarithmic-layer theory:

.. math::

   k = \frac{u_\tau^2}{\sqrt{c_\mu^0}}, \quad
   \varepsilon = \frac{(c_\mu^0)^3 k^{3/2}}{\kappa z}

where :math:`u_\tau` is the friction velocity and :math:`\kappa = 0.4` is the
von Kármán constant.

Algebraic Closures
------------------

Buoyancy Variance k_b
~~~~~~~~~~~~~~~~~~~~~

The algebraic equation for :math:`k_b` (§4.7.30) assumes equilibrium
:math:`P_b = \varepsilon_b`, giving (Eq. 172):

.. math::

   k_b = c_b \frac{k}{\varepsilon} P_b \point

Buoyancy Dissipation ε_b
~~~~~~~~~~~~~~~~~~~~~~~~~

The algebraic :math:`\varepsilon_b` (§4.7.32) follows from the constant
time-scale ratio :math:`r = c_b` (Eq. 179):

.. math::

   \varepsilon_b = \frac{1}{c_b} \frac{\varepsilon}{k} k_b \point

Velocity Variances
~~~~~~~~~~~~~~~~~~

The diagonal Reynolds stresses (§4.7.33, Eq. 180) are computed algebraically
from :math:`k`, :math:`\varepsilon`, shear production, and buoyancy production.

Wave-Breaking TKE Injection
-----------------------------

Following Craig and Banner (1994), the TKE flux from breaking surface waves
(§4.7.46, Eq. 209) is:

.. math::

   F_k = \eta u_{\tau s}^3 \comma

where :math:`\eta \approx 100` is the Craig–Banner coefficient ``cw``.

Internal-Wave and Shear-Instability Background Mixing
------------------------------------------------------

When :math:`k < k_{\mathrm{limiw}}` the Kantha-Clayson (1994) scheme
(§4.7.45) sets (Eqs. 204–208):

.. math::

   \nu_t^{IW} = 10^{-4}\,\text{m}^2\text{s}^{-1}, \quad
   \kappa_t^{IW} = 5\times10^{-5}\,\text{m}^2\text{s}^{-1} \comma

with additional shear-instability contributions depending on the gradient
Richardson number :math:`R_i`.

Stability Functions
-------------------

**Local weak-equilibrium (cmue_c, §4.7.38)**: Polynomials in :math:`\alpha_M`
and :math:`\alpha_N` (Canuto et al. 2001; Cheng et al. 2002).  Clipping:
``anLimitFact = 0.5``, ``asLimitFact = 1.0``.

**Quasi-equilibrium (cmue_d, §4.7.39)**: Uses the TKE equilibrium condition
:math:`(P+G)/\varepsilon = 1` to determine :math:`\alpha_M(\alpha_N)` before
evaluating the §4.7.38 polynomials (Umlauf and Burchard 2005).

API Reference
-------------

.. currentmodule:: pygotm.turbulence

.. autosummary::
   :nosignatures:

   alpha_mnb
   cmue_c
   cmue_d
   dissipationeq
   epsbalgebraic
   internal_wave
   kbalgebraic
   omegaeq
   production
   tkeeq
   variances
