Physics Overview
================

GOTM models a **one-dimensional water column** — the vertical dimension only.
Horizontal gradients of velocity, temperature, and salinity can be prescribed
as external forcing, but are not computed.  This is appropriate for open-ocean
and lake sites where horizontal homogeneity is a reasonable assumption over
the simulation period.

Governing Equations
-------------------

The state vector of a GOTM column consists of:

* Horizontal velocity components :math:`U(z,t)` and :math:`V(z,t)`
* Potential temperature :math:`\theta(z,t)`
* Salinity :math:`S(z,t)`
* Turbulent kinetic energy :math:`k(z,t)` and a second turbulence quantity
  (:math:`\varepsilon`, :math:`\omega`, or a length-scale :math:`l`)

Each quantity is governed by a 1D transport equation of the generic form

.. math::

   \frac{\partial \phi}{\partial t} =
   \frac{\partial}{\partial z}\left(\nu_\phi \frac{\partial \phi}{\partial z}\right)
   + S_\phi

where :math:`\nu_\phi` is the appropriate eddy diffusivity and :math:`S_\phi`
collects all remaining source/sink terms (Coriolis, pressure gradients,
radiation, etc.).

Vertical Grid
-------------

GOTM uses a **staggered finite-difference grid** with :math:`N_\mathrm{lev}`
layers:

* Scalar quantities (:math:`\theta`, :math:`S`, :math:`k`,
  :math:`\varepsilon`) are located at **layer centres** (indices
  :math:`1, \dots, N_\mathrm{lev}`).
* Fluxes and diffusivities are located at **layer interfaces** (indices
  :math:`0, \dots, N_\mathrm{lev}`).
* Layer thicknesses :math:`h_i` are uniform by default but can be configured.

.. figure:: ../figures/staggered_grid.png
   :align: center
   :width: 55%
   :alt: Staggered vertical grid

   **Figure 1** — Staggered vertical grid.  Filled circles mark cell
   interfaces (indices :math:`i = 0, \dots, N`) where turbulent quantities
   (:math:`k`, :math:`\varepsilon`, :math:`\nu_t`, :math:`\kappa_t`) are
   stored.  Open squares mark cell centres (indices :math:`i = 1, \dots, N`)
   where mean-flow quantities (:math:`U`, :math:`V`, :math:`\theta`, :math:`S`)
   are stored.  Layer thickness :math:`h_i` connects adjacent interfaces.
   *(After GOTM manual Fig. 1, p.25.)*

Solution Sequence
-----------------

Each timestep follows this sequence:

1. **Coriolis rotation** — rotate :math:`(U,V)` by angle :math:`f\Delta t`
   (Section 3.2.4).
2. **External pressure gradient** — add the depth-uniform barotropic pressure
   gradient (Section 3.2.7).
3. **Internal pressure gradient** — add the baroclinic pressure gradient
   from the density field (Section 3.2.8).
4. **U/V momentum equations** — advance :math:`U` and :math:`V` with
   implicit vertical diffusion (Sections 3.2.5–3.2.6).
5. **Ice thermodynamics** — update ice cover, thickness, albedo, and
   transmissivity; diagnose the ocean–ice heat flux that modifies the
   temperature upper boundary condition (see :doc:`ice_thermodynamics`).
6. **Temperature equation** — advance :math:`\theta` including short-wave
   radiation extinction with the ice-modified albedo and transmissivity
   (Section 3.2.10).
7. **Salinity equation** — advance :math:`S` (Section 3.2.11).
8. **Equation of state** — update density :math:`\rho` and buoyancy frequency
   :math:`N^2`.
9. **Shear frequency** — compute :math:`M^2` from the updated velocity field
   (Section 3.2.13).
10. **Turbulence closure** — advance :math:`k` and the second turbulence
    quantity; update stability functions and diffusivities
    (Chapter 4).

All section references are to the GOTM manual (Umlauf, Burchard & Bolding).

When FABM biogeochemistry is enabled, the physics loop above runs for a
*chunk* of timesteps (default: ~1 day), storing snapshots of
:math:`T`, :math:`S`, :math:`\rho`, :math:`h`, :math:`\nu_h`, and radiation.
The biogeochemical engine then steps through those snapshots at the same
:math:`\Delta t`.  See :doc:`biogeochemistry` for the full description of the
coupled loop.

See Also
--------

* :doc:`meanflow` — mean-flow equation details
* :doc:`turbulence` — turbulence closure details
* :doc:`airsea` — air–sea interaction and boundary conditions
* :doc:`ice_thermodynamics` — ice thermodynamics models (simple through Winton three-layer)
* :doc:`biogeochemistry` — pyfabm coupling and chunked biogeochemical loop
