Air–Sea Interaction
===================

Air–sea fluxes drive the surface boundary conditions for the momentum,
temperature, and salinity equations.

Momentum Flux
-------------

The surface wind stress :math:`(\tau_x, \tau_y)` is applied as a Neumann
boundary condition at the top interface of the momentum equations.  The
surface friction velocity is

.. math::

   u_{\tau s} = \sqrt{\frac{|\boldsymbol{\tau}|}{\rho_0}}

The bottom stress (drag) is computed from a logarithmic drag law using the
bottom roughness length :math:`z_{0b}`:

.. math::

   u_{\tau b} = \kappa \frac{|\boldsymbol{u}|_1}{\ln(z_1 / z_{0b})}

where :math:`\kappa = 0.4` is the von Kármán constant and
:math:`z_1` is the height of the first model level.

Heat Flux
---------

The net surface heat flux is split into two components:

* **Non-penetrative flux** :math:`Q_0` — applies at the surface as a Neumann
  boundary condition for the temperature equation.  Includes latent heat,
  sensible heat, and net long-wave radiation.
* **Short-wave radiation** :math:`I(z)` — penetrates into the water column
  and is absorbed according to a Beer–Lambert extinction law.  The extinction
  depth (e-folding scale) is configurable.

The divergence of the short-wave flux,
:math:`-\partial I / \partial z`, enters as a source term in the temperature
equation at every interior level.

Freshwater Flux
---------------

Precipitation minus evaporation (:math:`P - E`) is applied as a salt flux
(virtual salt flux method) at the surface.  No explicit dilution or
concentration of the water column is computed; the salt flux maintains
consistency with the prescribed surface salinity.

Configuring Air–Sea Forcing
----------------------------

All surface fluxes are read from the GOTM YAML configuration via the
``surface`` section.  Supported sources:

* Constant prescribed values
* Time series from file (linear interpolation between records)
* Bulk formulae (Kondo 1975, COARE) — available when the full air–sea module
  is enabled in the configuration

API Reference
-------------

See :doc:`../api/airsea` for the full API reference.
