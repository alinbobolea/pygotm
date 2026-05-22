Optional Extensions
===================

Optional physics extensions for specialised applications not needed for
core turbulence modelling.

Seagrass
--------

Seagrass canopy drag and turbulence generation.  Treats seagrass canopies as
Lagrangian tracers that advect with the horizontal current or rest at their
excursion limits and exert friction on the mean flow (Verduin and Backhaus,
2000).  Turbulence generation due to seagrass friction is included as an extra
TKE production term.

.. automodule:: pygotm.extras.seagrass.seagrass
   :members:
   :undoc-members:
   :show-inheritance:

Sediment
--------

Suspended sediment transport, settling, and resuspension.  Solves the
concentration transport equation with settling advection and turbulent
diffusion.  Supports Eulerian (default) and Lagrangian particle methods.

.. automodule:: pygotm.extras.sediment.sediment
   :members:
   :undoc-members:
   :show-inheritance:
