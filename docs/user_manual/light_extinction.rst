Light Extinction
================

The ``light_extinction`` section controls how incident shortwave radiation is
partitioned into visible and non-visible wavebands and attenuated as it
penetrates the water column.  pyGOTM implements the two-band exponential
scheme of Paulson and Simpson (1977) as used in GOTM.

The physics is implemented in :mod:`pygotm.airsea.shortwave_radiation`.
The parameters feed into :attr:`pygotm.gotm.runtime_params.RuntimeParams.light_A`,
``light_g1``, and ``light_g2``.

Parsed by :class:`pygotm.config.settings.LightExtinctionSettings`.

.. code-block:: yaml

   light_extinction:
     method: jerlov-i
     A:
       method: constant
       constant_value: 0.58
     g1:
       method: constant
       constant_value: 0.35
     g2:
       method: constant
       constant_value: 23.0

.. _yaml-light-method:

``light_extinction.method``
---------------------------

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Valid values**
     - ``jerlov-i``, ``jerlov-1-50m``, ``jerlov-ia``, ``jerlov-ib``,
       ``jerlov-ii``, ``jerlov-iii``, ``custom``
   * - **Default**
     - ``"jerlov-i"``

Selects the water-type preset or enables manual tuning.

Jerlov (1968) classified natural seawater into optical water types based on
measured spectral transmittance.  Each type has known values for the partition
coefficient *A* and the two e-folding depths *g1* and *g2*.

.. list-table:: Jerlov water type presets
   :header-rows: 1
   :widths: 20 20 20 40

   * - Method token
     - A
     - g1 (m)
     - g2 (m)
   * - ``jerlov-i``
     - 0.58
     - 0.35
     - 23.0
   * - ``jerlov-1-50m``
     - 0.68
     - 1.20
     - 28.0
   * - ``jerlov-ia``
     - 0.62
     - 0.60
     - 20.0
   * - ``jerlov-ib``
     - 0.67
     - 1.00
     - 17.0
   * - ``jerlov-ii``
     - 0.77
     - 1.50
     - 14.0
   * - ``jerlov-iii``
     - 0.78
     - 1.40
     - 7.9

When a preset is chosen, the values of ``A``, ``g1``, and ``g2`` in the YAML
are overridden by the preset constants.

``custom``
   Use the values of ``A``, ``g1``, and ``g2`` as specified in the YAML.
   Each can be a constant or a time-varying file input.

.. _yaml-light-A:

``light_extinction.A``
----------------------

Non-visible fraction of shortwave radiation (near-infrared band).

Follows the ``InputSetting`` pattern (``method: constant | file``).

.. list-table::
   :widths: 20 80

   * - **Units**
     - dimensionless (fraction, 0–1)
   * - **Default**
     - ``0.7``

The remaining fraction :math:`1-A` is the visible (photosynthetically active
radiation, PAR) band.  Both bands decay exponentially with depth.

The total shortwave irradiance at depth *z* (below surface) is:

.. math::

   I(z) = I_0 \left[ A \, e^{-z/g_1} + (1-A) \, e^{-z/g_2} \right]

where :math:`I_0` is the surface irradiance after albedo correction.

.. _yaml-light-g1:

``light_extinction.g1``
-----------------------

E-folding attenuation depth of the non-visible (red/near-infrared) band.

Follows the ``InputSetting`` pattern.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m
   * - **Range**
     - > 0.0
   * - **Default**
     - ``0.4``

Shorter ``g1`` values indicate more turbid water (higher non-visible
attenuation).  For clear open-ocean water (Jerlov type I), ``g1 ≈ 0.35 m``.

.. _yaml-light-g2:

``light_extinction.g2``
-----------------------

E-folding attenuation depth of the visible (blue/green) band.

Follows the ``InputSetting`` pattern.

.. list-table::
   :widths: 20 80

   * - **Units**
     - m
   * - **Range**
     - > 0.0
   * - **Default**
     - ``8.0``

Larger ``g2`` values represent clearer water.  For clear open-ocean water
(Jerlov type I), ``g2 ≈ 23 m``.

.. note::

   When ``method: custom``, all three parameters (``A``, ``g1``, ``g2``) can
   be read from a time-varying file to represent seasonal phytoplankton
   blooms or changing turbidity.  Use ``column`` to select the appropriate
   column from a shared extinction data file.

.. rubric:: References

Jerlov, N. G. (1968). *Optical Oceanography*. Elsevier.

Paulson, C. A., and J. J. Simpson (1977). Irradiance measurements in the
upper ocean. *J. Phys. Oceanogr.*, 7, 952–956.
