Global Keys
===========

These top-level keys appear at the root of ``gotm.yaml`` and are not nested
under any section block.

.. _yaml-version:

``version``
-----------

.. list-table::
   :widths: 20 80

   * - **Type**
     - integer
   * - **Default**
     - ``7``
   * - **Valid values**
     - ``7``

Configuration file schema version.  pyGOTM only supports version 7 (the
GOTM 6 YAML format).  Always set this to ``7``; omitting the key uses the
default.

.. code-block:: yaml

   version: 7

.. _yaml-title:

``title``
---------

.. list-table::
   :widths: 20 80

   * - **Type**
     - string
   * - **Default**
     - ``"GOTM simulation"``

A human-readable label for the simulation.  The title is written into the
NetCDF output file as a global attribute and is used in validation report
HTML to identify the run.

.. code-block:: yaml

   title: FLEX 1976 — North Sea spring experiment
