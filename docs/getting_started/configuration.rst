Configuration
=============

pyGOTM uses GOTM-compatible YAML configuration files (``gotm.yaml``). Any
official GOTM 6.0.7 test case YAML file can be used directly.

For a full reference of all configuration keys, see the official
`GOTM documentation <https://gotm.net/portfolio/documentation/>`__.

The 22 official test cases are located in ``validation/reference/``:

.. code-block:: text

   asics_med      blacksea    channel      couette       entrainment
   estuary        flex        gotland      lago_maggiore langmuir
   liverpool_bay  medsea_east medsea_west  nns_annual    nns_seasonal
   ows_papa       plume       resolute     reynolds      rouse
   seagrass       wave_breaking

.. note::
   The Pydantic config models that parse ``gotm.yaml`` are documented in
   :mod:`pygotm.config`.

.. note::
   Not all GOTM configurations are supported by the compiled runtime.
   Unsupported configurations raise ``UnsupportedConfigurationError`` at
   setup with a message identifying the blocking setting.
   See :doc:`../validation/test_cases` for the list of currently supported cases.
