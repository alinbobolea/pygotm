Citations
=========

pyGOTM ships a curated BibTeX database for kernel, turbulence, air-sea, ice,
light, and FABM references. The database lives inside the package as citation
metadata with a CC0 license note so downstream reports can redistribute the
generated bibliography without copying GPL code.

Commands
--------

Emit the full BibTeX database:

.. code-block:: bash

   conda run -n pygotm pygotm cite --all

Emit JSON records for a configuration:

.. code-block:: bash

   conda run -n pygotm pygotm cite --for-config path/to/gotm.yaml --json

Emit JSON records for an output file:

.. code-block:: bash

   conda run -n pygotm pygotm cite --for-output result.nc --json

Mapping Rules
-------------

The citation layer maps runtime labels to bibliography keys explicitly:

* Air-sea bulk algorithms: Kondo, Fairall/COARE, and Liu stability functions.
* Light extinction: Jerlov water types.
* Turbulence closures: Mellor-Yamada, k-epsilon, GLS, k-omega,
  Canuto/Cheng, Kantha-Clayson, and Craig-Banner where configured.
* Ice models: Lebedev, MyLake, Winton, and basal melt references.
* FABM: FABM framework citations plus known model paths from
  ``instances.*.model`` in ``fabm.yaml``.

For ``--for-output``, pyGOTM first tries to load the ``source_yaml`` path stored
in NetCDF attributes so citations reflect the original configuration. If the
source YAML is not available, it falls back to the provenance attributes stored
inside the NetCDF file.
