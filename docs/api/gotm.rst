GOTM Runtime
============

Runtime containers and the compiled timestep loop.  The Python/Numba boundary
is crossed exactly **once** per simulation run by calling the compiled
``time_loop_*_compiled()`` function.

Runtime Containers
------------------

.. automodule:: pygotm.gotm.runtime_state
   :members:
   :undoc-members:

.. automodule:: pygotm.gotm.runtime_params
   :members:
   :undoc-members:

.. automodule:: pygotm.gotm.runtime_work
   :members:
   :undoc-members:

.. automodule:: pygotm.gotm.runtime_output
   :members:
   :undoc-members:

.. automodule:: pygotm.gotm.runtime_forcing
   :members:
   :undoc-members:

Output Coordinates
------------------

The :class:`~pygotm.gotm.runtime_output.RuntimeOutput` buffers use two
vertical coordinates in the returned :class:`xarray.Dataset`:

* ``z`` — cell-centre depth levels (shape ``nlev``); used by scalar fields
  (:math:`\theta`, :math:`S`, :math:`k`, :math:`\varepsilon`, etc.)
* ``zi`` — interface depth levels (shape ``nlev + 1``); used by flux and
  diffusivity fields (:math:`\nu_t`, :math:`\kappa_t`, etc.)

Output scheduling (``output_every``, initial/final slots) is resolved before
entering the compiled loop.  ``runtime_output_to_dataset()`` maps the dense
buffers to GOTM-compatible variable names and attaches latitude/longitude as
scalar coordinates after the run completes.

Time Loop
---------

.. automodule:: pygotm.gotm.time_loop
   :members:
   :undoc-members:

Builder
-------

.. automodule:: pygotm.gotm.runtime_builder
   :members:
   :undoc-members:
