Warm Daemon Protocol
====================

``pygotm serve`` starts a warm kernel subprocess that speaks newline-delimited
JSON-RPC over stdin and stdout. It is not an HTTP server. Studio owns any local
FastAPI or WebSocket user interface; the kernel daemon remains behind process
and file boundaries.

Framing
-------

Each request is one JSON object on stdin. Each response is one JSON object on
stdout. Progress events and diagnostics are written to stderr so stdout remains
protocol-only.

Required request examples:

.. code-block:: json

   {"id":"1","method":"version"}
   {"id":"2","method":"schema_config"}
   {"id":"3","method":"schema_output","params":{"config_path":"gotm.yaml"}}
   {"id":"4","method":"run","params":{"config_path":"gotm.yaml","output_path":"out.nc","max_steps":3}}
   {"id":"5","method":"shutdown"}

Response shape:

.. code-block:: json

   {"id":"1","ok":true,"result":{"pygotm_version":"X.Y.Z"}}
   {"id":"4","ok":false,"error":{"code":10,"message":"..."}}

Methods
-------

``version``
   Returns the same manifest-shaped fields as ``pygotm version --json`` plus a
   ``warmup`` status. Startup warmup is self-contained and does not require
   validation reference data.

``schema_config``
   Returns the curated GOTM config schema used by visual editors.

``schema_output``
   Returns output variable metadata. Passing ``params.config_path`` allows the
   daemon to include state-dependent FABM or ice records.

``run``
   Runs one config and writes one NetCDF output. Progress events are emitted to
   stderr in the same shape as ``pygotm run --progress json``.

``shutdown``
   Returns one final success response and exits with code ``0``.

State
-----

Each ``run`` request creates a fresh ``GotmDriver`` run, finalizes model state,
and clears field registries the same way as one-shot ``pygotm run``. Daemon
warmth comes from the Python process and Numba compiled specializations, not
from hidden simulation state.
