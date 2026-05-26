Kernel Stability Contract
=========================

pyGOTM exposes a small kernel surface for external tools. The kernel remains a
simulation package; tools interact through commands, stdin/stdout JSON-RPC, and
NetCDF files.

Stable Public Surface
---------------------

The following contracts are stable after the first public kernel release:

* Public CLI command names and documented flags.
* JSON keys from ``pygotm version --json``.
* JSON-lines event shapes from ``pygotm run --progress json``.
* ``pygotm serve`` request, response, and event schema.
* NetCDF global attributes listed by ``pygotm schema netcdf-attrs --json``.
* ``pygotm_config_schema_version`` and ``pygotm_output_schema_version``.

Additive keys, commands, variables, or optional fields are minor-version
changes. Removing a key, renaming a key, or changing a documented meaning
requires a major-version bump or a documented deprecation period.

Exit Codes
----------

Click parse and usage errors keep Click's exit code ``2``. Runtime commands map
known failures to these public codes:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success.
   * - ``1``
     - Validation completed but the verdict is not full parity.
   * - ``2``
     - CLI usage or parse error from Click.
   * - ``10``
     - Config, YAML validation, or schema error.
   * - ``11``
     - Unsupported compiled-runtime configuration.
   * - ``12``
     - Runtime execution failure after setup.
   * - ``13``
     - Input/output file error.
   * - ``14``
     - Dependency or environment unavailable.
   * - ``70``
     - Unexpected internal error.

Tracebacks are hidden by default for public CLI use. Pass ``--debug`` to
``pygotm run`` when developing against the kernel and needing the original
Python traceback.

Progress Events
---------------

``pygotm run --progress json`` writes newline-delimited JSON events to stderr.
The default mode remains silent. Hydro-only runs currently report
``progress_mode: "indeterminate"`` for integration, because the compiled hydro
runtime is invoked as one call and does not have parity-tested chunked progress.
FABM-active runs can emit determinate chunk progress.

NetCDF Contract
---------------

Every pyGOTM output includes global attributes for version, environment,
configuration hashes, timestamps, turbulence closure, ice model, and FABM
status. The source YAML hash is the raw file-byte SHA-256. The effective YAML
hash is computed from a canonical materialized YAML document and is portable
across machine path relocations.
