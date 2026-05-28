# pyGOTM

pyGOTM is a Python reimplementation of the General Ocean Turbulence Model
(GOTM) with Numba-compiled CPU physics. It is a substantial source-level
translation of the Fortran model into a Python package while preserving GOTM's
scientific structure, YAML configuration style, and NetCDF validation target.

The project mirrors the GOTM source layout, reads GOTM-compatible YAML
configuration files, runs supported single-column cases through a compiled
runtime, and validates NetCDF output against GOTM 6.0.7 reference results.

## Current Status

pyGOTM is usable as a simulation kernel and validation target, but it is not yet
a complete replacement for every GOTM configuration.

- Runtime: compiled single-column Numba execution.
- Configuration: GOTM 6.x YAML files, parsed through the pyGOTM config layer.
- Output: xarray/NetCDF datasets with reproducibility metadata.
- CLI: `run`, `validate`, `version`, `schema`, `cite`, and `serve`.
- Integration surface: `pygotm serve`, a warm stdin/stdout JSON-RPC daemon for
  external tools. It is not a network server.
- Validation: the default Frechet validation set (`couette`, `channel`,
  `entrainment`) passes.
- Full reference suite: all 22 official cases execute, with the current local
  snapshot at partial parity: 15 PASS and 7 FAIL. The failures are documented in
  `docs/validation/test_cases.rst` and generated validation reports.
- Full-suite variable totals in the latest snapshot: 2316 PASS, 67 MARGINAL,
  31 DISCREPANT, and 0 BROKEN.
- Representative current validation wall times on the local 8-core Ryzen
  validation host: `couette` 3.0 s, `channel` 3.0 s, and `entrainment` 2.5 s.

The current parity achievement is strong for a Python translation of a mature
Fortran ocean model: every official case runs, most cases pass outright, and
the remaining differences are documented as deterministic Frechet deviations
rather than missing output schema. See `docs/validation/overview.rst` for the
validation method and `docs/validation/test_cases.rst` for the case table.

## What It Does

- Simulates one-dimensional vertical water-column physics for ocean and lake
  cases.
- Advances temperature, salinity, horizontal velocity, density, turbulence
  quantities, ice thermodynamics, and supported FABM coupling paths.
- Supports the current compiled-runtime GOTM turbulence paths used by the
  validated reference cases, including first-order and second-order closures
  such as k-epsilon, k-omega, GLS, and Mellor-Yamada variants where supported by
  the selected configuration.
- Reads time series, vertical profiles, and reference-case forcing files.
- Writes NetCDF output compatible with xarray and standard ocean-analysis tools.
- Produces Frechet-distance HTML and JSON validation reports against Fortran
  GOTM reference NetCDF files, including per-variable scores and case reports.

Unsupported compiled-runtime configurations fail during setup with an explicit
error. pyGOTM does not silently fall back to a legacy Python timestep loop during
parity runs.

## Integrated Science Modules

pyGOTM is not just a minimal hydrodynamic runner. The translation includes
substantial coupled physics and ecosystem machinery from GOTM's broader model
family.

**FABM / pyfabm biogeochemistry**

- Optional FABM coupling through `pyfabm`, driven by `fabm.yaml` next to the
  GOTM configuration.
- A chunked interleaved architecture: the Numba hydro loop advances a window of
  timesteps and stores hydrodynamic snapshots, then pyfabm advances biological
  state through the same timesteps with those physical dependencies.
- pyGOTM supplies FABM dependencies such as temperature, salinity, density, cell
  thickness, PAR/light, wind speed, day-of-year, and bottom stress.
- pyfabm state variables receive pyGOTM-owned vertical movement, sinking/rising
  advection, turbulent diffusion, surface exchange, bottom exchange, and
  source/sink rate updates.
- Bio-shading feedback is supported through FABM light-attenuation diagnostics,
  allowing biological attenuation to modify the PAR profile within the coupling
  loop.
- When `fabm.use: false`, pyfabm is not imported and the physics-only compiled
  runtime remains unchanged.
- FABM-enabled runs require the conda environment, which installs `pyfabm` from
  conda-forge. The wheel metadata intentionally does not advertise a
  `pygotm[fabm]` pip extra.

See `docs/physics/biogeochemistry.rst` and `docs/api/fabm.rst` for the coupling
design and API.

**Ice thermodynamics**

- Five ice modes are available through `pygotm.icethm`: no ice, the GOTM simple
  limiter, ice-shelf basal melt, Lebedev freezing-degree-day ice, MyLake slab
  ice, and Winton three-layer sea ice.
- The ice dispatcher is Numba-compiled and shares a common `IceState`
  container across models.
- Ice thermodynamics is coupled back into the ocean through the upper
  temperature boundary condition, albedo, transmissivity, and diagnosed
  ocean-ice heat and salt fluxes where the selected model provides them.
- The basal-melt implementation uses the Holland-Jenkins three-equation
  ice-shelf closure with McDougall-Jackett pressure-adjusted freezing
  temperature.
- The Winton model tracks snow, upper/lower ice layers, internal ice
  temperatures, surface melt, basal growth/melt, flooding, albedo, and
  penetrating shortwave radiation.

See `docs/physics/ice_thermodynamics.rst` and `docs/api/icethm.rst` for the
full model descriptions and references.

## Documentation and Validation System

The documentation is a compact technical collection that combines the pieces a
serious model user or developer needs in one place:

- Theory and methods pages for mean flow, turbulence, air-sea exchange, ice
  thermodynamics, and FABM/pyfabm coupling.
- A GOTM-compatible YAML user manual, including file formats, forcing inputs,
  output variables, and unsupported compiled-runtime paths.
- API reference pages generated from the pyGOTM package.
- Validation pages that summarize the current Frechet parity snapshot and link
  to generated per-case HTML reports.

The validation system is also a core part of the translation. It compares
pyGOTM NetCDF output against Fortran GOTM reference output with a discrete
Frechet-distance pipeline. The comparison aligns time axes, checks variable
presence and structure, computes raw and normalized Frechet distances, switches
to a relative raw score for tiny-signal variables, and reports PASS, MARGINAL,
DISCREPANT, or BROKEN statuses per variable. A separate peak-sensitive
diagnostic keeps localized turbulence differences visible without letting
post-decorrelation peak timing dominate the primary verdict.
Validation writes a docs-published report snapshot under `validation/report/`,
including `report.html`, `report.json`, and one per-case HTML page. The
per-variable `results.json` artifact is generated there for release triage but
is intentionally not tracked.

See `docs/validation/overview.rst` for the algorithm and
`docs/validation/test_cases.rst` for the current 22-case result table.

## Quick Start

From a fresh checkout:

```bash
conda env create -f pygotm-conda-env.yml
conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .
conda run -n pygotm pygotm --help
```

Seven canonical reference cases are vendored under `tests/fixtures/cases/`
(couette, channel, asics_med, rouse, seagrass, wave_breaking, entrainment) so
the test suite runs on a clean checkout with no external download required.
The top-level `validation/` directory is mostly local validation state:
`validation/reference/`, `validation/runs/`, NetCDFs, and debug outputs remain
ignored, while `validation/report/` is a curated report snapshot tracked for
complete documentation builds. To run the full 22-case sweep locally, provide
the external reference-data tree so files like this exist:

```text
validation/reference/couette/gotm.yaml
```

Maintainers may distribute a separate reference-data archive. For the in-tree
regression gate on a fresh checkout, use `python -m pytest` (everything
resolves through the bundled fixtures).

Run one case:

```bash
conda run -n pygotm pygotm run validation/reference/couette/gotm.yaml \
  --output couette.nc
```

Run the default validation set:

```bash
conda run -n pygotm pygotm validate
```

Validation runs serially by default for deterministic command-line behavior.
Use `--workers N` with `N > 1` when you want multi-case validation through a
local Dask cluster.

Run selected validation cases:

```bash
conda run -n pygotm pygotm validate --cases couette,channel,entrainment
```

`pygotm validate --all` currently completes the 22-case suite but exits with a
nonzero status because the full-suite verdict is partial parity. The detailed
case results are documented in `docs/validation/test_cases.rst`.

## Python API

```python
from pygotm.driver import GotmDriver

driver = GotmDriver("validation/reference/couette/gotm.yaml")
ds = driver.run(output_path="couette.nc")

print(ds)
print(list(ds.data_vars))
```

For performance checks that should exercise the compiled loop without material
NetCDF output:

```python
from pygotm.driver import GotmDriver

ds = GotmDriver("validation/reference/couette/gotm.yaml").run(output=False)
```

## Command Surface

```bash
conda run -n pygotm pygotm run CONFIG --output result.nc
conda run -n pygotm pygotm validate --cases couette,channel
conda run -n pygotm pygotm version --json
conda run -n pygotm pygotm schema config --json
conda run -n pygotm pygotm schema output --json --config CONFIG
conda run -n pygotm pygotm schema netcdf-attrs --json
conda run -n pygotm pygotm cite --all
conda run -n pygotm pygotm serve
```

`pygotm serve` speaks newline-delimited JSON-RPC on stdin/stdout. Progress
events and diagnostics go to stderr so stdout remains protocol-only.

## Development Gates

Run Python commands through the `pygotm` conda environment:

```bash
conda run -n pygotm python -m pytest -W error::RuntimeWarning
conda run -n pygotm mypy src/
conda run -n pygotm ruff format .
conda run -n pygotm ruff check .
conda run -n pygotm sphinx-build -W -b html docs docs/build
```

The full pytest suite is the regression gate and runs entirely from
`tests/fixtures/cases/`. No external reference-data download is required.
The GitHub Actions CI workflow runs the same conda-backed formatting,
linting, typechecking, pytest, documentation, and packaging gates. The separate
full-reference validation workflow is manual because it requires the external
`validation/reference/` data tree.

The conda environment owns third-party dependencies. The only permitted `pip`
use is the no-dependency editable install of the local checkout:

```bash
conda run -n pygotm python -m pip install --no-deps --no-build-isolation -e .
```

## Repository Layout

```text
src/pygotm/             Python package and translated GOTM modules
tests/                  Unit, integration, validation, docs, and CLI tests
tests/fixtures/cases/   Vendored reference cases used by the test suite
docs/                   Sphinx documentation source
docs/_intersphinx/      Vendored intersphinx inventories for offline builds
validation/report/      Tracked validation HTML/JSON snapshot for docs
validation/reference/   Local external reference data, ignored by git
validation/runs/        Local generated validation NetCDF outputs, ignored by git
gotm-model/             External Fortran GOTM checkout and reference cases
```

Except for `validation/report/`, `validation/` and `gotm-model/` are
intentionally not part of normal Git history. They are local data used for
validation and translation work.

## Build Documentation

Build the docs with:

```bash
conda run -n pygotm sphinx-build -W -b html docs docs/build
```

Open `docs/build/index.html` after a successful build.

## Upstream

pyGOTM is based on GOTM, the General Ocean Turbulence Model:

- Project: https://gotm.net
- Source: https://github.com/gotm-model/code

The original GOTM authors are Lars Umlauf, Hans Burchard, and Karsten Bolding.
GOTM remains the scientific authority for the equations, constants, and
reference behavior translated here.

## License

GPL-2.0-only, matching the original GOTM licensing. See `LICENSE`.

The citation database in `src/pygotm/citations/` carries its own CC0 license
notice for bibliography metadata.

## Roadmap

- [x] Numba infrastructure and core utility kernels.
- [x] Compiled single-column runtime for supported GOTM reference cases.
- [x] CLI, schema, citation, validation, and warm daemon surfaces.
- [x] Sphinx documentation and generated validation report integration.
- [ ] Resolve remaining full-suite parity cases.
- [ ] Continue performance and validation hardening of the compiled runtime.
