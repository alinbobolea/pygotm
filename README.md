# pyGOTM

**Python reimplementation of the General Ocean Turbulence Model (GOTM) using
Numba-compiled CPU physics.**

GOTM is a widely used 1D ocean turbulence model. pyGOTM translates the Fortran
source layout into Python while keeping scientific validation against GOTM
reference cases at the center of development.

## What it does

- Simulates the vertical structure of temperature, salinity, and turbulence in an ocean water column or lake
- Supports six turbulence closure models: k-ε, k-ω, GLS, Mellor-Yamada 2.5, KPP
- Reads GOTM 6.x YAML configuration files natively
- All 22 official GOTM validation cases built in
- Runs single-column physics through a compiled Numba runtime
- Outputs NetCDF (CF conventions) compatible with xarray, MATLAB, and all ocean toolboxes

## Why it exists

Fortran GOTM requires: a Fortran compiler, CMake, NetCDF libraries, Conda environment, command-line expertise.  
pyGOTM requires: a browser (SaaS) or `pip install pygotm` (local).

Target users: aquaculture site engineers, limnologists, coastal ocean researchers, environmental consultancies.

## Key Features

| Feature | Description |
|---------|-------------|
| Compiled runtime | Numba JIT single-column timestep loop with flat float64 arrays |
| GOTM parity | Validated against Fortran GOTM 6.0.7 with the project range-aware tolerance |
| Built-in cases | All 22 official GOTM test cases loadable in one click |
| REST API | FastAPI with async job submission + WebSocket progress streaming |
| Browser UI | NiceGUI + Plotly interactive vertical profiles and Hövmoller diagrams |
| Export | NetCDF, CSV, interactive HTML reports |

## Architecture

```
┌──────────────────────────────────────┐
│  NiceGUI UI (port 8080)              │
│  Plotly charts, YAML form editor     │
└────────────────┬─────────────────────┘
                 │ HTTP / WebSocket
┌────────────────▼─────────────────────┐
│  FastAPI REST API (port 8000)        │
│  POST /api/simulations               │
│  GET  /api/results/{id}              │
│  WS   /ws/{id} (live progress)       │
└────────────────┬─────────────────────┘
                 │ Python import
┌────────────────▼─────────────────────┐
│  GotmDriver (driver.py)              │
│  YAML → Pydantic config              │
│  Runtime setup → compiled loop       │
│  xarray output → NetCDF writer       │
└────────────────┬─────────────────────┘
                 │ Numba call
┌────────────────▼─────────────────────┐
│  gotm/time_loop.py                   │
│  compiled timestep runner            │
│  direct 1D physics routines          │
│  dense output buffers                │
│  post-run xarray conversion          │
└──────────────────────────────────────┘
```

## Quick Start

```bash
# Clone and create the conda environment
git clone https://github.com/<org>/pygotm.git
cd pygotm
conda env create -f pygotm-conda-env.yml
conda activate pygotm

# Run validation for a supported reference case
pygotm validate --case couette

# Benchmark the compiled runtime
pygotm benchmark --cases couette,channel
```

## Example Use Cases

- **Aquaculture:** Model thermal stratification and oxygen profiles for salmon farm site assessment
- **Lake management:** Predict algal bloom timing in drinking water reservoirs
- **Coastal research:** Standalone validation testbed for turbulence closure schemes
- **Education:** Interactive water column physics without installing Fortran

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Physics kernels | [Numba](https://numba.pydata.org/) >= 0.59 |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| UI | [NiceGUI](https://nicegui.io/) |
| Charts | [Plotly](https://plotly.com/python/) |
| Data I/O | xarray + NetCDF4 |
| Config | Pydantic v2 + PyYAML |
| Reports | Jinja2 |

## Upstream

Based on [GOTM](https://github.com/gotm-model/code) — the General Ocean Turbulence Model (Fortran 90).

## License

GPL-2.0 — same as the original GOTM Fortran model.

## Roadmap

- [x] Phase 1: Numba infrastructure and core utility kernels
- [x] Phase 2: Compiled Couette/channel runner for core emitted fields
- [ ] Phase 3: Expand compiled runtime forcing and profile-input support
- [ ] Phase 4: All turbulence closures validated
- [ ] Phase 5: Multi-column Dask + Numba batch mode
- [ ] Phase 6: FastAPI + NiceGUI web application
- [ ] Phase 7: SaaS deployment
