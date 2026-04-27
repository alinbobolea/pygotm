# pyGOTM — Project Mission

## Mission
Convert the General Ocean Turbulence Model (GOTM, Fortran 90) into a modern,
GPU-accelerated Python platform. Serve oceanographers, limnologists, and aquaculture
site assessors through a browser-based SaaS product — no Fortran compiler needed.

## Product Scope
- Single-column 1D ocean/lake turbulence model (pure vertical)
- Multi-column GPU mode: thousands of independent columns solved in parallel (Taichi)
- Multiple turbulence closures: k-epsilon, GLS, k-omega, Mellor-Yamada 2.5, KPP
- Surface forcing: COARE bulk fluxes (Fairall), Kondo 1975, solar absorption (Jerlov)
- GOTM-compatible YAML config format (direct import of all 22 official test cases)
- NetCDF output (CF conventions, xarray-compatible)
- FastAPI REST API + WebSocket progress streaming
- NiceGUI browser UI with Plotly interactive charts
- Validation parity with Fortran GOTM 6.0.7: pass criterion `|a−b| ≤ max(1e-7×ref_range, 1e-12) + 1e-6×|b|`
- Stokes drift / wave-averaged equations (Craig-Banner BC, Stokes drift profiles)
- FABM biogeochemistry coupling (gotm_fabm interface)
- Extras: seagrass drag parameterization, sediment model
- Horizontal advection / 3D dynamics (post-single-column, requires external 3D driver)

## Design Philosophy
1. **Scientific accuracy is non-negotiable.** Every kernel must validate against
   Fortran GOTM using the range-aware combined tolerance:
   `|a−b| ≤ max(1e-7×ref_range, 1e-12) + 1e-6×|b|`
   The signal-range atol floor prevents false failures at near-zero field values
   while rtol=1e-6 governs where the signal is non-negligible.
   No accuracy tradeoffs for performance before parity is established.
2. **Reproducibility is a feature.** Every simulation result must be fully
   reproducible from its YAML config. No hidden state.
3. **The kernel is the product.** The API and UI are wrappers. Keep `pygotm/src/`
   pure (no I/O, no web concerns).
4. **Double precision everywhere.** Use `ti.f64` throughout — GOTM uses REAL(8).
   Never downcast to f32 without a documented, tested justification.
5. **Parallelism is horizontal, not vertical.** Each vertical column (nlev ≈ 100)
   is solved serially (Thomas algorithm is inherently serial). Parallelism comes
   from solving thousands of independent columns simultaneously on GPU.
6. **Fail loudly on science errors.** If a field goes NaN or violates physical
   bounds, raise immediately with a diagnostic message. Never silently continue.

## Performance Goals
| Scenario | Target |
|----------|--------|
| Single column, 1 year simulation (dt=3600s) | < 5 seconds CPU |
| 10,000 columns, 1 year (GPU) | < 60 seconds |
| Browser UI: case setup → first result | < 2 minutes |

---

## Source Layout and Translation Mapping

Every GOTM Fortran subroutine lives in:

```
gotm-model/code/src/<module>/<file>.F90
```

Each file translates **one-to-one** to a Python file:

```
src/pygotm/<module>/<file>.py
```

The folder structure from `gotm-model/code/src/` is **preserved exactly** in
`src/pygotm/`. Module names remain the same (airsea, meanflow, turbulence,
util, etc.). A corresponding test file lives at:

```
tests/<module>/test_<file>.py
```

### Full Fortran → Python file inventory

#### airsea/ (10 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| airsea.F90 | airsea.py | Surface flux dispatcher |
| airsea_fluxes.F90 | airsea_fluxes.py | Flux computations |
| airsea_variables.F90 | airsea_variables.py | Module-level state fields |
| albedo_water.F90 | albedo_water.py | Water surface albedo |
| fairall.F90 | fairall.py | COARE bulk flux (Fairall et al.) |
| humidity.F90 | humidity.py | Humidity thermodynamics |
| kondo.F90 | kondo.py | Kondo 1975 bulk flux |
| longwave_radiation.F90 | longwave_radiation.py | Longwave radiation |
| shortwave_radiation.F90 | shortwave_radiation.py | Shortwave / Jerlov |
| solar_zenith_angle.F90 | solar_zenith_angle.py | Solar geometry |

#### config/ (1 file)
| Fortran | Python | Purpose |
|---------|--------|---------|
| settings.F90 | settings.py | Runtime configuration |

#### cvmix/ (1 file)
| Fortran | Python | Purpose |
|---------|--------|---------|
| gotm_cvmix.F90 | gotm_cvmix.py | CVMix turbulence interface |

#### extras/ (2 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| seagrass/seagrass.F90 | seagrass/seagrass.py | Seagrass parameterization |
| sediment/sediment.F90 | sediment/sediment.py | Sediment model |

#### fabm/ (2 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| gotm_fabm.F90 | gotm_fabm.py | FABM biogeochemistry interface |
| gotm_fabm_input.F90 | gotm_fabm_input.py | FABM input reader |

#### gotm/ (6 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| cmdline.F90 | cmdline.py | Command-line parsing |
| diagnostics.F90 | diagnostics.py | Diagnostic output |
| gotm.F90 | gotm.py | Top-level orchestrator |
| main.F90 | main.py | Entry point |
| print_version.F90 | print_version.py | Version info |
| register_all_variables.F90 | register_all_variables.py | Variable registration |

#### input/ (2 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| input.F90 | input.py | Generic input dispatcher |
| input_netcdf.F90 | input_netcdf.py | NetCDF input reader |

#### meanflow/ (13 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| coriolis.F90 | coriolis.py | Coriolis forcing |
| external_pressure.F90 | external_pressure.py | Barotropic pressure gradient |
| friction.F90 | friction.py | Bottom friction |
| internal_pressure.F90 | internal_pressure.py | Baroclinic pressure gradient |
| meanflow.F90 | meanflow.py | Meanflow dispatcher + state |
| salinity.F90 | salinity.py | Salinity diffusion equation |
| shear.F90 | shear.py | Shear production |
| stratification.F90 | stratification.py | Stratification / buoyancy |
| temperature.F90 | temperature.py | Temperature diffusion equation |
| uequation.F90 | uequation.py | U-momentum equation |
| updategrid.F90 | updategrid.py | Grid update (z-levels) |
| vequation.F90 | vequation.py | V-momentum equation |
| wequation.F90 | wequation.py | Vertical velocity |

#### observations/ (4 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| analytical_profile.F90 | analytical_profile.py | Analytical T/S profiles |
| const_NNS.F90 | const_nns.py | Constant salinity stratification |
| const_NNT.F90 | const_nnt.py | Constant temperature stratification |
| observations.F90 | observations.py | Observational forcing dispatcher |

#### stokes_drift/ (3 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| stokes_drift.F90 | stokes_drift.py | Stokes drift dispatcher |
| stokes_drift_exp.F90 | stokes_drift_exp.py | Exponential profile |
| stokes_drift_theory.F90 | stokes_drift_theory.py | Theory-based profile |

#### turbulence/ (29 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| algebraiclength.F90 | algebraiclength.py | Algebraic length scale |
| alpha_mnb.F90 | alpha_mnb.py | Stability function parameters |
| cmue_a.F90 | cmue_a.py | Stability functions (method a) |
| cmue_b.F90 | cmue_b.py | Stability functions (method b) |
| cmue_c.F90 | cmue_c.py | Stability functions (method c) |
| cmue_d.F90 | cmue_d.py | Stability functions (method d) |
| cmue_d_h15.F90 | cmue_d_h15.py | Stability functions (H15 variant) |
| cmue_ma.F90 | cmue_ma.py | Stability functions (MA) |
| cmue_sg.F90 | cmue_sg.py | Stability functions (SG) |
| compute_cpsi3.F90 | compute_cpsi3.py | GLS c_psi3 coefficient |
| compute_rist.F90 | compute_rist.py | Richardson number at surface |
| dissipationeq.F90 | dissipationeq.py | Epsilon (dissipation) equation |
| epsbalgebraic.F90 | epsbalgebraic.py | Algebraic epsilon balance |
| fk_craig.F90 | fk_craig.py | Craig-Banner wave-breaking flux |
| genericeq.F90 | genericeq.py | GLS generic length-scale eq |
| gotm_lib_version.F90 | gotm_lib_version.py | Library version string |
| internal_wave.F90 | internal_wave.py | Internal wave parameterization |
| kbalgebraic.F90 | kbalgebraic.py | Algebraic TKE |
| kbeq.F90 | kbeq.py | TKE equation |
| lengthscaleeq.F90 | lengthscaleeq.py | Length-scale equation |
| omegaeq.F90 | omegaeq.py | k-omega equation |
| potentialml.F90 | potentialml.py | Potential mixed-layer depth |
| production.F90 | production.py | TKE shear/buoyancy production |
| q2over2eq.F90 | q2over2eq.py | Mellor-Yamada q²/2 equation |
| r_ratio.F90 | r_ratio.py | Richardson number ratio |
| tkealgebraic.F90 | tkealgebraic.py | Algebraic TKE (shortcut) |
| tkeeq.F90 | tkeeq.py | Full TKE equation |
| turbulence.F90 | turbulence.py | Turbulence dispatcher + state |
| variances.F90 | variances.py | T/S variance equations |

#### util/ (14 files)
| Fortran | Python | Purpose |
|---------|--------|---------|
| adv_center.F90 | adv_center.py | Centered advection |
| compilation.F90 | compilation.py | Build info |
| convert_fluxes.F90 | convert_fluxes.py | Unit conversion for fluxes |
| density.F90 | density.py | Seawater density (UNESCO EOS) |
| diff_center.F90 | diff_center.py | Diffusion on cell centers |
| diff_face.F90 | diff_face.py | Diffusion on cell faces |
| gotm_version.F90 | gotm_version.py | GOTM version string |
| gridinterpol.F90 | gridinterpol.py | Grid interpolation |
| lagrange.F90 | lagrange.py | Lagrange interpolation |
| ode_solvers.F90 | ode_solvers.py | ODE solvers |
| ode_solvers_template.F90 | ode_solvers_template.py | ODE solver template |
| time.F90 | time.py | Time utilities |
| tridiagonal.F90 | tridiagonal.py | Thomas algorithm |
| util.F90 | util.py | General utilities |

**Total: 87 Fortran files → 87 Python files**

---

## Translation Rules

### File-level structure
Each `src/pygotm/<module>/<file>.py` must:
1. Begin with a module docstring that reproduces **all Fortran comments** from the
   corresponding `.F90` file verbatim (as a triple-quoted string).
2. Declare Taichi fields at module scope or inside the module's `init()` function
   (never inside `@ti.func` or `@ti.kernel`).
3. Export a clean public API via `__all__`.

### FORTRAN comment preservation
Every `!` comment in the Fortran source becomes part of the Python docstring for
the corresponding function or module. Do not discard any comment — they carry
physical derivations, paper references, and boundary condition explanations that
are critical for scientific correctness.

```fortran
! Compute buoyancy frequency squared N² (s⁻²)
! Uses UNESCO equation of state linearized about reference T, S
subroutine stratification(...)
```

becomes:

```python
def stratification(...):
    """
    Compute buoyancy frequency squared N² (s⁻²)
    Uses UNESCO equation of state linearized about reference T, S
    """
```

### Data types
| Fortran | Taichi | Python |
|---------|--------|--------|
| `REAL(kind=rk)` / `REAL(8)` / `DOUBLE PRECISION` | `ti.f64` | `float` |
| `REAL(4)` | `ti.f32` | `float` |
| `INTEGER` | `ti.i32` | `int` |
| `LOGICAL` | `ti.i32` (0/1 in kernels) | `bool` |

### Taichi usage philosophy
Leverage Taichi to the **maximum beneficial extent** — wherever it provides
computational speedup across a wide range of devices (CPU SIMD, GPU, Metal, Vulkan).

**Use Taichi when:**
- The operation is a hot loop over vertical levels or horizontal columns (parallelisable).
- The computation is pure math with no Python I/O or object allocation.
- The function will be called millions of times per simulation (time-stepping, per-column physics).

**Do NOT use Taichi when:**
- The operation is inherently serial with complex Python control flow (config parsing, I/O).
- The function is called once (init, cleanup, diagnostics).
- Taichi overhead (kernel launch, JIT compile) would dominate for tiny one-shot operations.

The goal is a codebase where the entire physics hot-path is composed of `@ti.func`
primitives called from `@ti.kernel` entry points — making the multi-column GPU path
a natural extension with zero algorithmic changes.

### Subroutine classification
| Fortran role | Taichi decorator | Notes |
|---|---|---|
| Pure computation, no I/O | `@ti.func` | Called from `@ti.kernel` only |
| Time-stepping entry point | `@ti.kernel` | One per physics module |
| I/O / init / finalize | Plain Python method | On driver class |

### Array mapping
- Single-column: `REAL(rk), DIMENSION(0:nlev) :: u` → `u = ti.field(ti.f64, shape=nlev+1)`
- Multi-column: `REAL(rk), DIMENSION(0:nlev) :: u` → `u = ti.field(ti.f64, shape=(n_cols, nlev+1))`
- All fields declared in `src/pygotm/<module>/<module>_variables.py`

### Implicit solver pattern (Crank-Nicolson)
```python
# Fortran:                          # Taichi:
# a(k) = -...                       a_x[k] = -...
# c(k) = -...                       c_x[k] = -...
# b(k) = 1 - a(k) - c(k) + ...     b_x[k] = 1 - a_x[k] - c_x[k] + ...
# r(k) = ...                        r_x[k] = ...
# CALL Thomas(a,b,c,r,x,nlev)       solve_tridiagonal(a_x,b_x,c_x,r_x,x_field,nlev)
```

### Math function mapping
| Fortran | Taichi |
|---------|--------|
| `SQRT(x)` | `ti.sqrt(x)` |
| `ABS(x)` | `ti.abs(x)` |
| `MAX(a,b)` | `ti.max(a,b)` |
| `MIN(a,b)` | `ti.min(a,b)` |
| `EXP(x)` | `ti.exp(x)` |
| `LOG(x)` | `ti.log(x)` |
| `TANH(x)` | `ti.tanh(x)` |
| `SIGN(a,b)` | `ti.math.sign(b) * ti.abs(a)` |

### Multi-column (GPU) parallelism
For any `@ti.kernel` that advances a prognostic variable, add a column index `col`
as the outermost parallel loop axis. The vertical loop over `k` is sequential
(Thomas algorithm). Example:

```python
@ti.kernel
def step_uequation(n_cols: ti.i32, nlev: ti.i32, dt: ti.f64):
    for col in range(n_cols):           # parallel across GPU threads
        for k in range(1, nlev):        # serial — Thomas algorithm
            ...
```

---

## Execution Lessons from Option A
Apply these rules to every remaining phase:

- Keep one production physics path per hot module: Taichi workspace + `@ti.kernel` only. Do not keep NumPy shadow solvers for tridiagonal, diffusion, or time-stepping code. Single-column runs must use the same kernel with `n_cols=1`.
- Keep a Python reference function only when it is thin, stateless, and solver-free. If it duplicates solver logic, remove it after migration.
- Test kernels directly. Use shared Taichi fixtures/helpers, compare against analytic or reference results, and always add multi-column parity with `n_cols=1` plus at least one `n_cols>1` identical-column case.
- Fix interface and sign-convention mismatches before migration. Preserve Fortran argument lists and saved state exactly; concrete examples already encountered are `temperature(..., wflux, hflux, ...)`, the atmospheric `hflux` sign convention, and friction's first-call `u_taub`/`u_taubo` behavior.
- For strict typing on the current stack (`Taichi 1.7.4`, `Python 3.13`, `mypy` strict), do not annotate checked kernels directly with raw `ti.template()` or `ti.types.ndarray()`. Use the local typing bridge in `pygotm.taichi_typing` (`TemplateArg`, `NdarrayArg`, `ti_kernel`) for all source and test kernels.
- In modules that declare `@ti.kernel` or `@ti.func`, do not use `from __future__ import annotations`; Taichi needs real annotations at import time. Also do not write `-> None` on `@ti.kernel`; void kernels must omit the return annotation on this stack.
- Keep Taichi runtime ownership suite-safe in tests. A test-local fixture may call `ti.reset()` only if it also restores `ti.init()` before later files run; otherwise full-suite `pytest` will fail even when targeted test files pass.
- Protect stack-specific Taichi constraints with regression tests. Keep explicit checks that forbid postponed annotations in Taichi callable modules and forbid `-> None` on `@ti.kernel`.
- Run validation pytest passes with `-W error::RuntimeWarning` so upstream binary-compatibility issues, numerical warnings, and import-time runtime warnings fail loudly instead of being buried in otherwise green runs. Investigate and fix the warning source or add a narrow, justified filter only when the upstream condition is understood.
- End each migration step with a validation gate: targeted tests for touched files, full `uv run pytest -W error::RuntimeWarning`, relevant `uv run mypy` passes (`src/` and touched `tests/`), and explicit checks that deleted APIs and dual-path residue are gone.

---

## Implementation Plan

Translate files in dependency order (utilities first, physics last, drivers last).
Each phase must be fully tested and validated before the next phase begins.

### Phase 0 — Infrastructure (no Fortran translation)
0.1 Create `src/pygotm/` directory tree mirroring `gotm-model/code/src/`  
0.2 Add `__init__.py` files to each package  
0.3 Create `src/pygotm/fields.py` — base class for Taichi field collections  
0.4 Create `src/pygotm/constants.py` — physical constants with source citations  
0.5 Create `tests/` directory tree mirroring `src/pygotm/`  
0.6 Verify `uv run pytest` passes (empty test suite)

### Phase 1 — Utilities (`util/`)
Translate in order:

1.1 `tridiagonal.py` — Thomas algorithm (`@ti.func`, single- and multi-column)  
1.2 `diff_center.py` — diffusion on cell centers  
1.3 `diff_face.py` — diffusion on cell faces  
1.4 `adv_center.py` — centered advection scheme  
1.5 `density.py` — UNESCO seawater EOS (critical: validate constants against UNESCO 1983)  
1.6 `convert_fluxes.py` — unit conversions  
1.7 `gridinterpol.py` — vertical grid interpolation  
1.8 `lagrange.py` — Lagrange interpolation  
1.9 `ode_solvers.py` + `ode_solvers_template.py` — ODE integrators  
1.10 `time.py` — time utilities (Python datetime wrappers; no Taichi needed)  
1.11 `util.py` — miscellaneous helpers  
1.12 `compilation.py`, `gotm_version.py` — metadata stubs  

**Test requirement for each util file:** unit tests covering all code paths,
boundary values (k=0, k=nlev), and known analytic solutions where available.

### Phase 2 — Meanflow (`meanflow/`)
2.1 `meanflow.py` — state fields and dispatcher  
2.2 `updategrid.py` — z-level update  
2.3 `shear.py` — shear production M²  
2.4 `stratification.py` — N², buoyancy flux  
2.5 `uequation.py` — U-momentum (Crank-Nicolson implicit)  
2.6 `vequation.py` — V-momentum  
2.7 `wequation.py` — vertical velocity (continuity)  
2.8 `external_pressure.py` — barotropic pressure gradient  
2.9 `internal_pressure.py` — baroclinic pressure gradient  
2.10 `coriolis.py` — Coriolis rotation  
2.11 `friction.py` — bottom friction  
2.12 `temperature.py` — temperature diffusion equation  
2.13 `salinity.py` — salinity diffusion equation  

**Test requirement:** each equation validated against known analytic solutions
(e.g., Couette flow for U-momentum, linear stratification for N²).

### Phase 3 — Turbulence (`turbulence/`)
3.1 `turbulence.py` — state fields and dispatcher  
3.2 `production.py` — shear and buoyancy production  
3.3 `tkeeq.py` — TKE equation (k-epsilon base)  
3.4 `dissipationeq.F90` → `dissipationeq.py` — epsilon equation  
3.5 `kbeq.py` — TKE (alternative form)  
3.6 `lengthscaleeq.py` — length-scale equation  
3.7 `genericeq.py` — GLS generic equation  
3.8 `omegaeq.py` — k-omega equation  
3.9 `q2over2eq.py` — Mellor-Yamada q²/2  
3.10 `tkealgebraic.py`, `kbalgebraic.py`, `epsbalgebraic.py` — algebraic closures  
3.11 `algebraiclength.py` — algebraic length scale  
3.12 `cmue_a.py` … `cmue_sg.py` — all stability function variants  
3.13 `alpha_mnb.py`, `compute_cpsi3.py`, `compute_rist.py`, `r_ratio.py` — auxiliary  
3.14 `internal_wave.py` — IW parameterization  
3.15 `potentialml.py` — potential mixed-layer depth  
3.16 `fk_craig.py` — Craig-Banner wave-breaking BC  
3.17 `variances.py` — T/S variance equations  
3.18 `gotm_lib_version.py` — version stub  

**Test requirement:** each closure tested against the Couette flow case
(analytical turbulent viscosity known), each stability function tested against
tabulated values from the source paper.

### Phase 4 — Air-Sea (`airsea/`)
4.1 `airsea_variables.py` — state fields  
4.2 `humidity.py` — specific/relative humidity conversions  
4.3 `solar_zenith_angle.py` — solar geometry  
4.4 `albedo_water.py` — surface albedo  
4.5 `shortwave_radiation.py` — shortwave / Jerlov water types  
4.6 `longwave_radiation.py` — longwave (bulk formula)  
4.7 `fairall.py` — COARE 3.0 bulk flux (Fairall et al. 2003)  
4.8 `kondo.py` — Kondo 1975 bulk flux  
4.9 `airsea_fluxes.py` — flux dispatcher  
4.10 `airsea.py` — top-level air-sea interface  

**Test requirement:** each bulk formula tested against published reference values
from the respective papers; shortwave radiation tested against known clear-sky
solar irradiance at specific lat/lon/time.

### Phase 5 — Observations and Config (`observations/`, `config/`)
5.1 `observations.py` — observational forcing dispatcher  
5.2 `analytical_profile.py` — analytical T/S profiles  
5.3 `const_nns.py`, `const_nnt.py` — constant stratification  
5.4 `settings.py` — runtime configuration (bridge to Pydantic config)  

### Phase 6 — Input / Output (`input/`, `gotm/` diagnostics)
6.1 `input.py` — generic YAML/NetCDF input dispatcher  
6.2 `input_netcdf.py` — NetCDF reader (wraps xarray)  
6.3 `diagnostics.py` — diagnostic variable output  
6.4 `register_all_variables.py` — variable registration for output  

### Phase 7 — Single-Column Driver
7.1 `src/pygotm/driver.py` — GotmDriver orchestrating time loop for one column  
7.2 `src/pygotm/config.py` — Pydantic YAML config models  
7.3 Validation: run GOTM test cases from `gotm-model/cases-runs/` (starting with
    couette, channel, entrainment; expanding to all 22), compare every prognostic
    field to Fortran GOTM output using the range-aware combined tolerance
    (`max(1e-7×ref_range, 1e-12) + 1e-6×|b|`), produce `validation/report.html`  

### Phase 8 — Multi-Column Driver
8.1 `src/pygotm/driver_multi.py` — GotmDriverMulti (n_cols on GPU)  
8.2 Extend all Phase 1–4 kernels with column-parallel outer loop  
8.3 Benchmark: 10,000 columns × 1 year on GPU, target < 60 s  

### Phase 9 — API and UI (post-physics)
9.1 `api/` — FastAPI endpoints  
9.2 `ui/` — NiceGUI browser interface  

### Phase 10 — Extras (post-MVP)
10.1 `extras/seagrass/seagrass.py`  
10.2 `extras/sediment/sediment.py`  
10.3 `stokes_drift/` — three files  
10.4 `fabm/` — two stub files  

---

## Testing Requirements

### Per-file unit tests (`tests/<module>/test_<file>.py`)
Every translated Python file must have a corresponding test file. Tests must:

1. **Import and instantiate** — the module initializes without error
2. **Smoke test** — call every public function with valid inputs
3. **Physical bounds** — assert outputs are within physically meaningful ranges
4. **Analytic verification** — where analytic solutions exist, assert rtol ≤ 1e-10
5. **Boundary conditions** — test k=0 and k=nlev explicitly
6. **Edge cases** — zero gradient, neutral stratification, maximum/minimum values
7. **Multi-column parity** — single-column and multi-column (n_cols=1) give identical results
8. **NaN / Inf guard** — assert no NaN or Inf in outputs for any valid input set

Test files must NOT depend on integration state; all inputs are constructed
explicitly in each test function.

### Integration tests (`tests/integration/`)
After Phase 7, run each of the 22 GOTM reference cases:

```
gotm-model/cases-runs/
  couette/        blacksea/       channel/
  entrainment/    estuary/        flex/
  gotland/        lago_maggiore/  langmuir/
  liverpool_bay/  medsea_east/    medsea_west/
  nns_annual/     nns_seasonal/   ows_papa/
  plume/          resolute/       reynolds/
  rouse/          seagrass/       wave_breaking/
  asics_med/
```

For each case:
1. Load YAML config from `gotm-model/cases-runs/<case>/gotm.yaml`
2. Run pyGOTM simulation; save output to `validation/runs/<case>/<case>.nc`
3. Load Fortran GOTM reference NetCDF from `gotm-model/cases-runs/<case>/`
4. Compare all shared numeric variables using the range-aware pass criterion:
   `|a−b| ≤ max(1e-7×ref_range, 1e-12) + 1e-6×|b|`
   Variables absent from pyGOTM output (FABM, ice model) are reported as
   `SKIP`, not `FAIL`.
5. Report per-variable metrics on failure:
   `max_abs_err`, `max_rel_err` ⚠, `max_range_err`, `mean_abs_err`,
   `mean_rel_err`, `RMSE`, `NRMSE`, `ref_range`
   (`max_range_err = max_abs_err/ref_range` and `NRMSE = RMSE/ref_range`
   are the primary metrics — reliable at near-zero field values where
   `max_rel_err` is undefined)

Integration test file: `tests/integration/test_gotm_cases.py`
Standalone runner:     `validation/run_validation.py` (3 cases → 22 cases)
Report:                `validation/report.html` + `validation/results.json`

---

## Validation Protocol
After converting each Fortran module, run the validation suite and inspect the report:

```bash
uv run python validation/run_validation.py --cases couette,channel,entrainment
# or with --no-run to re-compare without re-running simulations
```

Pass criterion (applied per variable):
```
|a − b| ≤ max(1e-7 × ref_range, 1e-12) + 1e-6 × |b|
```
- `atol_var = 1e-7 × ref_range` — signal-range floor that prevents false failures
  when the reference field is near zero (e.g., velocity at rest, TKE at surface)
- `rtol = 1e-6 × |b|` — standard relative tolerance where the field is non-negligible

Reported metrics per variable (in `validation/report.html`):
| Metric | Formula | Notes |
|--------|---------|-------|
| `max_abs_err` | max\|a−b\| | Raw maximum absolute error |
| `max_rel_err` ⚠ | max(\|a−b\|/\|b\|) | Unreliable when b≈0 |
| **`max_range_err`** | max\|a−b\| / ref_range | **Primary metric** — range-normalised, always valid |
| `mean_abs_err` | mean\|a−b\| | Mean absolute error |
| `mean_rel_err` | mean(\|a−b\|/\|b\|) | Mean relative error |
| `RMSE` | √(mean(a−b)²) | Root mean square error |
| **`NRMSE`** | RMSE / ref_range | **Unified normalised RMSE** |
| `ref_range` | ref_max − ref_min | Signal range context |

FAIL the validation if any variable does not satisfy the pass criterion.

---

## Architecture Principles
- `src/pygotm/` — pure Taichi physics, mirrors `gotm-model/code/src/` folder structure
- `src/pygotm/driver.py` — orchestrates time loop, calls kernels, writes output (single column)
- `src/pygotm/driver_multi.py` — multi-column GPU driver
- `src/pygotm/config.py` — Pydantic models + YAML deserialization
- `api/` — FastAPI, depends on pygotm package only
- `ui/` — NiceGUI, depends on api/ via HTTP (not directly on pygotm)
- No circular imports. Dependency order: src/ → driver → config → api → ui

---

## Coding Standards
- Python 3.12+, Pydantic v2, FastAPI, NiceGUI, Taichi ≥ 1.7.4
- `uv` for all package management (never pip)
- `uv run ruff format .` + `uv run black .` before commits
- `uv run mypy src/` must pass (strict mode)
- `uv run mypy tests/` must pass when changing test helpers, Taichi test kernels, or typing infrastructure
- `uv run pytest` must pass before any commit
- Line length: 88 chars maximum
- Type hints on all public functions and methods
- Docstrings on all public APIs

---

## Known GOTM Sign Conventions

- **`hflux` (non-solar surface heat flux):** GOTM uses the **atmospheric sign convention**.
  `hflux > 0` means the ocean is **losing heat** (upward flux to atmosphere).
  `hflux < 0` means the ocean is **gaining heat** (downward flux from atmosphere).
  This is the opposite of the standard physical-oceanography convention.
  In `temperature.F90`: `DiffTup = -hflux/(rho0*cp)`, so a negative `hflux` (ocean
  gains heat) produces a positive Neumann BC value that warms the surface layer.

---

## Working Relationship with Claude
- Ask clarifying questions rather than inventing physics assumptions.
- Never invent a correlation constant or parameterization coefficient without citing the source paper.
- Prefer deterministic, auditable logic over opaque heuristics.
- Every physical constant in kernels must have a comment with its source and units.
- If a design decision affects numerical accuracy or reproducibility, explain the tradeoff.
- Superpowers artifacts (plans, specs, brainstorm outputs) live in `.superpowers/` — never in `docs/superpowers/`.
- **Before implementing any physics module, thoroughly review the corresponding Fortran source in `gotm-model/code/src/`.** This is the authoritative reference for algorithms, constants, array conventions, and boundary conditions. Never guess at physics logic — read the Fortran first.

---

# Development Guidelines

## Core Development Rules

1. Package Management
   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Running tools: `uv run tool`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality
   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 88 chars maximum

3. Testing Requirements
   - Framework: `uv run pytest`
   - Validation and regression sweeps: run `uv run pytest -W error::RuntimeWarning` so runtime warnings fail the suite until investigated and fixed
   - Async testing: use anyio, not asyncio
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Code Style
    - PEP 8 naming (snake_case for functions/variables)
    - Class names in PascalCase
    - Constants in UPPER_SNAKE_CASE
    - Document with docstrings
    - Use f-strings for formatting

- For commits fixing bugs or adding features based on user reports add:
  ```bash
  git commit --trailer "Reported-by:<name>"
  ```
  Where `<name>` is the name of the user.

- For commits related to a Github issue, add
  ```bash
  git commit --trailer "Github-Issue:#<number>"
  ```
- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organisation**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## Code Formatting

1. Linter
   - Format: `uv run ruff format .`
   - Check: `uv run ruff check .`
   - Fix: `uv run ruff check . --fix`
   - Critical issues:
     - Line length (88 chars)
     - Import sorting (I001)
     - Unused imports

2. Type Checking
   - Tool: `uv run mypy .`
   - Requirements:
     - All functions must have explicit type annotations
     - Explicit None checks for Optional types (no implicit assumptions)
     - Use proper type narrowing (e.g., isinstance checks)
     - Avoid `Any` unless absolutely necessary
     - Third-party libraries must have type stubs or be explicitly ignored

3. Code Formatting
   - Tool: `uv run black .`
   - Requirements:
     - All Python code must be formatted with black before commit
     - No manual formatting that conflicts with black
     - Line length must follow project configuration

4. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Prettier (YAML/JSON), Ruff (Python)
