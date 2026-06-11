# Lake Erken — BGC Parity Status and Limitations

**Status: PARTIAL PARITY (`FAIL`).** The hypsography / inflow / outflow /
water-balance is at **exact parity** and the physics trajectory tracks the
reference closely. The residual `FAIL` is confined to the `selmaprotbas`
biogeochemistry (BGC) and is dominated by a single FABM-library build
limitation. This page records what is at parity, the fixes applied, and the
exact blocker, so anyone picking up the project knows precisely where it stands.

## The case

| Property | Value |
|---|---|
| Period / step | 1999-02-01 → 2020-12-31 (~22 yr), 3600 s hourly (~192k steps) |
| Grid | 21 m, 42 layers, hypsographic (cross-sectional area varies with depth) |
| Turbulence | k-ε (`tke_method=2`, `len_scale_method=8`, Schumann-Gerz stability) |
| BGC | FABM `selmaprotbas` — 13 interior + 5 benthic state variables |
| Output | **daily means** (`time_method: mean`) |
| FABM numerics | `ode_method=3` (RK4), `repair_state: true`, `bottom_everywhere: true`, feedbacks (shade/albedo/drag) **off** |
| Lineage | reference built from **gotm-lake** (older GOTM fork + its own bundled FABM); pyGOTM derives from **gotm-model** + conda `pyfabm 3.0.0` |

Because pyGOTM and the reference come from different lineages, mismatches may
originate in gotm-model vs gotm-lake, the translation, the pyFABM coupling, the
FABM library build, or the reporting — not every mismatch is a pyGOTM bug.

## What is at parity (the accomplishment)

- **Hypsography / water-balance is exact** (`d_norm = 0.0`): `Af`, `Qlayer`,
  `Qs`, `wq`, `FQ`, `Qres`, `zeta`, `h`, `salt`, plus the integrated
  `int_precip` / `int_evap` / `int_water_balance` terms. The momentum equations
  and bottom friction were also made hypsography-aware (area-weighted
  diffusion/advection, `wq` advection, per-layer bottom drag), mirroring
  gotm-lake.
- **The physics is excellent.** Direct NetCDF comparison gives surface
  temperature correlation **1.000** (the 0.12 `d_norm` is a small +0.5 °C warm
  bias, not a shape divergence) and surface `selmaprotbas_o2` correlation 0.972.
- All FABM→physics feedbacks are **off**, so physics is a genuine one-way driver
  of the BGC and pyGOTM's offline FABM coupling is architecturally valid here.
  **The BGC failures are FABM-coupling issues, not physics divergence.**

## Fixes applied (no library rebuild required)

Both live in `src/pygotm/fabm/fabm_loop.py` and are gated so that **only
lake_erken** is affected (all 22 other cases are byte-identical):

1. **Daily-mean FABM output** (gated on `output_reduce_mode`). The physics loop
   already averaged to daily means; the FABM loop was writing the instantaneous
   value at the output step — a **local-midnight snapshot** (the run starts at
   00:00 with hourly steps), which systematically biases diurnal quantities
   (GPP/NPP = 0 at night, O₂ at its daily minimum). The fix accumulates state and
   diagnostics over each output window and writes the window mean; the
   initial-condition slot stays instantaneous, matching GOTM.
2. **`repair_state` clipping** (gated on the `fabm: repair_state` flag, which
   only lake_erken sets). The reference clips state to `[minimum, maximum]` after
   transport and after integration (`do_repair_state` → `fabm_check_state`);
   pyGOTM's offline loop did not. Without it the explicit update drove
   `selmaprotbas_po` (phosphate) negative and the rate laws emitted **NaN by day
   ~96** (332,178 / 336,210 = 99 % of phosphate values). The fix calls
   `model.check_state(repair=True)` at both points.

**Before → after (lake_erken):**

| | PASS | MARGINAL | DISCREPANT | BROKEN |
|---|---|---|---|---|
| Baseline | 60 | 5 | 23 | 67 |
| mean + repair | 60 | 5 | 26 | **64** |

- Phosphate NaN eliminated (332,178 → 0; `po` now physical, 0.002–3.83).
- 25 BGC variables improved (`po`/`Pho` 0.86→0.28, `pb` 0.61→0.31, `PBR`
  0.80→0.55, `total_nitrogen` 0.62→0.43); 3 crossed BROKEN→DISCREPANT
  (`dd_n`, `dd_c`, `dd_p`); **0 status regressions**.
- Protected water-balance vars unchanged (exact); all other cases identical to
  baseline.
- Tests: `tests/fabm/test_fabm_loop.py` (+5 deterministic tests). Gate:
  `pytest -W error::RuntimeWarning` 1395 passed, `mypy src/` clean, `ruff` clean.

## The dominant blocker — `bottom_everywhere` needs a FABM build flag

The worst remaining variables are all **sediment-coupled**: `selmaprotbas_o2`
(0.91), `DO_mg` (0.89), `DNP` (1.0), and the benthic-regenerated nutrients. The
O₂ depth profile pins the cause:

```
selmaprotbas_o2, pyGOTM vs reference daily means (per depth):
  bottom : corr 0.82, +24–27 µmol/L too high
  mid    : corr 0.83, +13 too high
  surface: corr 0.98,  −7
```

pyGOTM's **hypolimnion O₂ is systematically too high** while the **surface tracks
at corr 0.98** — the signature of sediment O₂ demand applied at one layer instead
of distributed over the sloping bed.

gotm-lake's `bottom_everywhere` distributes the benthic flux over the whole bed:
for each layer `k`, `rhs(k) *= (Af(k)-Af(k-1))/Vc(k)` with per-layer benthic
state, via `set_bottom_index(k)` + per-layer `fabm_do_bottom` and
`kmax_bot = nlev`. The canonical port (loop `k=1..nlev`, `link_bottom_index(k)`,
per-layer `do_bottom`) requires the FABM library compiled with
**`_FABM_BOTTOM_INDEX_ = -1`** (variable bottom index). The conda
**`pyfabm 3.0.0` is built with the default `0`** — `variable_bottom_index` is a
read-only compile-time property and `link_bottom_index(...)` raises
`"the underlying FABM library has been compiled without support for variable
bottom indices"`. The reference runs `bottom_everywhere` only because gotm-lake
builds its **own** FABM with that flag.

**Decision: do not rebuild pyfabm locally.** A patched custom conda package would
diverge from conda-forge and add a maintenance burden on every FABM bump; the
variable-bottom build is non-default for the python host and unverified; the
per-layer evaluation costs `nlev × getRates` per step (~8M for this case); and it
would still likely not deliver full PASS (§ below). **Upstream ask instead:** a
conda-forge `pyfabm` variant (or documented build option) with
`variable_bottom_index` enabled, after which the per-layer loop drops in with no
conda divergence.

## Secondary residuals (available without a rebuild, deferred)

Real gotm-lake-vs-pyGOTM differences that could be implemented with no library
change but cannot change the verdict while `bottom_everywhere` stands, and which
the shape-based Frechet metric is largely insensitive to:

- **RK4 vs Euler** (`ode_method=3`) — better matches the nutrient integration;
  config-isolated to lake_erken.
- **FABM settling/diffusion area-weighting** (`Vc`/`Af`) — a consistency fix
  (physics T is already area-weighted); but the O₂ bias is sediment *demand*,
  not mixing.

## Would a rebuild achieve full parity?

Only partially. `bottom_everywhere` would attack the worst variables and move
many toward parity, but **full PASS (all variables `d_norm < 0.05` over 22 yr) is
unlikely** even then, given the small physics warm bias, the §-above integration
and transport differences, and the inherent difficulty of matching a 22-year BGC
*shape* across two model lineages. It is **necessary but probably not
sufficient** — which, with the cost above, is why a local rebuild is not pursued.

## Reproduction

```bash
# lake_erken only (≈4–5 min):
conda run -n pygotm python -m pygotm.validation.run_validation \
  --cases lake_erken --report-dir /tmp/val_le --output-dir validation
# re-compare existing NetCDFs without re-running:
conda run -n pygotm python -m pygotm.validation.run_validation --all --no-run
```

Localization diagnostic: compare `temp`, `selmaprotbas_o2`, `selmaprotbas_po`
between `validation/runs/lake_erken/lake_erken.nc` and
`validation/reference/lake_erken/output.nc` — physics tracks at corr ≈ 1.0; the
BGC divergence is at depth (sediment).

## Bottom line

Hypsography / water-balance: **exact parity**. Physics: **excellent**
(temp corr 1.000). BGC: two correct, isolated coupling fixes landed (daily-mean
output; `repair_state`, which eliminates a catastrophic phosphate NaN and
improves 25 variables) with no impact on other cases. The remaining
`selmaprotbas` failures are dominated by **`bottom_everywhere`**, which requires a
FABM library compiled with `variable_bottom_index` — a capability the conda
`pyfabm 3.0.0` lacks and that we choose not to rebuild locally. The actionable
path is upstream.
