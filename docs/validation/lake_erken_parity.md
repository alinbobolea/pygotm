# Lake Erken — Parity Status and Limitations

**Status: PARTIAL PARITY (`FAIL`).** The hypsography / inflow / outflow /
water-balance is at **exact parity**. The residual `FAIL` has **two independent
causes**: (1) a **mild, seasonal under-mixing of the summer thermocline** — a
real but small physical difference affecting the native turbulence/temperature
fields — and (2) the `selmaprotbas` biogeochemistry (BGC) is limited by a FABM
library build flag. This page records exactly what is at parity and what is not,
with source-level and data-level evidence, so anyone picking up the project knows
precisely where it stands.

## The case

| Property | Value |
|---|---|
| Period / step | 1999-02-01 → 2020-12-31 (~22 yr), 3600 s hourly (~192k steps) |
| Grid | 21 m, 42 layers, hypsographic (cross-sectional area varies with depth) |
| Turbulence | **second-order** closure (`turb_method=3`, `tke_method=2`, `len_scale_method=8`, `scnd_method=1` quasi-equilibrium, `scnd_coeff=7` = Cheng et al. 2002) |
| Light | very turbid (`light_extinction` custom: `A=0.58`, `g1=0.5 m`, `g2=1.74 m` — almost all shortwave absorbed in the top ~2 m → a sharp, shallow summer thermocline) |
| BGC | FABM `selmaprotbas` — 13 interior + 5 benthic state variables |
| Output | **daily means** (`time_method: mean`) |
| FABM numerics | `repair_state: true`, `bottom_everywhere: true`, feedbacks (shade/albedo/drag) **off** |
| Lineage | reference built from **gotm-lake** (older GOTM fork + its own bundled FABM); pyGOTM derives from **gotm-model** + conda `pyfabm 3.0.0` |

Because pyGOTM and the reference come from different lineages, mismatches may
originate in gotm-model vs gotm-lake, the translation, the pyFABM coupling, the
FABM library build, or the reporting — not every mismatch is a pyGOTM bug.

## Variable inventory (current)

114 native variables and 41 FABM variables are compared. Native:
**60 PASS / 5 MARGINAL / 23 DISCREPANT / 26 BROKEN**; FABM: **0 / 0 / 3 / 38**.
All 64 BROKEN variables are in `lake_erken`; no other case has any BROKEN.

The 26 **native** BROKEN variables all trace to **one** root residual — the
summer-thermocline under-mixing described below — either directly or through a
sensitive derived/diagnostic quantity. They split as:

- **Core turbulence/stratification, just over the 0.20 threshold:** `num` (0.245),
  `nuh`/`nus` (0.280), `cmue2` (0.238), `tke` (0.208), `as` (0.208), `SS` (0.217),
  `NN`/`NNT` (0.203), `G` (0.217), `avh` (0.288), `P` (0.323).
- **Sensitive derived diagnostics that amplify the same residual:** `Rig` (0.93 —
  `NN/SS`, which blows up wherever `SS → 0`), the Reynolds-stress diagnostics
  `uu`/`vv`/`ww` (0.75–0.79), and the SST-driven surface turbulent heat fluxes
  `qe`/`qh` (0.79/0.68) plus the ice fields `Hfrazil`/`Tice_surface`.
- **Momentum:** `u` (0.218), `taux`/`tauy` (0.26/0.30), `Ekin` (0.23).
- **Mixed-layer-depth diagnostics:** `mld_bott` (0.54), `mld_surf` (0.45) — see
  the diagnostic-method note below.

For calibration: `temp` itself is only **DISCREPANT** (`d_norm = 0.123`),
`rho = 0.141`, `eps = 0.101`. The BROKEN turbulence variables sit *just* above the
0.20 line because the normalized Fréchet metric is sensitive for these
floor-dominated, multi-decade fields; the underlying physical difference is small.

## What is at exact parity (verified)

- **Hypsography / water-balance is exact** (`d_norm = 0.0`): `Af`, `Qlayer`,
  `Qs`, `wq`, `FQ`, `Qres`, `zeta`, `h`, `salt`, and the integrated
  `int_precip` / `int_evap` / `int_inflow` / `int_outflow` / `int_water_balance`
  terms (the last three PASS within `d_norm < 0.003`). The momentum equations and
  bottom friction are hypsography-aware (area-weighted diffusion/advection, `wq`
  advection, per-layer bottom drag), mirroring gotm-lake. This is a genuine and
  significant accomplishment.

## The native residual: a mild, seasonal summer-thermocline under-mixing

A full-record (8005-day) comparison of the pyGOTM run against the reference
NetCDF localises the residual precisely. It is **seasonal**, not catastrophic:

| Season | temp \|err\| | interior `num` ratio (py/ref) | stratification (surface − bottom) |
|---|---|---|---|
| Winter / ice (Nov–Mar) | 0.06–0.27 °C | ~0.88–1.00 | matches |
| Summer stratified (Jun–Aug) | **0.86–1.38 °C** | **0.44–0.68** | py **~2 °C too strong** |

- **Annual** temp mean-abs-error is **0.55 °C** (median 0.35 °C). The largest
  single point is 6.8 °C, but that is a >99.9th-percentile transient: only ~2 % of
  points exceed 3 °C and ~0.2 % exceed 5 °C.
- **The signed bias is a clean dipole:** surface **+0.49 °C** (too warm), bottom
  **−0.20 °C** (too cold) on the annual mean; in midsummer the surface runs
  ~+1 °C and the bottom ~−1 to −1.5 °C. A representative 1-Aug profile: pyGOTM
  bottom 12.4 °C / surface 22.0 °C (ΔT 9.6 °C) vs reference 13.9 °C / 20.9 °C
  (ΔT 7.0 °C).
- The `num` deficit is concentrated at the **thermocline depth** (worst around
  interface level 32; near-surface and near-bed diffusivity match) and only in
  the **stratified months**.
- The **velocity field matches** the reference to ~90 % (near-bottom mean speed:
  pyGOTM `3.8e-3` vs reference `4.1e-3`), and the columns agree in winter and
  below the thermocline.

**Mechanism.** During the stratified season pyGOTM holds a **sharper, shallower
thermocline** than the reference: the cross-thermocline turbulent diffusivity is
lower, so less heat mixes downward, the surface stays warmer, the bottom stays
cooler, and the stratification is ~2 °C too strong. In winter, and below the
thermocline, the columns agree. Lake Erken is the case most exposed to this:
it is a shallow (21 m), very turbid lake whose shortwave is absorbed almost
entirely in the top ~2 m (`g2 = 1.74 m`), producing an unusually sharp, shallow
summer thermocline where any small cross-thermocline mixing difference is
maximally amplified. The same closure is only mildly different (MARGINAL/
DISCREPANT) in the deeper, weakly-stratified second-order ocean cases
(`gotland`, `nns_annual`, `resolute`).

## The second-order closure is identical between the lineages

The residual is **not** a closure-coefficient difference. A direct source diff
(`gotm-lake/.../turbulence` vs `gotm-model/code/src/turbulence`) shows the
closure is the same in both lineages:

| Closure component | gotm-lake vs gotm-model |
|---|---|
| Stability functions `cmue_d.F90`, `cmue_c.F90` | **byte-identical** |
| Cheng et al. 2002 coefficients (`CCH02`: `cc1..cc6`, `ct1..ct5`, `ctt`) | **byte-identical** |
| `alpha_mnb.F90`, `variances.F90` | identical **except** Stokes/Langmuir terms (`av`, `aw`, `Px`), which are **off** for this case |
| `turbulence.F90` init differences | all in **gotm-model-only additions** (k-omega namelist, Stokes vectors, renamed enums) that are inactive here |

Every closure knob is **correctly plumbed** from the YAML at runtime (verified
through the driver): `scnd_method = 1` (quasi-equilibrium), `scnd_coeff = 7`
(Cheng), `length_lim = true`, `galp = 0.53`, `compute_c3 = true`,
`Ri_st = 0.25`, `stab_method = 3`. Internal-wave mixing is **off in both** runs:
the case's `iw:` block sets only `alpha = 0.7` with no `method`, so `iw_model = 0`
in pyGOTM, and gotm-lake's `internal_wave.F90` only acts for `iw_model == 2`. The
reference's interior `num` shows **no** ~`1e-4` floor, confirming IW mixing is
inactive.

**Interpretation.** With identical closure coefficients and identical, correctly
applied configuration, the residual is most consistent with a small **numerical**
difference in how the two codes resolve a sharp, shallow thermocline
(advection/diffusion discretisation, accumulated over 22 years) — pyGOTM holds
the slightly sharper gradient. There is **no evidence-justified code fix**: the
closure is shared with the 14 passing ocean cases, and forcing pyGOTM to match
the reference here would mean **adding spurious cross-thermocline mixing**, which
would degrade those cases. It is recorded as a documented, bounded lineage
difference.

## Mixed-layer-depth diagnostic: a method-default difference (now configurable)

`mld_bott` and `mld_surf` are **post-hoc diagnostics** (`diagnostics.F90`,
`mld_method`); they do not feed back into the physics. Their `d_norm` reflects a
**diagnostic-method default difference** between the lineages:

- gotm-lake defaults `mld_method = 1` (TKE criterion, which also computes the
  *bottom* mixed layer); gotm-model (pyGOTM's basis) defaults `mld_method = 2`
  (critical-Richardson, which computes only the surface MLD and leaves
  `mld_bott = 0`). The case YAML overrides neither, so the **reference ran method
  1** (confirmed: its `mld_bott` reaches 21 m) while gotm-model's default is
  method 2 (`mld_bott ≡ 0`).
- pyGOTM now supports an optional **`mld:` block** (`method`/`diff_k`/`Ri_crit`),
  defaulting to the gotm-model value so the other 22 cases are unchanged. The
  lake_erken config sets `mld: method: 1` to use the **same diagnostic as the
  reference**. This is a documented input-config setting (the reference output is
  untouched); it is diagnostic-only and does not alter any physics field.
- Effect (verified): `mld_bott` `d_norm` is `0.5423` and `mld_surf` `0.4477` under
  method 1 (versus `1.0000` and `0.5950` under the gotm-model default). Both
  remain BROKEN, because once computed by the reference's method they faithfully
  reflect the **same** summer-thermocline `tke` residual as the other turbulence
  variables — they are part of the single coherent story rather than a separate
  artifact. The case verdict and the 26-native-BROKEN count are unchanged.

## FABM coupling fixes already applied (no library rebuild)

Two correct, gated fixes live in `src/pygotm/fabm/fabm_loop.py` (only `lake_erken`
is affected; the other 22 cases are byte-identical):

1. **Daily-mean FABM output** (gated on `output_reduce_mode`): accumulate FABM
   state and diagnostics over each output window and write the window mean,
   instead of an instantaneous local-midnight snapshot that biased diurnal
   quantities.
2. **`repair_state` clipping** (gated on the `fabm: repair_state` flag): call
   `model.check_state(repair=True)` after transport and after the source update,
   matching gotm-lake `do_repair_state`. This eliminates a catastrophic
   `selmaprotbas_po` phosphate NaN (verified: 0 NaN in the current output).

## pyfabm limitations preventing complete BGC validation

Two limitations of the stock conda `pyfabm` library (vs the custom FABM that
gotm-lake bundles) mean the `selmaprotbas` biogeochemistry **cannot** reach full
parity with the reference, independent of the native under-mixing above.

**1. `variable_bottom_index` (benthic coupling).** The worst FABM variables are
sediment-coupled (`selmaprotbas_o2`, `DO_mg`, benthic-regenerated nutrients).
gotm-lake's `bottom_everywhere` distributes the benthic flux over the whole
sloping bed (`rhs(k) *= (Af(k)-Af(k-1))/Vc(k)` per layer, via
`set_bottom_index(k)`), which requires a FABM library compiled with
**`_FABM_BOTTOM_INDEX_ = -1`** (variable bottom index). The conda
**`pyfabm 3.0.0` is built with the default `0`**; `link_bottom_index(...)` raises
`"… compiled without support for variable bottom indices"`. The reference runs
`bottom_everywhere` only because gotm-lake builds its own FABM with that flag.

**2. `selmaprotbas` `alpha`/`beta` phytoplankton parameters (interior BGC).** The
reference `fabm.yaml` sets two GOTM-lake-only parameters on each
`selmaprotbas/phytoplankton` instance — `alpha` (nutrient-uptake
half-saturation) and `beta` (temperature growth-correction). Conda `pyfabm` does
not declare them, so loading the reference YAML verbatim raises an "invalid
configuration" error. pyGOTM therefore **strips** `alpha`/`beta` (see
`src/pygotm/fabm/config.py` → `_normalized_fabm_config_path`) and runs with the
upstream defaults, so the interior phytoplankton growth differs from the
GOTM-lake-tuned reference. This is why the bundle's `fabm.yaml` is a
*materialized* (stripped) copy rather than a byte copy of the reference — the
validation runner stages exactly that materialized YAML, and its hash is
recorded as `fabm_yaml_sha256`, so re-running the bundle is reproducible.

**Decision: do not rebuild pyfabm locally** (conda-forge divergence + maintenance
burden on every FABM bump; the variable-bottom build is non-default for the
python host and unverified; per-layer evaluation costs `nlev × getRates` per
step; and even with both limitations resolved, full PASS is unlikely given the
native under-mixing above). The actionable path is upstream: a conda-forge
`pyfabm` variant that enables `variable_bottom_index` and exposes the
`selmaprotbas` `alpha`/`beta` parameters.

## Validation reporting — variable ownership

`PYGOTM_VARIABLES` (the native-vs-FABM section classifier in
`src/pygotm/validation/tolerances.py`) lists the native fields. It mirrors
`register_all_variables.py` plus the runtime lake/air-sea outputs (air-sea, ice,
observation, energetics and lake-hypsography fields are native). Normalization is
**decoupled** from ownership — full-range normalization is reserved for the
floor-dominated mean-flow/turbulence fields — so section labels and the
normalization choice are independent and every reported `d_norm` is unaffected by
the classification.

## Reproduction

```bash
# Re-run lake_erken and regenerate its bundle (~5 min):
conda run -n pygotm python -m pygotm.validation.run_validation --cases lake_erken
# Re-compare existing NetCDFs without re-running:
conda run -n pygotm python -m pygotm.validation.run_validation --all --no-run
```

Localization diagnostic: compare `temp`, `num`, `NN` at depth between
`validation/runs/lake_erken/lake_erken.nc` and
`validation/reference/lake_erken/output.nc`, grouped by month — the columns track
in winter and below the thermocline; the divergence is the summer thermocline.

## Bottom line

Hypsography / water-balance: **exact parity**. Native physics: a **mild, seasonal
under-mixing of the summer thermocline** (~0.5 °C annual-mean temp error;
stratification ~2 °C too strong in midsummer). The second-order closure source is
byte-identical between the lineages and every closure knob is correctly plumbed,
so the residual is a small numerical difference in resolving a sharp, shallow
thermocline; there is no safe, evidence-justified code fix and the shared closure
must not be distorted toward the reference. The `mld_bott`/`mld_surf` diagnostics
use the reference's `mld_method` via the new `mld:` config block. BGC: two correct
FABM-coupling fixes landed; the dominant FABM blocker is `bottom_everywhere`,
which needs a FABM library compiled with `variable_bottom_index`. The overall
case remains **FAIL**; the physics is **close but not at parity**.
