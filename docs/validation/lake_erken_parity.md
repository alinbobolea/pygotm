# Lake Erken — Parity Status and Limitations

**Status: PARTIAL PARITY (`FAIL`).** The hypsography / inflow / outflow /
water-balance is at **exact parity**. The residual `FAIL` has **two independent
causes**: (1) the native physical column is **under-mixed** — a real turbulence
divergence affecting 26 native variables — and (2) the `selmaprotbas`
biogeochemistry (BGC) is limited by a FABM-library build flag. This page records
exactly what is at parity and what is not, with source-level evidence, so anyone
picking up the project knows precisely where it stands.

> **Correction (supersedes earlier notes).** An earlier version of this page
> claimed "the physics is excellent" and that "the BGC failures are FABM-coupling
> issues, not physics divergence." That was based on **surface** quantities only
> and is **incorrect**. Direct full-depth comparison shows the column is
> systematically under-mixed; 26 *native* (non-FABM) variables are BROKEN. The
> assessment below is the corrected, evidence-based picture.

## The case

| Property | Value |
|---|---|
| Period / step | 1999-02-01 → 2020-12-31 (~22 yr), 3600 s hourly (~192k steps) |
| Grid | 21 m, 42 layers, hypsographic (cross-sectional area varies with depth) |
| Turbulence | **second-order** closure (`turb_method=3`, `tke_method=2`, `len_scale_method=8`, `scnd_coeff=7` = Cheng et al. 2002) |
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

The 26 **native** BROKEN variables are the under-mixing cascade — turbulence
(`num`, `nuh`, `nus`, `avh`, `tke`, `cmue2`, `as`, `P`, `G`, `Pb`→via `Rig`,
`SS`, `NN`, `NNT`), momentum (`u`, `uu`, `vv`, `ww`, `taux`, `tauy`, `Ekin`),
mixed-layer depth (`mld_surf`, `mld_bott`), and the downstream surface-flux / ice
fields (`qe`, `qh`, `Hfrazil`, `Tice_surface`). The 38 FABM BROKEN variables are
the `selmaprotbas` benthic/biogeochemistry fields.

## What is at exact parity (verified)

- **Hypsography / water-balance is exact** (`d_norm = 0.0`): `Af`, `Qlayer`,
  `Qs`, `wq`, `FQ`, `Qres`, `zeta`, `h`, `salt`, and the integrated
  `int_precip` / `int_evap` / `int_inflow` / `int_outflow` / `int_water_balance`
  terms (the last three PASS within `d_norm < 0.003`). The momentum equations and
  bottom friction are hypsography-aware (area-weighted diffusion/advection, `wq`
  advection, per-layer bottom drag), mirroring gotm-lake. This is a genuine and
  significant accomplishment and is **not** disturbed by any work here.

## Native physics: the column is under-mixed (the real native blocker)

Direct daily-mean NetCDF comparison localises a clear first divergence:

- **Day 0 (init): identical** (`NN`, `num`, `temp` match exactly — init is
  correct).
- **Day 1: already diverged at depth.** `temp` matches *exactly* in the lower
  column (levels 0–20) but diverges in the upper ~8 m; `rho` matches wherever
  `temp` matches, so **the equation of state is correct**. At mid/upper depth the
  turbulent diffusivity `num` is **~40× too small** and the column is ~2× too
  stratified.
- **By day 3, near the bed:** shear `SS` is ~0 in pyGOTM (`1e-15` vs ref
  `1.4e-5`), shear production `P` is ~0, `tke` is stuck at its floor; the
  near-bottom velocity is pinned (`u` ≈ `-3e-8` vs ref `-2e-3`).

**Mechanism.** pyGOTM's deep turbulence fails to ignite from day 1 → the column
over-stratifies → mixing is suppressed → momentum cannot penetrate → there is no
deep shear → it stays under-mixed (a self-reinforcing under-mixed state). The
consequences cascade: deep `temp` is up to ~7 °C too warm, `mld_bott` collapses
to 0 (ref is fully mixed bottom-up, 21 m), and the surface heat-flux and ice
fields drift. The over-large `drag` is a *consequence* of the pinned near-bottom
velocity (small `u_taub` → the hydraulically-smooth roughness term saturates →
large `rr²`), not an independent bug — drag is quadratic in velocity and cannot
pin a near-zero velocity.

**Source trace — every lake-specific kernel was checked and matches gotm-lake:**

| Subsystem | pyGOTM | vs gotm-lake |
|---|---|---|
| Area-weighted scalar diffusion | `diff_center_lake` | matches (interior + both Neumann BCs) |
| Lake momentum + per-layer bottom drag | `step_uequation_lake_single` / `friction` | matches |
| Lake temperature / salinity | dispatched when `lake != 0` | matches |
| Shear, production, TKE/eps equations | shear / production / tkeeq / dissipationeq | matches (only Stokes terms differ, off here) |
| EOS / density / `NN` | — | correct (ρ matches where T matches) |
| Turbulence dispatch | `turb_method=3` → second-order Cheng | correctly selected; `cmue_d`/`cmue_c` identical |

**Classification (C / B).** The lake-*specific* physics is correctly ported and
verified. The residual under-mixing lives in the **second-order turbulence
closure internals** (the coefficient setup in `alpha_mnb` / `turbulence.F90`),
where the **gotm-lake reference (older fork) differs from gotm-model (pyGOTM's
basis)**, amplified by the stably-stratified, ice-covered, low-energy regime that
is bistable between a mixed and an under-mixed state over 22 years. A smaller
component of the same closure difference is visible (mildly, MARGINAL/DISCREPANT)
in the other second-order ocean cases (`gotland`, `nns_annual`, `resolute`),
which use gotm-model references — a separate, pre-existing discrepancy. The
closure is **shared** with the 14 passing ocean cases, so it must **not** be
distorted toward the older lineage without sub-daily A/B evidence; doing so risks
regressing those cases.

**Exact next step (open):** instrument a sub-daily A/B of the turbulence state
(`tke`, `eps`, `cmue1/2`, `L`, `num`) over days 0–3 to isolate the precise
`alpha_mnb`/closure-init coefficient difference, then decide between (a) a
config-gated gotm-lake second-order variant for lake runs, or (b) accepting it as
a documented lineage limitation.

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

## Validation reporting — variable-ownership correction

`PYGOTM_VARIABLES` (the native-vs-FABM section classifier in
`src/pygotm/validation/tolerances.py`) was incomplete: native air-sea, ice,
observation, energetics and lake-hypsography fields were mis-reported under the
**PyFABM** section. The list is now completed to mirror
`register_all_variables.py` plus the runtime lake/air-sea outputs. Normalization
is **decoupled** from ownership (full-range normalization stays reserved for the
floor-dominated mean-flow/turbulence fields), so the correction changes section
labels only: **1096 variables reclassified across the suite, 0 status changes, 0
FABM failures hidden**, every `d_norm` identical.

## Reproduction

```bash
# Re-run lake_erken and regenerate its bundle (~5 min):
conda run -n pygotm python -m pygotm.validation.run_validation --cases lake_erken
# Re-compare existing NetCDFs without re-running:
conda run -n pygotm python -m pygotm.validation.run_validation --all --no-run
```

Localization diagnostic: compare `temp`, `num`, `NN` at depth between
`validation/runs/lake_erken/lake_erken.nc` and
`validation/reference/lake_erken/output.nc` — the surface tracks closely while the
deep column is under-mixed.

## Bottom line

Hypsography / water-balance: **exact parity**. Native physics: **under-mixed** —
26 native variables BROKEN, traced to the second-order turbulence-closure
difference between the gotm-lake reference lineage and pyGOTM's gotm-model basis,
amplified by the stratified low-energy regime; all lake-*specific* kernels are
verified correct. BGC: two correct FABM-coupling fixes landed; the dominant FABM
blocker is `bottom_everywhere`, which needs a FABM library compiled with
`variable_bottom_index`. The overall case remains **FAIL**; it is **not** correct
to say lake_erken's physics passes.
