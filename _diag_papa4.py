import xarray as xr
import numpy as np

ds_ref = xr.open_dataset(
    "/home/nick/projects/pygotm/gotm-model/cases-runs/ows_papa/ows_papa_keps_meteo.nc",
    decode_times=False,
)
ds_calc = xr.open_dataset(
    "/home/nick/projects/pygotm/validation/runs/ows_papa/ows_papa.nc",
    decode_times=False,
)

# Look at key variables in the first ~20 days
print("=== EARLY TIME DIVERGENCE TRACING ===")
print()
print("t | airt_diff | u10_diff | qh_diff | qe_diff | tau_diff | temp_diff | tke_diff")
for t in [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]:
    ref_airt = ds_ref.airt.values.squeeze()[t]
    calc_airt = ds_calc.airt.values.squeeze()[t]
    ref_u10 = ds_ref.u10.values.squeeze()[t]
    calc_u10 = ds_calc.u10.values.squeeze()[t]
    ref_qh = ds_ref.qh.values.squeeze()[t]
    calc_qh = ds_calc.qh.values.squeeze()[t]
    ref_qe = ds_ref.qe.values.squeeze()[t]
    calc_qe = ds_calc.qe.values.squeeze()[t]
    ref_tx = ds_ref.tx.values.squeeze()[t]
    calc_tx = ds_calc.tx.values.squeeze()[t]
    ref_t = ds_ref.temp.values.squeeze()[t, -1]
    calc_t = ds_calc.temp.values.squeeze()[t, -1]
    ref_tke = ds_ref.tke.values.squeeze()[t, -1]
    calc_tke = ds_calc.tke.values.squeeze()[t, -1]
    print(f"t={t:5d} | airt:{ref_airt:.3f}/{calc_airt:.3f} | u10:{ref_u10:.3f}/{calc_u10:.3f} | qh:{ref_qh:.4e}/{calc_qh:.4e} | qe:{ref_qe:.4e}/{calc_qe:.4e} | tx:{ref_tx:.4e}/{calc_tx:.4e} | temp:{ref_t:.3f}/{calc_t:.3f} | tke:{ref_tke:.4e}/{calc_tke:.4e}")

# Diagnostic: where is the FIRST big divergence in surface temp?
print("\n=== First significant temp divergence ===")
ref_temp = ds_ref.temp.values.squeeze()
calc_temp = ds_calc.temp.values.squeeze()
diff_surf = np.abs(calc_temp[:, -1] - ref_temp[:, -1])
for thresh in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    idx = np.argmax(diff_surf > thresh)
    if diff_surf[idx] > thresh:
        print(f"diff > {thresh}: first at t={idx}, ref={ref_temp[idx, -1]:.4f}, calc={calc_temp[idx, -1]:.4f}")

# Maximum divergence
print(f"\nMax surface temp diff: {diff_surf.max():.4f} at t={np.argmax(diff_surf)}")

# Surface taux time series, looking for sign or large mismatches
print("\n=== Surface taux time series ===")
ref_taux = ds_ref.taux.values.squeeze()
calc_taux = ds_calc.taux.values.squeeze()
diff_taux_surf = calc_taux[:, -1] - ref_taux[:, -1]
print(f"Max abs surface taux diff: {np.abs(diff_taux_surf).max():.6e} at t={np.argmax(np.abs(diff_taux_surf))}")
print(f"Mean abs surface taux diff: {np.mean(np.abs(diff_taux_surf)):.6e}")

# Compare es, qs (Fairall outputs)
print("\n=== es, qs first divergence ===")
ref_es = ds_ref.es.values.squeeze()
calc_es = ds_calc.es.values.squeeze()
ref_qs = ds_ref.qs.values.squeeze()
calc_qs = ds_calc.qs.values.squeeze()
ref_qa = ds_ref.qa.values.squeeze()
calc_qa = ds_calc.qa.values.squeeze()
for t in [0, 1, 5, 10, 100, 1000]:
    print(f"t={t}: es ref={ref_es[t]:.6e}/calc={calc_es[t]:.6e}, qs ref={ref_qs[t]:.6e}/calc={calc_qs[t]:.6e}, qa ref={ref_qa[t]:.6e}/calc={calc_qa[t]:.6e}")

# Check sst at t=0 and t=1
print("\n=== sst time series (early) ===")
ref_sst = ds_ref.sst.values.squeeze()
calc_sst = ds_calc.sst.values.squeeze()
for t in [0, 1, 2, 5, 10]:
    print(f"t={t}: sst ref={ref_sst[t]:.4f}, calc={calc_sst[t]:.4f}, diff={calc_sst[t]-ref_sst[t]:+.4f}")
