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

print("=== Time alignment ===")
print(f"Ref time:  {ds_ref.time.values[0]} to {ds_ref.time.values[-1]}, N={len(ds_ref.time)}")
print(f"Calc time: {ds_calc.time.values[0]} to {ds_calc.time.values[-1]}, N={len(ds_calc.time)}")
print(f"Ref time step: {ds_ref.time.values[1] - ds_ref.time.values[0]}s")

print("\n=== Surface taux ===")
ref_taux = ds_ref.taux.values.squeeze()
calc_taux = ds_calc.taux.values.squeeze()
surf_ref = ref_taux[:, -1]
surf_calc = calc_taux[:, -1]
print(f"Ref surface taux:  min={surf_ref.min():.6e}, max={surf_ref.max():.6e}, mean={surf_ref.mean():.6e}")
print(f"Calc surface taux: min={surf_calc.min():.6e}, max={surf_calc.max():.6e}, mean={surf_calc.mean():.6e}")

print("\n=== tx (input) ===")
ref_tx = ds_ref.tx.values.squeeze()
calc_tx = ds_calc.tx.values.squeeze()
print(f"Ref tx: shape={ref_tx.shape}, min={ref_tx.min():.6e}, max={ref_tx.max():.6e}, mean={ref_tx.mean():.6e}")
print(f"Calc tx: shape={calc_tx.shape}, min={calc_tx.min():.6e}, max={calc_tx.max():.6e}, mean={calc_tx.mean():.6e}")

print("\n=== heat ===")
ref_heat = ds_ref.heat.values.squeeze()
calc_heat = ds_calc.heat.values.squeeze()
print(f"Ref heat:  min={ref_heat.min():.3e}, max={ref_heat.max():.3e}, mean={ref_heat.mean():.3e}")
print(f"Calc heat: min={calc_heat.min():.3e}, max={calc_heat.max():.3e}, mean={calc_heat.mean():.3e}")

print("\n=== qh ===")
ref_qh = ds_ref.qh.values.squeeze()
calc_qh = ds_calc.qh.values.squeeze()
print(f"Ref qh: min={ref_qh.min():.3e}, max={ref_qh.max():.3e}, mean={ref_qh.mean():.3e}")
print(f"Calc qh: min={calc_qh.min():.3e}, max={calc_qh.max():.3e}, mean={calc_qh.mean():.3e}")

print("\n=== Surface temp ===")
ref_temp = ds_ref.temp.values.squeeze()
calc_temp = ds_calc.temp.values.squeeze()
print(f"Ref surface temp:  min={ref_temp[:, -1].min():.3f}, max={ref_temp[:, -1].max():.3f}, mean={ref_temp[:, -1].mean():.3f}")
print(f"Calc surface temp: min={calc_temp[:, -1].min():.3f}, max={calc_temp[:, -1].max():.3f}, mean={calc_temp[:, -1].mean():.3f}")

print("\n=== Per-time difference for taux surface ===")
diff = surf_calc - surf_ref
print(f"Diff: min={diff.min():.6e}, max={diff.max():.6e}, mean={diff.mean():.6e}, std={diff.std():.6e}")
idx_max = np.argmax(np.abs(diff))
print(f"Worst diff at t={idx_max}, ref={surf_ref[idx_max]:.6e}, calc={surf_calc[idx_max]:.6e}, diff={diff[idx_max]:.6e}")

print("\n=== tx == taux at surface? (Fortran ref) ===")
diff_tx_taux = ref_tx - surf_ref
print(f"ref tx - ref surf_taux: min={diff_tx_taux.min():.6e}, max={diff_tx_taux.max():.6e}, mean={diff_tx_taux.mean():.6e}")

print("\n=== First 10 values: t, ref_tx, ref_taux_surf, calc_tx, calc_taux_surf ===")
for i in range(10):
    print(f"  i={i}: ref_tx={ref_tx[i]:.6e}, ref_taux_surf={surf_ref[i]:.6e}, calc_tx={calc_tx[i]:.6e}, calc_taux_surf={surf_calc[i]:.6e}")
