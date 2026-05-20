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

ref_taux = ds_ref.taux.values.squeeze()
calc_taux = ds_calc.taux.values.squeeze()
print(f"Shapes: ref={ref_taux.shape}, calc={calc_taux.shape}")

# Find absolute biggest difference and its (time, level) location
diff = calc_taux - ref_taux
idx_flat = np.argmax(np.abs(diff))
t_idx, k_idx = np.unravel_index(idx_flat, diff.shape)
print(f"\nWorst taux difference at (t={t_idx}, k={k_idx})")
print(f"  ref = {ref_taux[t_idx, k_idx]:.6e}")
print(f"  calc = {calc_taux[t_idx, k_idx]:.6e}")
print(f"  diff = {diff[t_idx, k_idx]:.6e}")

# Show full profile at that time
print(f"\n=== taux profile at t={t_idx} ===")
print(f"k= 0 (bottom):  ref={ref_taux[t_idx, 0]:.6e}, calc={calc_taux[t_idx, 0]:.6e}")
print(f"k= 1:           ref={ref_taux[t_idx, 1]:.6e}, calc={calc_taux[t_idx, 1]:.6e}")
print(f"k= 75 (middle): ref={ref_taux[t_idx, 75]:.6e}, calc={calc_taux[t_idx, 75]:.6e}")
print(f"k=149:           ref={ref_taux[t_idx, 149]:.6e}, calc={calc_taux[t_idx, 149]:.6e}")
print(f"k=150 (surface):ref={ref_taux[t_idx, 150]:.6e}, calc={calc_taux[t_idx, 150]:.6e}")

# Now check bottom stress
print("\n=== Bottom taux time series ===")
bot_ref = ref_taux[:, 0]
bot_calc = calc_taux[:, 0]
print(f"Ref bottom taux:  min={bot_ref.min():.6e}, max={bot_ref.max():.6e}, mean={bot_ref.mean():.6e}, std={bot_ref.std():.6e}")
print(f"Calc bottom taux: min={bot_calc.min():.6e}, max={bot_calc.max():.6e}, mean={bot_calc.mean():.6e}, std={bot_calc.std():.6e}")

# Check level-1 (just above bottom)
print("\n=== Level 1 (just above bottom) taux ===")
l1_ref = ref_taux[:, 1]
l1_calc = calc_taux[:, 1]
print(f"Ref l1 taux:  min={l1_ref.min():.6e}, max={l1_ref.max():.6e}")
print(f"Calc l1 taux: min={l1_calc.min():.6e}, max={l1_calc.max():.6e}")

# Compare u_taub
print("\n=== u_taub ===")
ref_utb = ds_ref.u_taub.values.squeeze()
calc_utb = ds_calc.u_taub.values.squeeze()
print(f"Ref u_taub: shape={ref_utb.shape}, min={ref_utb.min():.6e}, max={ref_utb.max():.6e}, mean={ref_utb.mean():.6e}")
print(f"Calc u_taub: shape={calc_utb.shape}, min={calc_utb.min():.6e}, max={calc_utb.max():.6e}, mean={calc_utb.mean():.6e}")

# Compare taub
print("\n=== taub ===")
ref_tb = ds_ref.taub.values.squeeze()
calc_tb = ds_calc.taub.values.squeeze()
print(f"Ref taub: shape={ref_tb.shape}, min={ref_tb.min():.6e}, max={ref_tb.max():.6e}")
print(f"Calc taub: shape={calc_tb.shape}, min={calc_tb.min():.6e}, max={calc_tb.max():.6e}")
