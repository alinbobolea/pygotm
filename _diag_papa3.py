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

print("=== Variables shape ===")
for var in ["u", "v", "uu", "vv", "ww", "tke", "eps", "num", "nuh", "an", "as"]:
    if hasattr(ds_ref, var):
        print(f"{var}: ref shape={ds_ref[var].shape}, calc shape={ds_calc[var].shape}")

# Check u (velocity) at the worst time
t = 2452
print(f"\n=== u profile at t={t} ===")
ref_u = ds_ref.u.values.squeeze()
calc_u = ds_calc.u.values.squeeze()
print(f"ref shape: {ref_u.shape}")
print(f"k= 0 (bottom): ref={ref_u[t, 0]:.6e}, calc={calc_u[t, 0]:.6e}")
print(f"k= 75 (mid):   ref={ref_u[t, 75]:.6e}, calc={calc_u[t, 75]:.6e}")
print(f"k=149:         ref={ref_u[t, 149]:.6e}, calc={calc_u[t, 149]:.6e}")

# Look at the v
print(f"\n=== v profile at t={t} ===")
ref_v = ds_ref.v.values.squeeze()
calc_v = ds_calc.v.values.squeeze()
print(f"k= 0 (bottom): ref={ref_v[t, 0]:.6e}, calc={calc_v[t, 0]:.6e}")
print(f"k= 75 (mid):   ref={ref_v[t, 75]:.6e}, calc={calc_v[t, 75]:.6e}")
print(f"k=149:         ref={ref_v[t, 149]:.6e}, calc={calc_v[t, 149]:.6e}")

print("\n=== num profile at t=" + str(t) + " ===")
ref_num = ds_ref.num.values.squeeze()
calc_num = ds_calc.num.values.squeeze()
print(f"ref shape: {ref_num.shape}")
print(f"k= 0:           ref={ref_num[t, 0]:.6e}, calc={calc_num[t, 0]:.6e}")
print(f"k= 75 (mid):    ref={ref_num[t, 75]:.6e}, calc={calc_num[t, 75]:.6e}")
print(f"k=149:           ref={ref_num[t, 149]:.6e}, calc={calc_num[t, 149]:.6e}")
print(f"k=150 (surface):ref={ref_num[t, 150]:.6e}, calc={calc_num[t, 150]:.6e}")

print("\n=== tke profile at t=" + str(t) + " ===")
ref_tke = ds_ref.tke.values.squeeze()
calc_tke = ds_calc.tke.values.squeeze()
print(f"k= 0:           ref={ref_tke[t, 0]:.6e}, calc={calc_tke[t, 0]:.6e}")
print(f"k= 75 (mid):    ref={ref_tke[t, 75]:.6e}, calc={calc_tke[t, 75]:.6e}")
print(f"k=149:           ref={ref_tke[t, 149]:.6e}, calc={calc_tke[t, 149]:.6e}")
print(f"k=150 (surface):ref={ref_tke[t, 150]:.6e}, calc={calc_tke[t, 150]:.6e}")

print("\n=== u_taus ===")
ref_uts = ds_ref.u_taus.values.squeeze()
calc_uts = ds_calc.u_taus.values.squeeze()
print(f"u_taus: ref shape={ref_uts.shape}, min={ref_uts.min():.6e}, max={ref_uts.max():.6e}, mean={ref_uts.mean():.6e}")
print(f"u_taus: calc shape={calc_uts.shape}, min={calc_uts.min():.6e}, max={calc_uts.max():.6e}, mean={calc_uts.mean():.6e}")

print(f"\nu_taus at t={t}: ref={ref_uts[t]:.6e}, calc={calc_uts[t]:.6e}")

# Surface stress at t
print(f"\n=== Surface stress at t={t} ===")
print(f"tx_input: ref={ds_ref.tx.values.squeeze()[t]:.6e}, calc={ds_calc.tx.values.squeeze()[t]:.6e}")
print(f"ty_input: ref={ds_ref.ty.values.squeeze()[t]:.6e}, calc={ds_calc.ty.values.squeeze()[t]:.6e}")
ref_taux = ds_ref.taux.values.squeeze()
calc_taux = ds_calc.taux.values.squeeze()
print(f"taux[surf]: ref={ref_taux[t, 150]:.6e}, calc={calc_taux[t, 150]:.6e}")
print(f"taux[k=142]: ref={ref_taux[t, 142]:.6e}, calc={calc_taux[t, 142]:.6e}")
print(f"taux[k=141]: ref={ref_taux[t, 141]:.6e}, calc={calc_taux[t, 141]:.6e}")
print(f"taux[k=143]: ref={ref_taux[t, 143]:.6e}, calc={calc_taux[t, 143]:.6e}")
