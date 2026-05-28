# GOTM GSW SAAR Data

`saar_2011_gotm.npz` contains the Absolute Salinity Anomaly Ratio grid used by
GOTM's bundled GSW toolbox. It is generated from:

`gotm-model/code/extern/gsw/modules/gsw_mod_saar_data.f90`

The Fortran data file identifies the source as `gsw_data_v3_0.nc`, with
`gsw_version_date = "15th_May_2011"` and
`gsw_version_number = "3.05.6"`.

pyGOTM packages this data so runtime simulations and CI do not depend on a local
`gotm-model` checkout. The external Python `gsw` package is not used for SAAR
because its newer grid gives different salinity conversions and breaks GOTM
parity.

Regenerate the package data from a local GOTM checkout with:

```bash
conda run -n pygotm python scripts/generate_gotm_saar_data.py \
  gotm-model/code/extern/gsw/modules/gsw_mod_saar_data.f90 \
  src/pygotm/util/gsw/data/saar_2011_gotm.npz
```

Regression point:

`gsw_saar(76.5, 0.32, 58.91666) == 3.652974094910443e-05`
