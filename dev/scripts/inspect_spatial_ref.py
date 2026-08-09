"""Inspect spatial_ref attributes in extract_historical.nc and rlz_*_st_0.nc.

weathergenr's write_netcdf needs `x_dim` and `y_dim` attributes on the
template's `spatial_ref` variable. Confirm whether they're present in the
historical file (used as template by generate_weather), and whether they
get propagated to the realization output (used as template by
impose_climate_change).
"""

from pathlib import Path

from netCDF4 import Dataset

# Relative to the repo root, so run this from there. The two realization NCs
# are wrapped in temp() by rule 3.06, so they exist only DURING a WF3 run --
# "(not present)" for those two is the normal state between runs, not a stale
# path.
FIXTURE = Path("test_case/test_local")
EXPERIMENT = "experiment"
STORE_KEY = "era5_20000101_20201231"
paths = [
    FIXTURE / f"data/climate/historical/{STORE_KEY}/extract_historical.nc",
    FIXTURE / f"experiments/{EXPERIMENT}/climate/weathergenr/output/rlz_1_st_0.nc",
    FIXTURE / f"experiments/{EXPERIMENT}/climate/weathergenr/output/rlz_2_st_0.nc",
]

for p in paths:
    print(f"\n=== {p} ===")
    if not p.exists():
        print("  (not present)")
        continue
    with Dataset(p) as ds:
        print(f"  dims: {dict((d, len(ds.dimensions[d])) for d in ds.dimensions)}")
        if "spatial_ref" in ds.variables:
            attrs = ds.variables["spatial_ref"].ncattrs()
            print(f"  spatial_ref attrs: {attrs}")
            for a in attrs:
                v = ds.variables["spatial_ref"].getncattr(a)
                if isinstance(v, str) and len(v) > 80:
                    v = v[:80] + "..."
                print(f"    {a} = {v!r}")
        else:
            print("  no 'spatial_ref' variable")
