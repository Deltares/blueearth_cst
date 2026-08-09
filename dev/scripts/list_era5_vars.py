"""Quickly list the data variables in the staged era5_daily zarr.

Reads only metadata — does not stream values. Useful when picking the
correct variable names for stage_data.yml's `variables:` filter.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

# Use the staged copy if available; it's local and fast.
STAGED = Path("C:/data/wflow_global/hydromt/meteo/era5_daily.zarr")
SOURCE = Path("P:/wflow_global/hydromt/meteo/era5_daily.zarr")

target = STAGED if STAGED.exists() else SOURCE
print(f"Reading metadata from: {target}")

ds = xr.open_zarr(target, consolidated=True)
print("\nData variables (name : long_name | units):")
for v in ds.data_vars:
    attrs = ds[v].attrs
    long_name = attrs.get("long_name", "")
    units = attrs.get("units", "")
    print(f"  {v:<20s}  {long_name}  [{units}]")

print("\nCoordinates:", list(ds.coords))
print("Dimensions:", dict(ds.sizes))
