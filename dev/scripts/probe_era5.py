"""Definitive NaN count on the staged era5 t2m, plus sample reads.

If the bulk count disagrees with sample probes, the bug is in
inspect_era5_nan.py. If it agrees, something is genuinely wrong with the
staged data and we need to look at chunk-level integrity.
"""
import numpy as np
import xarray as xr

SRC = "P:/wflow_global/hydromt/meteo/era5_daily.zarr"
DST = "C:/data/wflow_global/hydromt/meteo/era5_daily.zarr"

print("=== Open staged t2m and load fully into memory ===")
dst = xr.open_zarr(DST, consolidated=True)
t2m = dst["t2m"].load().values  # eager numpy array
print(f"shape: {t2m.shape}  dtype: {t2m.dtype}")
print(f"NaN count: {int(np.isnan(t2m).sum())} / {t2m.size}")
print(f"Min: {np.nanmin(t2m):.3f}  Max: {np.nanmax(t2m):.3f}  Mean: {np.nanmean(t2m):.3f}")
print(f"All-NaN per timestep count: {int(np.all(np.isnan(t2m), axis=(1,2)).sum())} / {t2m.shape[0]}")

print("\n=== Sample STAGED t2m at multiple times/cells (no .sel slicing) ===")
print(f"  raw [0,0,0]: {t2m[0,0,0]}")
print(f"  raw [3000,4,5]: {t2m[3000,4,5]}")  # mid-2008
print(f"  raw [3800,4,5]: {t2m[3800,4,5]}")  # mid-2010
print(f"  raw [4200,4,5]: {t2m[4200,4,5]}")  # mid-2011
print(f"  raw [-1,-1,-1]: {t2m[-1,-1,-1]}")

print("\n=== Coord values ===")
print(f"  latitude: {dst['latitude'].values}")
print(f"  longitude: {dst['longitude'].values}")
print(f"  time[0]: {str(dst['time'].values[0])[:19]}  time[-1]: {str(dst['time'].values[-1])[:19]}")

print("\n=== Try the same slice as inspect_era5_nan.py (bbox + time) ===")
BBOX = (8.5, -0.5, 11.0, 1.5)  # W S E N
lat = "latitude"
lon = "longitude"
w, s, e, n = BBOX
lat_desc = dst[lat].values[0] > dst[lat].values[-1]
lat_slice = slice(n, s) if lat_desc else slice(s, n)
print(f"  lat_desc: {lat_desc}  lat_slice: {lat_slice}  lon_slice: slice({w}, {e})")

sliced = dst.sel({lat: lat_slice, lon: slice(w, e)})
sliced = sliced.sel(time=slice("2000-01-01", "2020-12-31"))
print(f"  sliced shape: {dict(sliced.sizes)}")

t2m_sliced = sliced["t2m"].load().values
print(f"  sliced NaN count: {int(np.isnan(t2m_sliced).sum())} / {t2m_sliced.size}")
print(f"  sliced [0,0,0]: {t2m_sliced[0,0,0]}")
print(f"  sliced sample: {t2m_sliced[3800,4,5]}")
