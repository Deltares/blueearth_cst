"""Diagnose where NaN values around 2010-2011 in the staged era5_daily zarr come from.

Opens the source and staged zarr stores with the same bbox + time_range that
stage_data.py applies, then prints yearly NaN counts per variable for both.
If the source already has NaN for those years, the staging is innocent;
if only the staged copy has them, subset_zarr() introduced them.

Run from repo root:
    python dev/scripts/inspect_era5_nan.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SOURCE = Path("P:/wflow_global/hydromt/meteo/era5_daily.zarr")
STAGED = Path("C:/data/wflow_global/hydromt/meteo/era5_daily.zarr")
BBOX = (8.5, -0.5, 11.0, 1.5)  # west, south, east, north — matches stage_data.yml
TIME_RANGE = ("2000-01-01", "2020-12-31")


def _spatial_slice(ds: xr.Dataset, bbox) -> xr.Dataset:
    """Apply bbox slice using whatever lat/lon coord names the dataset has."""
    lat = next(c for c in ds.coords if c.lower() in ("lat", "latitude", "y"))
    lon = next(c for c in ds.coords if c.lower() in ("lon", "longitude", "x"))
    w, s, e, n = bbox
    lat_desc = ds[lat].values[0] > ds[lat].values[-1]
    lat_slice = slice(n, s) if lat_desc else slice(s, n)
    return ds.sel({lat: lat_slice, lon: slice(w, e)})


def _time_slice(ds: xr.Dataset, time_range) -> xr.Dataset:
    tdim = next((d for d in ds.dims if d.lower() in ("time", "t")), "time")
    return ds.sel({tdim: slice(*time_range)})


def _nan_per_year(ds: xr.Dataset) -> dict[str, dict[int, tuple[int, int]]]:
    """Return {var_name: {year: (nan_count, total_count)}}.

    Loads each variable fully into memory before counting. The earlier
    lazy-eval pipeline (`xarray.isnull().sum() -> to_pandas() -> groupby`)
    miscounted on the chunked staged zarr (chunks larger than array shape),
    reporting 100% NaN where the underlying data was clean. Numpy on a
    materialised array is small enough here (~MB scale) and unambiguous.
    """
    tdim = next((d for d in ds.dims if d.lower() in ("time", "t")), "time")
    time_idx = pd.DatetimeIndex(ds[tdim].values)
    years = time_idx.year
    out: dict[str, dict[int, tuple[int, int]]] = {}
    for v in ds.data_vars:
        arr_np = ds[v].load().values
        spatial_axes = tuple(i for i, d in enumerate(ds[v].dims) if d != tdim)
        cells_per_step = int(np.prod([ds[v].sizes[d] for d in ds[v].dims if d != tdim]))
        per_step_nan = np.isnan(arr_np).sum(axis=spatial_axes)
        nan_by_year = pd.Series(per_step_nan, index=time_idx).groupby(years).sum()
        steps_by_year = pd.Series(1, index=time_idx).groupby(years).sum()
        out[v] = {
            int(y): (int(nan_by_year[y]), int(steps_by_year[y]) * cells_per_step)
            for y in nan_by_year.index
        }
    return out


def _open_clipped(path: Path) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=True)
    ds = _spatial_slice(ds, BBOX)
    ds = _time_slice(ds, TIME_RANGE)
    return ds


def _print_table(label: str, stats: dict[str, dict[int, tuple[int, int]]]) -> None:
    print(f"\n=== {label} ===")
    vars_ = list(stats.keys())
    years = sorted({y for v in stats.values() for y in v.keys()})
    header = "year   " + "  ".join(f"{v:>14s}" for v in vars_)
    print(header)
    print("-" * len(header))
    for y in years:
        cells = []
        for v in vars_:
            n, total = stats[v].get(y, (0, 0))
            cells.append(f"{n:>7d}/{total:<6d}" if total else "       —      ")
        flag = "  <<<" if any(stats[v].get(y, (0, 0))[0] > 0 for v in vars_) else ""
        print(f"{y}   " + "  ".join(cells) + flag)


def _diff_table(
    src: dict[str, dict[int, tuple[int, int]]],
    dst: dict[str, dict[int, tuple[int, int]]],
) -> None:
    print("\n=== diff (staged - source NaN counts; positive = introduced by staging) ===")
    vars_ = sorted(set(src.keys()) | set(dst.keys()))
    years = sorted({y for v in src.values() for y in v.keys()} |
                   {y for v in dst.values() for y in v.keys()})
    header = "year   " + "  ".join(f"{v:>10s}" for v in vars_)
    print(header)
    print("-" * len(header))
    for y in years:
        cells = []
        any_diff = False
        for v in vars_:
            sn = src.get(v, {}).get(y, (0, 0))[0]
            dn = dst.get(v, {}).get(y, (0, 0))[0]
            d = dn - sn
            any_diff = any_diff or (d != 0)
            cells.append(f"{d:>+10d}" if d != 0 else f"{'.':>10s}")
        flag = "  <<<" if any_diff else ""
        print(f"{y}   " + "  ".join(cells) + flag)


def main() -> None:
    print(f"source: {SOURCE}")
    print(f"staged: {STAGED}")
    print(f"bbox:   {BBOX}")
    print(f"time:   {TIME_RANGE[0]} -> {TIME_RANGE[1]}")

    print("\nOpening source (this can take a moment over SMB)...")
    src = _open_clipped(SOURCE)
    print(f"  vars: {list(src.data_vars)}")
    print(f"  shape: {dict(src.sizes)}")

    print("\nOpening staged...")
    dst = _open_clipped(STAGED)
    print(f"  vars: {list(dst.data_vars)}")
    print(f"  shape: {dict(dst.sizes)}")

    print("\nComputing NaN counts per year (source)...")
    src_stats = _nan_per_year(src)

    print("Computing NaN counts per year (staged)...")
    dst_stats = _nan_per_year(dst)

    _print_table("SOURCE: yearly NaN / total cells", src_stats)
    _print_table("STAGED: yearly NaN / total cells", dst_stats)
    _diff_table(src_stats, dst_stats)


if __name__ == "__main__":
    main()
