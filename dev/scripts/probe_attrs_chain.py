"""M2b attrs-loss diagnostic probe (R4 §5 / chain-audit, ext1-4 / ext2-3).

Localizes where CF metadata (``units``/``standard_name``/``long_name``) is lost
in the workflow-2 change-factor chain. Standalone, diagnostic-only: it does NOT
instrument any ``blueearth_cst/`` module — it re-executes the operation *sequence* of the
workflow-2 functions on a synthetic input carrying known non-empty CF attrs,
recording ``.attrs`` after each step, so the first attr-dropping operation is
pinnable without a real (temp-deleted) run.

Design contract: `dev/milestones/r04/climate-projections-design.md` §5 "M2b attrs
diagnostic probe". The checkpoint names (P0-P6 per-model, M1-M2 merge) match
that section. A parallel upstream check covers the get_stats reduction, since
P0 being clean routes the localization upstream (design P0 note).

Run: ``pixi run python dev/scripts/probe_attrs_chain.py``. It prints a table
and a one-line localization verdict; it writes nothing and moves no baseline.
"""

import os
import sys
import tempfile
from os.path import dirname, join, realpath

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, join(dirname(realpath(__file__)), "..", ".."))

KNOWN_ATTRS = {
    "units": "kg m-2 s-1",
    "standard_name": "precipitation_flux",
    "long_name": "Precipitation",
}


def _keys(x):
    return list(getattr(x, "attrs", {}).keys())


def _row(tag, x, rows):
    present = _keys(x)
    rows.append((tag, present))
    flag = "ok" if present else "DROPPED"
    print(f"  {tag:44s} attrs={present!s:60s} {flag}")


def probe_get_stats_upstream():
    """Upstream (get_stats_clim_projections op sequence): resample -> spatial
    mean -> round -> to_dataset."""
    print("\n[upstream] get_stats_climate_proj reduction chain")
    rows = []
    t = pd.date_range("2000-01-01", periods=24, freq="D")
    da = xr.DataArray(
        np.ones((24, 3, 3)),
        dims=("time", "lat", "lon"),
        coords={"time": t, "lat": [0, 1, 2], "lon": [0, 1, 2]},
        name="precip",
    )
    da.attrs = dict(KNOWN_ATTRS)
    _row("S0 input", da, rows)
    var_m = da.resample(time="MS").sum("time")
    _row("S1 resample(MS).sum", var_m, rows)
    scal = var_m.mean(["lat", "lon"])
    _row("S2 spatial .mean([x,y])", scal, rows)
    rnd = scal.round(decimals=2)
    _row("S3 .round(2)", rnd, rows)
    _row("S4 .to_dataset() var", rnd.to_dataset()["precip"], rows)
    return rows


def probe_per_model():
    """Per-model (get_change_annual_clim_proj op sequence), P0-P6, on a
    datetime64 index (the calendar-safe case; cftime is a separate defect)."""
    print("\n[per-model] get_change_annual_clim_proj chain (P0-P6)")
    rows = []
    t = pd.date_range("2000-01-01", periods=36, freq="MS")
    hist = xr.DataArray(np.ones(36), dims="time", coords={"time": t}, name="precip")
    hist.attrs = dict(KNOWN_ATTRS)
    _row("P0 input (known attrs)", hist, rows)
    w = hist.sel(time=slice("2000-01-01", "2002-12-01"))
    _row("P1 .sel(time=slice)", w, rows)
    r = w.resample(time="YS-JAN").sum("time")
    _row("P2 resample(YS).sum", r, rows)
    red = r.mean("time")
    _row("P3a stat reduction .mean('time')", red, rows)
    chg = (red - red) / red * 100
    _row("P3b change arithmetic", chg, rows)
    chg2 = chg.assign_coords({"stats": "mean"}).expand_dims("stats")
    _row("P4a assign_coords/expand_dims", chg2, rows)
    merged = xr.merge([chg2.to_dataset(name="precip")])
    _row("P4b to_dataset + xr.merge", merged["precip"], rows)
    d = tempfile.mkdtemp()
    p = os.path.join(d, "per_model.nc")
    merged.to_netcdf(p)
    _row("P5/P6 to_netcdf + reopen", xr.open_dataset(p)["precip"], rows)
    return rows


def probe_merge():
    """Merge (summary_climate_proj path), M1-M2: open_mfdataset(coords=minimal,
    preprocess) + to_netcdf + reopen."""
    print("\n[merge] summary open_mfdataset + write chain (M1-M2)")
    from blueearth_cst.projections.get_change_climate_proj_summary import (
        preprocess_coords,
    )

    rows = []
    d = tempfile.mkdtemp()

    def mkfile(path, model):
        da = xr.DataArray(
            np.ones((1, 1, 1, 1)),
            dims=("stats", "horizon", "scenario", "model"),
            coords={
                "stats": ["mean"],
                "horizon": ["near"],
                "scenario": ["ssp245"],
                "model": [model],
            },
            name="precip",
        )
        da.attrs = dict(KNOWN_ATTRS)
        da.to_dataset().to_netcdf(path)

    f1, f2 = os.path.join(d, "a.nc"), os.path.join(d, "b.nc")
    mkfile(f1, "GOOD1")
    mkfile(f2, "GOOD2")
    _row("M0 single file reopened", xr.open_dataset(f1)["precip"], rows)
    m = xr.open_mfdataset([f1, f2], coords="minimal", preprocess=preprocess_coords)
    _row("M1 open_mfdataset(min,preprocess)", m["precip"], rows)
    out = os.path.join(d, "summary.nc")
    m.to_netcdf(out)
    _row("M2 summary written + reopened", xr.open_dataset(out)["precip"], rows)
    return rows


def main():
    print("M2b attrs-loss probe -- synthetic known-attrs input, no real run")
    allrows = probe_get_stats_upstream() + probe_per_model() + probe_merge()
    dropped = [tag for tag, present in allrows if not present]
    print("\n" + "=" * 66)
    if dropped:
        print(f"LOCALIZATION: first attr-drop at -> {dropped[0]}")
    else:
        print(
            "LOCALIZATION: no workflow-2 pure-Python operation drops CF attrs.\n"
            "  The only remaining M2b candidate is the hydromt catalog read\n"
            "  (data_catalog.get_rasterdataset in get_stats_climate_proj.py) --\n"
            "  a dependency operation, matching the design's 'dependency\n"
            "  reproducer' end state. Attrs restoration stays OUT of R4\n"
            "  (candidate #1, deferred to a dedicated task)."
        )


if __name__ == "__main__":
    main()
