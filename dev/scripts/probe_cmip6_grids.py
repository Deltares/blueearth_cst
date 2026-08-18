"""Report which CMIP6 models hydromt's raster accessor will refuse.

hydromt's `.raster` accessor requires UNIFORMLY spaced coordinates
(`hydromt/gis/raster.py:441` -- `np.allclose(dys, dys[0], atol=5e-4)`), and a
large share of CMIP6 publishes `Amon` on a GAUSSIAN grid, whose latitudes are
Legendre roots and vary by ~1%. Those models raise
`ValueError: The 'raster' accessor only applies to regular grids` and are
therefore invisible to WF2 -- silently absent from an ensemble rather than
reported.

Measured 2026-08-18: 27 of 67 models, including CanESM5, every EC-Earth3
variant, MPI-ESM1-2-HR/LR, CNRM-CM6-1, MIROC6 and MRI-ESM2-0. Board item
`t2608182020` carries the full table and the options.

CHEAP ON PURPOSE: reads only the `lat`/`lon` coordinates of one `Amon/tas`
store per model -- consolidated metadata plus two small arrays -- rather than
opening through hydromt. Threads, not processes: this is pure object-store wait
with no hydromt, GDAL or HDF5 in the path.

Usage (from the repo root, inside pixi)::

    python dev/scripts/probe_cmip6_grids.py

Not part of a run: this is a diagnostic
(see AGENTS.md, "Three homes for executables").
"""

import os

# MUST precede any gcsfs import; see fetch_gcm_raw's note (14x slowdown).
os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")

# E402 throughout: the environment switch above MUST run before anything that
# transitively imports gcsfs, which is the whole point of the ordering.
import json  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

idx = json.load(open("config/catalogs/cmip6_store_index.json", encoding="utf-8"))[
    "sources"
]

# one store per MODEL, preferring historical
per_model = {}
for entry, members in idx.items():
    body = entry[len("cmip6_") :].rsplit("_{member}", 1)[0]
    model, _, scen = body.rpartition("_")
    for member, vars_ in (members or {}).items():
        for var, locs in (vars_ or {}).items():
            if var != "tas" or not locs:
                continue
            cur = per_model.get(model)
            if cur is None or (scen == "historical" and cur[0] != "historical"):
                per_model[model] = (
                    scen,
                    f"gs://cmip6/CMIP6/{'CMIP' if scen == 'historical' else 'ScenarioMIP'}/{model}/{scen}/{member}/Amon/tas/{locs[0]}",
                )


def check(item):
    model, (scen, uri) = item
    try:
        import xarray as xr

        ds = xr.open_zarr(uri, consolidated=True, chunks=None, decode_times=False)
        lat = np.asarray(ds["lat"].values)
        lon = np.asarray(ds["lon"].values)
        ds.close()
        dys, dxs = np.diff(lat), np.diff(lon)
        yreg = bool(np.allclose(dys, dys[0], atol=5e-4))
        xreg = bool(np.allclose(dxs, dxs[0], atol=5e-4))
        return (
            model,
            ("regular" if (xreg and yreg) else "IRREGULAR"),
            f"lat n={lat.size} dlat {dys.min():.4f}..{dys.max():.4f}  lon reg={xreg}",
        )
    except Exception as exc:
        return model, "error", f"{type(exc).__name__}: {str(exc)[:70]}"


rows = []
with ThreadPoolExecutor(max_workers=12) as pool:
    futs = {pool.submit(check, it): it[0] for it in sorted(per_model.items())}
    for i, f in enumerate(as_completed(futs), 1):
        rows.append(f.result())
        print(f"  [{i}/{len(futs)}] {rows[-1][0]:34s} {rows[-1][1]}", flush=True)

print("\n=== SUMMARY ===")
for kind in ("regular", "IRREGULAR", "error"):
    sel = [r for r in rows if r[1] == kind]
    print(f"{kind:10s} {len(sel):3d} of {len(rows)}")
    if kind != "regular":
        for m, _, d in sorted(sel):
            print(f"    {m:34s} {d}")
