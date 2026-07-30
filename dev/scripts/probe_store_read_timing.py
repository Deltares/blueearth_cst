"""Time a CMIP6 store read, split into OPEN / FETCH / REDUCE.

Answers two questions the fetch/reduce split (design §5.1-5.3) is gated on:

1. How much of a reduce job is remote work versus local arithmetic? A raw-slice
   cache is only worth its coherence surface if the local part is small.
2. Is one experiment (historically ``ssp585``) genuinely slower, or is the
   variance in the catalog's URI glob resolution?

The three phases are timed separately because they have very different fixes:

* **open** — ``get_rasterdataset``: resolves the catalog URI glob
  (``gs://cmip6/.../Amon/{variable}/*/*``) against the remote listing and reads
  store metadata. Measured at 1142 s for one source on 2026-07-30. **A raw cache
  does not avoid this unless the reduce stage never calls the catalog at all.**
* **fetch** — ``.load()``: the actual data transfer. Small (~19 s/source at
  ``Amon`` over a buffered basin bbox).
* **reduce** — the monthly resample/mean/round arithmetic. Negligible (~0.2 s).

Never part of a workflow run; a diagnostic only. Reads the fixture's region
polygon and writes raw slices to ``--out`` (default: a system temp dir), never
into ``project_dir``.

Usage
-----
    pixi run python dev/scripts/probe_store_read_timing.py --mode open
    pixi run python dev/scripts/probe_store_read_timing.py --mode full --experiments ssp585

``--mode open`` skips the transfer, so it isolates glob-resolution latency; it is
the cheap form to re-run when asking "is the store slow today?".
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_REGION = (
    REPO
    / "test_case/test_local/climate_historical/era5_20000101_20201231/store_region.geojson"
)
DEFAULT_CATALOG = REPO / "config/catalogs/cmip6_data.yml"

#: Matches series_identity.ACQUISITION_WINDOWS -- keep in step with it.
WINDOWS = {"historical": ("1950-01-01", "2014-12-31")}
SCENARIO_WINDOW = ("2015-01-01", "2100-12-31")


def reduce_arithmetic(data):
    """The reduction get_stats_clim_projections performs, on already-local data."""
    x_dim = next(d for d in ("x", "longitude", "lon", "long") if d in data.coords)
    y_dim = next(d for d in ("y", "latitude", "lat") if d in data.coords)
    for var in data.data_vars:
        if "precip" in var:
            var_m = data[var].resample(time="MS").sum("time")
        else:
            var_m = data[var].resample(time="MS").mean("time")
        var_m.mean([x_dim, y_dim]).round(decimals=2).load()
        var_m.groupby("time.month").mean("time").round(decimals=2).mean([x_dim, y_dim]).load()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("open", "full"), default="open")
    parser.add_argument("--model", default="INM/INM-CM4-8")
    parser.add_argument("--member", default="r1i1p1f1")
    parser.add_argument(
        "--experiments", nargs="+", default=["ssp245", "historical", "ssp585"]
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--region", type=Path, default=DEFAULT_REGION)
    parser.add_argument("--buffer", type=float, default=1.0)
    parser.add_argument("--variables", nargs="+", default=["precip", "temp"])
    parser.add_argument("--out", type=Path, default=None, help="raw-slice dir (full mode)")
    args = parser.parse_args()

    import geopandas as gpd
    import hydromt

    out_dir = args.out or Path(tempfile.mkdtemp(prefix="cst_probe_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    geom = gpd.read_file(args.region)
    bbox = list(geom.geometry.bounds.values[0])
    catalog = hydromt.DataCatalog(data_libs=str(args.catalog))

    print(f"model={args.model} member={args.member} buffer={args.buffer}")
    print(f"bbox={[round(b, 4) for b in bbox]} raw slices -> {out_dir}")
    header = f"{'pos':<5}{'experiment':<14}{'open_s':>10}"
    if args.mode == "full":
        header += f"{'fetch_s':>10}{'reduce_s':>10}{'raw_MB':>9}"
    print(header)

    for position, experiment in enumerate(args.experiments, start=1):
        window = WINDOWS.get(experiment, SCENARIO_WINDOW)
        entry = f"cmip6_{args.model}_{experiment}_{args.member}"

        t0 = time.perf_counter()
        data = catalog.get_rasterdataset(
            entry,
            bbox=bbox,
            buffer=args.buffer,
            time_range=window,
            variables=args.variables,
        )
        open_s = time.perf_counter() - t0
        row = f"{position:<5}{experiment:<14}{open_s:10.1f}"

        if args.mode == "full":
            data = data.sel(time=slice(*window))
            t0 = time.perf_counter()
            data = data.load()
            fetch_s = time.perf_counter() - t0

            raw_path = out_dir / f"raw_{experiment}.nc"
            data.to_netcdf(raw_path)

            t0 = time.perf_counter()
            reduce_arithmetic(data)
            reduce_s = time.perf_counter() - t0
            row += f"{fetch_s:10.1f}{reduce_s:10.1f}{raw_path.stat().st_size / 1e6:9.2f}"

        # flush per row: a slow open must not hide behind buffering, and a killed
        # probe should still have reported every completed source.
        print(row, flush=True)


if __name__ == "__main__":
    main()
