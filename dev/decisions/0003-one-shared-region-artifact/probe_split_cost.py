"""Measure ADR 0003 §8's acceptance gate: is the vector half materially cheaper?

§8 splits `prepare_spatial_maps` so WF2 and WF3 can declare only the vector half.
The whole decision rests on an **unmeasured** claim: that the hydrography read
plus delineation costs materially less than the thematic raster stack it leaves
behind. If it does not, WF2 pays roughly what declaring the unsplit rule would
have cost, the split bought nothing, and the simpler "declare it unsplit"
alternative wins.

So this times the two halves exactly as `prepare_spatial_products` calls them:

* **vector half** — hydrography raster read, `prepare_hydrography`, parent-basin
  delineation, gauge snapping, subbasin partitioning, rivers geodataframe;
* **thematic half** — `_thematic_maps`, i.e. the LULC / LAI / soil reads and
  their reprojections onto the model grid.

Read-only: it opens catalog sources and writes nothing but the region GeoJSON,
into a temp dir. It is **not** a pipeline run — no Snakemake state, no
`project_dir`, so it is safe to run from a worktree.

    pixi run python dev/decisions/0003-one-shared-region-artifact/probe_split_cost.py

Defaults to the tracked test config's region so the number is reproducible.
Pass `--config` to time a larger, more realistic basin — the ratio is what the
gate turns on, and a small basin may flatter both halves.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
# The repo is not installed; Snakefiles do the same prepend for `script:` modules.
sys.path.insert(0, str(REPO))
DEFAULT_CONFIG = REPO / "config" / "workflows" / "snake_config_model_test.yml"


@contextmanager
def timed(label: str, into: dict):
    start = time.perf_counter()
    yield
    into[label] = time.perf_counter() - start
    print(f"  {label:<28} {into[label]:8.2f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    import hydromt

    from blueearth_cst.spatial import products as P
    from blueearth_cst.spatial.config import parse_spatial_config
    from blueearth_cst.spatial.delineate_region import delineate_region
    from blueearth_cst.spatial.hydrography import prepare_hydrography

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    basin_cfg = raw["shared"]["basin"]
    model_cfg = raw.get("workflows", {}).get("model_creation", {}) or {}
    catalog_path = raw["project"]["data_sources"]
    config = parse_spatial_config(basin_cfg, model_cfg)

    # A project config lives OUTSIDE the repo by convention, so neither the
    # config path nor its catalog path can be assumed repo-relative.
    catalog_fn = Path(catalog_path)
    if not catalog_fn.is_absolute():
        catalog_fn = REPO / catalog_fn

    print(f"config   : {args.config}")
    print(f"region   : {config.region}")
    print(f"catalog  : {catalog_fn}\n")

    timings: dict[str, float] = {}
    catalog = hydromt.DataCatalog(data_libs=str(catalog_fn))

    # ignore_cleanup_errors: rasterio/xarray keep handles open past the block on
    # Windows, and a failing finalizer floods shutdown with excepthook noise
    # after the result has already printed.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        region_fn = Path(tmp) / "region.geojson"

        with timed("delineate_region", timings):
            delineate_region(
                config.region,
                str(catalog_fn),
                hydrography=config.hydrography,
                basin_index=config.basin_index,
                region_out=region_fn,
            )

        # ---- vector half: everything up to and including _delineate_spatial_units
        with timed("VECTOR half", timings):
            region = P._region_geometry(region_fn)
            source = catalog.get_rasterdataset(
                config.hydrography, geom=region, buffer=10, single_var_as_array=False
            )
            maps, flwdir = prepare_hydrography(
                P._as_dataset(source),
                region,
                config.resolution,
                config.river_uparea_km2,
            )
            basin_map, basins, outlet_by_basin = P._parent_basins(maps, flwdir, region)
            maps["basin_id"] = basin_map
            gauges = P._snap_gauge_points(
                P.read_gauge_points(config.gauge_points_path),
                maps,
                flwdir,
                config.gauge_snap_tolerance_m,
            )
            subbasin_map, *_ = P._delineate_spatial_units(
                maps,
                flwdir,
                basins,
                outlet_by_basin,
                gauges,
                config.max_automatic_subbasins,
            )
            maps["subbasin_id"] = subbasin_map
            rivers = catalog.get_geodataframe(config.sources.rivers, geom=basins)
            assert rivers is not None

        # ---- thematic half: the WF1-only raster stack the split leaves behind
        with timed("THEMATIC half", timings):
            P._thematic_maps(catalog, config, basins, maps["flow_direction"])

    # `delineate_region` is EXCLUDED from the comparison: WF2 and WF3 already
    # declare it (ADR 0003 §1-7, landed), so it is a cost they pay today either
    # way. The gate is about what §8 ADDS to them, against what declaring the
    # rule unsplit would add instead.
    already_paid = timings["delineate_region"]
    added_by_split = timings["VECTOR half"]
    avoided = timings["THEMATIC half"]
    added_unsplit = added_by_split + avoided

    print(f"\n  {'-' * 52}")
    print(
        f"  already paid by WF2/WF3 today   {already_paid:8.2f} s   (delineate_region)"
    )
    print(f"  §8 ADDS to WF2/WF3              {added_by_split:8.2f} s")
    print(f"  unsplit would add instead       {added_unsplit:8.2f} s")
    print(
        f"  §8 avoids                       {avoided:8.2f} s"
        f"   ({avoided / added_unsplit:5.1%} of the incremental cost)"
    )
    print(f"\n  grid: {dict(maps.sizes)}")
    print(
        "\nGate: §8 is worth it if it avoids the dominant share of what it would\n"
        "otherwise add. If the vector half is most of the incremental cost,\n"
        "declaring the rule unsplit in all three workflows is simpler and buys\n"
        "the same thing.\n"
        "\nCAVEAT: on a small basin both halves are mostly FIXED overhead — opening\n"
        "VRTs and tifs, parsing the catalog — not per-cell work. Re-run on a large\n"
        "basin before treating the ratio as general."
    )


if __name__ == "__main__":
    main()
