"""Extract historical climate data for a given region and time period.

Rule ``extract_climate_grid``'s script — the SINGLE producer of the shared
``data/climate/historical/<key>/`` store, declared identically in
``Snakefile_model_creation`` (1.10) and ``Snakefile_climate_experiment`` (3.02)
from ``snake_utils.climate_store_rule`` (R07 B1). The extraction extent stays
**model-free**: it comes from ``data/spatial/geoms/region.geojson``, the one
project region artifact delineated from ``shared.basin`` + the catalog by rule
``delineate_region`` (ADR 0003), so nothing here reads a built model.

Until ADR 0003 this script delineated that polygon itself and wrote a
per-store-key copy (``store_region.geojson``). The copy is gone; the store's
extent provenance now travels *in* ``extract_historical.nc`` as the
``region_bbox`` / ``region_geojson_sha256`` / ``region_source`` attributes,
which cannot be separated from the data they describe.
"""

import os
import warnings
from os.path import join
from pathlib import Path
import geopandas as gpd
import hydromt
import pandas as pd

from typing import Optional, Union

from dask.diagnostics import ProgressBar
from hydromt.model.processes.meteo import temp

from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.snake_utils import (
    MIN_HISTORICAL_YEARS,
    log_row,
    meets_min_historical_years,
)
from blueearth_cst.spatial.delineate_region import delineate_region, read_region

#: Grid cells kept AROUND the basin bbox when reading a source.
#:
#: Two, not one, since 2026-08-10. The store is becoming the forcing source for
#: rule 1.10 instead of that rule re-reading the global dataset from the
#: catalog, and hydromt reads precipitation for a model region with
#: ``buffer=2`` (``hydromt_wflow/wflow_sbm.py:3288``,
#: ``setup_precip_forcing``). A store built at ``buffer=1`` is one ring short of
#: what that reader sees, so the regrid onto the model grid would differ at the
#: basin edge -- silently, and worst on the small basins this toolbox targets.
#:
#: The ring is NOT free downstream: weathergenr averages every cell in the store
#: (``compute_area_averages``, an unweighted mean over ``n_grids``), so widening
#: alone would dilute the series driving the stress test. That is why the
#: extraction also writes ``basin_cells.csv`` -- see ``write_basin_cell_mask``.
BUFFER_CELLS = 2


def write_basin_cell_mask(climate_nc, region_gdf, out_csv):
    """List the store cells the basin TOUCHES, for weathergenr to average over.

    weathergenr resamples on a spatial mean of every cell handed to it
    (``compute_area_averages``: ``daily_mat / n_grids``, no mask, no weights).
    The store is a bbox read plus ``BUFFER_CELLS``, so most of those cells can
    lie outside the basin, and the wider buffer the forcing path needs makes
    that worse. Measured on gabon_1008: the basin spans 0.80 x 0.53 ERA5 cells,
    the ``buffer=1`` store held 6, and the basin touches 2 -- so two thirds of
    the series driving that stress test was neighbouring climate.

    Selection is INTERSECTS, and the reason is the same basin: a
    centre-in-polygon test picks **zero** cells there, because the basin is
    smaller than one 0.25-degree cell and contains no cell centre. Centre tests
    are the obvious implementation and they fail exactly on the small basins CST
    exists for.

    Every selected cell counts EQUALLY (owner ruling 2026-08-10). No fractional
    area weighting: the cell either touches the basin or it does not, which
    avoids area arithmetic entirely and leaves weathergenr's own unweighted mean
    CORRECT for the subset it is given.

    Writes ``latitude,longitude`` for the kept cells. The consumer matches on
    those coordinates rather than on index order, so it cannot be silently
    broken by a change in how either side enumerates the grid.
    """
    import xarray as xr
    from shapely.geometry import box

    with xr.open_dataset(climate_nc) as ds:
        lats = [float(v) for v in ds["latitude"].values]
        lons = [float(v) for v in ds["longitude"].values]

    geom = region_gdf.union_all() if hasattr(region_gdf, "union_all") else region_gdf.unary_union
    half_lat = abs(lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.0
    half_lon = abs(lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.0

    kept = [
        (la, lo)
        for la in lats
        for lo in lons
        if geom.intersects(box(lo - half_lon, la - half_lat, lo + half_lon, la + half_lat))
    ]
    if not kept:
        # Unreachable for a region inside its own bbox read, but a store whose
        # grid degenerated to a single cell has zero half-width above and the
        # boxes collapse to points. Falling back to the nearest centre keeps the
        # contract non-empty; an empty list would hand weathergenr no data at
        # all, twenty rules from anything that could explain it.
        centre = geom.centroid
        kept = [
            min(
                ((la, lo) for la in lats for lo in lons),
                key=lambda c: abs(c[0] - centre.y) + abs(c[1] - centre.x),
            )
        ]

    frame = pd.DataFrame(kept, columns=["latitude", "longitude"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    log_row(
        f"Basin touches {len(frame)} of {len(lats) * len(lons)} store cells "
        f"-> {os.path.basename(str(out_csv))}",
        module="extract",
    )
    return frame


def _check_window_coverage(ds, starttime, endtime, clim_source):
    """Check what the extraction ACTUALLY covers against three expectations.

    The parse-time guard (``snake_utils.validate_historical_window``) can only
    check what the config REQUESTS. What the staged source holds is knowable
    only here, and a silently-truncated record is the original defect: a config
    asking 1980..2010 against an era5 staging that starts in 2000 yielded 11
    years with no signal, and WF3 then died on weathergenr's wavelet minimum
    twenty rules away (dev/tasks/ R3, observed 2026-05-07).

    Two deliberately separate comparisons:

    * **shortfall vs requested** -- advisory, with a 31-day tolerance. A source
      that begins three weeks late is normal, not an error, and this says
      nothing about whether what arrived is long enough.
    * **below MIN_HISTORICAL_YEARS** -- ``ValueError``. The same floor the
      parse-time guard applies to the requested window, applied here to what was
      actually delivered. Failing in the producer names the cause; failing in a
      consumer does not.

    The tolerance belongs to the first check only -- a floor with a tolerance is
    not a floor.

    This runs in the SHARED store producer, so the floor applies to WF2's rule
    2.11 and WF3's rule 3.02 as well as WF1's 1.10. That is the point of a
    unified floor: the store is one artifact serving all three, and a record too
    short for a stress test is a misconfigured project regardless of which
    workflow happens to be running.
    """
    try:
        time_vals = ds.time.values
        actual_start = pd.Timestamp(pd.to_datetime(time_vals.min()))
        actual_end = pd.Timestamp(pd.to_datetime(time_vals.max()))
        req_start = pd.Timestamp(pd.to_datetime(starttime))
        req_end = pd.Timestamp(pd.to_datetime(endtime))
    except (AttributeError, ValueError, TypeError):
        return  # cannot introspect the time axis -> skip the checks
    actual_days = (actual_end - actual_start).days

    tol = pd.Timedelta(days=31)
    if actual_start > req_start + tol or actual_end < req_end - tol:
        warnings.warn(
            f"Extracted {clim_source} window "
            f"{actual_start.date()}..{actual_end.date()} is shorter than the "
            f"requested {req_start.date()}..{req_end.date()}; the staged source "
            f"may not cover the full period.",
            stacklevel=2,
        )

    if not meets_min_historical_years(actual_start, actual_end):
        raise ValueError(
            f"Extracted {clim_source} record covers "
            f"{actual_start.date()}..{actual_end.date()} "
            f"(~{actual_days / 365.25:.1f} years) for the requested "
            f"{req_start.date()}..{req_end.date()}, below the "
            f"{MIN_HISTORICAL_YEARS}-year minimum this toolbox requires "
            f"(weathergenr's wavelet decomposition needs at least "
            f"{MIN_HISTORICAL_YEARS} annual observations). The staged "
            f"{clim_source} source does not cover the configured "
            f"historical_window. Either stage data for that period, or move "
            f"shared.historical_window onto the years the source actually holds"
        )


def prep_historical_climate(
    region_fn: Optional[Union[str, Path]],
    fn_out: Union[str, Path],
    data_libs: Union[str, Path] = "deltares_data",
    clim_source: str = "era5",
    *,
    starttime: str,
    endtime: str,
    bbox=None,
    oro_out: Optional[Union[str, Path]] = None,
    region_sha256: Optional[str] = None,
    region_source: Optional[Union[str, Path]] = None,
):
    """
    Extract historical climate data for a given region and time period.

    If clim_source is chirps or chirps_global, then only precip is extracted and will be
    combined with other climate data from era5.

    Parameters
    ----------
    region_fn : str, Path, optional
        Path to the region geojson file. Exactly one of ``region_fn`` and
        ``bbox`` must be provided.
    fn_out : str, Path
        Path to the output netcdf file
    data_libs : str, Path
        Path to the data catalogs yaml file or pre-defined catalogs
    clim_source : str
        Name of the climate source to use
    starttime : str
        Start time of the forcing, format YYYY-MM-DDTHH:MM:SS
    endtime : str
        End time of the forcing, format YYYY-MM-DDTHH:MM:SS
    bbox : tuple of float, optional
        Extraction bounds (xmin, ymin, xmax, ymax) used instead of the region
        file's total bounds. The rule passes
        ``read_region(...).total_bounds``; ``region_fn`` remains for
        standalone/unit use.
    oro_out : str, Path, optional
        Destination for the chirps/chirps_global orography sidecar. The rule
        passes its declared ``oro_nc`` output (``<store>/orography.nc``) so the
        DAG edge, not a filename convention, carries the DEM/climate
        co-provenance contract. Defaults to the historical
        ``<dirname(fn_out)>/<clim_source>_orography.nc`` when omitted. Ignored
        outside the chirps branch.
    region_sha256 : str, optional
        Content digest of the region artifact the ``bbox`` came from, stamped
        on the extraction as ``region_geojson_sha256`` (ADR 0003).
    region_source : str, Path, optional
        Path of that artifact, stamped as ``region_source``.
    """
    if (region_fn is None) == (bbox is None):
        raise ValueError(
            "prep_historical_climate: exactly one of region_fn or bbox must be provided"
        )
    if bbox is None:
        # Read region
        region = gpd.read_file(region_fn)
        bbox = region.geometry.total_bounds
    # Read data catalog
    data_catalog = hydromt.DataCatalog(data_libs=data_libs)

    # Extract climate data
    log_row("Extracting historical climate grid", module="extract")
    if clim_source == "chirps" or clim_source == "chirps_global":  # precip only
        log_row(
            f"{clim_source} only contains precipitation data. Combining with climate data from era5",
            module="extract",
        )
        # Get precip first
        ds = data_catalog.get_rasterdataset(
            clim_source,
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=["precip"],
        ).to_dataset()
        # Get clim
        ds_clim = data_catalog.get_rasterdataset(
            "era5",
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=["temp", "temp_min", "temp_max", "kin", "kout", "press_msl"],
        )
        # Prepare orography data corresponding to chirps from merit hydro DEM
        # (needed for downscaling of climate variables)
        log_row(
            f"Preparing orography data for {clim_source} to downscale climate variables.",
            module="extract",
        )
        dem = data_catalog.get_rasterdataset(
            "merit_hydro",
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=["elevtn"],
        )
        dem = dem.raster.reproject_like(ds, method="average")
        # Resample other variables and add to ds_precip
        log_row(f"Downscaling era5 variables to the resolution of {clim_source}", module="extract")
        for var in ["press_msl", "kin", "kout"]:
            ds[var] = ds_clim[var].raster.reproject_like(ds, method="nearest_index")

        # Read era5 dem for temp downscaling
        dem_era5 = data_catalog.get_rasterdataset(
            "era5_orography",
            geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
            buffer=2,
            variables=["elevtn"],
        ).squeeze()
        for var in ["temp", "temp_min", "temp_max"]:
            ds[var] = temp(
                ds_clim[var],
                dem,
                dem_forcing=dem_era5,
                lapse_correction=True,
                freq=None,
                reproj_method="nearest_index",
                lapse_rate=-0.0065,
            )
        # Save dem grid to netcdf, at the caller's declared output when given.
        fn_dem = (
            os.fspath(oro_out)
            if oro_out is not None
            else os.path.join(os.path.dirname(fn_out), f"{clim_source}_orography.nc")
        )
        dem.to_netcdf(fn_dem, mode="w")

    else:
        # Here we can afford larger chunks as we only extract and save.
        # In hydromt 1.x the source schema changed: chunks lives under
        # driver.options instead of the old top-level driver_kwargs.
        data_catalog_temp = data_catalog.to_dict()
        source = data_catalog_temp[clim_source]
        driver = source.setdefault("driver", {})
        if isinstance(driver, str):
            driver = {"name": driver}
            source["driver"] = driver
        driver.setdefault("options", {})["chunks"] = "auto"
        data_catalog = hydromt.DataCatalog().from_dict(data_catalog_temp)

        ds = data_catalog.get_rasterdataset(
            clim_source,
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=BUFFER_CELLS,
            variables=[
                "precip",
                "temp",
                "temp_min",
                "temp_max",
                "kin",
                "kout",
                "press_msl",
            ],
        )

    _check_window_coverage(ds, starttime, endtime, clim_source)

    # The store's extent provenance, IN the extraction rather than beside it
    # (ADR 0003). `store_region.geojson` used to sit in the store directory as
    # the record of where the bbox came from; the region is now one shared
    # project artifact, so a copy per store key would be a second source of
    # truth that can drift. Attributes cannot be separated from the data they
    # describe, and the sha256 says WHICH polygon, not merely which numbers.
    ds.attrs["region_bbox"] = [float(value) for value in bbox]
    if region_sha256 is not None:
        ds.attrs["region_geojson_sha256"] = region_sha256
    if region_source is not None:
        ds.attrs["region_source"] = os.fspath(region_source)

    dvars = ds.raster.vars
    encoding = {k: {"zlib": True} for k in dvars}

    log_row("Saving to netcdf", module="extract")
    delayed_obj = ds.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
    with ProgressBar():
        delayed_obj.compute()
    # Release the store handles deterministically rather than leaving them to
    # the garbage collector. Good practice on its own terms.
    #
    # It does NOT silence the "Error in sys.excepthook: / Original exception
    # was:" cascade that follows this rule under Snakemake -- measured, 14 lines
    # before and after. Recorded so nobody retries it: the cascade reproduces
    # ONLY under Snakemake's `script:` execution (0 lines standalone, same data
    # and same tee), and a probe excepthook installed to capture the original
    # exception never fired -- which is itself the diagnosis. By the time these
    # fire, CPython finalization has already torn module globals down, so any
    # excepthook fails and the interpreter prints the bare marker pair. It is
    # post-success noise from interpreter shutdown, not from this workflow.
    ds.close()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            # Two declared inputs (ADR 0003). The catalog is the store's
            # freshness boundary (R07 ext2-01), so an in-place catalog edit
            # mtime-triggers exactly one re-extraction; the region is the shared
            # project artifact, delineated once by rule 1.01b/2.03b/3.01b rather
            # than re-derived here per store key.
            catalog = sm.input.catalog
            region_fn = sm.input.region_geojson
            gdf = read_region(region_fn)
            prep_historical_climate(
                region_fn=None,
                fn_out=sm.output.climate_nc,
                data_libs=catalog,
                clim_source=sm.params.clim_source,
                starttime=sm.params.starttime,
                endtime=sm.params.endtime,
                bbox=tuple(gdf.total_bounds),
                # Absent outside the chirps/chirps_global branch, where the spec
                # declares no oro_nc output.
                oro_out=getattr(sm.output, "oro_nc", None),
                region_sha256=file_sha256(region_fn),
                region_source=region_fn,
            )
            # Which of the extracted cells the basin actually touches. Written
            # HERE rather than derived by the consumer because this is the only
            # place holding both the grid and the region polygon, and because R
            # has no geometry library in this env (no sf/terra/sp) -- so a
            # consumer-side mask would need a new dependency to do what one CSV
            # answers.
            write_basin_cell_mask(sm.output.climate_nc, gdf, sm.output.basin_cells)
    else:
        # Standalone demo (no Snakemake). Point the paths and the region at your
        # own project before running; the shape mirrors the rule above.
        demo_dir = join(os.getcwd(), "_climate_store_demo")
        demo_gdf = delineate_region(
            "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
            "deltares_data",
            region_out=join(demo_dir, "region.geojson"),
        )
        prep_historical_climate(
            region_fn=None,
            fn_out=join(demo_dir, "extract_historical.nc"),
            data_libs="deltares_data",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",
            bbox=tuple(demo_gdf.total_bounds),
        )
