"""Extract historical climate data for a given region and time period.

Rule ``extract_climate_grid``'s script — the SINGLE producer of the shared
``climate_historical/<key>/`` store, declared identically in
``Snakefile_model_creation`` (1.10) and ``Snakefile_climate_experiment`` (3.02)
from ``snake_utils.climate_store_spec`` (R07 B1). The extraction extent is
derived **model-free**: the ``shared.basin`` region specification is delineated
against the data catalog via hydromt's ``parse_region_basin``, so nothing here
reads a built model. The delineated polygon is written as a declared output
(``store_region.geojson``) — the on-disk record of where the bbox came from.
"""

import ast
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
from hydromt.model.processes.region import parse_region_basin

from blueearth_cst.shared.snake_utils import (
    DEFAULT_BASIN_INDEX,
    DEFAULT_HYDROGRAPHY,
    MIN_HISTORICAL_YEARS,
    log_row,
    meets_min_historical_years,
)


def _check_window_coverage(ds, starttime, endtime, clim_source):
    """Check what the extraction ACTUALLY covers against three expectations.

    The parse-time guard (``snake_utils.validate_historical_window``) can only
    check what the config REQUESTS. What the staged source holds is knowable
    only here, and a silently-truncated record is the original defect: a config
    asking 1980..2010 against an era5 staging that starts in 2000 yielded 11
    years with no signal, and WF3 then died on weathergenr's wavelet minimum
    twenty rules away (dev/followups.md R3, observed 2026-05-07).

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


def delineate_store_region(
    model_region,
    data_libs: Union[str, Path],
    *,
    hydrography: str = DEFAULT_HYDROGRAPHY,
    basin_index: str = DEFAULT_BASIN_INDEX,
    region_out: Optional[Union[str, Path]] = None,
):
    """Delineate the store's region from the region spec + catalog (R07 B1).

    Model-free counterpart of the pre-R07 derivations, which read the extent
    either from the built model's ``staticmaps.nc`` (wf1) or from its
    ``staticgeoms/region.geojson`` (wf3). Both coupled a supposedly
    model-independent climate artifact to a hydrology build; this reads only
    ``shared.basin`` + the catalog, which is what lets one rule definition serve
    both workflows.

    ``hydrography``/``basin_index`` are catalog ENTRY NAMES, not paths —
    hydromt resolves them against ``data_libs`` itself (verified on the pinned
    hydromt 1.3.1). They default to the shipped build template's
    ``setup_basemaps`` values; rule 1.02 raises if the two ever disagree.

    Parameters
    ----------
    model_region : str | dict
        ``shared.basin.region``. A Python-dict-literal string (the form the
        snake config carries, e.g. ``"{'subbasin': [9.666, 0.4476],
        'uparea': 100}"``) is parsed with ``ast.literal_eval``, matching
        ``prepare_build_config.merge_build_config``.
    data_libs : str | Path
        Data catalog(s) to resolve the hydrography sources against.
    hydrography, basin_index : str
        Catalog entry names for the flow-direction data and its basin index.
    region_out : str | Path, optional
        When given, the delineated GeoDataFrame is written there as GeoJSON
        (parents created) — the store's ``store_region.geojson`` output.

    Returns
    -------
    geopandas.GeoDataFrame
        The delineated region; ``.total_bounds`` is the extraction bbox.
    """
    if isinstance(model_region, str):
        model_region = ast.literal_eval(model_region)

    data_catalog = hydromt.DataCatalog(data_libs=data_libs)
    log_row(f"Delineating store region {model_region} on {hydrography}", module="extract")
    gdf = parse_region_basin(
        model_region,
        data_catalog=data_catalog,
        hydrography_path=hydrography,
        basin_index_path=basin_index,
    )
    if region_out is not None:
        parent = os.path.dirname(os.fspath(region_out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        gdf.to_file(region_out, driver="GeoJSON")
        log_row(f"Wrote store region: {region_out}", module="extract")
    return gdf


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
        ``delineate_store_region(...).total_bounds``; ``region_fn`` remains for
        standalone/unit use.
    oro_out : str, Path, optional
        Destination for the chirps/chirps_global orography sidecar. The rule
        passes its declared ``oro_nc`` output (``<store>/orography.nc``) so the
        DAG edge, not a filename convention, carries the DEM/climate
        co-provenance contract. Defaults to the historical
        ``<dirname(fn_out)>/<clim_source>_orography.nc`` when omitted. Ignored
        outside the chirps branch.
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
            buffer=1,
            variables=["precip"],
        ).to_dataset()
        # Get clim
        ds_clim = data_catalog.get_rasterdataset(
            "era5",
            bbox=bbox,
            time_range=(starttime, endtime),
            buffer=1,
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
            buffer=1,
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
            buffer=1,
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
            # The catalog is the rule's single declared input (R07 ext2-01), so
            # an in-place catalog edit mtime-triggers exactly one re-extraction.
            catalog = sm.input.catalog
            gdf = delineate_store_region(
                sm.params.model_region,
                catalog,
                hydrography=sm.params.hydrography,
                basin_index=sm.params.basin_index,
                region_out=sm.output.region_geojson,
            )
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
            )
    else:
        # Standalone demo (no Snakemake). Point the paths and the region at your
        # own project before running; the shape mirrors the rule above.
        demo_dir = join(os.getcwd(), "_climate_store_demo")
        demo_gdf = delineate_store_region(
            "{'subbasin': [9.666, 0.4476], 'uparea': 100}",
            "deltares_data",
            region_out=join(demo_dir, "store_region.geojson"),
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
