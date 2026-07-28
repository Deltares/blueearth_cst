"""Extract historical climate data for a given region and time period."""

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

from blueearth_cst.shared.snake_utils import log_row


def _warn_if_window_truncated(ds, starttime, endtime, clim_source):
    """Warn when the extracted data covers a shorter span than requested.

    A silently-truncated window -- the staged source lacks the full requested
    range -- otherwise surfaces far downstream as a cryptic failure, e.g.
    weathergenr's 16-year wavelet minimum ('series' must have at least 16
    observations). Advisory only; never blocks extraction. (t260716a,
    dev/followups.md R3)
    """
    try:
        time_vals = ds.time.values
        actual_start = pd.Timestamp(pd.to_datetime(time_vals.min()))
        actual_end = pd.Timestamp(pd.to_datetime(time_vals.max()))
        req_start = pd.Timestamp(pd.to_datetime(starttime))
        req_end = pd.Timestamp(pd.to_datetime(endtime))
    except (AttributeError, ValueError, TypeError):
        return  # cannot introspect the time axis -> skip the advisory check
    tol = pd.Timedelta(days=31)
    if actual_start > req_start + tol or actual_end < req_end - tol:
        warnings.warn(
            f"Extracted {clim_source} window "
            f"{actual_start.date()}..{actual_end.date()} is shorter than the "
            f"requested {req_start.date()}..{req_end.date()}; the staged source "
            f"may not cover the full period. Downstream steps (e.g. weathergenr's "
            f"16-year minimum) can fail on a truncated record.",
            stacklevel=2,
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
        file's total bounds (P3-2a: the wf1 caller passes the staticmaps.nc
        model-grid bounds; wf3 keeps passing ``region_fn``).
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
        # Save dem grid to netcdf
        fn_dem = os.path.join(os.path.dirname(fn_out), f"{clim_source}_orography.nc")
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

    _warn_if_window_truncated(ds, starttime, endtime, clim_source)

    dvars = ds.raster.vars
    encoding = {k: {"zlib": True} for k in dvars}

    log_row("Saving to netcdf", module="extract")
    delayed_obj = ds.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
    with ProgressBar():
        delayed_obj.compute()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            prep_historical_climate(
                region_fn=sm.input.prj_region,
                fn_out=sm.output.climate_nc,
                data_libs=sm.params.data_sources,
                clim_source=sm.params.clim_source,
                starttime=sm.params.starttime,
                endtime=sm.params.endtime,
            )
    else:
        prep_historical_climate(
            region_fn=join(
                os.getcwd(),
                "test_case",
                "my_project",
                "hydrology_model",
                "staticgeoms",
                "region.geojson",
            ),
            fn_out=join(
                os.getcwd(),
                "test_case",
                "my_project",
                "climate_historical",
                "raw_data",
                "extract_historical.nc",
            ),
            data_libs="deltares_data",
            clim_source="era5",
            starttime="2000-01-01T00:00:00",
            endtime="2020-12-31T00:00:00",
        )
