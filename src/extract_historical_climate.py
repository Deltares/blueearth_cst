"""Extract historical climate data for a given region and time period."""

import os
from os.path import join
from pathlib import Path
import geopandas as gpd
import hydromt

from typing import Union

from dask.diagnostics import ProgressBar
from hydromt.workflows.forcing import temp


def prep_historical_climate(
    region_fn: Union[str, Path],
    fn_out: Union[str, Path],
    data_libs: Union[str, Path] = "deltares_data",
    clim_source: str = "era5",
    starttime: str = "1980-01-01T00:00:00",
    endtime: str = "2010-12-31T00:00:00",
):
    """
    Extract historical climate data for a given region and time period.

    If clim_source is chirps or chirps_global, then only precip is extracted and will be
    combined with other climate data from era5.

    Parameters
    ----------
    region_fn : str, Path
        Path to the region geojson file
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
    """
    # Read region
    region = gpd.read_file(region_fn)
    # Read data catalog
    data_catalog = hydromt.DataCatalog(data_libs=data_libs)

    # Extract climate data
    print("Extracting historical climate grid")
    if clim_source == "chirps" or clim_source == "chirps_global":  # precip only
        print(
            f"{clim_source} only contains precipitation data. Combining with climate data from era5"
        )
        # Get precip first
        ds = data_catalog.get_rasterdataset(
            clim_source,
            bbox=region.geometry.total_bounds,
            time_tuple=(starttime, endtime),
            buffer=1,
            variables=["precip"],
        ).to_dataset()
        # Get clim
        ds_clim = data_catalog.get_rasterdataset(
            "era5",
            bbox=region.geometry.total_bounds,
            time_tuple=(starttime, endtime),
            buffer=1,
            variables=["temp", "temp_min", "temp_max", "kin", "kout", "press_msl"],
        )
        # Prepare orography data corresponding to chirps from merit hydro DEM
        # (needed for downscaling of climate variables)
        print(
            f"Preparing orography data for {clim_source} to downscale climate variables."
        )
        dem = data_catalog.get_rasterdataset(
            "merit_hydro",
            bbox=region.geometry.total_bounds,
            time_tuple=(starttime, endtime),
            buffer=1,
            variables=["elevtn"],
        )
        dem = dem.raster.reproject_like(ds, method="average")
        # Resample other variables and add to ds_precip
        print(f"Downscaling era5 variables to the resolution of {clim_source}")
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
        ds = data_catalog.get_rasterdataset(
            clim_source,
            bbox=region.geometry.total_bounds,
            time_tuple=(starttime, endtime),
            buffer=1,
        )

    dvars = ds.raster.vars
    encoding = {k: {"zlib": True} for k in dvars}

    print("Saving to netcdf")
    delayed_obj = ds.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
    with ProgressBar():
        delayed_obj.compute()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        prep_historical_climate(
            region_fn=sm.input.prj_region,
            fn_out=sm.output.climate_nc,
            data_libs=sm.params.data_sources,
            clim_source=sm.params.clim_source,
            starttime="1980-01-01T00:00:00",
            endtime="2010-12-31T00:00:00",
        )
    else:
        prep_historical_climate(
            region_fn=join(
                os.getcwd(),
                "examples",
                "my_project",
                "hydrology_model",
                "staticgeoms",
                "region.geojson",
            ),
            fn_out=join(
                os.getcwd(),
                "examples",
                "my_project",
                "climate_historical",
                "raw_data",
                "extract_historical.nc",
            ),
            data_libs="deltares_data",
            clim_source="era5",
            starttime="1980-01-01T00:00:00",
            endtime="2010-12-31T00:00:00",
        )
