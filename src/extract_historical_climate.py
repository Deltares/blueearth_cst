"""Extract historical climate data for a given region and time period."""

import os
from pathlib import Path
import geopandas as gpd
import numpy as np
import hydromt

from typing import Union, List

from dask.diagnostics import ProgressBar
from hydromt.workflows.forcing import temp


def prep_historical_climate(
    region_fn: Union[str, Path],
    fn_out: Union[str, Path],
    data_libs: Union[str, Path] = "deltares_data",
    clim_source: str = "era5",
    climate_variables: List[str] = ["precip", "temp"],
    starttime: str = "1980-01-01T00:00:00",
    endtime: str = "2010-12-31T00:00:00",
    buffer: int = 1,
    combine_with_era5: bool = False,
):
    """
    Extract historical climate data for a given region and time period.

    If clim_source only contains precip, then only precip is extracted and will be
    combined with other climate data from era5 if combine_with_era5 is True.

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
    climate_variables : list
        List of climate variables to extract
    starttime : str
        Start time of the forcing, format YYYY-MM-DDTHH:MM:SS
    endtime : str
        End time of the forcing, format YYYY-MM-DDTHH:MM:SS
    buffer : int
        Buffer in km around the region to extract the data.
    combine_with_era5 : bool
        If True, missing variables will be extracted from era5 and downscaled to the
        resolution of the clim_source.
    """
    # Read region
    region = gpd.read_file(region_fn)
    # Read data catalog
    data_catalog = hydromt.DataCatalog(data_libs=data_libs)

    # Extract climate data
    # Here we can afford larger chunks as we only extract and save
    # data_catalog_temp = data_catalog.to_dict()
    # data_catalog_temp[clim_source]["driver_kwargs"]["chunks"] = "auto"
    # data_catalog = hydromt.DataCatalog().from_dict(data_catalog_temp)

    # Load the data
    ds = data_catalog.get_rasterdataset(
        clim_source,
        bbox=region.geometry.total_bounds,
        time_tuple=(starttime, endtime),
        buffer=buffer,
        single_var_as_array=False,
    )

    # Check the size
    if np.any(np.asarray(ds.raster.shape) == 1):
        raise ValueError(
            f"{clim_source} does not contain enough cells in the "
            "region (at least 2*2). Try increasing the buffer."
        )

    # Find which of climate_variables are in the dataset
    variables_in_ds = [v for v in climate_variables if v in ds.data_vars]
    if not variables_in_ds:
        raise ValueError(
            f"None of the climate variables {variables_in_ds} are present in the dataset"
        )
    # Select only the variables that are present in the dataset
    ds = ds[variables_in_ds]

    # For missing variables try to extract from era5 if allowed
    if len(variables_in_ds) < len(climate_variables) and combine_with_era5:
        # Load era5 data
        ds_clim = data_catalog.get_rasterdataset(
            "era5",
            bbox=region.geometry.total_bounds,
            time_tuple=(starttime, endtime),
            buffer=buffer,
            single_var_as_array=False,
        )
        # Find which of the missing climate_variables are in the era5 dataset
        variables_in_era5 = [
            v
            for v in climate_variables
            if v in ds_clim.data_vars and v not in ds.data_vars
        ]
        # Select only the variables that are present in the era5 dataset
        ds_clim = ds_clim[variables_in_era5]

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

    # Add clim_source to coordinates
    ds.coords["source"] = clim_source

    # Save to netcdf
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
            region_fn=sm.input.region_fn,
            fn_out=sm.output.climate_nc,
            data_libs=sm.params.data_sources,
            clim_source=sm.params.clim_source,
            climate_variables=sm.params.climate_variables,
            starttime=sm.params.starttime,
            endtime=sm.params.endtime,
            buffer=sm.params.buffer_km,
            combine_with_era5=sm.params.combine_with_era5,
        )
    else:
        print("This script should be run from a snakemake environment")
