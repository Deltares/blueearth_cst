import os
import geopandas as gpd
import xarray as xr
import hydromt

from dask.diagnostics import ProgressBar
from hydromt.workflows.forcing import temp

# Snake parameters
region_fn = snakemake.input.prj_region
fn_out = snakemake.output.climate_nc
starttime = "1990-01-01T00:00:00" #snakemake.params.starttime
endtime = "2020-12-31T00:00:00" #snakemake.params.endtime
data_libs = snakemake.params.data_sources

region = gpd.read_file(region_fn)

source = snakemake.params.clim_source

data_catalog = hydromt.DataCatalog(data_libs=data_libs)
print("Extracting historical climate grid")
if source == "chirps" or source == "chirps_global":  # precip only
    print(
        f"{source} only contains precipitation data. Combining with climate data from era5"
    )
    # Get precip first
    ds = data_catalog.get_rasterdataset(
        source,
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
    # Prepare orography data corresponding to chirps from merit hydro DEM (needed for downscaling of climate variables)
    print(f"Preparing orography data for {source} to downscale climate variables.")
    dem = data_catalog.get_rasterdataset(
        "merit_hydro",
        bbox=region.geometry.total_bounds,
        time_tuple=(starttime, endtime),
        buffer=1,
        variables=["elevtn"],
    )
    dem = dem.raster.reproject_like(ds, method="average")
    # Resample other variables and add to ds_precip
    print(f"Downscaling era5 variables to the resolution of {source}")
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
    fn_dem = os.path.join(os.path.dirname(fn_out), f"{source}_orography.nc")
    dem.to_netcdf(fn_dem, mode="w")

else:
    ds = data_catalog.get_rasterdataset(
        source,
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
