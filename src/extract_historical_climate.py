import os
import geopandas as gpd
import xarray as xr
import hydromt

from dask.diagnostics import ProgressBar

# Snake parameters
region_fn = snakemake.input.prj_region
fn_out = snakemake.output.climate_nc
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime

region = gpd.read_file(region_fn)

source = 'era5_daily'

data_catalog = hydromt.DataCatalog(deltares_data=True)
print("Extracting historical climate grid")
ds = data_catalog.get_rasterdataset(
    source, 
    bbox = region.geometry.total_bounds,
    time_tuple = (starttime, endtime),
    buffer = 1,
)

dvars = ds.raster.vars
encoding={k: {"zlib": True} for k in dvars}

print("Saving to netcdf")
delayed_obj = ds.to_netcdf(fn_out, encoding=encoding, mode="w", compute=False)
with ProgressBar():
    delayed_obj.compute()