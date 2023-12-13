from hydromt_wflow import WflowModel
from pathlib import Path
import os
import numpy as np

# Snakemake parameters
config_out_fn = snakemake.output.toml
fn_out = Path(snakemake.output.nc)
fn_in = snakemake.input.nc
data_libs = snakemake.input.data_sources
model_root = snakemake.params.model_dir

precip_source = snakemake.params.clim_source

# Time horizon climate experiment and number of hydrological model run
horizontime_climate = snakemake.params.horizontime_climate
wflow_run_length = snakemake.params.run_length
# Get start and end year
startyear = int(horizontime_climate - np.ceil(wflow_run_length / 2))
endyear = int(horizontime_climate + np.round(wflow_run_length / 2))
starttime = f"{startyear}-01-01T00:00:00"
endtime = f"{endyear}-12-31T00:00:00"

oro_source = f"{precip_source}_orography"
if precip_source == "eobs":
    pet_method = "makkink"
else:  # (chirps is precip only so combined with era5)
    pet_method = "debruin"

# Get name of climate scenario (rlz_*_cst_*)
fn_in_path = Path(fn_in, resolve_path=True)
climate_name = os.path.basename(fn_in_path).split(".")[0]

# Get options for toml file name
config_out_fn = Path(config_out_fn)
config_out_root = os.path.dirname(config_out_fn)
config_out_name = os.path.basename(config_out_fn)

# Hydromt ini dictionnaries for update options
update_options = {
    "setup_config": {
        "calendar": "noleap",
        "starttime": starttime,
        "endtime": endtime,
        "timestepsecs": 86400,
        "state.path_input": os.path.join("..", "instate", "instates.nc"),
        "state.path_output": f"outstates_{climate_name}.nc",
        "input.path_static": os.path.join("..", "staticmaps.nc"),
        "input.path_forcing": os.path.join("..", "..", "..", "..", fn_out),
        "csv.path": f"output_{climate_name}.csv",
    },
    "set_root": {
        "root": config_out_root,
        "mode": "r+",
    },
    "setup_precip_forcing": {
        "precip_fn": climate_name,
        "precip_clim_fn": None,
    },
    "setup_temp_pet_forcing": {
        "temp_pet_fn": climate_name,
        "press_correction": True,
        "temp_correction": True,
        "dem_forcing_fn": oro_source,
        "pet_method": pet_method,
    },
    # "write_forcing": {},
    "write_config": {
        "config_name": config_out_name,
        "config_root": config_out_root,
    },
}

### Run Hydromt update using update_options dict ###
# Instantiate model
mod = WflowModel(root=model_root, mode="r+", data_libs=data_libs)

# Update
mod.update(opt=update_options)

# The slicing of DateTimeNoLeap is not so well done by hydromt
# Implement here
for var in mod.forcing.keys():
    da = mod.forcing[var]
    da = da.sel(time=slice(starttime, endtime))
    mod.forcing[var] = da

# Write forcing
mod.write_forcing()

# Weagen has strange timestamps, update in the wflow config
start = da.time.values[0].strftime(format="%Y-%m-%dT%H:%M:%S")
end = da.time.values[-1].strftime(format="%Y-%m-%dT%H:%M:%S")

mod.set_config("starttime", start)
mod.set_config("endtime", end)
mod.write_config(config_name=config_out_name, config_root=config_out_root)
