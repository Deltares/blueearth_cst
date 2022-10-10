import hydromt
from hydromt_wflow import WflowModel
from pathlib import Path
import os

# Snakemake parameters
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime
config_out_fn = snakemake.output.toml
fn_out = snakemake.output.nc
fn_in = snakemake.input.nc
data_libs = snakemake.input.data_sources
model_root = snakemake.params.model_dir

precip_source = snakemake.params.clim_source

oro_source = f'{precip_source}_orography'
if precip_source == 'eobs':
    pet_method = 'makkink'
else: # (chirps is precip only so combined with era5)
    pet_method = 'debruin'

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
        "state.path_input": "../instate/instates.nc",
        "state.path_output": f"outstates_{climate_name}.nc",
        "input.path_static": "../staticmaps.nc",
        "input.path_forcing": f"../../../{fn_out}",
        "csv.path": f"output_{climate_name}.csv",
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
    "write_forcing":{},
    "setup_config1": {
        "input.path_forcing": f"../../../../{fn_out}", # the TOML will be saved one extra folder into model_root so need to re-update forcing path after writting
    },
    "write_config":{
        "config_name": config_out_name,
        "config_root": config_out_root, 
    },
}

### Run Hydromt update using update_options dict ###
# Instantiate model
mod = WflowModel(root=model_root, mode="r+", data_libs=data_libs)

# Update
mod.update(opt=update_options)
