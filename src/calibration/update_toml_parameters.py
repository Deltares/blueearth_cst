import os
import hydromt_wflow
from src.setuplog import setup_logging
import pandas as pd
from icecream import ic

#Read the data. SnakeMake arguments are automatically passed
logger = setup_logging("data/0-log/update_toml_parameters.log", "create_toml.log")

# Read the SnakeMake arguments
paramset = snakemake.params.calibration_parameters #instance
lnames = snakemake.params.lnames #the nc variable names
methods = snakemake.params.methods #the methods
wflow_vars = snakemake.params.wflow_vars #the wflow variables
setstring = snakemake.params.calibration_pattern #pattern
toml_default_fn = snakemake.input.toml_default
toml_out = snakemake.output.toml_fid
wflow_dir_input = snakemake.params.wflow_dir_input
calib_folder = snakemake.params.calib_folder
starttime = snakemake.params.starttime
endtime = snakemake.params.endtime
mode = snakemake.params.mode
## LOGGING
logger.info(f"Updating parameter string: {setstring}")     #string
logger.info(f"Updating parameter set: {paramset}")      #dictionary
logger.info(f"Updating parameter methods: {methods}")
logger.info(f"Updating parameter lnames: {lnames}")
logger.info(f"Updating parameter toml_default_fn: {toml_default_fn}")
logger.info(f"Updating parameter wflow_dir_input: {wflow_dir_input}")
logger.info(f"Updating parameter calib_folder: {calib_folder}")
logger.info(f"Updating parameter starttime: {starttime}")
logger.info(f"Updating parameter endtime: {endtime}")


# Instantiate wflow model and read/update config
root = os.path.dirname(toml_default_fn)
mod = hydromt_wflow.WflowModel(root, config_fn=os.path.basename(toml_default_fn), mode="r+")
# Parameters for which value is updated instead of a scale
parameters_value = ["tt", "ttm", "cfmax", "water_holding_capacity", "g_cfmax", "g_tt", "g_sifrac"]

def update_scale_toml(mod, param_name, param_value):
    # get param name in staticmaps netcdf
    param_netcdf = mod.config["input"]["vertical"][f"{param_name}"]
    if not isinstance(param_netcdf, str):
        if "netcdf" in param_netcdf:
            param_netcdf = param_netcdf["netcdf"]["variable"]["name"]
    mod.set_config(f"input.vertical.{param_name}.netcdf.variable.name", param_netcdf)
    mod.set_config(f"input.vertical.{param_name}.scale", float(param_value))

def update_toml(mod: hydromt_wflow.WflowModel, 
                method: str, 
                data_var: str, 
                wflow_var: str, 
                value: float) -> None:
    """
    Update the toml file with the new parameter value. Used only in calibration mode.

    - method (str): "scale", "set", "offset"
    - data_var (str): the name of the netcdf variable
    - wflow_var (str): the name of the wflow toml (config) reference
    - value (float): the new value for the parameter (combined with method)

    output: None
    """
    parts = wflow_var.split(".")
    param_name = parts[-1]
    logger.info(f"parts: {parts}, len: {len(parts)}")
    
    if method == "scale":
        logger.info(f"Updating parameter: {param_name}, scaling by: {value}")
        #TODO: Check if we need to pop the input.vertical ref to param_name 
        mod.set_config(f"{wflow_var}.netcdf.variable.name", data_var)
        mod.set_config(f"{wflow_var}.scale", float(value))    
    
    elif method == "set":
        logger.info(f"Updating parameter: {param_name}, setting to: {value}")
        #TODO: check if we need to add the var if it is just a constant
        mod.set_config(f"{wflow_var}.value", float(value))    
    
    elif method == "offset":
        logger.info(f"Updating parameter: {param_name}, offset by: {value}")
        mod.set_config(f"{wflow_var}.netcdf.variable.name", data_var)
        mod.set_config(f"{wflow_var}.offset", float(value))    

    
# elif isinstance(paramset, pd.DataFrame):
if mode == "sensitivity":
    """
    Adding this config argument to retain the original functionality i.e.
    csv based parameter sensitivity analysis, not the newer json based full grid parameter calibration. 
    These are customized instructions for setting up specific parameters for the sensitivity analysis.
    It's not super flexible in terms of setting up parameters, but it works. 
    """
    for param in paramset:
        if param == "ksathorfrac_BRT_250":
            mod.set_config("input.lateral.subsurface.ksathorfrac.netcdf.variable.name", param)
            mod.set_config("input.lateral.subsurface.ksathorfrac.scale", float(paramset[param]))
            # update_scale_toml(mod, "ksathorfrac", paramset[param])
        elif param == "f":
            if paramset[param] == "f_":
                mod.set_config(f"input.vertical.{param}", paramset[param])
            else: #scale
                update_scale_toml(mod, "f", paramset[param])
        elif param == "kv_0":
            # if paramset[param].startswith("KSatVer_"):
            if (paramset[param]== "KvB"): # | (paramset[param]== "KsatVer_Brakensiek_Bonetti") | (paramset[param]== "KsatVer_Cosby_Bonetti"):
                mod.set_config(f"input.vertical.{param}", paramset[param])
            else: #scale
                update_scale_toml(mod, "kv_0", paramset[param])
        elif param in parameters_value:
            mod.set_config(f"input.vertical.{param}", {"value": float(paramset[param])})
        else:
            update_scale_toml(mod, param, paramset[param])


elif mode == "calibration":
    snames = list(paramset.keys()) # e.g. ["ksat", "f"...]
    #shortnames (keys in paramset) related to methods
    name_method = {snames[i]: methods[i] for i in range(len(snames))} # e.g. {"ksat": "scale", "f": "set" ... }
    #shortnames (keys in paramset) related to gridfile data vars
    col_ds = {snames[i]: lnames[i] for i in range(len(snames))} # e.g. {"ksat": "ksathorfrac_BRT_250", "f": "f_" ... }
    #shortnames (keys in paramset) related to wflow parameter concepts
    col_wflow_var = {snames[i]: wflow_vars[i] for i in range(len(snames))} # e.g. {"ksat": "input.lateral.subsurface.ksathorfrac", "f": "input.vertical.f" ... }
    for param in paramset:
        method = name_method[param]
        data_var = col_ds[param]
        wflow_var = col_wflow_var[param]
        value = paramset[param]
        update_toml(mod, method, data_var, wflow_var, value)

# Find the realtive path of toml_out to the wflow_dir_input
rel_path = os.path.relpath(wflow_dir_input, os.path.dirname(toml_out) )

# Update parameters for saving outputs
setting_toml = {
    "starttime": starttime,
    "endtime": endtime,
    "state.path_output": f"outstates_{setstring}.nc",
    "csv.path": f"output_{setstring}.csv",
    "dir_input": rel_path,
    "dir_output": "",
}

for option in setting_toml:
    mod.set_config(option, setting_toml[option])

# Remove the output section from the config dict
mod._config.pop("output", None)

# Write new toml file
# toml_root = os.path.join(os.path.dirname(toml_default_fn))
toml_root = os.path.join(wflow_dir_input, calib_folder)
toml_name = f"wflow_sbm_{setstring}.toml"
mod.write_config(config_name=toml_name, config_root=toml_root)
