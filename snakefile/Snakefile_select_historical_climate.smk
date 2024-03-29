import sys

# Get the snake_config file from the command line
args = sys.argv
config_path = args[args.index("--configfile") + 1]

# Parsing the Snakemake config file (options for basins to build, data catalog, model output directory)
#configfile: "config/snake_config_test.yml"

# Function to get argument from config file and return default value if not found
def get_config(config, arg, default=None, optional=True):
    """
    Function to get argument from config file and return default value if not found

    Parameters
    ----------
    config : dict
        config file
    arg : str   
        argument to get from config file
    default : str/int/float/list, optional
        default value if argument not found, by default None
    optional : bool, optional
        if True, argument is optional, by default True
    """
    if arg in config:
        return config[arg]
    elif optional:
        return default
    else:
        raise ValueError(f"Argument {arg} not found in config file")

project_dir = get_config(config, 'project_dir', optional=False)
basin_dir = f"{project_dir}/hydrology_model"

clim_historical_add = get_config(config, "clim_historical_add", optional=False)

#requires to have run Snakefile_model_creation first

# Rule to prepare the yml for each clim dataset with time horizon 
rule setup_runtime:
    input:
        gauges_fid = f"{basin_dir}/staticgeoms/gauges.geojson"
    output:
        forcing_yml = (project_dir + "/config/wflow_build_forcing_historical_{clim_historical_add}.yml")
    params:
        starttime = get_config(config, "starttime", optional=False),
        endtime = get_config(config, "endtime", optional=False),
        clim_source = "{clim_historical_add}",
    script: "../src/setup_time_horizon.py" #todo update path forcing name nc in script

# Rule to update the model for each additional forcing dataset - todo
rule add_forcing:
    input:
        forcing_ini = f"{project_dir}/config/wflow_build_forcing_historical.yml"
    output:
        forcing_fid = f"{project_dir}/climate_historical/wflow_data/inmaps_historical.nc"
    shell:
        """hydromt update wflow "{basin_dir}" -i "{input.forcing_ini}" -d "{DATA_SOURCES}" -vv"""

# Rule to run the wflow model for each additional forcing dataset - todo
rule run_wflow:
    input:
        forcing_fid = f"{project_dir}/climate_historical/wflow_data/inmaps_historical.nc"
    output:
        csv_file = f"{basin_dir}/run_default/output.csv"
    params:
        toml_fid = f"{basin_dir}/wflow_sbm.toml"
    shell:
        """ julia --threads 4 -e "using Wflow; Wflow.run()" "{params.toml_fid}" """