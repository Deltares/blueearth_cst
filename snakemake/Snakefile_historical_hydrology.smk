import sys
import numpy as np 

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
model_region = get_config(config, 'model_region', optional=False)
model_resolution = get_config(config, 'model_resolution', 0.00833333)
model_build_config = get_config(config, 'model_build_config', 'config/cst_api/wflow_build_model.yml')
waterbodies_config = get_config(config, 'waterbodies_config', 'config/cst_api/wflow_update_waterbodies.yml')
climate_sources = get_config(config, "clim_historical", optional=False)
climate_sources_colors = get_config(config, "climate_sources_colors", optional=True)
DATA_SOURCES = get_config(config, "data_sources", optional=False)
DATA_SOURCES = np.atleast_1d(DATA_SOURCES).tolist() #make sure DATA_SOURCES is a list format (even if only one DATA_SOURCE)

output_locations = get_config(config, "output_locations", None)
observations_timeseries = get_config(config, "observations_timeseries", None)

wflow_outvars = get_config(config, "wflow_outvars", ['river discharge'])

# Master rule: end with all model run and analysed with saving a output plot
rule all:
    input: 
        f"{project_dir}/plots/wflow_model_performance/hydro_wflow_1.png",
        f"{project_dir}/plots/wflow_model_performance/basin_area.png",
        expand((project_dir + "/plots/wflow_model_performance/{climate_source}/precip.png"), climate_source = climate_sources),
        f"{project_dir}/config/snake_config_model_creation.yml",

# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_build = model_build_config,
        config_snake = config_path,
        config_waterbodies = waterbodies_config,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "model_creation",
    output:
        config_snake_out = f"{project_dir}/config/snake_config_model_creation.yml",
    script:
        "../src/copy_config_files.py"

# Rule to build model hydromt build wflow
rule create_model:
    input:
        hydromt_ini = model_build_config,
    output:
        basin_nc = f"{basin_dir}/staticmaps.nc",
    params:
        data_catalogs = [f"-d {cat} " for cat in DATA_SOURCES]  
    shell:
        """hydromt build wflow "{basin_dir}" --region "{model_region}" --opt setup_basemaps.res="{model_resolution}" -i "{input.hydromt_ini}" {params.data_catalogs} --fo -vv"""

# Rule to add reservoirs, lakes and glaciers to the built model (temporary hydromt fix)
# Can be moved back to create_model rule when hydromt is updated
rule add_reservoirs_lakes_glaciers:
    input:
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc")
    output:
        text_out = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt"
    params:
        data_catalog = DATA_SOURCES,
        config = waterbodies_config,
    script:
        "../src/setup_reservoirs_lakes_glaciers.py"

# Rule to add gauges to the built model
rule add_gauges_and_outputs:
    input:
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"),
        text = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt"
    output:
        gauges_fid = f"{basin_dir}/staticgeoms/gauges.geojson"
    params:
        output_locs = output_locations,
        outputs = wflow_outvars,
        data_catalog = DATA_SOURCES
    script:
        "../src/setup_gauges_and_outputs.py"

# Rule to prepare the yml for each clim dataset with time horizon 
rule setup_runtime:
    input:
        gauges_fid = f"{basin_dir}/staticgeoms/gauges.geojson"
    output:
        forcing_yml = (project_dir + "/config/wflow_build_forcing_historical_{climate_source}.yml")
    params:
        starttime = get_config(config, "starttime", optional=False),
        endtime = get_config(config, "endtime", optional=False),
        clim_source = "{climate_source}",
        suffix=True, #add climate_source name to config_name and forcing_path name. 
    script: "../src/setup_time_horizon.py" 

# Rule to update the model for each additional forcing dataset 
rule add_forcing:
    input:
        forcing_ini = (project_dir + "/config/wflow_build_forcing_historical_{climate_source}.yml")
    output:
        forcing_fid = (project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc")
    params:
        data_catalogs = [f"-d {cat} " for cat in DATA_SOURCES] 
    shell:
        """hydromt update wflow "{basin_dir}" -i "{input.forcing_ini}" {params.data_catalogs} -vv"""

#Rule to run the wflow model for each additional forcing dataset 
rule run_wflow:
    input:
        forcing_fid = (project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc")
    output:
        csv_file = (basin_dir + "/run_default_{climate_source}/output.csv")
    params:
        toml_fid = (basin_dir + "/wflow_sbm_{climate_source}.toml"),
    shell:
        """ julia --threads 4 -e "using Wflow; Wflow.run()" "{params.toml_fid}" """

# Rule to analyse and plot wflow model run results --> final output
rule plot_results:
   input:
       csv_file = expand((basin_dir + "/run_default_{climate_source}/output.csv"), climate_source = climate_sources),
       script = "src/plot_results.py"
   output: 
       output_png = f"{project_dir}/plots/wflow_model_performance/hydro_wflow_1.png",
   params:
       project_dir = f"{project_dir}",
       observations_file = observations_timeseries,
       gauges_output_fid = output_locations,
       climate_sources = climate_sources,
       climate_sources_colors = climate_sources_colors,
   script: "../src/plot_results.py"

# Rule to plot the wflow basin, rivers, gauges and DEM on a map
rule plot_map:
    input:
        gauges_fid = f"{basin_dir}/staticgeoms/gauges.geojson"
    output:
        output_map_png = f"{project_dir}/plots/wflow_model_performance/basin_area.png",
    params:
        project_dir = f"{project_dir}",
        output_locations = output_locations,
    script: "../src/plot_map.py"

# Rule to plot the forcing on a map
rule plot_forcing:
    input:
        # forcing_fid = expand((project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc"), climate_source = climate_sources),
        forcing_fid = (project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc"),
    output:
        output_forcing_map = (project_dir + "/plots/wflow_model_performance/{climate_source}/precip.png"),
    params:
        project_dir = f"{project_dir}",
        gauges_fid = output_locations,
        config_fn=("wflow_sbm_{climate_source}.toml"),
        climate_source="{climate_source}",
    script: "../src/plot_map_forcing.py"