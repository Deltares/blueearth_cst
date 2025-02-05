import argparse as ap
import numpy as np 
from pathlib import Path
from get_config import get_config

args = sys.argv
config_path = args[args.index("--configfile") + 1]
config_dir = Path(config_path).parent
project_dir = get_config(config, 'project_dir', optional=False)
basin_dir = f"{project_dir}/hydrology_model"
model_region = get_config(config, 'model_region', optional=False)
model_resolution = get_config(config, 'model_resolution', default=0.00833333)
model_build_config = get_config(config, 'model_build_config', default='config/cst_api/wflow_build_model.yml')
waterbodies_config = get_config(config, 'waterbodies_config', default='config/cst_api/wflow_update_waterbodies.yml')
climate_sources = list(get_config(config, "forcing_options", optional=False).keys())
DATA_SOURCES = get_config(config, "data_sources", optional=False)
DATA_SOURCES = np.atleast_1d(DATA_SOURCES).tolist() #make sure DATA_SOURCES is a list format (even if only one DATA_SOURCE)
DATA_SOURCES = [f"{project_dir}/{cat}" for cat in DATA_SOURCES]
output_locations = get_config(config, "output_locations", default=None)
observations_timeseries = get_config(config, "observations_timeseries", default=None)

wflow_outvars = get_config(config, "wflow_outvars", default=['river discharge'])
has_gridded_outputs = len(get_config(config, "wflow_outvars_gridded", default=[])) > 0

#paralellisation on HPC
total_mem = int(get_config(config, "total_mem", default=32768))
group_modelruns = int(get_config(config, "group_modelruns", default=4))
threads_available = int(get_config(config, "threads_available", default=4))

### Custom Python functions (here to access dictionnary elements from the config based on wildcards)
def get_forcing_options(wildcards):
    if "forcing_options" in config:
        if wildcards.climate_source in config["forcing_options"]:
            opts = config["forcing_options"][wildcards.climate_source]
    else:
        opts = {}
    return opts

def get_climate_sources_colors(config, climate_sources):
    colors = []
    for source in climate_sources:
        color = get_config(config, f"forcing_options.{source}.color", default=None)
        if color is not None:
            colors.append(color)
    # Check that colors were found for all sources
    if len(colors) != len(climate_sources):
        print("Not all climate sources have a color defined in the config file, using defaults")
        colors = None
    return colors

# Master rule: end with all model run and analysed with saving a output plot
rule all:
    input: 
        f"{project_dir}/plots/wflow_model_performance/plot_results.txt",
        f"{project_dir}/plots/wflow_model_performance/basin_area.png",
        expand((project_dir + "/plots/wflow_model_performance/{climate_source}/precip.png"), climate_source = climate_sources),
        f"{basin_dir}/snake_config_model_creation.yml",
        f"{project_dir}/plots/wflow_model_performance/gridded_output.txt",

# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_build = f"{config_dir}/{model_build_config}",
        config_waterbodies = f"{config_dir}/{waterbodies_config}",
        config_snake = config_path,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "model_creation",
    output:
        config_snake_out = f"{basin_dir}/wflow_build_model.yml",
    localrule: True
    script:
        "../src/copy_config_files.py"

# Rule to build model hydromt build wflow
rule create_model:
    input:
        hydromt_ini = f"{basin_dir}/wflow_build_model.yml",
    output:
        basin_nc = f"{basin_dir}/staticmaps.nc",
        model_created = touch(f"{basin_dir}/model.created")
    params:
        data_catalogs = [f'-d "{cat}" ' for cat in DATA_SOURCES]  
    localrule: True
    shell:
        """hydromt build wflow "{basin_dir}" --region "{model_region}" --opt setup_basemaps.res="{model_resolution}" -i "{input.hydromt_ini}" {params.data_catalogs} --fo -vv"""

# Rule to add reservoirs, lakes and glaciers to the built model (temporary hydromt fix)
# Can be moved back to create_model rule when hydromt is updated
rule add_reservoirs_lakes_glaciers:
    input:
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"), #should be ancient as it is regularly updated
        model_created = f"{basin_dir}/model.created" #let this timestamp control the rule
    output:
        text_out = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt",
        waterbodies_added = touch(f"{basin_dir}/waterbodies.added")
    params:
        data_catalog = DATA_SOURCES,
        config = f"{basin_dir}/{waterbodies_config}",
    localrule: True
    script:
        "../src/setup_reservoirs_lakes_glaciers.py"

# Rule to add gauges to the built model
rule add_gauges_and_outputs:
    input:
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"),
        text = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt",
        waterbodies_added = f"{basin_dir}/waterbodies.added"
    output:
        gauges_fid = f"{basin_dir}/staticgeoms/gauges.geojson",
        gauges_added = touch(f"{basin_dir}/gauges.added")
    params:
        output_locs = output_locations,
        outputs = wflow_outvars,
        outputs_gridded = get_config(config, "wflow_outvars_gridded", default=None),
        data_catalog = DATA_SOURCES,
    localrule: True
    script:
        "../src/setup_gauges_and_outputs.py"

# Rule to prepare the yml for each clim dataset with time horizon 
rule setup_runtime:
    input:
        gauges_fid = f"{basin_dir}/staticgeoms/gauges.geojson",
        gauges_added = f"{basin_dir}/gauges.added"
    output:
        forcing_yml = (f"{basin_dir}"+"/wflow_build_forcing_historical_{climate_source}.yml")
    params:
        starttime = get_config(config, "starttime", optional=False),
        endtime = get_config(config, "endtime", optional=False),
        clim_source = "{climate_source}",
        forcing_options = get_forcing_options,
        basin_dir = basin_dir,
    localrule: True
    script: "../src/setup_time_horizon.py"

# Rule to update the model for each additional forcing dataset 
rule add_forcing:
    input:
        forcing_ini = ancient(f"{basin_dir}"+"/wflow_build_forcing_historical_{climate_source}.yml")
    output:
        forcing_fid = (project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc")
    params:
        data_catalogs = [f'-d "{cat}" ' for cat in DATA_SOURCES] 
    localrule: False
    resources:
        mem_mb=total_mem,
        threads=threads_available #Fastest way to generate the fo
    shell:
        """hydromt update wflow "{basin_dir}" -i "{input.forcing_ini}" {params.data_catalogs} -vv"""

# Make sure all the forcing files are updated before running the model
rule confirm_forcing_update:
    input:
        forcing_fid = expand((project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc"), climate_source = climate_sources)
    output:
        forcing_fid = touch(f"{basin_dir}/forcing.updated")

#Rule to run the wflow model for each additional forcing dataset 
rule run_wflow:
    input:
        forcing_fid = (project_dir + "/climate_historical/wflow_data/inmaps_historical_{climate_source}.nc"),
        forcing_updated = f"{basin_dir}/forcing.updated"
    output:
        csv_file = (basin_dir + "/run_default/output_{climate_source}.csv"),
        nc_file = (basin_dir + "/run_default/output_{climate_source}.nc") if has_gridded_outputs else []
    params:
        toml_fid = (basin_dir + "/run_default/wflow_sbm_{climate_source}.toml")
    localrule: False
    group: "run_wflow"
    resources:
        threads = int(threads_available/group_modelruns),
        mem_mb= int(total_mem/group_modelruns)
    shell:
        """ julia --threads {resources.threads} -e "using Wflow; Wflow.run()" "{params.toml_fid}" """

# Rule to analyse and plot wflow model run results --> final output
rule plot_results:
   input:
       csv_file = expand((basin_dir + "/run_default/output_{climate_source}.csv"), climate_source = climate_sources),
   output: 
       output_txt = f"{project_dir}/plots/wflow_model_performance/plot_results.txt",
   params:
       project_dir = f"{project_dir}",
       observations_file = f"{project_dir}/{observations_timeseries}",
       gauges_output_fid = output_locations,
       climate_sources = f"{project_dir}/{climate_sources}",
       climate_sources_colors = get_climate_sources_colors(config, climate_sources),
       add_budyko_plot = get_config(config, "historical_hydrology_plots.plot_budyko", default=False),
       max_nan_year = get_config(config, "historical_hydrology_plots.flow.max_nan_per_year", default=60),
       max_nan_month = get_config(config, "historical_hydrology_plots.flow.max_nan_per_month", default=5),
       skip_precip_sources = get_config(config, "historical_hydrology_plots.clim.skip_precip_sources", default=[]),
       skip_temp_pet_sources = get_config(config, "historical_hydrology_plots.clim.skip_temp_pet_sources", default=[]),
   localrule: True
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
        output_locations_legend = get_config(config, "historical_hydrology_plots.basin_map.output_locations_legend", default="output locations"),
        data_catalog = DATA_SOURCES,
        meteo_locations = get_config(config, "climate_locations", default=None),
        buffer_km = get_config(config, "region_buffer", default=2),
        legend_loc = get_config(config, "historical_hydrology_plots.basin_map.legend_loc", default="lower right"),
    localrule: True
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
    localrule: True
    script: "../src/plot_map_forcing.py"

rule plot_gridded_results:
    input:
        nc_files = expand((basin_dir + "/run_default/output_{climate_source}.nc"), climate_source = climate_sources) if has_gridded_outputs else [],
    output:
        output_txt = (project_dir + "/plots/wflow_model_performance/gridded_output.txt")
    params:
        project_dir = f"{project_dir}",
        climate_sources = climate_sources,
        data_catalog = DATA_SOURCES,
        observations_snow = get_config(config, "observations_snow", default=None),
    localrule: True
    script: "../src/plot_results_grid.py"   