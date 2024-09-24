import sys
import numpy as np 
import glob

from get_config import get_config

# Get the snake_config file from the command line
args = sys.argv
config_path = args[args.index("--configfile") + 1]

# Parsing the Snakemake config file (options for basins to build, data catalog, model output directory)
#configfile: "config/snake_config_test.yml"

project_dir = get_config(config, 'project_dir', optional=False)
basin_dir = f"{project_dir}/hydrology_model"
DATA_SOURCES = get_config(config, "data_sources", optional=False)
DATA_SOURCES = np.atleast_1d(DATA_SOURCES).tolist() #make sure DATA_SOURCES is a list format (even if only one DATA_SOURCE)

config_model_historical = get_config(config, 'config_model_historical', optional=False)
config_model_historical_fn = f"{basin_dir}/{config_model_historical}"
config_basename = os.path.basename(config_model_historical).split(".")[0]

clim_project = get_config(config, 'clim_project', optional=False)
gcms_selected = get_config(config, 'gcm_selected', optional=False)
scenarios_selected = get_config(config, 'scenarios_selected', optional=False)

clim_project_dir = f"{project_dir}/climate_projections/{clim_project}"

output_locations = get_config(config, "output_locations", default=None)
observations_timeseries = get_config(config, "observations_timeseries", default=None)

wflow_outvars = get_config(config, "wflow_outvars", default=['river discharge'])
has_gridded_outputs = len(get_config(config, "wflow_outvars_gridded", default=[])) > 0

# Master rule: end with all model run and analysed with saving a output plot
rule all:
    input: 
        f"{project_dir}/config/snake_config_future_hydrology_delta_change.yml",
        f"{project_dir}/plots/model_delta_runs/flow/1/qhydro_1.png",
        f"{project_dir}/plots/model_delta_runs/other/gridded_output.txt",
        # expand((basin_dir + "/run_delta_{model}_{scenario}_near/output.csv"), model = gcms_selected, scenario = scenarios_selected),


#1. copy config
#2. downscale selected delta change (GCM and ssp and future_period selection) - reproject_like(method = nearest)
#3. update toml - use outstate historical run (config file - selected toml historical run - path to toml relative to basin root)
#4. run near (assume "near" and "far" names are fixed) 
#5. update toml (du historical) of far using states of near (change dir_output, input_path, state_path, reinit)
#6. run far 
#7. plots comparison. 


# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_snake = config_path,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "future_hydrology_delta_change",
    output:
        config_snake_out = f"{project_dir}/config/snake_config_future_hydrology_delta_change.yml",
    script:
        "../src/copy_config_files.py"

# Rule to downscale the monthly delta change factor for the near future
rule downscale_monthly_delta_change_grids_near:
    input:
        staticmaps_fid = ancient(f"{basin_dir}/staticmaps.nc"),
        monthly_change_mean_grid = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_near.nc"),
    output:
        delta_change_downscale_near_nc = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_near_downscaled.nc"),
    script: "../src/downscale_delta_change.py"

# Rule to downscale the monthly delta change factor for the far future
rule downscale_monthly_delta_change_grids_far:
    input:
        staticmaps_fid = ancient(f"{basin_dir}/staticmaps.nc"),
        monthly_change_mean_grid = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_far.nc"),
    output:
        delta_change_downscale_far_nc = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_far_downscaled.nc"),
    script: "../src/downscale_delta_change.py"

# Rule to prepare the yml for each clim dataset with time horizon 
rule setup_toml_near:
    input:
        config_model_historical_fn = config_model_historical_fn,
        monthly_change_mean_grid = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_near_downscaled.nc"),
    output:
        config_model_out_fn = (basin_dir + "/run_delta_change/" + config_basename + "_delta_{model}_{scenario}_near.toml"),
    params:
        model_name = "{model}",
        scenario_name = "{scenario}",
        horizon = "near",
        ref_time = get_config(config, "historical", default=None),
    script: "../src/setup_config_future.py"


#Rule to run the wflow model for each additional forcing dataset 
rule run_wflow_near:
    input:
        config_model_near = (basin_dir + "/run_delta_change/" + config_basename + "_delta_{model}_{scenario}_near.toml"),
        delta_change_downscale_near_nc = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_near_downscaled.nc"),
    output:
        csv_file_near = (basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_near.csv"), 
        state_near_nc = (basin_dir + "/run_delta_change/outstate/outstates_{model}_{scenario}_near.nc"),
        nc_file_near = (basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_near.nc") if has_gridded_outputs else []
    shell:
        """ julia --threads 4 "./src/wflow/run_wflow_change_factors.jl" "{input.config_model_near}" """

# Rule to prepare the yml for each clim dataset with time horizon 
rule setup_toml_far:
    input:
        config_model_historical_fn = (basin_dir + "/run_delta_change/" + config_basename + "_delta_{model}_{scenario}_near.toml"),
        state_near_nc = (basin_dir + "/run_delta_change/outstate/outstates_{model}_{scenario}_near.nc"),
        monthly_change_mean_grid = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_far_downscaled.nc"),
    output:
        config_model_out_fn = (basin_dir + "/run_delta_change/" + config_basename + "_delta_{model}_{scenario}_far.toml"),
    params:
        model_name = "{model}",
        scenario_name = "{scenario}",
        horizon = "far",
        ref_time = get_config(config, "historical", default=None),
    script: "../src/setup_config_future.py"

#Rule to run the wflow model for each additional forcing dataset 
rule run_wflow_far:
    input:
        config_model_far = (basin_dir + "/run_delta_change/" + config_basename + "_delta_{model}_{scenario}_far.toml"),
        delta_change_downscale_far_nc = (clim_project_dir + "/monthly_change_grid/{model}_{scenario}_far_downscaled.nc"),
    output:
        csv_file_far = (basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_far.csv"),
        nc_file_far = (basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_far.nc") if has_gridded_outputs else [],
    shell:
        """ julia --threads 4 "./src/wflow/run_wflow_change_factors.jl" "{input.config_model_far}" """

# Rule to analyse and plot wflow model run results --> final output
rule plot_results:
   input:
       csv_file_near = expand((basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_near.csv"), model = gcms_selected, scenario = scenarios_selected), 
       csv_file_far = expand((basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_far.csv"), model = gcms_selected, scenario = scenarios_selected),
   output: 
       output_png = f"{project_dir}/plots/model_delta_runs/flow/1/qhydro_1.png",
   params:
        wflow_hist_run_config = config_model_historical_fn,
        wflow_delta_runs_config = [f"{basin_dir}/run_delta_change/{config_basename}_delta_{model}_{scenario}_{hz}.toml" for model in gcms_selected for scenario in scenarios_selected for hz in ["near", "far"]],
        gauges_locs = output_locations,
        start_month_hyd_year = "JAN",
        project_dir = f"{project_dir}",
        future_horizons = get_config(config, "future_horizons", optional=False),
        add_plots_with_all_lines = get_config(config, "future_hydrology_plots.add_plots_with_all_lines", default=False),
   script: "../src/plot_results_delta.py"

rule plot_results_grid:
    input:
        nc_file_near = expand((basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_near.nc"), model = gcms_selected, scenario = scenarios_selected) if has_gridded_outputs else [],
        nc_file_far = expand((basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_far.nc"), model = gcms_selected, scenario = scenarios_selected) if has_gridded_outputs else [],
    output:
        output_txt = f"{project_dir}/plots/model_delta_runs/other/gridded_output.txt",
    params:
        project_dir = f"{project_dir}",
        future_horizons = get_config(config, "future_horizons", optional=False),
        scenarios = scenarios_selected,
        gcms = gcms_selected,
        config_historical = config_model_historical_fn,
    script: "../src/plot_results_grid_delta.py"

rule compute_change_statistics:
    input:
        csv_file_near = expand((basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_near.csv"), model = gcms_selected, scenario = scenarios_selected), 
        csv_file_far = expand((basin_dir + "/run_delta_change/output_delta_{model}_{scenario}_far.csv"), model = gcms_selected, scenario = scenarios_selected),
    output:
        output_txt = f"{project_dir}/plots/model_delta_runs/other/change_statistics.txt",
    params:
        wflow_hist_run_config = config_model_historical_fn,
        wflow_delta_runs_config = [f"{basin_dir}/run_delta_change/{config_basename}_delta_{model}_{scenario}_{hz}.toml" for model in gcms_selected for scenario in scenarios_selected for hz in ["near", "far"]],
        gauges_locs = output_locations,
        start_month_hyd_year = "JAN",
        project_dir = f"{project_dir}",
        future_horizons = get_config(config, "future_horizons", optional=False),
        scenarios = scenarios_selected,
        gcms = gcms_selected,
    script: "../src/compute_change_statistics.py"