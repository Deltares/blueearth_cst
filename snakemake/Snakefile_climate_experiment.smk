import sys
import numpy as np

from get_config import get_config

# Parsing the Snakemake config file (options for basins to build, data catalog, model output directory)
#configfile: "config/snake_config_test.yml"
# read path of the config file to give to the weagen scripts
args = sys.argv
config_path = args[args.index("--configfile") + 1]

project_dir = get_config(config, 'project_dir', optional=False)
experiment = get_config(config, 'experiment_name', optional=False)
RLZ_NUM = get_config(config, 'realizations_num', default=1)
ST_NUM = (get_config(config['temp'], 'step_num', default=1) + 1) * (get_config(config['precip'], 'step_num', default=1) + 1)
run_hist = get_config(config,"run_historical", default=False)
if run_hist:
    ST_START = 0
else:
    ST_START = 1

basin_dir = f"{project_dir}/hydrology_model"
exp_dir = f"{project_dir}/climate_{experiment}"

DATA_SOURCES = get_config(config, "data_sources", optional=False)
DATA_SOURCES = np.atleast_1d(DATA_SOURCES).tolist() #make sure DATA_SOURCES is a list format (even if only one DATA_SOURCE)
DATA_SOURCES_ALL = DATA_SOURCES.copy()
DATA_SOURCES_ALL.append(f"{exp_dir}/data_catalog_climate_experiment.yml")

# Data catalog historical
clim_source = get_config(config, "clim_reference", optional=False)
# Time horizon climate experiment and number of hydrological model run
horizontime_climate = get_config(config, 'horizontime_climate', optional=False)
wflow_run_length = get_config(config, 'run_length', default=20)

# Master rule: end with all model run and analysed with saving a output plot
rule all:
    input:
        Qstats = f"{exp_dir}/model_results/Qstats.csv",
        basin = f"{exp_dir}/model_results/basin.csv",
        snake_config = f"{project_dir}/config/snake_config_climate_experiment.yml",

# Rule to copy config files to the project_dir/config folder
rule copy_config:
    input:
        config_snake = config_path,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "climate_experiment",
    output:
        config_snake_out = f"{project_dir}/config/snake_config_climate_experiment.yml",
    script:
        "../src/copy_config_files.py"

# Rule to extract historic climate data at native resolution for the project area
rule extract_climate_grid:
    input:
        region_fn = ancient(f"{basin_dir}/staticgeoms/region.geojson"),
    params:
        data_sources = DATA_SOURCES,
        clim_source = clim_source,
        starttime = "1980-01-01T00:00:00",
        endtime = "2010-12-31T00:00:00",
        buffer_km = 1,
        climate_variables = ["precip", "temp", "temp_min", "temp_max", "kin", "kout", "press_msl"],
        combine_with_era5 = True,
        add_source_to_coords = False,
    output:
        climate_nc = f"{project_dir}/climate_historical/raw_data/extract_historical.nc",
    script:
        "../src/extract_historical_climate.py"

# Prepare stress test experiment
rule climate_stress_parameters:
    input:
        config = ancient(config_path),
    output:
        st_csv_fns = [f"{exp_dir}/stress_test/cst_{st_num}.csv" for st_num in np.arange(1, ST_NUM+1)]
    script:
        "../src/prepare_cst_parameters.py"

# Prepare config files for the weather generator: generate
rule prepare_weagen_config:
    output:
        weagen_config = f"{exp_dir}/weathergen_config.yml",
    params:
        cftype = "generate",
        snake_config = config_path,
        default_config = get_config(config, "weathergen_config", default="config/cst_api/weathergen_config.yml"),
        output_path = f"{exp_dir}/", 
        middle_year = horizontime_climate,
        sim_years = wflow_run_length,
        nc_file_prefix = "rlz"
    script:
        "../src/prepare_weagen_config.py"

# Prepare config files for the weather generator: climate change
rule prepare_weagen_config_st:
    output:
        weagen_config = f"{exp_dir}/realization_"+"{rlz_num}"+"/weathergen_config_rlz_"+"{rlz_num}"+"_cst_"+"{st_num}"+".yml",
    params:
        cftype = "stress_test",
        snake_config = config_path,
        output_path = f"{exp_dir}/realization_"+"{rlz_num}"+"/",
        nc_file_prefix = "rlz_"+"{rlz_num}"+"_cst",
        nc_file_suffix = "{st_num}",
    script:
        "../src/prepare_weagen_config.py"

# Generate climate realization
rule generate_weather_realization:
    input:
        climate_nc = ancient(f"{project_dir}/climate_historical/raw_data/extract_historical.nc"),
        weagen_config = f"{exp_dir}/weathergen_config.yml",
    output:
        temp([f"{exp_dir}/realization_{rlz_num}/rlz_{rlz_num}_cst_0.nc" for rlz_num in np.arange(1, RLZ_NUM+1)])
    shell:
        """Rscript --vanilla src/weathergen/generate_weather.R {input.climate_nc} {input.weagen_config} """

# Generate climate stress tests
rule generate_climate_stress_test:
    input:
        rlz_nc = f"{exp_dir}/realization_"+"{rlz_num}"+"/rlz_"+"{rlz_num}"+"_cst_0.nc",
        st_csv = f"{exp_dir}/stress_test/cst_"+"{st_num}"+".csv",
        weagen_config = f"{exp_dir}/realization_"+"{rlz_num}"+"/weathergen_config_rlz_"+"{rlz_num}"+"_cst_"+"{st_num}"+".yml",
    output:
        rlz_st_nc = temp(f"{exp_dir}/realization_"+"{rlz_num}"+"/rlz_"+"{rlz_num}"+"_cst_"+"{st_num}"+".nc")
    shell:
        """Rscript --vanilla src/weathergen/impose_climate_change.R {input.rlz_nc} {input.weagen_config} {input.st_csv}"""

# Prepare data catalog of the climate files
rule climate_data_catalog:
    input:
        cst_nc = expand((f"{exp_dir}/realization_"+"{rlz_num}"+"/rlz_"+"{rlz_num}"+"_cst_"+"{st_num}"+".nc"), rlz_num = np.arange(1, RLZ_NUM+1), st_num = np.arange(1, ST_NUM+1)),
        rlz_nc = [f"{exp_dir}/realization_{rlz_num}/rlz_{rlz_num}_cst_0.nc" for rlz_num in np.arange(1, RLZ_NUM+1)]
    output:
        clim_data = f"{exp_dir}/data_catalog_climate_experiment.yml"
    params:
        data_sources = DATA_SOURCES,
        clim_source = clim_source,
    script:
        "../src/prepare_climate_data_catalog.py"

# Downscale climate forcing for use with wflow
rule downscale_climate_realization:
    input:
        nc = f"{exp_dir}/realization_"+"{rlz_num}"+"/rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".nc",
        clim_data = f"{exp_dir}/data_catalog_climate_experiment.yml",
    output:
        nc = temp(f"{exp_dir}/realization_"+"{rlz_num}"+"/inmaps_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".nc"),
        toml = f"{basin_dir}/run_climate_{experiment}/wflow_sbm_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".toml"
    params:
        model_dir = basin_dir,
        clim_source = clim_source,
        horizontime_climate = horizontime_climate,
        run_length = wflow_run_length,
        data_sources = DATA_SOURCES_ALL,
    script:
        "../src/downscale_climate_forcing.py"

# Run Wflow for all climate forcing
rule run_wflow:
    input:
        forcing_fid = f"{exp_dir}/realization_"+"{rlz_num}"+"/inmaps_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".nc",
        toml_fid = f"{basin_dir}/run_climate_{experiment}/wflow_sbm_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".toml"
    output:
        csv_file = f"{basin_dir}/run_climate_{experiment}/output_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".csv",
        state_file = temp(f"{basin_dir}/run_climate_{experiment}/outstates_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".nc")
    shell:
        """ julia --threads 4 -e "using Wflow; Wflow.run()" "{input.toml_fid}" """

# Export wflow results
rule export_wflow_results:
    input:
        rlz_csv_fns = expand((f"{basin_dir}/run_climate_{experiment}/output_rlz_"+"{rlz_num}"+"_cst_"+"{st_num2}"+".csv"), rlz_num=np.arange(1, RLZ_NUM+1), st_num2=np.arange(ST_START, ST_NUM+1)),
    output:
        Qstats = f"{exp_dir}/model_results/Qstats.csv",
        basin = f"{exp_dir}/model_results/basin.csv",
    params:
        exp_dir = exp_dir,
        aggr_rlz = get_config(config, 'aggregate_rlz', default=True),
        st_num = ST_NUM,
        Tlow = get_config(config,"Tlow", default=2),
        Tpeak = get_config(config,"Tpeak", default=10),
    script:
        "../src/export_wflow_results.py"
