import shutil
import os
import sys
from os.path import join
from snakemake.utils import Paramspace
sys.path.append(os.getcwd())
from src.calibration.create_params import create_set
import pandas as pd
from pathlib import Path
import json


#Eliminate the need to have linux configs by formatting the DRIVE to absolute paths in config file
if sys.platform.startswith("win"):
    DRIVE="p:"
elif sys.platform.startswith("linux"):
    DRIVE="/p"
else:
    raise ValueError(f"Unsupported platform for formatting drive location: {sys.platform}")

# get options from config file
wflow_root = config["wflow_root"].format(DRIVE)

basin = config["basin"]
mode = config["mode"]

calibration_parameters = config["calibration_parameters"].format(DRIVE)
#absolute
calibration_parameters = join(wflow_root, calibration_parameters)

toml_default = config["toml_default_run"]
toml_default = join(wflow_root, toml_default)
calibration_runs_folder = config["calibration_runs_folder"]
calib_folder = join(wflow_root, calibration_runs_folder)
plot_folder = config["plot_folder"]
plot_folder = join(wflow_root, plot_folder)
calibration_parameters = join(wflow_root, config["calibration_parameters"])


#trying to distinguish betweeen these two distinct phases/approaches
if 'recipe' and 'json' in str(calibration_parameters):
    lnames, methods, df, wflow_vars = create_set(calibration_parameters)

#both working
elif 'csv' in str(calibration_parameters) and mode == "sensitivity":
    df = pd.read_csv(calibration_parameters)
    lnames = None
    methods = None
    wflow_vars = None

paramspace = Paramspace(df, filename_params="*")

# Master rule: end with all model run and analysed with saving a output plot
rule all:
    input: 
        expand((join(calib_folder, "output_{params}.csv")), params=paramspace.instance_patterns),
        expand((join(plot_folder, "per_run/txt/plot_{params}.txt")), params=paramspace.instance_patterns),
        expand((join(calib_folder, "performance_{params}.nc")), params=paramspace.instance_patterns),
        f"{calib_folder}/performance_appended.txt",
        f"{plot_folder}/best_params/interactive/best_params_timeseries.html",
        f"{plot_folder}/combined/combined_plot.txt"

# Rule to update TOML file settings for calibration
rule update_toml:
    input:
        staticmaps_fn = f"{wflow_root}/staticmaps.nc",
        toml_default = toml_default,
    localrule: True
    output:
        toml_fid = f"{calib_folder}/wflow_sbm_{paramspace.wildcard_pattern}.toml"
    params:
        calibration_parameters = paramspace.instance,
        mode = mode,
        lnames = lnames,
        methods = methods,
        wflow_vars = wflow_vars,
        calibration_pattern = paramspace.wildcard_pattern,
        wflow_dir_input = wflow_root,
        calib_folder = calib_folder,
        starttime = config["starttime"],
        endtime = config["endtime"]
    script:
        "../src/calibration/update_toml_parameters.py"

# Rule to run the wflow model
rule: 
    name: f"run_wflow_{basin}"
    group: "run_wflow"
    input:
        toml_fid = f"{calib_folder}/wflow_sbm_{paramspace.wildcard_pattern}.toml"
    output:
        csv_file = f"{calib_folder}/output_{paramspace.wildcard_pattern}.csv",
    params:
        project = Path("wflow", "Project.toml"),
        basin = basin,
    resources:
        partition = "4pcpu",
        threads = 1,
        time = "0-24:00:00",
        mem_mb = 8000
    localrule: False
    shell:
        """ julia --project={params.project}\
         --threads {resources.threads}\
          -e "using Pkg; Pkg.instantiate(); using Wflow; Wflow.run()" "{input.toml_fid}" """
        # """ julia --threads 4 --project=c:\Users\bouaziz\.julia\environments\reinfiltration -e "using Wflow; Wflow.run()" "{input.toml_fid}" """

rule evaluate_per_run:
    input:
        result = f"{calib_folder}/output_{paramspace.wildcard_pattern}.csv",
        sim = f"{calib_folder}/wflow_sbm_{paramspace.wildcard_pattern}.toml",
    params:
        observed = config["observations_timeseries"].format(DRIVE),
        gauges = config["observations_locations"].format(DRIVE),
        params = paramspace.wildcard_pattern,
        calib_folder = calib_folder,
        starttime = config["starttime"],
        endtime = config["endtime"],
        metrics = config["metrics"],
        weights = config["weights"],
        outflow = config["outflow"],
    output:
        csv_file = f"{calib_folder}/performance_{paramspace.wildcard_pattern}.nc",
    params:
        mode = mode,
    script:
        "../src/calibration/evaluate_runs.py"


rule touch_performance:
    input:
        expand(f"{calib_folder}/performance_"+"{params}"+".nc", params=paramspace.instance_patterns),
    params:
        calib_folder = calib_folder,
        observed = config["observations_timeseries"].format(DRIVE),
    output: #P:/11210673-fao/14%20Subbasins/Bhutan_Damchhu_500m_v2/plots/calibration/era5_imdaa_clim_soil_cal/best_params/interactive/best_params_timeseries.html
        f"{calib_folder}/performance_appended.txt"

rule assess_best_params:
    input:
        f"{calib_folder}/performance_appended.txt",
    params:
        calib_folder = calib_folder,
        gauges = config["observations_locations"].format(DRIVE),
    output:
        f"{plot_folder}/best_params/interactive/best_params_timeseries.html"
    script:
        "../src/calibration/best_params.py"

# Rule to analyse and plot wflow model run results
rule plot_results_per_run:
    input:
        csv_file = f"{calib_folder}/output_{paramspace.wildcard_pattern}.csv",
        toml_fid = f"{calib_folder}/wflow_sbm_{paramspace.wildcard_pattern}.toml",
    params:
        observations_locations = config["observations_locations"].format(DRIVE),
        observations_timeseries = config["observations_timeseries"].format(DRIVE),
        calib_run = paramspace.wildcard_pattern,
    output: 
        output_txt = f"{plot_folder}/per_run/txt/plot_{paramspace.wildcard_pattern}.txt"
    localrule: True
    script: 
        "../src/plot_results_calib.py"

# Rule to plot different model runs
rule plot_results_combined:
    input:
        csv_files = expand(join(calib_folder, "output_{params}.csv"), params=paramspace.instance_patterns),
        toml_files = expand(join(calib_folder, "wflow_sbm_{params}.toml"), params=paramspace.instance_patterns),
    params:
        observations_locations = config["observations_locations"].format(DRIVE),
        observations_timeseries = config["observations_timeseries"].format(DRIVE),
        calibration_runs_selection = config["calibration_runs_selection"],
        uncalibrated_run = config["uncalibrated_run"],
    output: 
        output_txt = f"{plot_folder}/combined/combined_plot.txt"
    localrule: True
    script: 
        "../src/plot_results_calib_combined.py"

