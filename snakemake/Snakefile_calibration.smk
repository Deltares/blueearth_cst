import shutil
import os
from os.path import join
from snakemake.utils import Paramspace
import pandas as pd
from pathlib import Path

#Eliminate the need to have linux configs
if sys.platform.startswith("win"):
    DRIVE="p:"
elif sys.platform.startswith("linux"):
    DRIVE="/p"
else:
    raise ValueError(f"Unsupported platform for formatting drive location: {sys.platform}")

# get options from config file
wflow_root = config["wflow_root"].format(DRIVE)

# get options from config file
wflow_root = config["wflow_root"].format(DRIVE)
basin = config["basin"]

calibration_parameters = config["calibration_parameters"].format(DRIVE)
print(calibration_parameters)
calibration_parameters = join(wflow_root, calibration_parameters)

toml_default = config["toml_default_run"]
toml_default = join(wflow_root, toml_default)

calibration_runs_folder = config["calibration_runs_folder"]
calib_folder = join(wflow_root, calibration_runs_folder)
os.makedirs(calib_folder, exist_ok=True)

plot_folder = config["plot_folder"]
plot_folder = join(wflow_root, plot_folder)

# Paramspace with model calibration parameters
paramspace = Paramspace(pd.read_csv(calibration_parameters, sep=","), filename_params="*")

# Master rule: end with all model run and analysed with saving a output plot
rule all:
    input: 
        expand((join(calib_folder, "output_{params}.csv")), params=paramspace.instance_patterns),
        expand((join(plot_folder, "per_run/txt/plot_{params}.txt")), params=paramspace.instance_patterns),
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
        calibration_pattern = paramspace.wildcard_pattern,
        wflow_dir_input = wflow_root,
        calib_folder = calib_folder,
        starttime = config["starttime"],
        endtime = config["endtime"]
    script:
        "../src/calibration/update_toml_parameters.py"

# Rule to run the wflow model
rule: 
    name: f"Wflow_{basin}"
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
        threads = 4,
        time = "0-24:00:00",
        mem_mb = 32000
    localrule: False
    shell:
        """ julia --project={params.project}\
         --threads {resources.threads}\
          -e "using Pkg; Pkg.instantiate(); using Wflow; Wflow.run()" "{input.toml_fid}" """
        # """ julia --threads 4 --project=c:\Users\bouaziz\.julia\environments\reinfiltration -e "using Wflow; Wflow.run()" "{input.toml_fid}" """

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

