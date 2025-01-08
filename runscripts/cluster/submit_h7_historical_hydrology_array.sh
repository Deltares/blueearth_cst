#!/bin/bash
#SBATCH --job-name=hist_hydro
#SBATCH --output=./data/0-log/cluster/%x_%A_%a.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=4vcpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4

echo "current working directory: $PWD"
# //////////////////////////////////////
script="scripts/A_historical_hydrology.sh"
#force_rule="run_wflow"
echo "RUNNING: $script"
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

stem= ## FOLDER WITH BUILD CONFIGS ##
country=("afg" "bhu" "pak" "nep")
basin=("ali" "dam" "swa" "set")

yaml_files=(
    "$stem/wflow_build_model_${country[0]}_${basin[0]}.yml"
    "$stem/wflow_build_model_${country[1]}_${basin[1]}.yml"
    "$stem/wflow_build_model_${country[2]}_${basin[2]}.yml"
    "$stem/wflow_build_model_${country[3]}_${basin[3]}.yml"
)

# Get the YAML file for this array task
# this works by using the SLURM_ARRAY_TASK_ID to index into the yaml_files array
yaml_file="${yaml_files[$SLURM_ARRAY_TASK_ID]}" #FOR USE WITH ARRAY JOBS

cd "$cwd"
chmod +x "$script"
"$script" "$cwd" "$yaml_file"
