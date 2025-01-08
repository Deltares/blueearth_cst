#!/bin/bash
#SBATCH --job-name=build=
#SBATCH --output=./data/0-log/cluster/%x_%A_%a.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=4vcpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-0%1           #change this to 0-3%1

echo "current working directory: $PWD"
# //////////////////////////////////////
script="scripts/A_historical_hydrology.sh"
#force_rule="run_wflow"
echo "RUNNING: $script"
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

stem=           ## STEM OF DIRECTORY WITH BUILD CONFIGS ##
country="afg"
basin="ali"
yaml_file="$stem/wflow_build_model_${country}_${basin}.yml"

cd "$cwd"
chmod +x "$script"
"$script" "$cwd" "$yaml_file"

