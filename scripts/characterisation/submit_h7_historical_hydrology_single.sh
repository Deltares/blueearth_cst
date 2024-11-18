#!/bin/bash
#SBATCH --job-name=FAOhisthyd
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/characterisation/%x_%A_%a.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=4pcpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=none
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --array=0-0%1

echo "current working directory: $PWD"
# //////////////////////////////////////
script="scripts/A_historical_hydrology.sh"
echo "RUNNING: $script"
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

stem="/p/11210673-fao/14 Subbasins/run_configs/1_characterisation"
cwd="/u/ohanrah/documents/FAO/"
country=("afg" "bhutan" "pakistan" "nepal")
basin=("alishing" "damchhu" "swat" "seti")
suff=("v2_linux" "v3_linux" "v2_linux" "v3_linux")

yaml_files=(
    # "$stem/snake_config_fao_${country[0]}_${basin[0]}_${suff[0]}.yml"
    # "$stem/snake_config_fao_${country[1]}_${basin[1]}_${suff[1]}.yml"
    "$stem/snake_config_fao_${country[2]}_${basin[2]}_${suff[2]}.yml"
    # "$stem/snake_config_fao_${country[3]}_${basin[3]}_${suff[3]}.yml"
)

for file in "${yaml_files[@]}"; do
    echo "Processing $file"
done

# Get the YAML file for this array task
# this works by using the SLURM_ARRAY_TASK_ID to index into the yaml_files array
yaml_file="${yaml_files[$SLURM_ARRAY_TASK_ID]}" #FOR USE WITH ARRAY JOBS

cd "$cwd"
chmod +x "$script"
"$script" "$cwd" "$yaml_file"
