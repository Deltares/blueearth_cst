#!/bin/bash
#SBATCH --job-name=FAOcal_test
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/%x_%A_%a.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=none
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --array=0-1%1

echo "current working directory: $PWD"
stem="/p/11210673-fao/14 Subbasins/run_configs/2_calibration"
cwd="/u/ohanrah/documents/FAO/"
phase="soil_cal"
basin=("damchhu" "swat" "seti")
yaml_files=(
    "$stem/snake_calibration_config_${basin[0]}_${phase}.yml"
    "$stem/snake_calibration_config_${basin[1]}_${phase}.yml"
    "$stem/snake_calibration_config_${basin[2]}_${phase}.yml"
)

for file in "${yaml_files[@]}"; do
    echo "Processing $file"
done

# Get the YAML file for this array task
# this works by using the SLURM_ARRAY_TASK_ID to index into the yaml_files array
yaml_file="${yaml_files[$SLURM_ARRAY_TASK_ID]}" #FOR USE WITH ARRAY JOBS

echo "Processing $yaml_file"
./scripts/B_run_calib.sh "$cwd" "$yaml_file"
