#!/bin/bash
#SBATCH --job-name=FAOcal_test
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/%x_%A.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=none
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
##SBATCH --array=0-1%1

echo "current working directory: $PWD"
stem="/p/11210673-fao/14 Subbasins/"
cwd="/u/ohanrah/documents/FAO/"

yaml_files=(
    "$stem/snake_calibration_config_damchhu_linux_snow.yml"
)

# Get the YAML file for this array task
# this works by using the SLURM_ARRAY_TASK_ID to index into the yaml_files array
# yaml_file="${yaml_files[$SLURM_ARRAY_TASK_ID]}" #FOR USE WITH ARRAY JOBS
yaml_file="${yaml_files[0]}" #FOR USE WITHOUT ARRAY JOBS

echo "Processing $yaml_file"
./scripts/B_run_long_calib.sh "$cwd" "$yaml_file"
