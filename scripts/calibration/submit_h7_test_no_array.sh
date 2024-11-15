#!/bin/bash
#SBATCH --job-name=FAOcal_test
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/test_%x_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --partition=test
#SBATCH --ntasks=1
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=none
#SBATCH --mail-user=michael.ohanrahan@deltares.nl

echo "current working directory: $PWD"
stem="/p/11210673-fao/14 Subbasins/run_configs/2_calibration"
cwd="/u/ohanrah/documents/FAO/"
phase="soil_cal"
basin="swat"
#p:\11210673-fao\14 Subbasins\run_configs\2_calibration\snake_calibration_config_damchhu_snow.yml
yaml_files=(
    "$stem/snake_calibration_config_${basin}_${phase}.yml"
)

# Get the YAML file for this array task
for yaml_file in "${yaml_files[@]}"; do
    echo "Processing $yaml_file"
    ./scripts/B_run_calib.sh "$cwd" "$yaml_file"
done
echo "All done on TEST partition"