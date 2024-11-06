#!/bin/bash
#SBATCH --job-name=FAOcal
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/%x_%A.log
#SBATCH --cpus-per-task=4
#SBATCH --partition 4pcpu
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=michael.ohanrahan@deltares.nl


echo "current working directory: $PWD"
stem="/p/11210673-fao/14 Subbasins"
cwd="/u/ohanrah/documents/FAO/"
#p:\11210673-fao\14 Subbasins\run_configs\2_calibration\snake_calibration_config_damchhu_snow.yml
yaml_files=(
    "$stem/run_configs/2_calibration/snake_calibration_config_damchhu_snow.yml"
)
yaml_file="${yaml_files[0]}"
echo "Processing $yaml_file"
chmod +x "$cwd/scripts/B_run_calib.sh"
"$cwd/scripts/B_run_calib.sh" "$cwd" "$yaml_file"

