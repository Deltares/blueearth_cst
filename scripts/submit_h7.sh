#!/bin/bash
#SBATCH --job-name=FAOcal
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/%x_%A_%a.log
#SBATCH --cpus-per-task=4
#SBATCH --partition 4pcpu
#SBATCH --ntasks=1
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --array=0-3


echo "current working directory: $PWD"
stem="/p/11210673-fao/14 Subbasins/"
cwd="/u/ohanrah/documents/FAO/"
names=(damchhu_snow1, damchhu_snow2, seti_snow, swat_snow)
yaml_files=(
    "$stem/snake_calibration_config_damchhu_linux.yml"
    "$stem/snake_calibration_config_damchhu_linux_02.yml"
    "$stem/snake_calibration_config_seti_linux.yml"
    "$stem/snake_calibration_config_swat_500m_linux.yml"
    # Add more YAML files as needed
)
yaml_file=${yaml_files[$SLURM_ARRAY_TASK_ID]}
echo "Processing $yaml_file"
./scripts/B_run_long_calib.sh "$cwd" "$yaml_file"

