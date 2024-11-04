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
stem="/p/11210673-fao/14 Subbasins/"
cwd="/u/ohanrah/documents/FAO/"
names=(damchhu_snow1, damchhu_snow2, seti_snow, swat_snow)
yaml_files=(
    "$stem/snake_calibration_config_damchhu_linux_snow.yml"
)
yaml_file="${yaml_files[0]}"
echo "Processing $yaml_file"
./scripts/B_run_calib.sh "$cwd" "$yaml_file"

