#!/bin/bash
#SBATCH --job-name=FAOcal
#SBATCH --account=hyd
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/slurm/%x_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --partition 4pcpu
#SBATCH --ntasks=1
#SBATCH --time=1-12:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=michael.ohanrahan@deltares.nl


echo "current working directory: $PWD"

chmod +x scripts/loop_configs_for_snow_glacier.sh
scripts/loop_configs_for_snow_glacier.sh
