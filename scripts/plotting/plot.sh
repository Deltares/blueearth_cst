#!/bin/bash
#SBATCH --job-name=alishing_plot  	# Job name
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/NBS/alishing_nbs_%x_%A_%a.log
#SBATCH --time=0-10:00:00			# Job duration (d-hh:mm:ss)
#SBATCH --account=hyd
#SBATCH --cpus-per-task=4
#SBATCH --partition 4pcpu			# 1 2 4 16 24 44   type node met aantal cores (per core 8gb geheugen)
#SBATCH --ntasks=1					# Number of tasks (analyses) to run -- hoeveel nodes 
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --mail-type=FAIL
#SBATCH --get-user-env
#SBATCH --array=0-1%2    			#e.g. 0-3%2 commands 4 nodes and runs 2 nodes at one time

pixi run 