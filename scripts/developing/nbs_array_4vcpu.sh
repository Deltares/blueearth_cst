#!/bin/bash
#SBATCH --job-name=damchhu_nbs  	# Job name
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/NBS/damchhu_nbs_%x_%A_%a.log
#SBATCH --time=1-00:00:00			# Job duration (d-hh:mm:ss)
#SBATCH --account=hyd
#SBATCH --cpus-per-task=4
#SBATCH --partition 4vcpu			# 1 2 4 16 24 44   type node met aantal cores (per core 8gb geheugen)
#SBATCH --ntasks=1					# Number of tasks (analyses) to run -- hoeveel nodes 
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --mail-type=FAIL
#SBATCH --get-user-env
#SBATCH --array=0-4%4    			#e.g. 0-3%2 commands 4 nodes and runs 2 nodes at one time

# if [[ ! -d ./logging ]]; then
# 	mkdir logging
# fi 

ROOT="/p/11210673-fao/14 Subbasins/Bhutan_Damchhu_500m_v2/hydrology_model/run_nbs" 
cd "${ROOT}"

#list the config files 
# 1:: SENSITIVITY
# cfgs=("wflow_sbm_calib_pond_0p3.toml" "wflow_sbm_calib_pond_0p5.toml")

# 2:: INCREASE PONDING
cfgs=(	"wflow_sbm_calib_pond_sum_1.toml"
		"wflow_sbm_calib_pond_sum_1.5.toml"
		"wflow_sbm_calib_pond_sum_2.toml"  
		"wflow_sbm_calib_pond_sum_2.5.toml" 
		"wflow_sbm_calib_pond_sum_3.toml")

#index the slurm array task ID to the configfile array
cfg="${cfgs[$SLURM_ARRAY_TASK_ID]}"

#RUN ARRAY TASK
julia -t 4 -e "using Wflow; Wflow.run()" "$cfg"


