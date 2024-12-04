#!/bin/bash
#SBATCH --job-name=alishing_nbs_test  	# Job name
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/NBS/alishing_nbs_%x_%A_%a.log
#SBATCH --time=0-00:30:00			# Job duration (d-hh:mm:ss)
#SBATCH --account=hyd
#SBATCH --cpus-per-task=4
#SBATCH --partition test			# 1 2 4 16 24 44   type node met aantal cores (per core 8gb geheugen)
#SBATCH --ntasks=1					# Number of tasks (analyses) to run -- hoeveel nodes 
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --mail-type=FAIL
#SBATCH --get-user-env


# if [[ ! -d ./logging ]]; then
# 	mkdir logging
# fi 

ROOT="/p/11210673-fao/14 Subbasins/Afghanistan_Alishing_500m_v2/hydrology_model/run_nbs" 
cd "${ROOT}"
PWD=$(pwd)
echo $PWD
#list the config files 
# 1:: SENSITIVITY
# cfgs=("wflow_sbm_calib_pond_0p3.toml" "wflow_sbm_calib_pond_0p5.toml")

# 2:: INCREASE PONDING
#"${ROOT}/wflow_sbm_calib_bunds_0pt3m_sum_agroforestry.toml"
# cfgs=(	
# 		"${ROOT}/wflow_sbm_calib_bunds_0pt3m.toml"
# )

#index the slurm array task ID to the configfile array
# cfg="${cfgs[$SLURM_ARRAY_TASK_ID]}"
cfg="${ROOT}/wflow_sbm_calib_bunds_0pt3m.toml"
#RUN ARRAY TASK
julia -t 4 -e "using Wflow; Wflow.run()" "$cfg"


