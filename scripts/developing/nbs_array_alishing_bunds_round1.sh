#!/bin/bash
#SBATCH --job-name=ali_nbs  	# Job name
#SBATCH --output=/u/ohanrah/documents/FAO/data/0-log/NBS/%x_%A_%a.log
#SBATCH --time=0-10:00:00			# Job duration (d-hh:mm:ss)
#SBATCH --account=hyd
#SBATCH --cpus-per-task=4
#SBATCH --partition 4pcpu			# 1 2 4 16 24 44   type node met aantal cores (per core 8gb geheugen)
#SBATCH --ntasks=1					# Number of tasks (analyses) to run -- hoeveel nodes 
#SBATCH --mail-user=michael.ohanrahan@deltares.nl
#SBATCH --mail-type=FAIL
#SBATCH --get-user-env
#SBATCH --array=0-1%2    			#e.g. 0-3%2 commands 4 nodes and runs 2 nodes at one time

# if [[ ! -d ./logging ]]; then
# 	mkdir logging
# fi 
#p:\11210673-fao\14 Subbasins\Afghanistan_Alishing_500m_v2\hydrology_model\run_nbs\
ROOT="/p/11210673-fao/14 Subbasins/Afghanistan_Alishing_500m_v2/hydrology_model/run_nbs" 
cd "${ROOT}"

#list the config files 
# 1:: SENSITIVITY
# cfgs=("wflow_sbm_calib_pond_0p3.toml" "wflow_sbm_calib_pond_0p5.toml")

# 2:: INCREASE PONDING
cfgs=(	"${ROOT}/wflow_sbm_calib_bunds_0pt3m.toml"
		# "${ROOT}/wflow_sbm_calib_agroforestry.toml"
		# "${ROOT}/wflow_sbm_calib_pond_cropland.toml"
		"${ROOT}/wflow_sbm_calib_pond_gentle_slopes_forest.toml"
		# "${ROOT}/wflow_sbm_calib_pond_cropland_0.02.toml"
		# "${ROOT}/wflow_sbm_calib_pond_cropland_0.04.toml"
		# "${ROOT}/wflow_sbm_calib_pond_cropland_0.1.toml"
		# "${ROOT}/wflow_sbm_calib_pond_cropland_0.2.toml"
		# "${ROOT}/wflow_sbm_calib_pond_grassland_terracing_hr.toml"
		# "${ROOT}/wflow_sbm_calib_ponding_cropland_reservoirs.toml"
		# "${ROOT}/wflow_sbm_calib_water_tanks.toml"
		# "${ROOT}/wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.02.toml"
		# "${ROOT}/wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.04.toml"
		# "${ROOT}/wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.1.toml"
		# "${ROOT}/wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.2.toml"
		# "${ROOT}/wflow_sbm_calib_bunds_0pt3m_bunded_sum_measures.toml"
)

#index the slurm array task ID to the configfile array
cfg="${cfgs[$SLURM_ARRAY_TASK_ID]}"

#RUN ARRAY TASK
julia --project="/u/ohanrah/documents/FAO/wflow" -t 4 -e "using Wflow; Wflow.run()" "$cfg"


