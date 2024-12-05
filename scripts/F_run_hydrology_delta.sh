cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_future_hydrology_delta_change.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/"  --unlock 
pixi run snakemake -s "snakemake/Snakefile_future_hydrology_delta_change.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/"  -n
pixi run snakemake -s "snakemake/Snakefile_future_hydrology_delta_change.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/" 

#pixi run snakemake -s "snakemake/Snakefile_calibration.smk" -c 4 --configfile "p:/11210673-fao/14 Subbasins//snake_calibration_config_damchhu_linux_snow.yml" --quiet rules --profile "./slurm/" -n
