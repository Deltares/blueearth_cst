cwd="$1"
yaml_filename="$2"
cd "$cwd"

#configfile is the path to the yaml file with the run config
#config_path is a supplemented entry in the config file 
#profile is the executor profile to use

pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/" --unlock
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/" -n
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/"