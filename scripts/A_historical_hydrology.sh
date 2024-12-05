cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/" --unlock
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/" -R run_wflow --rerun-triggers input -n
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --profile "./slurm/" -R run_wflow --rerun-triggers input
