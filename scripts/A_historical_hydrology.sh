cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --unlock 
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --quiet rules --profile "./slurm/" -n --until add_forcing
pixi run snakemake -s "snakemake/Snakefile_historical_hydrology.smk" -c 4 --configfile "$yaml_filename" --profile "./slurm/" --until add_forcing
