cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_climate_historical.smk" -c 4 --configfile "$yaml_filename" --unlock 
pixi run snakemake -s "snakemake/Snakefile_climate_historical.smk" -c 4 --configfile "$yaml_filename" --quiet rules --profile "./slurm/" 
# pixi run snakemake -s "snakemake/Snakefile_climate_historical.smk" -c 4 --configfile "$yaml_filename" --profile "./slurm/" --until add_forcing
