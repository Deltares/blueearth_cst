cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_climate_historical.smk" -c 4 --configfile "$yaml_filename" --unlock 
pixi run snakemake -s "snakemake/Snakefile_climate_historical.smk" -c 4 --configfile "$yaml_filename" --quiet rules -n
pixi run snakemake -s "snakemake/Snakefile_climate_historical.smk" -c 4 --configfile "$yaml_filename" 
