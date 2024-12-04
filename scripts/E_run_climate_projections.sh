cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_climate_projections.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --unlock 
pixi run snakemake -s "snakemake/Snakefile_climate_projections.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename" --quiet rules -n
pixi run snakemake -s "snakemake/Snakefile_climate_projections.smk" -c 4 --configfile "$yaml_filename" --config config_path="$yaml_filename"
