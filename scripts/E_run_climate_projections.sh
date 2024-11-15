cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_climate_projections.smk" -c 4 --configfile "$yaml_filename" --unlock 
pixi run snakemake -s "snakemake/Snakefile_climate_projections.smk" -c 4 --configfile "$yaml_filename" --quiet rules -n
pixi run snakemake -s "snakemake/Snakefile_climate_projections.smk" -c 4 --configfile "$yaml_filename" 

#pixi run snakemake -s "snakemake/Snakefile_calibration.smk" -c 4 --configfile "p:/11210673-fao/14 Subbasins//snake_calibration_config_damchhu_linux_snow.yml" --quiet rules --profile "./slurm/" -n
