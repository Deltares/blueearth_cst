cwd="$1"
yaml_filename="$2"
cd "$cwd"
pixi run snakemake -s "snakemake/Snakefile_calibration.smk" -c 4 --configfile "$yaml_filename" --unlock 
pixi run snakemake -s "snakemake/Snakefile_calibration.smk" -c 4 --configfile "$yaml_filename" --quiet rules --profile "./slurm/" -n
pixi run snakemake -s "snakemake/Snakefile_calibration.smk" -c 4 --configfile "$yaml_filename" --profile "./slurm/"