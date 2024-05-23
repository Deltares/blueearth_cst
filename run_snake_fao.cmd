call activate blueearth-cst

rem snakemake/Snakefile_run_historical_datasets.smk
snakemake -s snakemake/Snakefile_historical_hydrology.smk --configfile config/snake_config_model_fao.yml  --dag | dot -Tpng > dag_model_datasets.png
snakemake --unlock -s snakemake/Snakefile_historical_hydrology.smk --configfile config/snake_config_model_fao.yml
snakemake all -c 1 -s snakemake/Snakefile_historical_hydrology.smk --configfile config/snake_config_model_fao.yml --rerun-incomplete 
rem --until create_model
rem --report --dryrun

rem Snakefile climate_projections
rem snakemake -s Snakefile_climate_projections --configfile config/snake_config_model_fao.yml --dag | dot -Tpng > dag_projections.png
rem snakemake --unlock -s Snakefile_climate_projections --configfile config/snake_config_model_fao.yml
rem snakemake all -c 1 -s Snakefile_climate_projections --configfile config/snake_config_model_fao.yml --keep-going 


rem snakemake -s snakemake/Snakefile_run_historical_datasets.smk all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
rem --delete-temp-output - delete the temp files after the run
pause
