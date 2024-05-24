call activate blueearth-cst

rem Snakefile_historical_climate
snakemake -s snakemake/Snakefile_climate_historical.smk --configfile tests/snake_config_fao_test.yml  --dag | dot -Tpng > dag_climate_historical.png
snakemake --unlock -s snakemake/Snakefile_climate_historical.smk --configfile tests/snake_config_fao_test.yml
snakemake all -c 1 -s snakemake/Snakefile_climate_historical.smk --configfile tests/snake_config_fao_test.yml

rem snakemake/Snakefile_run_historical_datasets.smk
snakemake -s snakemake/Snakefile_historical_hydrology.smk --configfile tests/snake_config_fao_test.yml  --dag | dot -Tpng > dag_model_datasets.png
snakemake --unlock -s snakemake/Snakefile_historical_hydrology.smk --configfile tests/snake_config_fao_test.yml
snakemake all -c 1 -s snakemake/Snakefile_historical_hydrology.smk --configfile tests/snake_config_fao_test.yml --rerun-incomplete 
rem --until create_model
rem --report --dryrun

rem Snakefile climate_projections
snakemake -s snakemake/Snakefile_climate_projections.smk --configfile tests/snake_config_fao_test.yml --dag | dot -Tpng > dag_projections.png
snakemake --unlock -s snakemake/Snakefile_climate_projections.smk --configfile tests/snake_config_fao_test.yml
snakemake all -c 1 -s snakemake/Snakefile_climate_projections.smk --configfile tests/snake_config_fao_test.yml --keep-going 


rem snakemake -s snakemake/Snakefile_run_historical_datasets.smk all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
rem --delete-temp-output - delete the temp files after the run
pause
