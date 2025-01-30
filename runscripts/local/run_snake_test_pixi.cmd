
rem Snakefile_model_creation
pixi run snakemake -s snakemake/Snakefile_historical_hydrology.smk --configfile tests/snake_config_cst_test.yml --dag | dot -Tpng > dag_model.png
pixi run snakemake --unlock -s snakemake/Snakefile_historical_hydrology.smk --configfile tests/snake_config_cst_test.yml
pixi run snakemake all -c 1 -s snakemake/Snakefile_historical_hydrology.smk --configfile tests/snake_config_cst_test.yml

rem Snakefile climate_projections
pixi run snakemake -s snakemake/Snakefile_climate_projections.smk --configfile tests/snake_config_cst_test.yml --dag | dot -Tpng > dag_projections.png
pixi run snakemake --unlock -s snakemake/Snakefile_climate_projections.smk --configfile tests/snake_config_cst_test.yml
pixi run snakemake all -c 1 -s snakemake/Snakefile_climate_projections.smk --configfile tests/snake_config_cst_test.yml --keep-going

rem Snakefile_climate_experiment
pixi run snakemake -s snakemake/Snakefile_climate_experiment.smk --configfile tests/snake_config_cst_test.yml --dag | dot -Tpng > dag_climate.png
pixi run snakemake --unlock -s snakemake/Snakefile_climate_experiment.smk --configfile tests/snake_config_cst_test.yml
pixi run snakemake all -c 1 -s snakemake/Snakefile_climate_experiment.smk --configfile tests/snake_config_cst_test.yml

rem snakemake -s Snakefile_model_creation all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
rem --delete-temp-output - delete the temp files after the run
rem --notemp do not delete the temp files after the run
pause
