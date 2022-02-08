call activate blueearth-cst

rem Snakefile_model_creation
snakemake -s Snakefile_model_creation --configfile config/snake_config_model_test.yml  --dag | dot -Tpng > dag_all.png

snakemake --unlock -s Snakefile_model_creation --configfile config/snake_config_model_test.yml
snakemake all -c 1 -s Snakefile_model_creation --configfile config/snake_config_model_test.yml

rem Snakefile_climate_experiment
snakemake -s Snakefile_climate_experiment --configfile config/snake_config_model_test.yml  --dag | dot -Tpng > dag_climate.png

snakemake --unlock -s Snakefile_climate_experiment --configfile config/snake_config_model_test.yml
snakemake all -c 1 -s Snakefile_climate_experiment --configfile config/snake_config_model_test.yml --dry-run

rem snakemake -s Snakefile_models all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
pause
