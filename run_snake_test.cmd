call activate blueearth-cst

rem Snakefile_model_creation
snakemake -s Snakefile_model_creation --configfile config/snake_config_model_test.yml  --dag | dot -Tpng > dag_model.png

snakemake --unlock -s Snakefile_model_creation --configfile config/snake_config_model_test.yml
snakemake all -c 1 -s Snakefile_model_creation --configfile config/snake_config_model_test.yml

rem Snakefile climate_projections
snakemake -s Snakefile_climate_projections --configfile config/snake_config_model_test.yml --dag | dot -Tpng > dag_projections.png
snakemake --unlock -s Snakefile_climate_projections --configfile config/snake_config_model_test.yml

snakemake all -c 1 -s Snakefile_climate_projections --configfile config/snake_config_model_test.yml --keep-going 

rem Snakefile_climate_experiment
snakemake -s Snakefile_climate_experiment --configfile config/snake_config_model_test.yml  --dag | dot -Tpng > dag_climate.png

snakemake --unlock -s Snakefile_climate_experiment --configfile config/snake_config_model_test.yml
snakemake all -c 1 -s Snakefile_climate_experiment --configfile config/snake_config_model_test.yml

rem snakemake -s Snakefile_model_creation all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
pause
