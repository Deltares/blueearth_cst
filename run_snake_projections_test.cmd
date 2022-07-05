call activate blueearth-cst

rem snakemake --unlock -s Snakefile_climate_projections --configfile config/snake_config_projections_test.yml
rem snakemake -s Snakefile_climate_projections --configfile config/snake_config_projections_test.yml --dag | dot -Tpng > dag_projections.png
rem snakemake all -c 1 -s Snakefile_climate_projections --configfile config/snake_config_projections_test.yml --keep-going 
rem --until monthly_stats_hist 

rem CMIP5
snakemake --unlock -s Snakefile_climate_projections --configfile config/snake_config_modeltest.yml
snakemake -s Snakefile_climate_projections --configfile config/snake_config_model_test.yml --dag | dot -Tpng > dag_projections.png

snakemake all -c 1 -s Snakefile_climate_projections --configfile config/snake_config_model_test.yml --keep-going 



rem snakemake all -c 1 -s Snakefile_model_creation --configfile config/snake_config_model_test.yml
rem --allow-ambiguity
rem snakemake -s Snakefile_models all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
rem pause
