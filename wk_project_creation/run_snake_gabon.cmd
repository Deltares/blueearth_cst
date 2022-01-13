call activate blueearth-cst

snakemake --configfile config/Gabon/snake_config_Gabon.yml  --dag | dot -Tpng > dag_all.png

snakemake --unlock --configfile config/Gabon/snake_config_Gabon.yml 
snakemake all -c 1 --configfile config/Gabon/snake_config_Gabon.yml 

rem snakemake -s Snakefile_models all -c 1 --keep-going --until add_gauges --report --dryrun 
rem keep going is when parallel runs to keep going parallel if one series goes wrong
rem dryrun is to tell what it will be doing without actually running
rem until - still the whole workflow but not all jobs 
pause
