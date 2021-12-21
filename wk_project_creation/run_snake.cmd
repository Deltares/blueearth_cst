call activate blueearth-cst

snakemake --dag | dot -Tpng > dag_all.png

snakemake --unlock
snakemake all -c 1

pause
