#!/bin/bash
docker_root='/root/work'
volumeargs=(
    "-v $(pwd)/config:${docker_root}/config"
    "-v $(pwd)/examples:${docker_root}/examples"
    "-v $(pwd)/Snakefile_model_creation:${docker_root}/Snakefile_model_creation"
    "-v $(pwd)/Snakefile_climate_experiment:${docker_root}/Snakefile_climate_experiment"
    "-v $(pwd)/Snakefile_climate_projections:${docker_root}/Snakefile_climate_projections"
    "-v $(pwd)/src:${docker_root}/src"
    "-v $(pwd)/singularity:${docker_root}/singularity"
    "-v /mnt/p/wflow_global/hydromt:/p/wflow_global/hydromt"
    "-v /mnt/p/i1000365-007-blue-earth/ClimateChange/hydromt:/p/i1000365-007-blue-earth/ClimateChange/hydromt"
    "-v $(pwd)/.snakemake:${docker_root}/.snakemake"
)

# uncomment to explore snakemake container:
# docker run $(echo ${volumeargs[@]}) -ti --entrypoint='' snakemake-singularity bash
# exit 0

singularity_volumeargs=(
    "-B ${docker_root}/config:${docker_root}/config"
    "-B ${docker_root}/examples:${docker_root}/examples"
    "-B ${docker_root}/Snakefile_model_creation:${docker_root}/Snakefile_model_creation"
    "-B ${docker_root}/Snakefile_climate_experiment:${docker_root}/Snakefile_climate_experiment"
    "-B ${docker_root}/Snakefile_climate_projections:${docker_root}/Snakefile_climate_projections"
    "-B ${docker_root}/src:${docker_root}/src"
    "-B ${docker_root}/singularity:${docker_root}/singularity"
    "-B /p/wflow_global/hydromt:/p/wflow_global/hydromt"
    "-B /p/i1000365-007-blue-earth/ClimateChange/hydromt:/p/i1000365-007-blue-earth/ClimateChange/hydromt"
    "-B ${docker_root}/.snakemake:${docker_root}/.snakemake"
)
docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    snakemake-singularity \
    snakemake all \
    --use-singularity \
    --singularity-args "$(echo ${singularity_volumeargs[@]})" \
    -c 1 \
    -s ${docker_root}/Snakefile_model_creation \
    --configfile ${docker_root}/config/snake_config_model_test_linux.yml

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    snakemake-singularity \
    snakemake all \
    --use-singularity \
    --singularity-args "$(echo ${singularity_volumeargs[@]})" \
    -c 1 \
    -s ${docker_root}/Snakefile_climate_experiment \
    --configfile ${docker_root}/config/snake_config_model_test_linux.yml

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    snakemake-singularity \
    snakemake all \
    --use-singularity \
    --singularity-args "$(echo ${singularity_volumeargs[@]})" \
    -c 1 \
    -s ${docker_root}/Snakefile_climate_projections \
    --configfile ${docker_root}/hdata/snake_config_model_test.yml