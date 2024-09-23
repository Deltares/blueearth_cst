#!/bin/bash
workflow_image="cst-workflow:0.0.1"
docker_root='/root/work'
volumeargs=(
    "-v $(pwd)/config:${docker_root}/config"
    "-v $(pwd)/examples:${docker_root}/examples"
    "-v $(pwd)/data:${docker_root}/data"
    "-v /mnt/p/wflow_global/hydromt:/mnt/p/wflow_global/hydromt"
    "-v $(pwd)/.snakemake:${docker_root}/.snakemake"
)

# uncomment to explore snakemake container:
# docker run $(echo ${volumeargs[@]}) -ti --entrypoint='' snakemake-singularity bash
# exit 0

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    ${workflow_image} \
    snakemake all \
    -F \
    -c 4 \
    -s ${docker_root}/Snakefile_model_creation \
    --configfile ${docker_root}/config/snake_config_model_test_linux.yml

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    ${workflow_image} \
    snakemake all \
    -F \
    -c 4 \
    -s ${docker_root}/Snakefile_climate_experiment \
    --configfile ${docker_root}/config/snake_config_model_test_linux.yml

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    ${workflow_image} \
    snakemake all \
    -F \
    -c 4 \
    -s ${docker_root}/Snakefile_climate_projections \
    --configfile ${docker_root}/config/snake_config_model_test_linux.yml
