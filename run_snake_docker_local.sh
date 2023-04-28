#!/bin/bash
set -e # fail on errors

docker_root='/root/work'
volumeargs=(
    "-v $(pwd)/config:${docker_root}/config"
    "-v $(pwd)/examples:${docker_root}/examples"
    "-v $(pwd)/hdata:${docker_root}/hdata"
    "-v $(pwd)/.snakemake:${docker_root}/.snakemake"
)

# # uncomment to explore snakemake container:
# docker run $(echo ${volumeargs[@]}) -ti --privileged --entrypoint='' snakemake-singularity bash
# exit 0

singularity_volumeargs=(
    "-B ${docker_root}/config:${docker_root}/config"
    "-B ${docker_root}/examples:${docker_root}/examples"
    "-B ${docker_root}/src:${docker_root}/src"
    "-B ${docker_root}/hdata:${docker_root}/hdata"
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
    --configfile ${docker_root}/hdata/snake_config_model_test.yml

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
    --configfile ${docker_root}/hdata/snake_config_model_test.yml

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
