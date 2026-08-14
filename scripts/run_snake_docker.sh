#!/bin/bash
workflow_image="cst-workflow:0.0.1"
docker_root='/root/work'
volumeargs=(
    "-v $(pwd)/config:${docker_root}/config"
    "-v $(pwd)/examples:${docker_root}/examples"
    "-v /mnt/p/wflow_global/hydromt:/mnt/p/wflow_global/hydromt"
    "-v $(pwd)/.snakemake:${docker_root}/.snakemake"
)

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    ${workflow_image} \
    snakemake all \
    -F \
    -c 4 \
    -s ${docker_root}/build_model.smk \
    --configfile ${docker_root}/test_case/snake_config_baseline_linux.yml

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    ${workflow_image} \
    snakemake all \
    -F \
    -c 4 \
    -s ${docker_root}/run_stress_test.smk \
    --configfile ${docker_root}/test_case/snake_config_baseline_linux.yml

docker run \
    $(echo ${volumeargs[@]}) \
    --privileged \
    --entrypoint='' \
    ${workflow_image} \
    snakemake all \
    -F \
    -c 4 \
    -s ${docker_root}/analyze_projections.smk \
    --configfile ${docker_root}/test_case/snake_config_baseline_linux.yml
