#!/bin/bash

stem="/p/11210673-fao/14 Subbasins/"
cwd="/u/ohanrah/documents/FAO/"

chmod +x scripts/plot_snow_glacier.sh

# List of YAML files
yaml_files=(
    # "$stem/snake_calibration_config_damchhu_linux.yml"
    # "$stem/snake_calibration_config_damchhu_linux_02.yml"
    # "$stem/snake_calibration_config_seti_linux.yml"
    "$stem/snake_calibration_config_swat_500m_linux.yml"
    # Add more YAML files as needed
)

# Loop through the YAML files and run plot_snow_glacier.sh for each
for yaml_file in "${yaml_files[@]}"; do
    echo "Processing $yaml_file"
    ./scripts/plot_snow_glacier.sh "$cwd" "$yaml_file"
    
    echo "------------------------"
done

echo "All YAML files have been processed."