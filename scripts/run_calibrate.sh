echo "current working directory: $PWD"
stem="/p/11210673-fao/14 Subbasins/"
cwd="/u/ohanrah/documents/FAO/"
rule="plot_results_combined"

yaml_files=(
    # "$stem/snake_calibration_config_damchhu_linux.yml"
    # "$stem/snake_calibration_config_damchhu_linux_02.yml"
    # "$stem/snake_calibration_config_seti_linux.yml"
    "$stem/snake_calibration_config_swat_500m_linux.yml"
    # Add more YAML files as needed
)
for yaml_file in "${yaml_files[@]}"; do
    echo "Processing $yaml_file"
    ./scripts/C_run_specific_rule.sh "$cwd" "$yaml_file" "$rule"
done