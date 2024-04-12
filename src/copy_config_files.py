"""Copy snake config and other config files to the output directory."""
import os
from os.path import join, dirname
from pathlib import Path
from typing import Union, List


def copy_config_files(
    config: Union[str, Path],
    output_dir: Union[str, Path],
    config_out_name: str = None,
    other_config_files: List[Union[str, Path]] = [],
):
    """
    Copy snake config and other config files to the output directory.

    If config_out_name is provided, the name of the output config will be changed.

    Parameters
    ----------
    config : Union[str, Path]
        path to the snake config file
    output_dir : Union[str, Path]
        path to the output directory
    config_out_name : str, optional
        name of the output snake config file, by default None to use the same name
        as the input config
    other_config_files : List[Union[str, Path]], optional
        list of paths to other config files to copy, by default []

    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the name of the output snake config file
    if config_out_name is None:
        config_out_name = os.path.basename(config)
    # Copy the snake config file to the output directory
    print(f"Copying {config_out_name} to {output_dir}")
    with open(config, "r") as f:
        snake_config = f.read()
    with open(join(output_dir, config_out_name), "w") as f:
        f.write(snake_config)

    # Copy other config files to the output directory
    for config_file in other_config_files:
        # Check if the file does exist
        # (eg predefined catalogs of hydromt do not have a path)
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = f.read()
            config_name = os.path.basename(config_file)
            print(f"Copying {config_name} to {output_dir}")
            with open(join(output_dir, config_name), "w") as f:
                f.write(config)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        # Get the in and out path of the snake (main) config file
        config_snake = sm.input.config_snake
        config_snake_out = sm.output.config_snake_out

        # Derive output dir from the output path of the snake config file
        output_dir = dirname(config_snake_out)
        # Get new file name for the snake config file from config_snake_out
        config_snake_out_name = os.path.basename(config_snake_out)

        # Get other config files to copy based on workflow name
        workflow_name = sm.params.workflow_name
        other_config_files = []
        if workflow_name == "model_creation":
            # Get the in and out path of the model build config file
            config_build = sm.input.config_build
            config_wb = sm.input.config_waterbodies
            data_sources = sm.params.data_catalogs
            other_config_files.extend([config_build, config_wb])  
            other_config_files.extend(data_sources) #data sources is already a list 
        elif (
            workflow_name == "climate_projections"
            or workflow_name == "climate_experiment"
        ):
            data_sources = sm.params.data_catalogs
            other_config_files.extend([data_sources])

        # Call the main function
        copy_config_files(
            config=config_snake,
            output_dir=output_dir,
            config_out_name=config_snake_out_name,
            other_config_files=other_config_files,
        )

    else:
        copy_config_files(
            config="config/snake_config_model_test.yml",
            output_dir="examples/test/config",
            config_out_name=None,
            other_config_files=[],
        )
