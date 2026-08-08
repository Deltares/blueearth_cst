import os
import sys
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

from typing import Union, List

# Import the shared grid helper regardless of the working directory. The
# Snakefile prepends its basedir to sys.path before invoking script: rules, but
# guard here so the module is import-clean for unit tests too.
# parents[2] is the REPO ROOT (file -> experiment/ -> blueearth_cst/ ->
# root); parent.parent stopped at the package dir, from which
# `import blueearth_cst.shared...` cannot resolve (O-07).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from blueearth_cst.shared.snake_utils import stress_test_grid


def prep_cst_parameters(
    config_fn: Union[str, Path],
    csv_fns: List[Union[str, Path]],
):
    """
    Prepare a csv file for each stress test scenario.

    Parameters
    ----------
    config_fn : str, Path
        Path to the config file
    csv_fns : List[str, Path]
        List of paths to the output csv files. If None saves in same directory as
        config_fn and names from stress test parameters.
    """

    # Read the yaml config (R01 sectioned schema)
    with open(config_fn, "r") as stream:
        yml = yaml.load(stream, Loader=yaml.FullLoader)

    stress_test_cfg = yml["workflows"]["climate_experiment"]["stress_test"]

    # Grid step counts + total via the shared helper (single source of truth,
    # strict on a missing step_num). temp_step_num / precip_step_num are the
    # per-axis counts (step_num + 1) that size the linspaces and the loops below.
    temp_step_num, precip_step_num, ST_NUM = stress_test_grid(stress_test_cfg)

    # Temperature change attributes
    delta_temp_mean_min = stress_test_cfg["temp"]["mean"]["min"]
    delta_temp_mean_max = stress_test_cfg["temp"]["mean"]["max"]

    # Precip change attributes
    delta_precip_mean_min = stress_test_cfg["precip"]["mean"]["min"]
    delta_precip_mean_max = stress_test_cfg["precip"]["mean"]["max"]
    delta_precip_variance_min = stress_test_cfg["precip"]["variance"]["min"]
    delta_precip_variance_max = stress_test_cfg["precip"]["variance"]["max"]
    # Stress test values per variables
    temp_values = np.linspace(
        delta_temp_mean_min, delta_temp_mean_max, temp_step_num, axis=1
    )
    precip_values = np.linspace(
        delta_precip_mean_min, delta_precip_mean_max, precip_step_num, axis=1
    )
    precip_var_values = np.linspace(
        delta_precip_variance_min, delta_precip_variance_max, precip_step_num, axis=1
    )

    # Generate csv file for each stress test scenario
    i = 0
    for j in range(temp_step_num):
        temp_j = temp_values[:, j]
        for k in range(precip_step_num):
            precip_k = precip_values[:, k]
            precip_var_k = precip_var_values[:, k]

            # Create df and save to csv
            data = {
                "temp_mean": temp_j,
                "precip_mean": precip_k,
                "precip_variance": precip_var_k,
            }
            df = pd.DataFrame(data=data, dtype=np.float32, index=np.arange(1, 13))
            df.index.name = "month"
            if csv_fns is None:
                csv_fn = join(
                    os.path.dirname(config_fn),
                    f"st_{i+1}.csv",
                )
            else:
                csv_fn = csv_fns[i]
            df.to_csv(csv_fn)

            i += 1


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            prep_cst_parameters(
                config_fn=sm.input.config,
                csv_fns=sm.output.st_csv_fns,
            )
    else:
        prep_cst_parameters(
            config_fn=join(os.getcwd(), "config", "snake_config_model_test.yml"),
        )
