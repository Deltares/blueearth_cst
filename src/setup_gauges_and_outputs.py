"""Function to update a wflow model and add gauges and outputs"""

from hydromt_wflow import WflowModel
import os
from os.path import join
from pathlib import Path
from typing import Union, List


# Supported wflow outputs
WFLOW_VARS = {
    "river discharge": "lateral.river.q_av",
    "precipitation": "vertical.precipitation",
    "overland flow": "lateral.land.q_av",
    "actual evapotranspiration": "vertical.actevap",
    "groundwater recharge": "vertical.recharge",
    "snow": "vertical.snowwater",
}


def update_wflow_gauges_outputs(
    wflow_root: Union[str, Path],
    data_catalog: Union[str, Path] = "deltares_data",
    gauges_fn: Union[str, Path, None] = None,
    outputs: List[str] = ["river discharge"],
):
    """
    Update wflow model with output and optionnally gauges locations

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder
    data_catalog : str
        Name of the data catalog to use
    gauges_fn : Union[str, Path, None], optional
        Path to the gauges locations file, by default None
    outputs : List[str], optional
        List of outputs to add to the model, by default ["river discharge"]
        Available outputs are:
            - "river discharge"
            - "precipitation"
            - "overland flow"
            - "actual evapotranspiration"
            - "groundwater recharge"
            - "snow"
    """

    # Instantiate wflow model
    mod = WflowModel(wflow_root, mode="r+", data_libs=data_catalog)

    # Add outlets
    mod.setup_outlets(
        river_only=True,
        gauge_toml_header=["Q"],
        gauge_toml_param=["lateral.river.q_av"],
    )

    # Add gauges
    if gauges_fn is not None and os.path.isfile(gauges_fn):
        mod.setup_gauges(
            gauges_fn=gauges_fn,
            snap_to_river=True,
            derive_subcatch=True,
            toml_output="csv",
            gauge_toml_header=["Q", "P"],
            gauge_toml_param=["lateral.river.q_av", "vertical.precipitation"],
        )

    # Add additional outputs to the config
    # For now assumes basin-average timeseries apart for river.q_av which is saved by default for all outlets and gauges
    if "river discharge" in outputs:
        outputs.remove("river discharge")

    for var in outputs:
        if var in WFLOW_VARS:
            mod.config["csv"]["column"].append(
                {
                    "header": f"{var}_basavg",
                    "reducer": "mean",
                    "parameter": WFLOW_VARS[var],
                }
            )

    mod.write()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        update_wflow_gauges_outputs(
            wflow_root=os.path.dirname(sm.input.basin_nc),
            data_catalog=sm.params.data_catalog,
            gauges_fn=sm.params.output_locs,
            outputs=sm.params.outputs,
        )
    else:
        update_wflow_gauges_outputs(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            gauges_fn=None,
            outputs=["river discharge"],
        )
