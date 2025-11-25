"""Function to update a wflow model and add gauges and outputs"""

from hydromt_wflow import WflowModel
from hydromt import DataCatalog
import os
from os.path import join
from pathlib import Path
from typing import Union, List, Optional


# Supported wflow outputs
WFLOW_VARS = {
    "river discharge": "lateral.river.q_av",
    "precipitation": "vertical.precipitation",
    "overland flow": "lateral.land.q_av",
    "actual evapotranspiration": "vertical.actevap",
    "groundwater recharge": "vertical.recharge",
    "snow": "vertical.snow",
    "glacier": "vertical.glacierstore",
}


def update_wflow_gauges_outputs(
    wflow_root: Union[str, Path],
    data_catalog: Union[str, Path] = "deltares_data",
    output_point_locations: str = None,
    output_area_locations: Union[str, List[str]] = None,
    outputs: List[str] = ["river discharge"],
    outputs_gridded: Optional[List[str]] = None,
):
    """
    Update wflow model with output and optionally gauges locations

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder
    data_catalog : str
        Name of the data catalog to use
    gauges_fn : Union[str, Path, None], optional
        Path to the gauges locations file, by default None
    outputs : List[str], optional
        List of scalar outputs to add to the model, by default ["river discharge"]
        Available outputs are:
            - "river discharge"
            - "precipitation"
            - "overland flow"
            - "actual evapotranspiration"
            - "groundwater recharge"
            - "snow"
            - "glacier"
    outputs_gridded : Optional[List[str]], optional
        List of gridded outputs to add to the model, by default None to save no gridded
        outputs. Available outputs are the same as in `outputs`.
    """

    # Instantiate wflow model
    mod = WflowModel(wflow_root, mode="r+", data_libs=data_catalog)
    mod.read_config()
    mod.read_grid()
    
    # Add outlets
    mod.setup_outlets(
        river_only=True,
        gauge_toml_header=["Q"],
        gauge_toml_param=["lateral.river.q_av"],
    )

    # Add gauges
    if output_point_locations is not None:
        
        mod.setup_gauges(
            gauges_fn=output_point_locations, #was gauges_fn
            index_col="wflow_id",
            snap_to_river=True,
            derive_subcatch=True,
            toml_output="csv",
            gauge_toml_header=["Q", "P"],
            gauge_toml_param=["lateral.river.q_av", "vertical.precipitation"],
        )
    
    if output_area_locations is not None:
        oal = output_area_locations
        if isinstance(oal, str):
            oal = [oal]
        for oal_fn in oal:
            mod.setup_areamap(
                area_fn=oal_fn,
                col2raster=oal_fn, #just name the unique int column the same as the entry in the datacatalog
            )
            mod.setup_config_output_timeseries(
                mapname=oal_fn,
                toml_output="csv",
                header=[f"{var.replace(' ', '_')}_area_avg" for var in outputs],
                param=[WFLOW_VARS[var] for var in outputs],
                reducer=["mean"]*len(outputs),
            )
    
    # Add additional outputs to the config
    # For now assumes basin-average timeseries apart for river.q_av which is saved
    # by default for all outlets and gauges
    if "river discharge" in outputs:
        outputs.remove("river discharge")

    # If glacier check that there are included in the model
    if "glacier" in outputs:
        has_glacier = mod.get_config("model.glacier", fallback=False)
        if not has_glacier:
            print(
                "Glacier output requested but no glacier model found, removing glacier from outputs"
            )
            outputs.remove("glacier")

    for var in outputs:
        if var in WFLOW_VARS:
            mod.config["csv"]["column"].append(
                {
                    "header": f"{var.replace(' ', '_')}_basavg",
                    "reducer": "mean",
                    "parameter": WFLOW_VARS[var],
                }
            )

    # Add gridded outputs
    if outputs_gridded is not None:
        for var in outputs_gridded:
            if var in WFLOW_VARS:
                opt = f"output.{WFLOW_VARS[var]}"
                mod.set_config(opt, var)

    mod.write()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        update_wflow_gauges_outputs(
            wflow_root=os.path.dirname(sm.input.basin_nc),
            data_catalog=sm.params.data_catalog,
            output_point_locations=sm.params.output_point_locations, #was gauges_fn
            output_area_locations=sm.params.output_area_locations,
            outputs=sm.params.outputs,
            outputs_gridded=sm.params.outputs_gridded,
        )
    else:
        update_wflow_gauges_outputs(
            wflow_root=join(os.getcwd(), "examples", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            gauges_fn=None,
            outputs=["river discharge"],
        )
