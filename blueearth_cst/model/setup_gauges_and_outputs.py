"""Function to update a wflow model and add gauges and outputs."""
import os
from os.path import join
from pathlib import Path
from typing import List, Union


# Map user-facing semantic names to Wflow.jl 1.x CSDMS variable names.
# Mapping derived from
# .pixi/envs/default/Lib/site-packages/hydromt_wflow/version_upgrade.py.
WFLOW_VARS = {
    "river discharge": "river_water__volume_flow_rate",
    "precipitation": "atmosphere_water__precipitation_volume_flux",
    "overland flow": "land_surface_water__volume_flow_rate",
    "actual evapotranspiration": "land_surface__evapotranspiration_volume_flux",
    "groundwater recharge": "soil_water_saturated_zone_top__net_recharge_volume_flux",
    "snow": "snowpack_liquid_water__depth",
}


def update_wflow_gauges_outputs(
    wflow_root: Union[str, Path],
    data_catalog: Union[str, Path] = "deltares_data",
    gauges_fn: Union[str, Path, None] = None,
    outputs: List[str] = ["river discharge"],
):
    """
    Update wflow model with output and optionally gauges locations
    """
    # Validate up front: an unknown output name used to be silently dropped
    # (the `extras` filter skipped it), producing no output and no error. Fail
    # loudly on a config typo instead. Output-neutral on a valid config.
    unknown = [v for v in outputs if v not in WFLOW_VARS]
    if unknown:
        raise ValueError(
            f"Unknown wflow_outvars {unknown!r}; valid names are "
            f"{sorted(WFLOW_VARS)}. Fix the wflow_outvars config key."
        )

    # Lazy import: hydromt_wflow is heavy and only needed once we actually
    # touch the model, so importing this module (e.g. for the validation above)
    # stays light and does not require the plugin to be importable.
    from hydromt_wflow import WflowSbmModel

    mod = WflowSbmModel(wflow_root, mode="r+", data_libs=data_catalog)

    river_q_csdms = WFLOW_VARS["river discharge"]

    mod.setup_outlets(
        river_only=True,
        gauge_toml_header=["Q"],
        gauge_toml_param=[river_q_csdms],
    )

    if gauges_fn is not None and os.path.isfile(gauges_fn):
        mod.setup_gauges(
            gauges_fn=gauges_fn,
            snap_to_river=True,
            derive_subcatch=True,
            toml_output="csv",
            gauge_toml_header=["Q", "P"],
            gauge_toml_param=[
                river_q_csdms,
                WFLOW_VARS["precipitation"],
            ],
        )

    # Basin-average timeseries for any extra outputs (river discharge already
    # covered above for outlets/gauges). `outputs` is validated above, so every
    # entry is a known WFLOW_VARS key.
    extras = [v for v in outputs if v != "river discharge"]
    if extras:
        mod.setup_config_output_timeseries(
            mapname="subcatchment",
            toml_output="csv",
            header=[f"{v}_basavg" for v in extras],
            param=[WFLOW_VARS[v] for v in extras],
            reducer=["mean"] * len(extras),
        )

    mod.write()
    # mod.close() commits deferred staticmaps writes — without it,
    # hydromt 1.x leaves the new variables in `staticmaps_<hash>.nc`
    # temp files instead of swapping them into the real file.
    mod.close()


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            update_wflow_gauges_outputs(
                wflow_root=os.path.dirname(sm.input.basin_nc),
                data_catalog=sm.params.data_catalog,
                gauges_fn=getattr(sm.input, "output_locations", None),
                outputs=sm.params.outputs,
            )
    else:
        update_wflow_gauges_outputs(
            wflow_root=join(os.getcwd(), "test_case", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            gauges_fn=None,
            outputs=["river discharge"],
        )
