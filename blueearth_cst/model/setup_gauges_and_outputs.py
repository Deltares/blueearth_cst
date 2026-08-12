"""Function to update a wflow model and add gauges and outputs."""

import os
from os.path import join
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

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

from blueearth_cst.shared.wflow_outputs import code_for  # noqa: E402


def update_wflow_gauges_outputs(
    wflow_root: Union[str, Path],
    data_catalog: Union[str, Path] = "deltares_data",
    location_registry: Union[str, Path, None] = None,
    outputs: List[str] = ["river discharge"],
):
    """
    Add output declarations to registry-indexed maps created by P2.
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

    # Clear any previously declared csv columns FIRST, so what this rule writes
    # is a pure function of `wflow_outvars` rather than the union of every
    # config the model was ever built with.
    #
    # `setup_config_output_timeseries` APPENDS. Nothing removed a stale entry,
    # so editing wflow_outvars only ever added: a model rebuilt after renaming
    # the recharge header carried BOTH declarations, and its output.csv held
    # eight recharge columns -- four under each name, the same numbers twice
    # (measured 2026-08-10). Dropping a variable from wflow_outvars likewise
    # kept emitting it forever. `errors="ignore"` because a first build has no
    # such key to remove.
    mod.config.remove("output.csv.column", errors="ignore")

    staticmaps = mod.staticmaps.data
    if "outlets" not in staticmaps:
        raise ValueError("Built Wflow model lacks the P1-inherited outlets map")
    mod.setup_config_output_timeseries(
        mapname="outlets",
        toml_output="csv",
        header=["Q"],
        param=[river_q_csdms],
    )

    if location_registry is not None and os.path.isfile(location_registry):
        registry = pd.read_csv(location_registry)
        if registry["wflow_id"].duplicated().any():
            raise ValueError("location_registry contains duplicate wflow_id values")
        if "gauges_locations" not in staticmaps:
            raise ValueError("Built Wflow model lacks the registry-indexed gauge map")
        gauge_values = np.asarray(staticmaps["gauges_locations"].values)
        gauge_ids = {
            int(value)
            for value in np.unique(
                gauge_values[np.isfinite(gauge_values) & (gauge_values > 0)]
            )
        }
        registry_ids = set(registry["wflow_id"].astype(int))
        if gauge_ids != registry_ids:
            raise ValueError(
                "Wflow gauges_locations IDs disagree with location_registry"
            )
        # Discharge ONLY. A `P` column used to be appended here unconditionally,
        # independent of `wflow_outvars`, so a config asking for
        # `['river discharge']` still got one precipitation column per gauge.
        #
        # It carried nothing either. This map takes no `reducer`, so wflow's
        # default `only` applies and the column is a POINT sample at the gauge's
        # own grid cell -- i.e. the forcing the model was handed, echoed back
        # out of `forcing/inmaps_historical.nc`. Measured on a real basin
        # 2026-08-10: all four gauges reported an identical value every step,
        # because they share one coarse ERA5 cell, and the four P_ columns were
        # 4 of the 9 in the file. Nothing reads them -- plot_results.py takes
        # Q_outlets, the Q gauges variable and the `_basavg` extras, and
        # export_wflow_results keeps only `Q_` columns.
        #
        # Basin-average precipitation is still available, through the designed
        # route: put `precipitation` in `wflow_outvars` and the extras block
        # below emits `precipitation_basavg` as a subcatchment MEAN, which is a
        # derived quantity rather than the input handed back.
        mod.setup_config_output_timeseries(
            mapname="gauges_locations",
            toml_output="csv",
            header=["Q"],
            param=[river_q_csdms],
        )

    # Basin-average timeseries for any extra outputs (river discharge already
    # covered above for outlets/gauges). `outputs` is validated above, so every
    # entry is a known WFLOW_VARS key.
    extras = [v for v in outputs if v != "river discharge"]
    if extras:
        # SHORT CODES, not the semantic label. The header used to be
        # f"{v}_basavg", so `groundwater recharge` produced columns named
        # `groundwater recharge_basavg_101` -- 32 characters and a space before
        # the id, which then propagated into figure filenames. The code map is
        # shared/wflow_outputs.CODES; `gwr_101` says the same thing.
        mod.setup_config_output_timeseries(
            mapname="subcatchment",
            toml_output="csv",
            header=[code_for(v) for v in extras],
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
                location_registry=sm.input.location_registry,
                outputs=sm.params.outputs,
            )
    else:
        update_wflow_gauges_outputs(
            wflow_root=join(os.getcwd(), "test_case", "my_project", "hydrology_model"),
            data_catalog="deltares_data",
            location_registry=None,
            outputs=["river discharge"],
        )
