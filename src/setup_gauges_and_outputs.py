import hydromt
from hydromt_wflow import WflowModel
import os

root = os.path.dirname(snakemake.input.basin_nc)
data_catalog = snakemake.params.data_catalog
gauges_fn = snakemake.params.output_locs
outputs = snakemake.params.outputs

# Supported wflow outputs
WFLOW_VARS = {
    "river discharge": "lateral.river.q_av",
    "precipitation": "vertical.precipitation",
    "overland flow": "lateral.land.q_av",
    "actual evapotranspiration": "vertical.actevap",
    "groundwater recharge": "vertical.recharge",
    "snow": "vertical.snowwater",
}

# Instantiate wflow model
mod = WflowModel(root, mode="r+", data_libs=data_catalog)

# Add outlets
mod.setup_outlets(
    river_only=True,
    gauge_toml_header=["Q"],
    gauge_toml_param=["lateral.river.q_av"],
)

# Add gauges
if os.path.isfile(gauges_fn):
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
