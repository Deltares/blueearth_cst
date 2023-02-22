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
    "overland flow": "lateral.land.q_av",
    "actual evapotranspiration": "vertical.actevap",
    "groundwater recharge": "vertical.recharge",
    "snow": "vertical.snow",
}

# Instantiate wflow model
mod = WflowModel(root, mode="r+", data_libs=data_catalog)

# Add gauges
mod.setup_gauges(
    gauges_fn=gauges_fn,
    derive_subcatch=True,
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
