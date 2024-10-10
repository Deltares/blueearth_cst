import os
from os.path import join, basename, dirname
from pathlib import Path
import matplotlib.pyplot as plt
import hydromt
from hydromt_wflow import WflowModel

from typing import Union

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from wflow_utils import get_wflow_results
    from plot_utils import plot_hydro
else:
    from .wflow_utils import get_wflow_results
    from .plot_utils import plot_hydro

def plot_results_calib_run(
    wflow_toml: Union[str, Path],
    plot_folder: Union[str, Path],
    observations_locations: Union[str, Path],
    observations_timeseries: Union[str, Path],
    calib_run_name: str = "calibration",

):
    """Plot results of a calib run."""
    
    # 1. Read wflow results
    qsim, ds_clim, ds_basin = get_wflow_results(
        wflow_root=dirname(wflow_toml),
        config_fn=basename(wflow_toml),
        gauges_locs=observations_locations,
        remove_warmup=True,
    )

    # 2. Read observations
    # Get wflow basins to clip observations
    wflow = WflowModel(root=dirname(wflow_toml), config_fn=basename(wflow_toml), mode="r")

    # Read
    gdf_obs = hydromt.io.open_vector(
        observations_locations, crs=4326, sep=",", geom=wflow.basins
    )
    da_ts_obs = hydromt.io.open_timeseries_from_table(
        observations_timeseries, name="Q", index_dim="wflow_id", sep=";"
    )
    ds_obs = hydromt.vector.GeoDataset.from_gdf(
        gdf_obs, da_ts_obs, merge_index="inner"
    )
    # Rename wflow_id to index
    ds_obs = ds_obs.rename({"wflow_id": "index"})
    qobs = ds_obs["Q"].load()
    # Sel qobs to the time in qsim
    qobs = qobs.sel(time=slice(qsim.time[0], qsim.time[-1]))

    # 3. Plot results
    # Loop per station
    for station_id, station_name in zip(qsim.index.values, qsim.station_name.values):
        # Select the station
        qsim_i = qsim.sel(index=station_id)
        if station_id in qobs.index.values:
            qobs_i = qobs.sel(index=station_id)
        else:
            # Skip if no obs
            continue

        # Plot hydrographs
        print(f"Plot hydrographs at wflow station {station_name}")
        plot_hydro(
            qsim=qsim_i,
            qobs=qobs_i,
            Folder_out=plot_folder,
            station_name=station_name,
            color="steelblue",
            lw=0.8,
            fs=7,
            max_nan_year=60,
            max_nan_month=5,
            file_postfix=f"_{calib_run_name}",
        )
        plt.close()



if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        text_out = sm.output.output_txt

        plot_results_calib_run(
            wflow_toml = sm.input.toml_fid,
            plot_folder = join(dirname(text_out), ".."),
            observations_locations = sm.params.observations_locations,
            observations_timeseries = sm.params.observations_timeseries,
            calib_run_name = sm.params.calib_run,
        )
        # Write a file for snakemake tracking
        if not os.path.exists(dirname(text_out)):
            os.makedirs(dirname(text_out))
        with open(text_out, "w") as f:
            f.write(f"Plotted wflow results.\n")
    else:
        print("This script should be run from snakemake.")