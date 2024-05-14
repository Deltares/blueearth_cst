"""
Open monthly change files for all models/scenarios/horizon and compute/plot statistics
"""

import os
from pathlib import Path
import seaborn as sns
import xarray as xr
import numpy as np

from typing import Union, List, Dict, Optional


def preprocess_coords(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess function to remove unwanted coords"""
    coords_to_remove = ["height"]
    for coord in coords_to_remove:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    return ds


def summary_climate_proj(
    clim_dir: Union[Path, str],
    clim_files: List[Union[Path, str]],
    horizons: Dict[str, str],
):
    """
    Compute climate change statistics for all models/scenario/horizons.

    Also prepare response surface plot.

    Output in ``clim_dir``:
    - annual_change_scalar_stats_summary.nc/.csv: all change statistics (netcdf or csv)
    - annual_change_scalar_stats_summary_mean.csv: only mean change
    - plots/projected_climate_statistics.png: surface response plot

    Parameters
    ----------
    clim_dir: Path
        Path to the projected climate directory of the project
    clim_files: List[Path, str]
        Path to the netcdf files of results per climate model / scenario / horizons
    horizons: Dict
        Time horizon names and start and end year separated with a comma.
        E.g {"far": "2070, 2100", "near": "2030, 2060"}
    """
    if not os.path.exists(clim_dir):
        os.makedirs(clim_dir)

    # merge summary maps across models, scnearios and horizons.
    prefix = "annual_change_scalar_stats"
    # for prefix in prefixes:
    print(f"merging netcdf files {prefix}")
    # open annual scalar summary and merge
    list_files_not_empty = []
    for file in clim_files:
        ds_f = xr.open_dataset(file)
        # don't read in the dummy datasets
        if len(ds_f) > 0:
            list_files_not_empty.append(file)
    ds = xr.open_mfdataset(
        list_files_not_empty, coords="minimal", preprocess=preprocess_coords
    )
    dvars = ds.data_vars
    name_nc_out = f"{prefix}_summary.nc"
    ds.to_netcdf(
        os.path.join(clim_dir, name_nc_out),
        encoding={k: {"zlib": True} for k in dvars},
    )

    # write as a csv
    ds.to_dataframe().to_csv(
        os.path.join(clim_dir, "annual_change_scalar_stats_summary.csv")
    )

    # just keep mean for temp and precip for response surface plots
    df = ds.sel(stats="mean").to_dataframe()
    df.to_csv(os.path.join(clim_dir, "annual_change_scalar_stats_summary_mean.csv"))

    # plot change
    if not os.path.exists(os.path.join(clim_dir, "plots")):
        os.mkdir(os.path.join(clim_dir, "plots"))

    # Rename horizon names to the middle year of the period
    hz_list = df.index.levels[df.index.names.index("horizon")].tolist()
    for hz in horizons:
        # Get start and end year
        period = horizons[hz].split(",")
        period = [int(i) for i in period]
        horizon_year = int((period[0] + period[1]) / 2)
        # Replace hz values by horizon_year in hz_list
        hz_list = [horizon_year if h == hz else h for h in hz_list]

    # Set new values in multiindex dataframe
    df.index = df.index.set_levels(hz_list, level="horizon")

    scenarios = np.unique(df.index.get_level_values("scenario"))
    clrs = []
    for s in scenarios:
        if s == "ssp126":
            clrs.append("#003466")
        if s == "ssp245":
            clrs.append("#f69320")
        if s == "ssp370":
            clrs.append("#df0000")
        elif s == "ssp585":
            clrs.append("#980002")
    g = sns.JointGrid(
        data=df,
        x="precip",
        y="temp",
        hue="scenario",
    )
    g.plot_joint(
        sns.scatterplot, s=100, alpha=0.5, data=df, style="horizon", palette=clrs
    )
    g.plot_marginals(sns.kdeplot, palette=clrs)
    g.set_axis_labels(
        xlabel="Change in mean precipitation (%)",
        ylabel="Change in mean temperature (degC)",
    )
    g.ax_joint.grid()
    g.ax_joint.legend(loc="right", bbox_to_anchor=(1.5, 0.5))
    g.savefig(os.path.join(clim_dir, "plots", "projected_climate_statistics.png"))


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        # Snakemake options
        clim_project_dir = sm.params.clim_project_dir
        list_files = sm.input.stats_nc_change
        horizons = sm.params.horizons

        # Call the main function
        summary_climate_proj(
            clim_dir=clim_project_dir,
            clim_files=list_files,
            horizons=horizons,
        )
    else:
        raise ValueError("This script should be run from a snakemake environment")
