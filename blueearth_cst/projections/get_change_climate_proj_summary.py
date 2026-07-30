"""
Open monthly change files for all models/scenarios/horizon and compute/plot statistics
"""

import hydromt  # noqa: F401 -- registers the xarray .raster accessor (ds.raster.vars below)
import os
from pathlib import Path
import seaborn as sns
import xarray as xr
import numpy as np

from typing import Union, List, Dict

from blueearth_cst.shared.snake_utils import log_row


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
    horizons: Dict,
):
    """
    Compute climate change statitistics for all models/scenario/horizons.

    Also prepare response surface plot.

    Output under ``clim_dir`` (R07 B3: summary artifacts in ``summary/``,
    figures in ``plots/``):
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
    # merge summary maps across models, scnearios and horizons.
    prefix = "annual_change_scalar_stats"
    # for prefix in prefixes:
    log_row(f"merging netcdf files {prefix}", module="change")
    # Step 4c: filter_nonempty is GONE. It existed to drop the dummy empty
    # netCDFs stage A used to write for absent sources; since 4a an unresolved
    # combination never becomes a job, so every file in this list carries data and
    # dropping any of them would be silently shrinking the ensemble.
    # Eager, and closed — for two reasons, both learned the hard way on win-64.
    #
    # 1. HANDLES. Lazily-opened members stay open until the dataset is collected,
    #    so the caller cannot delete the files it just passed in. Step 4d made the
    #    per-point files job-internal (a TemporaryDirectory instead of Snakemake
    #    temp()), and its cleanup died with WinError 32 on exactly that. The leak
    #    predates 4d; it was invisible only because Snakemake deleted those files
    #    after the process had exited.
    # 2. DEADLOCK. `open_mfdataset` + `to_netcdf` reading from dask's thread pool
    #    parks forever on the HDF5 global lock — the failure diagnosed in bf1f4a5
    #    for plot_climate_proj_timeseries, fixed there the same way. This call site
    #    has the identical shape and had not been fixed.
    #
    # Value-neutral: `.load()` changes when the bytes are read, not what they are.
    with xr.open_mfdataset(
        clim_files, coords="minimal", preprocess=preprocess_coords
    ) as _ds_lazy:
        ds = _ds_lazy.load()
    dvars = ds.raster.vars
    # R07 B3: processed-vs-summary tiering. The three summary artifacts live
    # under summary/; the figures stay in plots/ (only these three move).
    summary_dir = os.path.join(clim_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    name_nc_out = f"{prefix}_summary.nc"
    ds.to_netcdf(
        os.path.join(summary_dir, name_nc_out),
        encoding={k: {"zlib": True} for k in dvars},
    )

    # write as a csv
    ds.to_dataframe().to_csv(
        os.path.join(summary_dir, "annual_change_scalar_stats_summary.csv")
    )

    # just keep mean for temp and precip for response surface plots
    df = ds.sel(stats="mean").to_dataframe()
    df.to_csv(
        os.path.join(summary_dir, "annual_change_scalar_stats_summary_mean.csv")
    )

    # plot change
    if not os.path.exists(os.path.join(clim_dir, "plots")):
        os.mkdir(os.path.join(clim_dir, "plots"))

    # Rename horizon names to the middle year of the period
    hz_list = df.index.levels[df.index.names.index("horizon")].tolist()
    for hz in horizons:
        # Get start and end year.
        # R01 delivers future_horizons as lists ([2030, 2060]); pre-R01 configs
        # delivered comma-separated strings ("2030, 2060"). Accept both.
        period = horizons[hz]
        period = period.split(",") if isinstance(period, str) else period
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


# NOTE: this module no longer runs as a Snakemake `script:`. Step 4d merged rules
# 2.04/2.05 into `derive_change_factors`, which imports the functions above and
# owns the orchestration. The former `__main__` block is deleted rather than left
# dead: it was a second copy of the per-point procedure, and two copies of the
# same arithmetic is how they drift apart. The functions stay here — they are the
# tested surface (tests/test_get_change_climate_proj*.py).
