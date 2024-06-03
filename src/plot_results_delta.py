# -*- coding: utf-8 -*-
"""
Plot wflow results of delta change runs. 
"""

import xarray as xr
import numpy as np
import os
from os.path import join, dirname
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import hydromt
from hydromt_wflow import WflowModel
import seaborn as sns
import matplotlib.patches as mpatches
import glob

from typing import Union, List, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_change_delta_runs import (
        plot_near_far_abs,
        plot_near_far_rel,
        get_df_seaborn,
        get_sum_annual_and_monthly,
        make_boxplot_monthly,
        get_plotting_position,
        plot_plotting_position,
    )
else:
    from .plot_utils.plot_change_delta_runs import (
        plot_near_far_abs,
        plot_near_far_rel,
        get_df_seaborn,
        get_sum_annual_and_monthly,
        make_boxplot_monthly,
        get_plotting_position,
        plot_plotting_position,
    )


# Supported wflow outputs
WFLOW_VARS = {
    "overland flow": {
        "resample": "mean",
        "legend": "Overland Flow (m$^3$s$^{-1}$)",
        "legend_annual": "Overland Flow (m$^3$s$^{-1}$)",
    },
    "actual evapotranspiration": {
        "resample": "sum",
        "legend": "Actual Evapotranspiration (mm month$^{-1}$)",
        "legend_annual": "Actual Evapotranspiration (mm year$^{-1}$)",
    },
    "groundwater recharge": {
        "resample": "sum",
        "legend": "groundwater recharge (mm month$^{-1}$)",
        "legend_annual": "groundwater recharge (mm year$^{-1}$)",
    },
    "snow": {
        "resample": "sum",
        "legend": "Snowpack (mm month$^{-1}$)",
        "legend_annual": "Snowpack (mm year$^{-1}$)",
    },
}


def get_wflow_results(
    wflow_root: Union[str, Path],
    config_fn: str = "wflow_sbm.toml",
    gauges_locs: Union[Path, str] = None,
):
    """
    Get wflow results as xarray.Dataset for simulated discharges, simulated flux/states as basin averages.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    wflow_config_fn : str, optional
        Name of the wflow configuration file, by default "wflow_sbm.toml". Used to read
        the right results files from the wflow model.
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .

    Returns
    ----------
    qsim: xr.Dataset
        simulated discharge at wflow basin locations and at observation locations
    ds_basin: xr.Dataset
        basin average flux and state variables

    """
    mod = WflowModel(
        root=wflow_root,
        mode="r",
        config_fn=config_fn,
    )

    # Q at wflow locations
    qsim = mod.results["Q_gauges"].rename("Q")
    qsim = qsim.assign_coords(
        station_name=(
            "index",
            ["wflow_" + x for x in list(qsim["index"].values.astype(str))],
        )
    )

    # Discharge at the gauges_locs if present
    if gauges_locs is not None and os.path.exists(gauges_locs):
        # Get name of gauges dataset from the gauges locations file
        gauges_output_name = os.path.basename(gauges_locs).split(".")[0]
        if f"Q_gauges_{gauges_output_name}" in mod.results:
            qsim_gauges = mod.results[f"Q_gauges_{gauges_output_name}"].rename("Q")
            # Add station_name > bug for reading geoms if dir_input in toml is not None
            if f"gauges_{gauges_output_name}" not in mod.geoms:
                dir_geoms = dirname(
                    join(
                        mod.root,
                        mod.get_config("dir_input", abs_path=False),
                        mod.get_config("input.path_static", abs_path=False),
                    )
                )
                dir_geoms = join(dir_geoms, "staticgeoms")
                mod.read_geoms(dir_geoms)
            gdf_gauges = (
                mod.geoms[f"gauges_{gauges_output_name}"]
                .rename(columns={"wflow_id": "index"})
                .set_index("index")
            )
            qsim_gauges = qsim_gauges.assign_coords(
                station_name=(
                    "index",
                    list(gdf_gauges["station_name"][qsim_gauges.index.values].values),
                )
            )
    else:
        qsim_gauges = None

    # merge qsim and qsim_gauges
    qsim = xr.concat([qsim, qsim_gauges], dim="index")

    # Other catchment average outputs
    ds_basin = xr.merge(
        [mod.results[dvar] for dvar in mod.results if "_basavg" in dvar]
    )
    ds_basin = ds_basin.squeeze(drop=True)
    # If precipitation, skip as this will be plotted with the other climate data
    if "precipitation_basavg" in ds_basin:
        ds_basin = ds_basin.drop_vars("precipitation_basavg")

    return qsim, ds_basin


def analyse_wflow_delta(
    wflow_hist_run_config: Path,
    wflow_delta_runs_config: List[Path],
    models: List[str],
    scenarios: List[str],
    gauges_locs: Union[Path, str] = None,
    plot_dir: Union[str, Path] = None,
    start_month_hyd_year: str = "JAN",
):
    """
    Evaluate impact of climate change for delta change runs compared to historical.

    Model results should include the following keys: Q_gauges,
    Q_gauges_{basename(gauges_locs)}, snow_basavg.

    For each streamflow station, the following plots are made:

    - plot of cumulative streamflow for historical and scenarios for near and far future
    - plot of mean monthly streamflow for historical and scenarios for near and far future
    - plot of mean annual streamflow for historical and scenarios for near and far future
    - plot of annual maximum streamflow for historical and scenarios for near and far future
    - plot of annual minimum 7days streamflow for historical and scenarios for near and far future
    - plot of timeseries of daily streamflow for historical and scenarios for near and far future
    - plot of plotting position maximum annual streamflow for historical and scenarios for near and far future
    - plot of plotting position min 7days annual streamflow for historical and scenarios for near and far future

    - plot of relative change of mean annual streamflow for scenarios for near and far future compared to historical
    - plot of relative change of maximum annual streamflow for scenarios for near and far future compared to historical
    - plot of relative change of minimum 7days annual streamflow for scenarios for near and far future compared to historical

    - boxplot of monthly streamflow (absolute values and relative change) - to show monthly variability over the full period

    - boxplot of monthly snowpack (absolute values and relative change) averaged over basin if in ds_basin
    - plot of relative change of mean monthly and annual basin average variables for scenarios for near and far future compared to historical
    - plot of absolute values of mean monthly and annual basin average variables for historical and scenarios for near and far future

    Parameters
    ----------

    wflow_hist_run_config : Union[str, Path]
        Path to the wflow model config file of the historical run
        NB: it is important that the historical run is initialized with a warm state as the full period is used to make the plots!
    wflow_runs_toml : List[Path]
        List of paths of config files for the delta change runs.
    models: List[str]
        List of climate models
    scenarios: List[str]
        List of climate scenarios
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    plot_dir : Union[str, Path], optional
        Path to the output folder. If None (default), create a folder "plots"
        in the wflow_hist_run_config folder.
    start_month_hyd_year: str, optional
        start month for hydrological year. default is "JAN"

    """
    ### 1. Prepare output and plotting options ###

    # If plotting dir is None, create
    if plot_dir is None:
        wflow_root = os.path.dirname(wflow_hist_run_config)
        plot_dir = os.path.join(wflow_root, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Plotting options
    fs = 7
    lw = 0.8

    # Other plot options
    linestyle = "-"
    marker = "o"

    # read model results for historical
    root = os.path.dirname(wflow_hist_run_config)
    config_fn = os.path.basename(wflow_hist_run_config)
    qsim_hist, ds_basin_hist = get_wflow_results(root, config_fn, gauges_locs)

    # read the model results and merge to single netcdf
    qsim_delta = []
    ds_basin_delta = []
    for delta_config in wflow_delta_runs_config:
        model = os.path.basename(delta_config).split(".")[0].split("_")[-3]
        scenario = os.path.basename(delta_config).split(".")[0].split("_")[-2]
        horizon = os.path.basename(delta_config).split(".")[0].split("_")[-1]
        root = os.path.dirname(delta_config)
        config_fn = os.path.basename(delta_config)
        qsim_delta_run, ds_basin_delta_run = get_wflow_results(
            root, config_fn, gauges_locs
        )
        qsim_delta_run = qsim_delta_run.assign_coords(
            {"horizon": horizon, "model": model, "scenario": scenario}
        ).expand_dims(["horizon", "model", "scenario"])
        ds_basin_delta_run = ds_basin_delta_run.assign_coords(
            {"horizon": horizon, "model": model, "scenario": scenario}
        ).expand_dims(["horizon", "model", "scenario"])
        qsim_delta.append(qsim_delta_run)
        ds_basin_delta.append(ds_basin_delta_run)
    qsim_delta = xr.merge(qsim_delta)
    ds_basin_delta = xr.merge(ds_basin_delta)

    cmap = sns.color_palette("Set2", len(np.atleast_1d(scenarios).tolist()))

    # make plots per station
    for index in qsim_delta.index.values:

        # plot cumsum
        plot_near_far_abs(
            qsim_delta["Q"].cumsum("time").sel(index=index),
            qsim_hist.cumsum("time").sel(index=index),
            plot_dir=plot_dir,
            ylabel="Q",
            figname_prefix=f"cumsum_{index}",
            cmap=cmap,
            fs=fs,
        )

        # plot mean monthly flow
        plot_near_far_abs(
            qsim_delta["Q"].groupby("time.month").mean("time").sel(index=index),
            qsim_hist.groupby("time.month").mean("time").sel(index=index),
            plot_dir=plot_dir,
            ylabel="Q (m$^3$s$^{-1}$)",
            figname_prefix=f"mean_monthly_Q_{index}",
            cmap=cmap,
            fs=fs,
        )

        # plot nm7q timeseries
        qsim_delta_nm7q = (
            qsim_delta["Q"]
            .rolling(time=7)
            .mean()
            .resample(time="YS")
            .min("time")
            .sel(index=index)
        )
        qsim_hist_nm7q = (
            qsim_hist.rolling(time=7)
            .mean()
            .resample(time="YS")
            .min("time")
            .sel(index=index)
        )
        qsim_delta_nm7q_rel = (qsim_delta_nm7q - qsim_hist_nm7q) / qsim_hist_nm7q * 100
        plot_near_far_abs(
            qsim_delta_nm7q,
            qsim_hist_nm7q,
            plot_dir=plot_dir,
            ylabel="NM7Q (m$^3$s$^{-1}$)",
            figname_prefix=f"nm7q_{index}",
            cmap=cmap,
            fs=fs,
        )

        # plot maxq timeseries
        qsim_delta_maxq = (
            qsim_delta["Q"]
            .resample(time=f"YS-{start_month_hyd_year}")
            .max("time")
            .sel(index=index)
        )
        qsim_hist_maxq = (
            qsim_hist.resample(time=f"YS-{start_month_hyd_year}")
            .max("time")
            .sel(index=index)
        )
        qsim_delta_maxq_rel = (qsim_delta_maxq - qsim_hist_maxq) / qsim_hist_maxq * 100
        plot_near_far_abs(
            qsim_delta_maxq,
            qsim_hist_maxq,
            plot_dir=plot_dir,
            ylabel="max annual Q (m$^3$s$^{-1}$)",
            figname_prefix=f"max_annual_q_{index}",
            cmap=cmap,
            fs=fs,
        )

        # plot mean annual flow
        qsim_delta_meanq = (
            qsim_delta["Q"]
            .resample(time=f"YS-{start_month_hyd_year}")
            .mean("time")
            .sel(index=index)
        )
        qsim_hist_meanq = (
            qsim_hist.resample(time=f"YS-{start_month_hyd_year}")
            .mean("time")
            .sel(index=index)
        )
        qsim_delta_meanq_rel = (
            (qsim_delta_meanq - qsim_hist_meanq) / qsim_hist_meanq * 100
        )
        plot_near_far_abs(
            qsim_delta_meanq,
            qsim_hist_meanq,
            plot_dir=plot_dir,
            ylabel="mean annual Q (m$^3$s$^{-1}$)",
            figname_prefix=f"mean_annual_q_{index}",
            cmap=cmap,
            fs=fs,
        )

        # plot timeseries daily q
        qsim_delta_d = qsim_delta["Q"].sel(index=index)
        qsim_hist_d = qsim_hist.sel(index=index)
        plot_near_far_abs(
            qsim_delta_d,
            qsim_hist_d,
            plot_dir=plot_dir,
            ylabel="Q (m$^3$s$^{-1}$)",
            figname_prefix=f"qhydro_{index}",
            cmap=cmap,
            fs=fs,
        )

        # plot relative change mean, max, min q
        for dvar, prefix in zip(
            [qsim_delta_meanq_rel, qsim_delta_maxq_rel, qsim_delta_nm7q_rel],
            ["mean", "max", "nm7q"],
        ):
            plot_near_far_rel(
                dvar,
                plot_dir=plot_dir,
                ylabel=f"Change {prefix} (Qfut-Qhist)/Qhist (%)",
                figname_prefix=f"{prefix}_annual_q_{index}",
                cmap=cmap,
                fs=fs,
            )

        # plotting position maxq
        plot_plotting_position(
            qsim_delta_maxq,
            qsim_hist_maxq,
            plot_dir,
            f"maxq_{index}",
            cmap,
            "Annual maximum discharge (m$^3$s$^{-1}$)",
            ascending=True,
        )

        # plotting position nm7q
        plot_plotting_position(
            qsim_delta_nm7q,
            qsim_hist_nm7q,
            plot_dir,
            f"nm7q_{index}",
            cmap,
            "Annual min 7 days discharge (m$^3$s$^{-1}$)",
            ascending=False,
        )

        # plot boxplot monthly - abs
        qsim_delta_m = qsim_delta_d.resample(time="M").mean("time")
        qsim_hist_m = qsim_hist_d.resample(time="M").mean("time")
        qsim_delta_m_rel = (qsim_delta_m - qsim_hist_m) / qsim_hist_m * 100

        df_hist = pd.DataFrame(qsim_hist_m.to_dataframe().unstack()["Q"].rename("Q"))
        df_hist["month"] = df_hist.index.month
        df_hist["scenario"] = "hist"

        # absolute monthly q
        df_delta_near, df_delta_far = get_df_seaborn(qsim_delta_m.dropna("time"), "Q")
        # merge with df_hist
        df_delta_near = pd.concat([df_delta_near, df_hist])
        df_delta_far = pd.concat([df_delta_far, df_hist])

        # boxplot monthly q abs values
        palette_with_hist = ["grey"] + cmap
        hue_order_with_hist = ["hist"] + scenarios
        make_boxplot_monthly(
            df_delta_near,
            df_delta_far,
            plot_dir,
            f"q_abs_{index}",
            "Q (m$^3$s$^{-1}$)",
            palette_with_hist,
            hue_order_with_hist,
        )

        # boxplot relative monthly q
        df_delta_near_rel, df_delta_far_rel = get_df_seaborn(
            qsim_delta_m_rel.dropna("time"), "Q"
        )
        # boxplot relative change q
        palette = cmap
        hue_order = scenarios
        make_boxplot_monthly(
            df_delta_near_rel,
            df_delta_far_rel,
            plot_dir,
            f"q_rel_{index}",
            "change monthly Q (%)",
            palette,
            hue_order,
            relative=True,
        )

    # boxplot relative monthly snow
    if "snow_basavg" in ds_basin_delta.data_vars:
        ds_basin_delta_m = ds_basin_delta.resample(time="M").mean("time")
        ds_basin_hist_m = ds_basin_hist.resample(time="M").mean("time")
        ds_basin_delta_m_rel = (
            (ds_basin_delta_m - ds_basin_hist_m) / ds_basin_hist_m * 100
        )
        # NB: remove nan otherwise boxplot fails!
        df_delta_near_rel, df_delta_far_rel = get_df_seaborn(
            ds_basin_delta_m_rel.dropna("time"), "snow_basavg"
        )
        # boxplot relative change snow
        palette = cmap
        hue_order = scenarios
        make_boxplot_monthly(
            df_delta_near_rel,
            df_delta_far_rel,
            plot_dir,
            "snow_rel",
            "change monthly snow (%)",
            palette,
            hue_order,
            var_y="snow_basavg",
            relative=True,
        )

    # plot basinavg monthly and annual
    for dvar in ds_basin_delta.data_vars:
        print(dvar)
        resample = WFLOW_VARS[dvar.split("_")[0]]["resample"]
        sum_monthly_delta, sum_annual_delta, mean_monthly_delta = (
            get_sum_annual_and_monthly(ds_basin_delta, dvar, resample=resample)
        )
        sum_monthly_hist, sum_annual_hist, mean_monthly_hist = (
            get_sum_annual_and_monthly(ds_basin_hist, dvar, resample=resample)
        )

        mean_monthly_delta_rel = (
            (mean_monthly_delta - mean_monthly_hist) / mean_monthly_hist * 100
        )
        sum_annual_delta_rel = (
            (sum_annual_delta - sum_annual_hist) / sum_annual_hist * 100
        )

        # mean monthly sum or mean
        plot_near_far_abs(
            mean_monthly_delta,
            mean_monthly_hist,
            plot_dir=plot_dir,
            ylabel=WFLOW_VARS[dvar.split("_")[0]]["legend"],
            figname_prefix=f"mean_monthly_{dvar}",
            cmap=cmap,
            fs=fs,
        )

        # mean annual sum or mean
        plot_near_far_abs(
            sum_annual_delta,
            sum_annual_hist,
            plot_dir=plot_dir,
            ylabel=WFLOW_VARS[dvar.split("_")[0]]["legend_annual"],
            figname_prefix=f"sum_annual_{dvar}",
            cmap=cmap,
            fs=fs,
        )

        # relative mean monthly sum or mean
        plot_near_far_rel(
            mean_monthly_delta_rel,
            plot_dir=plot_dir,
            ylabel=f"Change monthly {dvar} \n (fut-hist)/hist (%)",
            figname_prefix=f"mean_monthly_{dvar}",
            cmap=cmap,
            fs=fs,
        )

        # relative mean annual sum or mean
        plot_near_far_rel(
            sum_annual_delta_rel,
            plot_dir=plot_dir,
            ylabel=f"Change annual {dvar} \n (fut-hist)/hist (%)",
            figname_prefix=f"sum_annual_{dvar}",
            cmap=cmap,
            fs=fs,
        )

    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        project_dir = sm.params.project_dir
        Folder_plots = f"{project_dir}/plots/model_delta_runs"
        root = f"{project_dir}/hydrology_model"

        analyse_wflow_delta(
            wflow_hist_run_config=sm.params.wflow_hist_run_config,
            wflow_delta_runs_config=sm.params.wflow_delta_runs_config,
            models=sm.params.models,
            scenarios=sm.params.scenarios,
            gauges_locs=sm.params.gauges_locs,
            plot_dir=Folder_plots,
            start_month_hyd_year=sm.params.start_month_hyd_year,
        )
    else:
        print("run with snakemake please")
