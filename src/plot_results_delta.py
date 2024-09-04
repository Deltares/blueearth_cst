"""Plot wflow results of delta change runs."""

import xarray as xr
import os
from os.path import join, dirname, basename
from pathlib import Path
import pandas as pd
from hydromt_wflow import WflowModel

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
        plot_plotting_position,
    )
    from wflow.wflow_utils import get_wflow_results
else:
    from .plot_utils.plot_change_delta_runs import (
        plot_near_far_abs,
        plot_near_far_rel,
        get_df_seaborn,
        get_sum_annual_and_monthly,
        make_boxplot_monthly,
        plot_plotting_position,
    )
    from .wflow.wflow_utils import get_wflow_results


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
        "resample": "mean",
        "legend": "Snowpack (mm)",
        "legend_annual": "Snowpack (mm)",
    },
    "glacier": {
        "resample": "mean",
        "legend": "Glacier water volume (mm)",
        "legend_annual": "Glacier water volume (mm)",
    },
}


def analyse_wflow_delta(
    wflow_hist_run_config: Path,
    wflow_delta_runs_config: List[Path],
    gauges_locs: Optional[Union[Path, str]] = None,
    plot_dir: Optional[Union[str, Path]] = None,
    start_month_hyd_year: str = "JAN",
    near_legend: str = "near future",
    far_legend: str = "far future",
    add_plots_with_all_lines: bool = False,
):
    """
    Evaluate impact of climate change for delta change runs compared to historical.

    Model results should include the following keys: Q_gauges,
    Q_gauges_{basename(gauges_locs)}, {variable}_basavg.

    Future time horizons should be called near and far. The legends can be updated.

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
    near_legend: str, optional
        legend for near future, default is "near future"
    far_legend: str, optional
        legend for far future, default is "far future"
    add_plots_with_all_lines: bool, optional
        add another version of the abs/rel plots with all lines on top on the mean and
        min-max shaded ones. Default is False.
    """
    ### 1. Prepare output and plotting options ###
    # If plotting dir is None, create
    if plot_dir is None:
        wflow_root = dirname(wflow_hist_run_config)
        plot_dir = join(wflow_root, "plots")
    plot_dir_flow = join(plot_dir, "flow")
    if not os.path.exists(plot_dir_flow):
        os.makedirs(plot_dir_flow)
    plot_dir_other = join(plot_dir, "other")
    if not os.path.exists(plot_dir_other):
        os.makedirs(plot_dir_other)

    # Plotting options
    fs = 7
    lw = 0.6

    # read model results for historical
    root = dirname(wflow_hist_run_config)
    config_fn = basename(wflow_hist_run_config)
    qsim_hist, ds_clim_hist, ds_basin_hist = get_wflow_results(
        root, config_fn, gauges_locs
    )

    # read the model results and merge to single netcdf
    qsim_delta = []
    ds_basin_delta = []
    for delta_config in wflow_delta_runs_config:
        model = basename(delta_config).split(".")[0].split("_")[-3]
        scenario = basename(delta_config).split(".")[0].split("_")[-2]
        horizon = basename(delta_config).split(".")[0].split("_")[-1]
        root = dirname(delta_config)
        config_fn = basename(delta_config)
        qsim_delta_run, ds_clim_delta_run, ds_basin_delta_run = get_wflow_results(
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

    # Slice historical reference run (may be longer than the future one) before plotting
    qsim_hist = qsim_hist.sel(time=slice(qsim_delta["time"][0], qsim_delta["time"][-1]))
    ds_basin_hist = ds_basin_hist.sel(
        time=slice(ds_basin_delta["time"][0], ds_basin_delta["time"][-1])
    )

    # make plots per station
    for index in qsim_delta["index"].values:
        plot_dir_flow_id = join(plot_dir_flow, str(index))
        # plot cumsum
        plot_near_far_abs(
            qsim_delta["Q"].cumsum("time").sel(index=index),
            qsim_hist.cumsum("time").sel(index=index),
            plot_dir=plot_dir_flow_id,
            ylabel="Q",
            figname_prefix=f"cumsum_{index}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                qsim_delta["Q"].cumsum("time").sel(index=index),
                qsim_hist.cumsum("time").sel(index=index),
                plot_dir=plot_dir_flow_id,
                ylabel="Q",
                figname_prefix=f"cumsum_{index}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

        # plot mean monthly flow
        plot_near_far_abs(
            qsim_delta["Q"].groupby("time.month").mean("time").sel(index=index),
            qsim_hist.groupby("time.month").mean("time").sel(index=index),
            plot_dir=plot_dir_flow_id,
            ylabel="mean monthly Q (m$^3$s$^{-1}$)",
            figname_prefix=f"mean_monthly_Q_{index}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                qsim_delta["Q"].groupby("time.month").mean("time").sel(index=index),
                qsim_hist.groupby("time.month").mean("time").sel(index=index),
                plot_dir=plot_dir_flow_id,
                ylabel="mean monthly Q (m$^3$s$^{-1}$)",
                figname_prefix=f"mean_monthly_Q_{index}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
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
            plot_dir=plot_dir_flow_id,
            ylabel="NM7Q (m$^3$s$^{-1}$)",
            figname_prefix=f"nm7q_{index}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                qsim_delta_nm7q,
                qsim_hist_nm7q,
                plot_dir=plot_dir_flow_id,
                ylabel="NM7Q (m$^3$s$^{-1}$)",
                figname_prefix=f"nm7q_{index}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
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
            plot_dir=plot_dir_flow_id,
            ylabel="max annual Q (m$^3$s$^{-1}$)",
            figname_prefix=f"max_annual_q_{index}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                qsim_delta_maxq,
                qsim_hist_maxq,
                plot_dir=plot_dir_flow_id,
                ylabel="max annual Q (m$^3$s$^{-1}$)",
                figname_prefix=f"max_annual_q_{index}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
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
            plot_dir=plot_dir_flow_id,
            ylabel="mean annual Q (m$^3$s$^{-1}$)",
            figname_prefix=f"mean_annual_q_{index}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                qsim_delta_meanq,
                qsim_hist_meanq,
                plot_dir=plot_dir_flow_id,
                ylabel="mean annual Q (m$^3$s$^{-1}$)",
                figname_prefix=f"mean_annual_q_{index}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

        # plot timeseries daily q
        qsim_delta_d = qsim_delta["Q"].sel(index=index)
        qsim_hist_d = qsim_hist.sel(index=index)
        plot_near_far_abs(
            qsim_delta_d,
            qsim_hist_d,
            plot_dir=plot_dir_flow_id,
            ylabel="Q (m$^3$s$^{-1}$)",
            figname_prefix=f"qhydro_{index}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                qsim_delta_d,
                qsim_hist_d,
                plot_dir=plot_dir_flow_id,
                ylabel="Q (m$^3$s$^{-1}$)",
                figname_prefix=f"qhydro_{index}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

        # plot relative change mean, max, min q
        for dvar, prefix in zip(
            [qsim_delta_meanq_rel, qsim_delta_maxq_rel, qsim_delta_nm7q_rel],
            ["mean", "max", "nm7q"],
        ):
            plot_near_far_rel(
                dvar,
                plot_dir=plot_dir_flow_id,
                ylabel=f"Change {prefix} (Qfut-Qhist)/Qhist (%)",
                figname_prefix=f"{prefix}_annual_q_{index}",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
            )
            if add_plots_with_all_lines:
                plot_near_far_rel(
                    dvar,
                    plot_dir=plot_dir_flow_id,
                    ylabel=f"Change {prefix} (Qfut-Qhist)/Qhist (%)",
                    figname_prefix=f"{prefix}_annual_q_{index}_all_lines",
                    fs=fs,
                    lw=lw,
                    near_legend=near_legend,
                    far_legend=far_legend,
                    show_all_lines=True,
                )

        # plotting position maxq
        plot_plotting_position(
            qsim_delta_maxq,
            qsim_hist_maxq,
            plot_dir_flow_id,
            f"maxq_{index}",
            "Annual maximum discharge (m$^3$s$^{-1}$)",
            ascending=True,
            near_legend=near_legend,
            far_legend=far_legend,
        )

        # plotting position nm7q
        plot_plotting_position(
            qsim_delta_nm7q,
            qsim_hist_nm7q,
            plot_dir_flow_id,
            f"nm7q_{index}",
            "Annual min 7 days discharge (m$^3$s$^{-1}$)",
            ascending=False,
            near_legend=near_legend,
            far_legend=far_legend,
        )

        # plot boxplot monthly - abs
        qsim_delta_m = qsim_delta_d.resample(time="M").mean("time")
        qsim_hist_m = qsim_hist_d.resample(time="M").mean("time")
        qsim_delta_m_rel = (qsim_delta_m - qsim_hist_m) / qsim_hist_m * 100

        df_hist = pd.DataFrame(qsim_hist_m.to_dataframe().unstack()["Q"].rename("Q"))
        df_hist["month"] = df_hist.index.month
        df_hist["scenario"] = "historical"

        # absolute monthly q
        df_delta_near, df_delta_far = get_df_seaborn(qsim_delta_m.dropna("time"), "Q")
        # merge with df_hist
        df_delta_near = pd.concat([df_delta_near, df_hist])
        df_delta_far = pd.concat([df_delta_far, df_hist])

        # boxplot monthly q abs values
        make_boxplot_monthly(
            df_delta_near,
            df_delta_far,
            plot_dir_flow_id,
            f"q_abs_{index}",
            "Q (m$^3$s$^{-1}$)",
            near_legend=near_legend,
            far_legend=far_legend,
        )

        # boxplot relative monthly q
        df_delta_near_rel, df_delta_far_rel = get_df_seaborn(
            qsim_delta_m_rel.dropna("time"), "Q"
        )
        # boxplot relative change q
        make_boxplot_monthly(
            df_delta_near_rel,
            df_delta_far_rel,
            plot_dir_flow_id,
            f"q_rel_{index}",
            "change monthly Q (%)",
            relative=True,
            near_legend=near_legend,
            far_legend=far_legend,
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
            ds_basin_delta_m_rel[["snow_basavg"]].dropna("time"), "snow_basavg"
        )
        # boxplot relative change snow
        make_boxplot_monthly(
            df_delta_near_rel,
            df_delta_far_rel,
            plot_dir_other,
            "snow_rel",
            "change monthly snow (%)",
            var_y="snow_basavg",
            relative=True,
            near_legend=near_legend,
            far_legend=far_legend,
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
            plot_dir=plot_dir_other,
            ylabel=WFLOW_VARS[dvar.split("_")[0]]["legend"],
            figname_prefix=f"mean_monthly_{dvar}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                mean_monthly_delta,
                mean_monthly_hist,
                plot_dir=plot_dir_other,
                ylabel=WFLOW_VARS[dvar.split("_")[0]]["legend"],
                figname_prefix=f"mean_monthly_{dvar}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

        # mean annual sum or mean
        plot_near_far_abs(
            sum_annual_delta,
            sum_annual_hist,
            plot_dir=plot_dir_other,
            ylabel=WFLOW_VARS[dvar.split("_")[0]]["legend_annual"],
            figname_prefix=f"sum_annual_{dvar}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_abs(
                sum_annual_delta,
                sum_annual_hist,
                plot_dir=plot_dir_other,
                ylabel=WFLOW_VARS[dvar.split("_")[0]]["legend_annual"],
                figname_prefix=f"sum_annual_{dvar}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

        # relative mean monthly sum or mean
        plot_near_far_rel(
            mean_monthly_delta_rel,
            plot_dir=plot_dir_other,
            ylabel=f"Change monthly {dvar} \n (fut-hist)/hist (%)",
            figname_prefix=f"mean_monthly_{dvar}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_rel(
                mean_monthly_delta_rel,
                plot_dir=plot_dir_other,
                ylabel=f"Change monthly {dvar} \n (fut-hist)/hist (%)",
                figname_prefix=f"mean_monthly_{dvar}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

        # relative mean annual sum or mean
        plot_near_far_rel(
            sum_annual_delta_rel,
            plot_dir=plot_dir_other,
            ylabel=f"Change annual {dvar} \n (fut-hist)/hist (%)",
            figname_prefix=f"sum_annual_{dvar}",
            fs=fs,
            lw=lw,
            near_legend=near_legend,
            far_legend=far_legend,
        )
        if add_plots_with_all_lines:
            plot_near_far_rel(
                sum_annual_delta_rel,
                plot_dir=plot_dir_other,
                ylabel=f"Change annual {dvar} \n (fut-hist)/hist (%)",
                figname_prefix=f"sum_annual_{dvar}_all_lines",
                fs=fs,
                lw=lw,
                near_legend=near_legend,
                far_legend=far_legend,
                show_all_lines=True,
            )

    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        # output folder
        project_dir = sm.params.project_dir
        Folder_plots = f"{project_dir}/plots/model_delta_runs"
        root = f"{project_dir}/hydrology_model"
        # time horizons legend
        future_horizons = sm.params.future_horizons
        near_horizon = future_horizons["near"].replace(", ", "-")
        far_horizon = future_horizons["far"].replace(", ", "-")

        analyse_wflow_delta(
            wflow_hist_run_config=sm.params.wflow_hist_run_config,
            wflow_delta_runs_config=sm.params.wflow_delta_runs_config,
            gauges_locs=sm.params.gauges_locs,
            plot_dir=Folder_plots,
            start_month_hyd_year=sm.params.start_month_hyd_year,
            near_legend=f"Horizon {near_horizon}",
            far_legend=f"Horizon {far_horizon}",
            add_plots_with_all_lines=sm.params.add_plots_with_all_lines,
        )
    else:
        print("run with snakemake please")
