import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import xarray as xr
import seaborn as sns

import matplotlib.patches as mpatches

from typing import List

__all__ = [
    "plot_near_far_abs",
    "plot_near_far_rel",
    "get_sum_annual_and_monthly",
    "get_df_seaborn",
    "make_boxplot_monthly",
    "get_plotting_position",
    "plot_plotting_position",
]

COLORS = {
    "ssp126": "#003466",
    "ssp245": "#f69320",
    "ssp370": "#df0000",
    "ssp585": "#980002",
}

MONTHS_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def plot_near_far_abs(
    qsim_delta_metric: xr.DataArray,
    q_hist_metric: xr.DataArray,
    plot_dir: str,
    ylabel: str,
    figname_prefix: str,
    fs: int = 8,
    lw: int = 0.6,
    near_legend: str = "near future",
    far_legend: str = "far future",
):
    """
    subplots of variable for near and far future showing scenarios and historical output.

    Parameters
    ----------
    qsim_delta_metric: xr.DataArray,
        data to plot for delta runs. Should include horizon (near and far), model and scenario as coordinates.
    q_hist_metric: xr.DataArray,
        data to plot for historical run
    plot_dir: str,
        directory to save figures
    ylabel: str,
        name of ylabel axis
    figname_prefix: str,
        prefix name of figure (suffix is .png)
        if qhydro in prefix -- 2 rows in subplots instead of 2 columns for near and far future
        if mean_monthly in prefix -- xticks are 1 to 12.
    fs: int = 8,
        fontsize
    lw: int = 0.6,
        linewidth
    near_legend: str = "near future",
        legend for near future
    far_legend: str = "far future",
        legend for far future
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    if "qhydro" in figname_prefix:
        fig, axes = plt.subplots(
            2, 1, figsize=(16 / 2.54, 12 / 2.54), sharex=True, sharey=True
        )
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(16 / 2.54, 8 / 2.54), sharex=True, sharey=True
        )
    axes = axes.flatten()

    for scenario in qsim_delta_metric.scenario.values:
        for ax, horizon in zip(axes, ["near", "far"]):
            # Find the min, max and mean over the models
            qsim_delta_metric_min = (
                qsim_delta_metric.sel(horizon=horizon)
                .sel(scenario=scenario)
                .min("model")
            )
            qsim_delta_metric_max = (
                qsim_delta_metric.sel(horizon=horizon)
                .sel(scenario=scenario)
                .max("model")
            )
            qsim_delta_metric_mean = (
                qsim_delta_metric.sel(horizon=horizon)
                .sel(scenario=scenario)
                .mean("model")
            )
            # plot the mean
            qsim_delta_metric_mean.plot(
                label=f"{scenario}", ax=ax, color=COLORS[scenario], linewidth=lw
            )
            # plot the min and max with a fill between
            dim0 = qsim_delta_metric_mean.dims[0]
            ax.fill_between(
                qsim_delta_metric_mean[dim0].values,
                qsim_delta_metric_min.values,
                qsim_delta_metric_max.values,
                color=COLORS[scenario],
                alpha=0.4,
                # label = f"{scenario} min-max",
            )
            # # plot all lines
            # qsim_delta_metric.sel(horizon=horizon).sel(scenario=scenario).plot(
            #     hue="model",
            #     ax=ax,
            #     color=COLORS[scenario],
            #     add_legend=False,
            #     linewidth=lw,
            #     alpha=0.5,
            # )

    for ax in axes:
        q_hist_metric.plot(label="historical", color="k", ax=ax, linewidth=lw)
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_xlabel("")
        ax.legend(fontsize=fs)
        if "mean_monthly" in figname_prefix:
            ax.set_xticks(ticks=np.arange(1, 13), labels=MONTHS_LABELS)
    axes[0].set_ylabel(f"{ylabel}", fontsize=fs)
    axes[1].set_ylabel("")
    axes[0].set_title(near_legend, fontsize=fs)
    axes[1].set_title(far_legend, fontsize=fs)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{figname_prefix}.png"), dpi=300)
    plt.close()


def plot_near_far_rel(
    qsim_delta_metric: xr.DataArray,
    plot_dir: str,
    ylabel: str,
    figname_prefix: str,
    fs: int = 8,
    lw: int = 0.6,
    near_legend: str = "near future",
    far_legend: str = "far future",
):
    """
    subplots of variable for near and far future showing relative change in scenarios compared to historical output

    Parameters
    ----------
    qsim_delta_metric: xr.DataArray,
        data to plot for delta runs. Should include horizon (near and far), model and scenario as coordinates.
    plot_dir: str,
        directory to save figures
    ylabel: str,
        name of ylabel axis
    figname_prefix: str,
        prefix name of figure (prefix before prefix is rel_ and suffix is .png)
        if mean_monthly in prefix -- xticks are 1 to 12.
    fs: int = 8,
        fontsize
    lw: int = 0.6,
        linewidth
    near_legend: str = "near future",
        legend for near future
    far_legend: str = "far future",
        legend for far future
    """
    if figname_prefix == "qhydro":
        fig, axes = plt.subplots(
            2, 1, figsize=(16 / 2.54, 12 / 2.54), sharex=True, sharey=True
        )
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(16 / 2.54, 8 / 2.54), sharex=True, sharey=True
        )
    axes = axes.flatten()

    for scenario in qsim_delta_metric.scenario.values:
        for ax, horizon in zip(axes, ["near", "far"]):
            # Find the min, max and mean over the models
            qsim_delta_metric_min = (
                qsim_delta_metric.sel(horizon=horizon)
                .sel(scenario=scenario)
                .min("model")
            )
            qsim_delta_metric_max = (
                qsim_delta_metric.sel(horizon=horizon)
                .sel(scenario=scenario)
                .max("model")
            )
            qsim_delta_metric_mean = (
                qsim_delta_metric.sel(horizon=horizon)
                .sel(scenario=scenario)
                .mean("model")
            )
            # plot the mean
            qsim_delta_metric_mean.plot(
                label=f"{scenario}", ax=ax, color=COLORS[scenario], linewidth=lw
            )
            # plot the min and max with a fill between
            dim0 = qsim_delta_metric_mean.dims[0]
            ax.fill_between(
                qsim_delta_metric_mean[dim0].values,
                qsim_delta_metric_min.values,
                qsim_delta_metric_max.values,
                color=COLORS[scenario],
                alpha=0.4,
                # label = f"{scenario} min-max",
            )
            # # plot all lines
            # qsim_delta_metric.sel(horizon=horizon).sel(scenario=scenario).plot(
            #     hue="model",
            #     ax=ax,
            #     color=COLORS[scenario],
            #     add_legend=False,
            #     linewidth=lw,
            #     alpha=0.5,
            # )

    for ax in axes:
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_xlabel("")
        ax.axhline(0, linestyle="--", color="lightgrey")
        ax.legend(fontsize=fs)
        if "mean_monthly" in figname_prefix:
            ax.set_xticks(ticks=np.arange(1, 13), labels=MONTHS_LABELS)

    axes[0].set_ylabel(f"{ylabel}", fontsize=fs)
    axes[1].set_ylabel("")
    axes[0].set_title(near_legend, fontsize=fs)
    axes[1].set_title(far_legend, fontsize=fs)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rel_{figname_prefix}.png"), dpi=300)
    plt.close()


def get_sum_annual_and_monthly(ds: xr.Dataset, dvar: str, resample: str = "mean"):
    """
    get sum monthly, sum annual and sum monthly mean of variables.

    Parameters
    ----------
    ds: xr.DataSet,
        data of specific variable
    dvar: str,
        name of data_vars in ds
    resample: str,
        mean or sum resampling.

    Returns
    ---------
    sum_monthly: xr.DataArray
        monthly sum of variable
    sum_annual: xr.DataArray
        annual sum of variable
    sum_monthly_mean: xr.DataArray
        mean monthy value of variable

    """
    if resample == "sum":
        sum_monthly = ds[dvar].resample(time="ME").sum("time")
        sum_annual = ds[dvar].resample(time="YS").sum("time")
    else:  # assume mean
        sum_monthly = ds[dvar].resample(time="ME").mean("time")
        sum_annual = ds[dvar].resample(time="YS").mean("time")

    sum_monthly_mean = sum_monthly.groupby("time.month").mean("time")

    return sum_monthly, sum_annual, sum_monthly_mean


def get_plotting_position(dsq_max: xr.DataArray, ascending=True):
    """
    get plotting position and sorted values of variable

    Parameters
    ----------
    dsq_max: xr.DataArray,
        data with maximum (or minimum) values
    ascending: str, optional
        default is True (to get plotting position for max values). if plotting position of min values, ascending should be False.

    Returns
    ---------
    gumbel_p1: xr.DataArray
        plotting position
    plotting_positions: xr.DataArray
        sorted values of dsq_max
    max_y: float
        maximum value of dsq_max

    """
    a = 0.3
    b = 1.0 - 2.0 * a
    max_y = np.round(dsq_max.max().values)
    ymin, ymax = 0, max_y
    p1 = ((np.arange(1, len(dsq_max.time) + 1.0) - a)) / (len(dsq_max.time) + b)
    RP1 = 1 / (1 - p1)
    gumbel_p1 = -np.log(-np.log(1.0 - 1.0 / RP1))
    plotting_positions = dsq_max.sortby(dsq_max, ascending=ascending)
    return gumbel_p1, plotting_positions, max_y


def get_df_seaborn(
    ds_delta: xr.Dataset,
    var: str = "Q",
):
    """
    Parameters
    -----------
    ds_delta: xr.Dataset
        dataset with var
        assumes ds_delta has the coordinates model, scenario and horizon (near and far)
    var: str, optional
        name of data_var in ds_delta. default is Q

    Returns
    df_delta_near: pd.DataFrame
        dataframe with columns for month, scenario and monthly variable for the near future which can be used to make monthly boxplot with seaborn.
    df_delta_far: pd.DataFrame
        dataframe with columns for month, scenario and monthly variable for the far future which can be used to make monthly boxplot with seaborn.

    """

    # near fut
    df_delta_near = pd.DataFrame()
    for model in ds_delta.model.values:
        for scenario in ds_delta.scenario.values:
            df_1 = ds_delta.sel(
                horizon="near", model=model, scenario=scenario
            ).to_dataframe()
            df_1["month"] = df_1.index.month
            df_delta_near = pd.concat([df_delta_near, df_1[["scenario", "month", var]]])
    # far fut
    df_delta_far = pd.DataFrame()
    for model in ds_delta.model.values:
        for scenario in ds_delta.scenario.values:
            df_1 = ds_delta.sel(
                horizon="far", model=model, scenario=scenario
            ).to_dataframe()
            df_1["month"] = df_1.index.month
            df_delta_far = pd.concat([df_delta_far, df_1[["scenario", "month", var]]])

    return df_delta_near, df_delta_far


def make_boxplot_monthly(
    df_delta_near: pd.DataFrame,
    df_delta_far: pd.DataFrame,
    plot_dir: str,
    figname_suffix: str,
    ylabel: str,
    var_x: str = "month",
    var_y: str = "Q",
    var_hue: str = "scenario",
    relative: str = False,
    fs: int = 8,
    near_legend: str = "near future",
    far_legend: str = "far future",
):
    """
    make monthly subplots for near and far future. The width of the boxplot represents the number of years for each month

    Parameters
    ---------
    df_delta_near: pd.DataFrame,
        dataframe with columns for month, scenario and monthly variable for the near future which can be used to make monthly boxplot with seaborn.
    df_delta_far: pd.DataFrame,
        dataframe with columns for month, scenario and monthly variable for the far future which can be used to make monthly boxplot with seaborn.
    plot_dir: str,
        directory to save plot
    figname_suffix: str,
        suffix for figname (boxplot_{figname_suffix}.png)
    ylabel: str,
        ylabel description
    palette: List,
        cmap which should match the number of scenarios
    hue_order: List,
        hue_order for the scenarios
    var_x: str = "month",
        name of column of the dataframe which is shown on the xaxis. default is month
    var_y: str = "Q",
        name of column of the dataframe which is shown on the yaxis. default is Q
    var_hue:str = "scenario",
        name of column of the dataframe which is shown as hue. default is scenario
    relative: str = False,
        if df_delta_near contains relative values of change, a dashed line at y=0 is shown in the plot. default is False.
    fs:int=8,
        fontsize. default is 8
    near_legend: str = "near future",
        legend for near future
    far_legend: str = "far future",
        legend for far future
    """
    # Define palette and hue_order
    hue_order = np.unique(df_delta_near["scenario"].values).tolist()
    palette = [COLORS[scenario] for scenario in hue_order if scenario != "historical"]
    if "historical" in hue_order:
        palette = ["grey"] + palette

    fig, axes = plt.subplots(
        2, 1, figsize=(16 / 2.54, 12 / 2.54), sharex=True, sharey=True
    )
    sns.boxplot(
        data=df_delta_near,
        x=var_x,
        y=var_y,
        hue=var_hue,
        ax=axes[0],
        fliersize=0.5,
        linewidth=0.8,
        width=0.8,
        palette=palette,
        hue_order=hue_order,
    )
    sns.boxplot(
        data=df_delta_far,
        x=var_x,
        y=var_y,
        hue=var_hue,
        ax=axes[1],
        fliersize=0.5,
        linewidth=0.8,
        width=0.8,
        palette=palette,
        hue_order=hue_order,
    )
    for ax in axes:
        ax.legend("", frameon=False)
        ax.set_xlabel("")
        ax.set_ylabel(f"{ylabel}", fontsize=fs)
        ax.tick_params(axis="both", labelsize=fs)
        if relative == True:
            ax.axhline(0, linestyle="--", color="lightgrey")
    axes[0].set_title(near_legend, fontsize=fs)
    axes[1].set_title(far_legend, fontsize=fs)
    axes[0].legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=fs
    )
    axes[1].set_xticklabels(MONTHS_LABELS, fontsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"boxplot_{figname_suffix}.png"), dpi=300)


def plot_plotting_position(
    qsim_delta_var: xr.DataArray,
    qsim_hist_var: xr.DataArray,
    plot_dir: str,
    figname_suffix: str,
    ylabel: str,
    ascending: bool = True,
    ts: List[int] = [2.0, 5.0, 10.0, 30.0],
    fs: int = 8,
    near_legend: str = "near future",
    far_legend: str = "far future",
):
    """
    Returns a plot of plotting position as a funciton of max (or min) of a specific variable

    Parameters
    ----------
    qsim_delta_var: xr.DataArray,
        dataarray containing data of min or max values of the delta change runs
    qsim_hist_var: xr.DataArray,
        dataarray containing data of min or max values of the historical run
    plot_dir: str,
        directory to save plot
    figname_suffix: str,
        name of suffix for figname (plotting_pos_{figname_suffix}.png)
    ylabel: str,
        ylabel description
    ascending: bool = True,
        ascending is typically True for maximum values and False for minimum values
    ts: List[int] = [2., 5.,10.,30.],
        list of return period values to show on the xaxis
    fs: int = 8,
        fontsize
    near_legend: str = "near future",
        legend for near future
    far_legend: str = "far future",
        legend for far future
    """
    do_boxplot = False
    show_all_markers = False
    hue_order = np.unique(qsim_delta_var.scenario.values).tolist()
    palette = [COLORS[scenario] for scenario in hue_order if scenario != "historical"]

    fig, axes = plt.subplots(
        1, 2, figsize=(16 / 2.54, 10 / 2.54), sharex=True, sharey=True
    )
    axes = axes.flatten()
    for ax, horizon in zip(axes, ["near", "far"]):
        df_gumbel = pd.DataFrame()
        for scenario in qsim_delta_var.scenario.values:
            df_gumbel_scenario = pd.DataFrame()
            for model in qsim_delta_var.model.values:
                # near
                qsim_delta_var_sel = qsim_delta_var.sel(
                    model=model, scenario=scenario, horizon=horizon
                )
                gumbel_p1, plotting_positions_hist, max_y_hist = get_plotting_position(
                    qsim_delta_var_sel, ascending=ascending
                )
                if not do_boxplot and show_all_markers:
                    ax.plot(
                        gumbel_p1,
                        plotting_positions_hist,
                        marker="o",
                        color=COLORS[scenario],
                        linestyle="None",
                        label="hist.",
                        markersize=5,
                    )
                else:
                    # convert to df for seaborn
                    df_gumbel_m = pd.DataFrame(
                        {
                            "return_period": gumbel_p1,
                            "var": plotting_positions_hist.values,
                            "model": np.repeat(model, len(gumbel_p1)),
                            "scenario": np.repeat(scenario, len(gumbel_p1)),
                        }
                    )
                    df_gumbel = pd.concat([df_gumbel, df_gumbel_m])
                    df_gumbel_scenario = pd.concat([df_gumbel_scenario, df_gumbel_m])
            if not do_boxplot and not show_all_markers:
                # compute the min, mean, max over the models
                df_gumbel_scenario = df_gumbel_scenario.groupby("return_period").agg(
                    {"var": ["min", "mean", "max"]}
                )
                # plot the mean with markers
                df_gumbel_scenario["var"]["mean"].plot(
                    label=f"{scenario}",
                    ax=ax,
                    color=COLORS[scenario],
                    linestyle="None",
                    marker="o",
                    markersize=5,
                )
                # fill between min and max
                ax.fill_between(
                    df_gumbel_scenario.index,
                    df_gumbel_scenario["var"]["min"],
                    df_gumbel_scenario["var"]["max"],
                    color=COLORS[scenario],
                    alpha=0.4,
                )
        # Add historical
        gumbel_p1, plotting_positions_hist, max_y_hist = get_plotting_position(
            qsim_hist_var, ascending=ascending
        )
        if do_boxplot:
            df_hist = pd.DataFrame(
                {
                    "return_period": gumbel_p1,
                    "var": plotting_positions_hist.values,
                    "model": np.repeat("hist.", len(gumbel_p1)),
                    "scenario": np.repeat("historical", len(gumbel_p1)),
                }
            )
            df_gumbel = pd.concat([df_gumbel, df_hist])
            sns.boxplot(
                data=df_gumbel,
                x="return_period",
                y="var",
                hue="scenario",
                ax=ax,
                fliersize=0.5,
                linewidth=0.8,
                width=0.8,
                palette=["grey"] + palette,
                hue_order=["historical"] + hue_order,
            )
        else:
            # add historical
            ax.plot(
                gumbel_p1,
                plotting_positions_hist,
                marker="+",
                color="k",
                linestyle="None",
                label="hist.",
                markersize=6,
            )

    axes[0].set_ylabel(f"{ylabel}", fontsize=fs)
    axes[0].set_title(near_legend, fontsize=fs)
    axes[1].set_title(far_legend, fontsize=fs)
    for ax in axes:
        ax.set_xlabel("Return period (1/year)", fontsize=fs)
        ax.xaxis.set_ticks([-np.log(-np.log(1 - 1.0 / t)) for t in ts])
        ax.xaxis.set_ticklabels([t for t in ts])
        ax.tick_params(axis="both", labelsize=fs)

    ll = []
    for scenario in qsim_delta_var.scenario.values:
        l1 = mpatches.Patch(color=COLORS[scenario], label=f"{scenario}")
        ll.append(l1)
    l3 = mpatches.Patch(color="k", label="Historical")
    plt.legend(handles=ll + [l3], fontsize=fs)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"plotting_pos_{figname_suffix}.png"), dpi=300)
    plt.close()
