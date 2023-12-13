# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:18:38 2021

@author: bouaziz
"""

# %%
import hydromt
from hydromt.stats import skills
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import xarray as xr

from typing import List, Union


# %%
# Supported wflow outputs
WFLOW_VARS = {
    "overland flow": {"resample": "mean", "legend": "Overland Flow (m$^3$s$^{-1}$)"},
    "actual evapotranspiration": {
        "resample": "sum",
        "legend": "Actual Evapotranspiration (mm month$^{-1}$)",
    },
    "groundwater recharge": {
        "resample": "sum",
        "legend": "groundwater recharge (mm month$^{-1}$)",
    },
    "snow": {"resample": "sum", "legend": "Snowpack (mm month$^{-1}$)"},
}


def rsquared(x, y):
    """Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2


def plot_signatures(
    dsq: xr.Dataset,
    Folder_out: Union[Path, str],
    station_name: str = "station",
    labels: List = ["Mod."],
    colors: List = ["orange"],
    linestyles: List = ["-"],
    markers: List = ["o"],
    lw: float = 0.8,
    fs: int = 8,
) -> pd.DataFrame:
    """
    Compute and plot hydrological signatures and performance metrics.

    Parameters
    ----------
    dsq : xr.Dataset
        Dataset with simulated and observed streamflow.

        * Required variables: [Q]
        * Required dimensions: [time, runs] and Obs. in runs
        * Required attributes: [station_name]
    Folder_out : Union[Path, str]
        Output folder to save plots.
    station_name : str, optional
        Station name, by default "station"
    labels : List, optional
        List of labels for the different runs, by default ["Mod."]
    colors : List, optional
        List of colors for the different runs, by default ["orange"]
    linestyles : List, optional
        List of linestyles for the different runs, by default ["-"]
    markers : List, optional
        List of markers for the different runs, by default ["o"]
    lw : float, optional
        Line width, by default 0.8
    fs : int, optional
        Font size, by default 8

    Returns
    -------
    pd.DataFrame
        Dataframe with performance metrics for each run.
    """

    ### 1. Calculate performance metrics based on daily and monthly timeseries ###
    # first calc some signatures
    dsq["metrics"] = ["KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE"]
    dsq["time_type"] = ["daily", "monthly"]
    dsq["performance"] = (
        ("runs", "metrics", "time_type"),
        np.zeros((len(dsq.runs), len(dsq.metrics), len(dsq.time_type))) * np.nan,
    )

    # perf metrics for single station
    for label in labels:
        # Select data and resample to monthly timeseries as well
        qsim_daily = dsq["Q"].sel(runs=label)
        qsim_monthly = qsim_daily.resample(time="M").mean("time")
        qobs_daily = dsq["Q"].sel(runs="Obs.")
        qobs_monthly = qobs_daily.resample(time="M").mean("time")

        # nse
        nse = skills.nashsutcliffe(qsim_daily, qobs_daily)
        dsq["performance"].loc[dict(runs=label, metrics="NSE", time_type="daily")] = nse
        nse_m = skills.nashsutcliffe(qsim_monthly, qobs_monthly)
        dsq["performance"].loc[
            dict(runs=label, metrics="NSE", time_type="monthly")
        ] = nse_m

        # nse logq
        nselog = skills.lognashsutcliffe(qsim_daily, qobs_daily)
        dsq["performance"].loc[
            dict(runs=label, metrics="NSElog", time_type="daily")
        ] = nselog
        nselog_m = skills.lognashsutcliffe(qsim_monthly, qobs_monthly)
        dsq["performance"].loc[
            dict(runs=label, metrics="NSElog", time_type="monthly")
        ] = nselog_m

        # kge
        kge = skills.kge(qsim_daily, qobs_daily)
        dsq["performance"].loc[
            dict(runs=label, metrics="KGE", time_type="daily")
        ] = kge["kge"]
        kge_m = skills.kge(qsim_monthly, qobs_monthly)
        dsq["performance"].loc[
            dict(runs=label, metrics="KGE", time_type="monthly")
        ] = kge_m["kge"]

        # rmse
        rmse = skills.rmse(qsim_daily, qobs_daily)
        dsq["performance"].loc[
            dict(runs=label, metrics="RMSE", time_type="daily")
        ] = rmse
        rmse_m = skills.rmse(qsim_monthly, qobs_monthly)
        dsq["performance"].loc[
            dict(runs=label, metrics="RMSE", time_type="monthly")
        ] = rmse_m

        # mse
        mse = skills.mse(qsim_daily, qobs_daily)
        dsq["performance"].loc[dict(runs=label, metrics="MSE", time_type="daily")] = mse
        mse_m = skills.mse(qsim_monthly, qobs_monthly)
        dsq["performance"].loc[
            dict(runs=label, metrics="MSE", time_type="monthly")
        ] = mse_m

        # pbias
        pbias = skills.percentual_bias(qsim_daily, qobs_daily)
        dsq["performance"].loc[dict(runs=label, metrics="Pbias", time_type="daily")] = (
            pbias * 100
        )
        pbias_m = skills.percentual_bias(qsim_monthly, qobs_monthly)
        dsq["performance"].loc[
            dict(runs=label, metrics="Pbias", time_type="monthly")
        ] = (pbias_m * 100)

        # ve (volumetric efficiency)
        # TODO: replace when moved to hydromt
        def _ve(sim, obs, axis=-1):
            """Volumetric efficiency."""
            return 1 - np.nansum(np.absolute(sim - obs), axis=axis) / np.nansum(
                obs, axis=axis
            )

        kwargs = dict(
            input_core_dims=[["time"], ["time"]],
            dask="parallelized",
            output_dtypes=[float],
        )
        ve = xr.apply_ufunc(_ve, qsim_daily, qobs_daily, **kwargs)
        ve.name = "ve"
        dsq["performance"].loc[dict(runs=label, metrics="VE", time_type="daily")] = ve
        ve_m = xr.apply_ufunc(_ve, qsim_monthly, qobs_monthly, **kwargs)
        ve_m.name = "ve"
        dsq["performance"].loc[
            dict(runs=label, metrics="VE", time_type="monthly")
        ] = ve_m

    ### 2. Convert to dataframe ###
    df_perf = None
    for label in labels:
        df = (
            dsq["performance"]
            .sel(
                runs=label,
                metrics=["NSE", "NSElog", "KGE", "RMSE", "MSE", "Pbias", "VE"],
            )
            .to_dataframe()
        )
        station_name = df["station_name"].iloc[0]
        if len(labels) > 1:
            station_name = f"{station_name}_{label}"
        df = df[["performance"]]
        df = df.rename(columns={"performance": station_name})
        if df_perf is None:
            df_perf = df
        else:
            df_perf = df_perf.join(df)

    ### 3. Plot signatures ###
    # Depending on number of years of data available, skip plotting position
    nb_years = np.unique(dsq["time.year"].values).size
    if nb_years > 5:
        nrows = 5
    else:
        nrows = 4
    fig = plt.figure(figsize=(16 / 2.54, 22 / 2.54), tight_layout=True)
    axes = fig.subplots(nrows=nrows, ncols=2)
    axes = axes.flatten()

    # daily against each other axes[0]
    for label, color in zip(labels, colors):
        axes[0].plot(
            dsq["Q"].sel(runs="Obs."),
            dsq["Q"].sel(runs=label),
            marker="o",
            linestyle="None",
            linewidth=lw,
            label=label,
            color=color,
            markersize=3,
        )
    max_y = np.round(dsq["Q"].max().values)
    axes[0].plot([0, max_y], [0, max_y], color="0.5", linestyle="--", linewidth=1)
    axes[0].set_xlim([0, max_y])
    axes[0].set_ylim([0, max_y])
    axes[0].set_ylabel("Simulated Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[0].set_xlabel("Observed Q (m$^3$s$^{-1}$)", fontsize=fs)
    #     axes[0].legend(frameon=True, fontsize = fs, )
    axes[0].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
        fontsize=fs,
    )

    # r2
    text_label = ""
    for label in labels:
        r2_score = rsquared(dsq["Q"].sel(runs="Obs."), dsq["Q"].sel(runs=label))
        text_label = text_label + f"R$_2$ {label} = {r2_score:.2f} \n"
    # axes[0].text(max_y/2, max_y/8, text_label, fontsize=fs)
    axes[0].text(0.2, 0.7, text_label, transform=axes[0].transAxes, fontsize=fs)
    # axes[0].text(max_y/2, max_y/8, f"R$_2$ {label} = {r2_score:.2f} \nR$_2$ {label_01} = {r2_score_new:.2f} ", fontsize=fs)

    # streamflow regime axes[1]
    for label, color in zip(labels, colors):
        dsq["Q"].sel(runs=label).groupby("time.month").mean("time").plot(
            ax=axes[1], linewidth=lw, label=label, color=color
        )
    dsq["Q"].sel(runs="Obs.").groupby("time.month").mean("time").plot(
        ax=axes[1], linewidth=lw, label="Obs.", color="k", linestyle="--"
    )
    axes[1].tick_params(axis="both", labelsize=fs)
    axes[1].set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[1].set_xlabel("month", fontsize=fs)
    axes[1].set_title("")
    axes[1].set_xticks(np.arange(1, 13))
    axes[1].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        fontsize=fs,
    )

    # FDC axes[2]
    for label, color, linestyle in zip(labels, colors, linestyles):
        axes[2].plot(
            np.arange(0, len(dsq.time)) / (len(dsq.time) + 1),
            dsq.Q.sel(runs=label).sortby(
                dsq.Q.sel(
                    runs=label,
                ),
                ascending=False,
            ),
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            label=label,
        )
    axes[2].plot(
        np.arange(0, len(dsq.time)) / (len(dsq.time) + 1),
        dsq.Q.sel(runs="Obs.").sortby(
            dsq.Q.sel(
                runs="Obs.",
            ),
            ascending=False,
        ),
        color="k",
        linestyle=":",
        linewidth=lw,
        label="Obs.",
    )
    axes[2].set_xlabel("Exceedence probability (-)", fontsize=fs)
    axes[2].set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)

    # FDClog axes[3]
    for label, color, linestyle in zip(labels, colors, linestyles):
        axes[3].plot(
            np.arange(0, len(dsq.time)) / (len(dsq.time) + 1),
            np.log(
                dsq.Q.sel(runs=label).sortby(
                    dsq.Q.sel(
                        runs=label,
                    ),
                    ascending=False,
                )
            ),
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            label=label,
        )
    axes[3].plot(
        np.arange(0, len(dsq.time)) / (len(dsq.time) + 1),
        np.log(
            dsq.Q.sel(runs="Obs.").sortby(
                dsq.Q.sel(
                    runs="Obs.",
                ),
                ascending=False,
            )
        ),
        color="k",
        linestyle=":",
        linewidth=lw,
        label="Obs.",
    )
    axes[3].set_xlabel("Exceedence probability (-)", fontsize=fs)
    axes[3].set_ylabel("log(Q)", fontsize=fs)

    # max annual axes[4]
    if len(dsq.time) > 365:
        dsq_max = (
            dsq.sel(
                time=slice(
                    f"{str(dsq['time.year'][0].values)}-09-01",
                    f"{str(dsq['time.year'][-1].values)}-08-31",
                )
            )
            .resample(time="AS-Sep")
            .max("time")
        )
    else:
        # Less than a year of data, max over the whole timeseries
        dsq_max = dsq.max("time")
    for label, color, marker in zip(labels, colors, markers):
        axes[4].plot(
            dsq_max.Q.sel(runs="Obs."),
            dsq_max.Q.sel(runs=label),
            color=color,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=label,
        )
    axes[4].plot(
        [0, max_y * 1.1], [0, max_y * 1.1], color="0.5", linestyle="--", linewidth=1
    )
    axes[4].set_xlim([0, max_y * 1.1])
    axes[4].set_ylim([0, max_y * 1.1])
    # R2 score
    text_label = ""
    for label in labels:
        if len(dsq.time) > 365:
            r2_score = rsquared(
                dsq_max["Q"].sel(runs="Obs."), dsq_max["Q"].sel(runs=label)
            )
            text_label = text_label + f"R$_2$ {label} = {r2_score:.2f} \n"
        else:
            text_label = text_label + f"{label}\n"
    axes[4].text(0.5, 0.05, text_label, transform=axes[4].transAxes, fontsize=fs)

    # add MHQ
    if len(dsq.time) > 365:
        mhq = dsq_max.mean("time")
    else:
        mhq = dsq_max.copy()
    for label in labels:
        axes[4].plot(
            mhq.Q.sel(runs="Obs."),
            mhq.Q.sel(runs=label),
            color="black",
            marker=">",
            linestyle="None",
            linewidth=lw,
            label=label,
            markersize=6,
        )
    # labels
    axes[4].set_ylabel("Sim. max annual Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[4].set_xlabel("Obs. max annual Q (m$^3$s$^{-1}$)", fontsize=fs)

    # nm7q axes[5]
    dsq_nm7q = dsq.rolling(time=7).mean().resample(time="A").min("time")
    max_ylow = dsq_nm7q["Q"].max().values
    for label, color, marker in zip(labels, colors, markers):
        axes[5].plot(
            dsq_nm7q.Q.sel(runs="Obs."),
            dsq_nm7q.Q.sel(runs=label),
            color=color,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=label,
        )
    axes[5].plot(
        [0, max_ylow * 1.1],
        [0, max_ylow * 1.1],
        color="0.5",
        linestyle="--",
        linewidth=1,
    )
    axes[5].set_xlim([0, max_ylow * 1.1])
    axes[5].set_ylim([0, max_ylow * 1.1])
    # #R2 score
    text_label = ""
    for label in labels:
        r2_score = rsquared(
            dsq_nm7q["Q"].sel(runs="Obs."), dsq_nm7q["Q"].sel(runs=label)
        )
        text_label = text_label + f"R$_2$ {label} = {r2_score:.2f} \n"
    axes[5].text(0.5, 0.05, text_label, transform=axes[5].transAxes, fontsize=fs)
    # labels
    axes[5].set_ylabel("Simulated NM7Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[5].set_xlabel("Observed NM7Q (m$^3$s$^{-1}$)", fontsize=fs)

    # cum axes[6]
    dsq["Q"].sel(runs="Obs.").cumsum("time").plot(
        ax=axes[6], color="k", linestyle=":", linewidth=lw, label="Obs."
    )
    for label, color, linestyle in zip(labels, colors, linestyles):
        dsq["Q"].sel(runs=label).cumsum("time").plot(
            ax=axes[6], color=color, linestyle=linestyle, linewidth=lw, label=label
        )
    axes[6].set_xlabel("")
    axes[6].set_ylabel("Cum. Q (m$^3$s$^{-1}$)", fontsize=fs)

    # performance measures NS, NSlogQ, KGE, axes[7]
    # nse
    for label, color, marker in zip(labels, colors, markers):
        axes[7].plot(
            0.8,
            dsq["performance"].loc[dict(runs=label, metrics="NSE", time_type="daily")],
            color=color,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=label,
        )
    # nselog
    for label, color, marker in zip(labels, colors, markers):
        axes[7].plot(
            2.8,
            dsq["performance"].loc[
                dict(runs=label, metrics="NSElog", time_type="daily")
            ],
            color=color,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=label,
        )
    # kge
    for label, color, marker in zip(labels, colors, markers):
        axes[7].plot(
            4.8,
            dsq["performance"].loc[dict(runs=label, metrics="KGE", time_type="daily")],
            color=color,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=label,
        )
    axes[7].set_xticks([1, 3, 5])
    axes[7].set_xticklabels(["NSE", "NSElog", "KGE"])
    axes[7].set_ylim([0, 1])
    axes[7].set_ylabel("Performance", fontsize=fs)

    # gumbel high axes[8]
    # Only if more than 5 years of data
    if nb_years > 5:
        a = 0.3
        b = 1.0 - 2.0 * a
        ymin, ymax = 0, max_y
        p1 = ((np.arange(1, len(dsq_max.time) + 1.0) - a)) / (len(dsq_max.time) + b)
        RP1 = 1 / (1 - p1)
        gumbel_p1 = -np.log(-np.log(1.0 - 1.0 / RP1))
        ts = [2.0, 5.0, 10.0, 30.0]  # ,30.,100.,300.,1000.,3000.,10000.,30000.]
        # plot
        axes[8].plot(
            gumbel_p1,
            dsq_max["Q"].sel(runs="Obs.").sortby(dsq_max["Q"].sel(runs="Obs.")),
            marker="+",
            color="k",
            linestyle="None",
            label="Obs.",
            markersize=6,
        )
        for label, color, marker in zip(labels, colors, markers):
            axes[8].plot(
                gumbel_p1,
                dsq_max["Q"].sel(runs=label).sortby(dsq_max["Q"].sel(runs=label)),
                marker=marker,
                color=color,
                linestyle="None",
                label=label,
                markersize=4,
            )

        for t in ts:
            axes[8].vlines(-np.log(-np.log(1 - 1.0 / t)), ymin, ymax, "0.5", alpha=0.4)
            axes[8].text(
                -np.log(-np.log(1 - 1.0 / t)),
                ymax * 0.2,
                "T=%.0f y" % t,
                rotation=45,
                fontsize=fs,
            )

        axes[8].set_ylabel("max. annual Q (m$^3$s$^{-1}$)", fontsize=fs)
        axes[8].set_xlabel(
            "Plotting position and associated return period", fontsize=fs
        )

    # gumbel low axes[9]
    # Only if more than 5 years of data
    if nb_years > 5:
        a = 0.3
        b = 1.0 - 2.0 * a
        ymin, ymax = 0, max_ylow
        p1 = ((np.arange(1, len(dsq_nm7q.time) + 1.0) - a)) / (len(dsq_nm7q.time) + b)
        RP1 = 1 / (1 - p1)
        gumbel_p1 = -np.log(-np.log(1.0 - 1.0 / RP1))
        ts = [2.0, 5.0, 10.0, 30.0]  # ,30.,100.,300.,1000.,3000.,10000.,30000.]
        # plot
        axes[9].plot(
            gumbel_p1,
            dsq_nm7q["Q"]
            .sel(runs="Obs.")
            .sortby(dsq_nm7q["Q"].sel(runs="Obs."), ascending=False),
            marker="+",
            color="k",
            linestyle="None",
            label="Obs.",
            markersize=6,
        )
        for label, color, marker in zip(labels, colors, markers):
            axes[9].plot(
                gumbel_p1,
                dsq_nm7q["Q"]
                .sel(runs=label)
                .sortby(dsq_nm7q["Q"].sel(runs=label), ascending=False),
                marker=marker,
                color=color,
                linestyle="None",
                label=label,
                markersize=4,
            )

        for t in ts:
            axes[9].vlines(-np.log(-np.log(1 - 1.0 / t)), ymin, ymax, "0.5", alpha=0.4)
            axes[9].text(
                -np.log(-np.log(1 - 1.0 / t)),
                ymax * 0.8,
                "T=%.0f y" % t,
                rotation=45,
                fontsize=fs,
            )

        axes[9].set_ylabel("NM7Q (m$^3$s$^{-1}$)", fontsize=fs)
        axes[9].set_xlabel(
            "Plotting position and associated return period", fontsize=fs
        )

    for ax in axes:
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_title("")
    # fig.set_tight_layout(True)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace = 0.5, hspace = 0.6)
    plt.savefig(os.path.join(Folder_out, f"signatures_{station_name}.png"), dpi=300)

    return df_perf


def plot_hydro(
    dsq,
    start_long,
    end_long,
    year_wet,
    year_dry,
    labels,
    colors,
    Folder_out,
    station_name,
    lw=0.8,
    fs=7,
):
    fig, axes = plt.subplots(5, 1, figsize=(16 / 2.54, 23 / 2.54))
    # long period
    for label, color in zip(labels, colors):
        dsq["Q"].sel(runs=label, time=slice(start_long, end_long)).plot(
            ax=axes[0], label=label, linewidth=lw, color=color
        )
    if "Obs." in dsq["runs"]:
        dsq["Q"].sel(runs="Obs.", time=slice(start_long, end_long)).plot(
            ax=axes[0], label="Obs.", linewidth=lw, color="k", linestyle="--"
        )

    # annual Q
    for label, color in zip(labels, colors):
        dsq.sel(runs=label).resample(time="A").sum().Q.plot(
            ax=axes[1], label=label, linewidth=lw, color=color
        )
    if "Obs." in dsq["runs"]:
        dsq.sel(runs="Obs.").resample(time="A").sum().Q.plot(
            ax=axes[1], label="Obs.", linewidth=lw, color="k", linestyle="--"
        )

    # monthly Q
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for label, color in zip(labels, colors):
        dsqM = dsq.sel(runs=label).resample(time="M").sum()
        dsqM = dsqM.groupby(dsqM.time.dt.month).mean()
        dsqM.Q.plot(ax=axes[2], label=label, linewidth=lw, color=color)
    if "Obs." in dsq["runs"]:
        dsqMo = dsq.sel(runs="Obs.").resample(time="M").sum()
        dsqMo = dsqMo.groupby(dsqMo.time.dt.month).mean()
        dsqMo.Q.plot(ax=axes[2], label="Obs.", linewidth=lw, color=color)
    axes[2].set_title("Average monthly sum")
    axes[2].set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=5)

    # wettest year
    for label, color in zip(labels, colors):
        dsq["Q"].sel(runs=label, time=year_wet).plot(
            ax=axes[3], label=label, linewidth=lw, color=color
        )
    if "Obs." in dsq["runs"]:
        dsq["Q"].sel(runs="Obs.", time=year_wet).plot(
            ax=axes[3], label="Obs.", linewidth=lw, color="k", linestyle="--"
        )

    # driest year
    for label, color in zip(labels, colors):
        dsq["Q"].sel(runs=label, time=year_dry).plot(
            ax=axes[4], label=label, linewidth=lw, color=color
        )
    if "Obs." in dsq["runs"]:
        dsq["Q"].sel(runs="Obs.", time=year_dry).plot(
            ax=axes[4], label="Obs.", linewidth=lw, color="k", linestyle="--"
        )

    titles = [
        "Daily time-series",
        "Annual time-series",
        "Annual cycle",
        "Wettest year",
        "Driest year",
    ]
    for ax, title in zip(axes, titles):
        ax.tick_params(axis="both", labelsize=fs)
        if ax == axes[1]:
            ax.set_ylabel("Q (m$^3$yr$^{-1}$)", fontsize=fs)
        elif ax == axes[2]:
            ax.set_ylabel("Q (m$^3$month$^{-1}$)", fontsize=fs)
        else:
            ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel("", fontsize=fs)
    axes[0].legend(fontsize=fs)
    plt.tight_layout()

    plt.savefig(os.path.join(Folder_out, f"hydro_{station_name}.png"), dpi=300)


def plot_hydro_1y(
    dsq, start_long, end_long, labels, colors, Folder_out, station_name, lw=0.8, fs=8
):
    fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 5 / 2.54))
    # long period
    for label, color in zip(labels, colors):
        dsq["Q"].sel(runs=label, time=slice(start_long, end_long)).plot(
            ax=ax, label=label, linewidth=lw, color=color
        )
    if "Obs." in dsq["runs"]:
        dsq["Q"].sel(runs="Obs.", time=slice(start_long, end_long)).plot(
            ax=ax, label="Obs.", linewidth=lw, color="k", linestyle="--"
        )

    ax.tick_params(axis="both", labelsize=fs)
    ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
    ax.set_xlabel("", fontsize=fs)
    ax.set_title("")
    ax.legend(fontsize=fs)
    plt.tight_layout()
    fig.set_tight_layout(True)
    plt.savefig(os.path.join(Folder_out, f"hydro_{station_name}.png"), dpi=300)


def plot_clim(ds_clim, Folder_out, station_name, period, lw=0.8, fs=8):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16 / 2.54, 15 / 2.54), sharex=True
    )

    if period == "year":
        resampleper = "A"
    else:
        resampleper = "M"

    # temp
    if period == "month":
        T_mean_monthly_mean = (
            ds_clim["T_subcatchment"].groupby(f"time.{period}").mean("time")
        )
        T_mean_monthly_q25 = (
            ds_clim["T_subcatchment"].groupby(f"time.{period}").quantile(0.25, "time")
        )
        T_mean_monthly_q75 = (
            ds_clim["T_subcatchment"].groupby(f"time.{period}").quantile(0.75, "time")
        )
        # plot
        T_mean_monthly_mean.plot(ax=ax1, color="red")
        ax1.fill_between(
            np.arange(1, 13), T_mean_monthly_q25, T_mean_monthly_q75, color="orange"
        )
        #    T_mean_monthly_mean.to_series().plot.line(ax=ax2, color = 'orange')
    else:
        T_mean_year = ds_clim["T_subcatchment"].resample(time=resampleper).mean("time")
        T_mean_year.plot(ax=ax1, color="red")

        x = T_mean_year.time.dt.year
        z = np.polyfit(x, T_mean_year, 1)
        p = np.poly1d(z)
        r2_score = rsquared(p(x), T_mean_year)
        ax1.plot(T_mean_year.time, p(x), ls="--", color="lightgrey")
        ax1.text(
            T_mean_year.time[0], T_mean_year.min(), f"$R^2$ = {round(r2_score, 3)}"
        )

    # precip and evap
    for climvar, clr, clr_range, ax in zip(
        ["P_subcatchment", "EP_subcatchment"],
        ["steelblue", "forestgreen"],
        ["lightblue", "lightgreen"],
        [ax2, ax3],
    ):
        var_sum_monthly = ds_clim[climvar].resample(time=resampleper).sum("time")

        if period == "month":
            var_sum_monthly_mean = var_sum_monthly.groupby(f"time.{period}").mean(
                "time"
            )
            var_sum_monthly_q25 = var_sum_monthly.groupby(f"time.{period}").quantile(
                0.25, "time"
            )
            var_sum_monthly_q75 = var_sum_monthly.groupby(f"time.{period}").quantile(
                0.75, "time"
            )

            var_sum_monthly_mean.plot(ax=ax, color=clr)
            ax.fill_between(
                np.arange(1, 13),
                var_sum_monthly_q25,
                var_sum_monthly_q75,
                color=clr_range,
            )
        else:
            x = var_sum_monthly.time.dt.year
            z = np.polyfit(x, var_sum_monthly, 1)
            p = np.poly1d(z)
            r2_score = rsquared(p(x), var_sum_monthly)

            ax.plot(var_sum_monthly.time, p(x), ls="--", color="lightgrey")
            ax.text(
                var_sum_monthly.time[0],
                var_sum_monthly.min(),
                f"$R^2$ = {round(r2_score, 3)}",
            )
            var_sum_monthly.plot(ax=ax, color=clr)

    for ax, title_name, ylab in zip(
        [ax1, ax2, ax3],
        ["Temperature", "Precipitation", "Potential evaporation"],
        [
            "T (deg C)",
            f"P (mm {period}$^{-1}$)",
            f"E$_P$ (mm {period}$^{-1}$)",
        ],
    ):
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_xlabel("", fontsize=fs)
        ax.set_title(title_name)
        ax.grid(alpha=0.5)
        ax.set_ylabel(ylab, fontsize=fs)

    if period == "month":
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        ax3.set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=fs)

    plt.tight_layout()
    fig.set_tight_layout(True)
    plt.savefig(os.path.join(Folder_out, f"clim_{station_name}_{period}.png"), dpi=300)


def plot_basavg(ds, Folder_out, fs=10):
    dvars = [dvar for dvar in ds.data_vars]
    n = len(dvars)

    for i in range(n):
        dvar = dvars[i]

        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(11, 4))
        # axes = [axes] if n == 1 else axes

        if WFLOW_VARS[dvar.split("_")[0]]["resample"] == "sum":
            sum_monthly = ds[dvar].resample(time="M").sum("time")
        else:  # assume mean
            sum_monthly = ds[dvar].resample(time="M").mean("time")
        sum_monthly_mean = sum_monthly.groupby("time.month").mean("time")
        sum_monthly_q25 = sum_monthly.groupby("time.month").quantile(0.25, "time")
        sum_monthly_q75 = sum_monthly.groupby("time.month").quantile(0.75, "time")

        # plot
        sum_monthly_mean.plot(ax=ax, color="darkblue")
        ax.fill_between(
            np.arange(1, 13), sum_monthly_q25, sum_monthly_q75, color="lightblue"
        )
        legend = WFLOW_VARS[dvar.split("_")[0]]["legend"]
        ax.set_ylabel(legend, fontsize=fs)

        ax.tick_params(axis="both", labelsize=fs)
        ax.set_xlabel("", fontsize=fs)
        ax.set_title("")
        ax.grid(alpha=0.5)

        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        ax.set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=fs)

        plt.tight_layout()
        fig.set_tight_layout(True)
        plt.savefig(os.path.join(Folder_out, f"{dvar}.png"), dpi=300)


# %%
