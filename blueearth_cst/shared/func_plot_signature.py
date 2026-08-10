# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:18:38 2021

@author: bouaziz
"""

# %%
from hydromt.stats import skills
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import xarray as xr

from typing import Union

from blueearth_cst.shared.snake_utils import save_figure


# %%
# Supported wflow outputs. The resample rule and axis legend live beside the
# code map they are keyed by, in shared/wflow_outputs.py — one table rather than
# two that had to be kept in agreement by hand.
from blueearth_cst.shared.wflow_outputs import (  # noqa: E402
    PLOT_META,
    code_from_variable,
)


# --- publication figure style -------------------------------------------
# Mirrors the constants in shared/plot_map.py. Both should move to a shared
# style module; kept local for now so the two redesigns stay independent.
_FIG_WIDTH_MM = 180.0
_MM_PER_IN = 25.4
_PT_BODY, _PT_SECONDARY, _PT_CREDIT = 8, 7, 6
_CLIM_RC = {
    "font.size": _PT_BODY,
    "axes.labelsize": _PT_BODY,
    "legend.fontsize": _PT_SECONDARY,
    "xtick.labelsize": _PT_SECONDARY,
    "ytick.labelsize": _PT_SECONDARY,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
}
# Okabe-Ito: CVD-safe, and each hue keeps a distinct lightness so the panels
# stay separable in greyscale.
_CLIM_COLORS = {"T": "#D55E00", "P": "#0072B2", "EP": "#009E73"}


def _sen_trend(years, values):
    """Theil-Sen slope with a Mann-Kendall significance test.

    Ordinary least squares plus R^2 — what this figure reported before —
    answers "how tightly do the points sit on a line", not "is there a
    trend". R^2 is also sensitive to the outliers that annual climate series
    routinely carry. Theil-Sen (robust slope) with Mann-Kendall (rank-based
    significance) is the standard pairing for hydro-climate trend detection,
    and both come from scipy, already a dependency.
    """
    result = stats.theilslopes(values, years)
    tau, p_value = stats.kendalltau(years, values)
    return result.slope, result.intercept, p_value


def _format_trend(slope_per_year, p_value, unit, decimals=2):
    """Trend annotation: magnitude per decade plus significance."""
    per_decade = slope_per_year * 10.0
    if p_value < 0.001:
        significance = "$p$ < 0.001"
    elif p_value < 0.05:
        significance = f"$p$ = {p_value:.3f}"
    else:
        significance = "not significant"
    # U+2212 MINUS SIGN, not the ASCII hyphen "+/-" formatting emits
    magnitude = f"{per_decade:+.{decimals}f}".replace("-", "−")
    return f"{magnitude} {unit} decade$^{{-1}}$ ({significance})"


def rsquared(x, y):
    """Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2


def compute_metrics(
    qsim: xr.DataArray,
    qobs: xr.DataArray,
    station_name: str = "station",
) -> pd.DataFrame:
    """
    Compute performance metrics.

    Calculated metrics for daily and montly timeseries are:
    - Nash-Sutcliffe efficiency (NSE)
    - Nash-Sutcliffe efficiency on log-transformed data (NSElog)
    - Kling-Gupta efficiency (KGE)
    - Root mean squared error (RMSE)
    - Mean squared error (MSE)
    - Percentual bias (Pbias)
    - Volumetric error (VE)

    Parameters
    ----------
    qsim : xr.DataArray
        Dataset with simulated streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    qobs : xr.DataArray
        Dataset with observed streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    station_name : str, optional
        Station name, by default "station"

    Returns
    -------
    pd.DataFrame
        Dataframe with performance metrics for this station.
    """
    ### 1. Calculate performance metrics based on daily and monthly timeseries ###
    # Initialize performance array
    metrics = ["KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE"]
    time_type = ["daily", "monthly"]
    da_perf = xr.DataArray(
        np.zeros((len(metrics), len(time_type))),
        coords=[metrics, time_type],
        dims=["metrics", "time_type"],
    )

    # Select data and resample to monthly timeseries as well
    qsim_monthly = qsim.resample(time="ME").mean("time")
    qobs_monthly = qobs.resample(time="ME").mean("time")

    # compute perf metrics
    # nse
    nse = skills.nashsutcliffe(qsim, qobs)
    da_perf.loc[dict(metrics="NSE", time_type="daily")] = nse
    nse_m = skills.nashsutcliffe(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="NSE", time_type="monthly")] = nse_m

    # nse logq
    nselog = skills.lognashsutcliffe(qsim, qobs)
    da_perf.loc[dict(metrics="NSElog", time_type="daily")] = nselog
    nselog_m = skills.lognashsutcliffe(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="NSElog", time_type="monthly")] = nselog_m

    # kge
    kge = skills.kge(qsim, qobs)
    da_perf.loc[dict(metrics="KGE", time_type="daily")] = kge["kge"]
    kge_m = skills.kge(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="KGE", time_type="monthly")] = kge_m["kge"]

    # rmse
    rmse = skills.rmse(qsim, qobs)
    da_perf.loc[dict(metrics="RMSE", time_type="daily")] = rmse
    rmse_m = skills.rmse(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="RMSE", time_type="monthly")] = rmse_m

    # mse
    mse = skills.mse(qsim, qobs)
    da_perf.loc[dict(metrics="MSE", time_type="daily")] = mse
    mse_m = skills.mse(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="MSE", time_type="monthly")] = mse_m

    # pbias
    pbias = skills.percentual_bias(qsim, qobs)
    da_perf.loc[dict(metrics="Pbias", time_type="daily")] = pbias
    pbias_m = skills.percentual_bias(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="Pbias", time_type="monthly")] = pbias_m

    # ve (volumetric efficiency)
    ve = skills.volumetric_error(qsim, qobs)
    da_perf.loc[dict(metrics="VE", time_type="daily")] = ve
    ve_m = skills.volumetric_error(qsim_monthly, qobs_monthly)
    da_perf.loc[dict(metrics="VE", time_type="monthly")] = ve_m

    ### 2. Convert to dataframe ###
    df_perf = da_perf.to_dataframe(name=station_name)

    return df_perf


def plot_signatures(
    qsim: xr.DataArray,
    qobs: xr.DataArray,
    Folder_out: Union[Path, str],
    station_name: str = "station",
    label: str = "simulated",
    color: str = "orange",
    linestyle: str = "-",
    marker: str = "o",
    lw: float = 0.8,
    fs: int = 8,
) -> pd.DataFrame:
    """
    Plot hydrological signatures.

    Plot the following signatures:
    - Daily against each other
    - Streamflow regime (monthly average)
    - Flow duration curve
    - Flow duration curve on log-transformed data
    - Annual Maxima against each other
    - NM7Q against each other
    - Cumulative flow
    - Performance metrics (NSE, NSElog, KGE)
    - Gumbel high (if 5+ years of data)
    - Gumbel low (if 5+ years of data)


    Parameters
    ----------
    qsim : xr.DataArray
        Dataset with simulated streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    qobs : xr.DataArray
        Dataset with observed streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    Folder_out : Union[Path, str]
        Output folder to save plots.
    station_name : str, optional
        Station name, by default "station"
    label : str, optional
        Labels for the simulated run, by default "simulated"
    color : str, optional
        Color for the simulated run, by default "orange"
    linestyle : str, optional
        Linestyle for the simulated run, by default "-"
    marker : str, optional
        Marker for the simulated run, by default "o"
    lw : float, optional
        Line width, by default 0.8
    fs : int, optional
        Font size, by default 8
    """
    # Depending on number of years of data available, skip plotting position
    nb_years = np.unique(qsim["time.year"].values).size
    nb_year_threshold = 5
    if nb_years >= nb_year_threshold:
        nrows = 5
    else:
        nrows = 4
    fig = plt.figure(figsize=(16 / 2.54, 22 / 2.54), tight_layout=True)
    axes = fig.subplots(nrows=nrows, ncols=2)
    axes = axes.flatten()

    ### 1. daily against each other axes[0] ###
    axes[0].plot(
        qobs,
        qsim,
        marker="o",
        linestyle="None",
        linewidth=lw,
        label=label,
        color=color,
        markersize=3,
    )
    max_y = np.round(qobs.max().values)
    axes[0].plot([0, max_y], [0, max_y], color="0.5", linestyle="--", linewidth=1)
    axes[0].set_xlim([0, max_y])
    axes[0].set_ylim([0, max_y])

    axes[0].set_ylabel("Simulated Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[0].set_xlabel("Observed Q (m$^3$s$^{-1}$)", fontsize=fs)
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
    r2_score = rsquared(qobs, qsim)
    text_label = text_label + f"R$_2$ {label} = {r2_score:.2f} \n"
    axes[0].text(0.2, 0.7, text_label, transform=axes[0].transAxes, fontsize=fs)

    ### 2. streamflow regime axes[1] ###
    qsim.groupby("time.month").mean("time").plot(
        ax=axes[1], linewidth=lw, label=label, color=color
    )
    qobs.groupby("time.month").mean("time").plot(
        ax=axes[1], linewidth=lw, label="observed", color="k", linestyle="--"
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

    ### 3. FDC axes[2] ###
    axes[2].plot(
        np.arange(0, len(qsim.time)) / (len(qsim.time) + 1),
        qsim.sortby(qsim, ascending=False),
        color=color,
        linestyle=linestyle,
        linewidth=lw,
        label=label,
    )
    axes[2].plot(
        np.arange(0, len(qobs.time)) / (len(qobs.time) + 1),
        qobs.sortby(qobs, ascending=False),
        color="k",
        linestyle=":",
        linewidth=lw,
        label="observed",
    )
    axes[2].set_xlabel("Exceedence probability (-)", fontsize=fs)
    axes[2].set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)

    ### 4. FDClog axes[3] ###
    axes[3].plot(
        np.arange(0, len(qsim.time)) / (len(qsim.time) + 1),
        np.log(qsim.sortby(qsim, ascending=False)),
        color=color,
        linestyle=linestyle,
        linewidth=lw,
        label=label,
    )
    axes[3].plot(
        np.arange(0, len(qobs.time)) / (len(qobs.time) + 1),
        np.log(qobs.sortby(qobs, ascending=False)),
        color="k",
        linestyle=":",
        linewidth=lw,
        label="observed",
    )
    axes[3].set_xlabel("Exceedence probability (-)", fontsize=fs)
    axes[3].set_ylabel("log(Q)", fontsize=fs)

    ### 5. max annual axes[4] ###
    if len(qsim.time) > 365:
        start = f"{str(qsim['time.year'][0].values)}-09-01"
        end = f"{str(qsim['time.year'][-1].values)}-08-31"
        qsim_max = qsim.sel(time=slice(start, end)).resample(time="YS-SEP").max("time")
        qobs_max = qobs.sel(time=slice(start, end)).resample(time="YS-SEP").max("time")
    else:
        # Less than a year of data, max over the whole timeseries
        qsim_max = qsim.max("time")
        qobs_max = qobs.max("time")
    axes[4].plot(
        qobs_max,
        qsim_max,
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
    if len(qsim.time) > 365:
        r2_score = rsquared(qobs_max, qsim_max)
        text_label = text_label + f"R$_2$ {label} = {r2_score:.2f} \n"
    else:
        text_label = text_label + f"{label}\n"
    axes[4].text(0.5, 0.05, text_label, transform=axes[4].transAxes, fontsize=fs)

    # add MHQ
    if len(qsim.time) > 365:
        mhq = qsim_max.mean("time")
        mhq_obs = qobs_max.mean("time")
    else:
        mhq = qsim_max.copy()
        mhq_obs = qobs_max.copy()
    axes[4].plot(
        mhq_obs,
        mhq,
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

    ### 6. nm7q axes[5] ###
    qsim_nm7q = qsim.rolling(time=7).mean().resample(time="YE").min("time")
    qobs_nm7q = qobs.rolling(time=7).mean().resample(time="YE").min("time")
    max_ylow = max(qsim_nm7q.max().values, qobs_nm7q.max().values)

    axes[5].plot(
        qobs_nm7q,
        qsim_nm7q,
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
    r2_score = rsquared(qobs_nm7q, qsim_nm7q)
    text_label = text_label + f"R$_2$ {label} = {r2_score:.2f} \n"
    axes[5].text(0.5, 0.05, text_label, transform=axes[5].transAxes, fontsize=fs)
    # labels
    axes[5].set_ylabel("Simulated NM7Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[5].set_xlabel("Observed NM7Q (m$^3$s$^{-1}$)", fontsize=fs)

    ### 7. cum axes[6] ###
    qobs.cumsum("time").plot(
        ax=axes[6], color="k", linestyle=":", linewidth=lw, label="observed"
    )
    qsim.cumsum("time").plot(
        ax=axes[6], color=color, linestyle=linestyle, linewidth=lw, label=label
    )
    axes[6].set_xlabel("")
    axes[6].set_ylabel("Cum. Q (m$^3$s$^{-1}$)", fontsize=fs)

    ### 8. performance measures NS, NSlogQ, KGE, axes[7] ###
    # nse
    axes[7].plot(
        0.8,
        skills.nashsutcliffe(qsim, qobs).values,
        color=color,
        marker=marker,
        linestyle="None",
        linewidth=lw,
        label=label,
    )
    # nselog
    axes[7].plot(
        2.8,
        skills.lognashsutcliffe(qsim, qobs).values,
        color=color,
        marker=marker,
        linestyle="None",
        linewidth=lw,
        label=label,
    )
    # kge
    axes[7].plot(
        4.8,
        skills.kge(qsim, qobs)["kge"].values,
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

    ### 9. gumbel high axes[8] ###
    # Only if more than 5 years of data
    if nb_years >= nb_year_threshold:
        a = 0.3
        b = 1.0 - 2.0 * a
        ymin, ymax = 0, max_y
        p1 = ((np.arange(1, len(qobs_max.time) + 1.0) - a)) / (len(qobs_max.time) + b)
        RP1 = 1 / (1 - p1)
        gumbel_p1 = -np.log(-np.log(1.0 - 1.0 / RP1))
        ts = [2.0, 5.0, 10.0, 30.0]  # ,30.,100.,300.,1000.,3000.,10000.,30000.]
        # plot
        axes[8].plot(
            gumbel_p1,
            qobs_max.sortby(qobs_max),
            marker="+",
            color="k",
            linestyle="None",
            label="observed",
            markersize=6,
        )
        axes[8].plot(
            gumbel_p1,
            qsim_max.sortby(qsim_max),
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

    ### 10. gumbel low axes[9] ###
    # Only if more than 5 years of data
    if nb_years >= nb_year_threshold:
        a = 0.3
        b = 1.0 - 2.0 * a
        ymin, ymax = 0, max_ylow
        p1 = ((np.arange(1, len(qobs_nm7q.time) + 1.0) - a)) / (len(qobs_nm7q.time) + b)
        RP1 = 1 / (1 - p1)
        gumbel_p1 = -np.log(-np.log(1.0 - 1.0 / RP1))
        ts = [2.0, 5.0, 10.0, 30.0]  # ,30.,100.,300.,1000.,3000.,10000.,30000.]
        # plot
        axes[9].plot(
            gumbel_p1,
            qobs_nm7q.sortby(qobs_nm7q, ascending=False),
            marker="+",
            color="k",
            linestyle="None",
            label="observed",
            markersize=6,
        )
        axes[9].plot(
            gumbel_p1,
            qsim_nm7q.sortby(qsim_nm7q, ascending=False),
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

    ### Common axis settings ###
    for ax in axes:
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_title("")

    ### Save plot ###
    save_figure(os.path.join(Folder_out, f"signatures_{station_name}.png"), dpi=300)


def plot_hydro(
    qsim: xr.DataArray,
    Folder_out: Union[Path, str],
    qobs: xr.DataArray = None,
    label: str = "simulated",
    color: str = "steelblue",
    station_name: str = "station_1",
    lw: float = 0.8,
    fs: int = 7,
):
    """
    Plot hydrograph for a specific location.

    If observations ``qobs`` are provided, the plot will include the observations.
    If the simulation is less than 3 years, the plot will be a single panel with the
    hydrograph for that year.
    If it is more than 3 years, the plot will be a 5 panel plot with the following:
    - Daily time-series
    - Annual time-series
    - Annual cycle (montly average)
    - Wettest year
    - Driest year

    Parameters
    ----------
    qsim : xr.DataArray
        Simulated streamflow.
    Folder_out : Union[Path, str]
        Output folder to save plots.
    qobs : xr.DataArray, optional
        Observed streamflow, by default None
    label : str, optional
        Label for the simulated run, by default ["simulated"]
    color : str, optional
        Color for the simulated run, by default ["steelblue"]
    station_name : str, optional
        Station name, by default "station_1"
    lw : float, optional
        Line width, by default 0.8
    fs : int, optional
        Font size, by default 7
    """
    # Settings for obs
    labobs = "observed"
    colobs = "k"

    # Number of years available
    nb_years = np.unique(qsim["time.year"].values).size

    # If less than 3 years, plot a single panel
    if nb_years <= 3:
        nb_panel = 1
        titles = ["Daily time-series"]
    else:
        nb_panel = 5
        titles = [
            "Daily time-series",
            "Annual time-series",
            "Annual cycle",
            "Wettest year",
            "Driest year",
        ]
        # Get the wettest and driest year
        qyr = qsim.resample(time="YE").sum()
        qyr["time"] = qyr["time.year"]
        # Get the year for the minimum as an integer
        year_dry = str(qyr.isel(time=qyr.argmin()).time.values)
        year_wet = str(qyr.isel(time=qyr.argmax()).time.values)

    fig, axes = plt.subplots(nb_panel, 1, figsize=(16 / 2.54, 23 / 2.54))
    axes = [axes] if nb_panel == 1 else axes

    # 1. long period
    qsim.plot(ax=axes[0], label=label, linewidth=lw, color=color)
    if qobs is not None:
        qobs.plot(ax=axes[0], label=labobs, linewidth=lw, color=colobs, linestyle="--")

    if nb_panel == 5:
        # 2. annual Q
        qsim.resample(time="YE").sum().plot(
            ax=axes[1], label=label, linewidth=lw, color=color
        )
        if qobs is not None:
            qobs.resample(time="YE").sum().plot(
                ax=axes[1], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

        # 3. monthly Q
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        dsqM = qsim.resample(time="ME").sum()
        dsqM = dsqM.groupby(dsqM.time.dt.month).mean()
        dsqM.plot(ax=axes[2], label=label, linewidth=lw, color=color)
        if qobs is not None:
            dsqMo = qobs.resample(time="ME").sum()
            dsqMo = dsqMo.groupby(dsqMo.time.dt.month).mean()
            dsqMo.plot(ax=axes[2], label=labobs, linewidth=lw, color=colobs)
        axes[2].set_title("Average monthly sum")
        axes[2].set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=5)

        # 4. wettest year
        qsim.sel(time=year_wet).plot(ax=axes[3], label=label, linewidth=lw, color=color)
        if qobs is not None:
            qobs.sel(time=year_wet).plot(
                ax=axes[3], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

        # 5. driest year
        qsim.sel(time=year_dry).plot(ax=axes[4], label=label, linewidth=lw, color=color)
        if qobs is not None:
            qobs.sel(time=year_dry).plot(
                ax=axes[4], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

    # Axes settings
    for ax, title in zip(axes, titles):
        ax.tick_params(axis="both", labelsize=fs)
        if ax == axes[0]:
            ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
        elif nb_panel == 5:
            if ax == axes[1]:
                ax.set_ylabel("Q (m$^3$yr$^{-1}$)", fontsize=fs)
            elif ax == axes[2]:
                ax.set_ylabel("Q (m$^3$month$^{-1}$)", fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel("", fontsize=fs)
    axes[0].legend(fontsize=fs)
    plt.tight_layout()

    save_figure(os.path.join(Folder_out, f"hydro_{station_name}.png"), dpi=300)


def plot_basavg(ds, Folder_out, fs=10):
    """One monthly-climatology panel PER SUBCATCHMENT, per variable.

    These series are per-subcatchment means (wflow's `subcatchment` map with
    `reducer = "mean"`), so on a basin with four control points there are four
    of them, not one. This drew them as a single object until 2026-08-10, which
    worked only while a basin had ONE subcatchment: `plot_results` squeezes a
    size-1 index away, so the array arrived 1-D and the line plot was correct by
    accident. With four, `.plot()` saw a 2-D (index, time) array, dispatched to
    `pcolormesh`, and raised on wflow's unsorted index -- and `fill_between`
    would have broken next, since it assumes exactly 12 values.

    Faceting rather than one file per subcatchment is a Snakemake constraint,
    not a preference: the subcatchment count is not known at DAG-parse time, so
    N separate PNGs could not be declared as outputs. One file per variable
    keeps the declaration in `_basavg_pngs` exact.
    """
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for dvar in ds.data_vars:
        meta = PLOT_META[code_from_variable(dvar)]
        series = ds[dvar]
        resampled = (
            series.resample(time="ME").sum("time")
            if meta["resample"] == "sum"
            else series.resample(time="ME").mean("time")
        )

        # Sorted, so the panels read in subcatchment order rather than wflow's
        # internal one (measured [103, 101, 104, 102] on a four-point basin).
        units = (
            sorted(np.atleast_1d(series["index"].values).tolist())
            if "index" in series.dims
            else [None]
        )
        fig, axes = plt.subplots(
            len(units), 1, sharex=True, figsize=(11, 3 * len(units)), squeeze=False
        )

        for ax, unit in zip(axes[:, 0], units):
            one = resampled if unit is None else resampled.sel(index=unit)
            grouped = one.groupby("time.month")
            ax.plot(np.arange(1, 13), grouped.mean("time"), color="darkblue")
            ax.fill_between(
                np.arange(1, 13),
                grouped.quantile(0.25, "time"),
                grouped.quantile(0.75, "time"),
                color="lightblue",
            )
            ax.set_ylabel(meta["legend"], fontsize=fs)
            ax.tick_params(axis="both", labelsize=fs)
            ax.set_xlabel("")
            ax.grid(alpha=0.5)
            if unit is not None:
                ax.set_title(f"subcatchment {int(unit)}", fontsize=fs)

        axes[-1, 0].set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=fs)
        plt.tight_layout()
        save_figure(os.path.join(Folder_out, f"{dvar}.png"), dpi=300)


# %%
