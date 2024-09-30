# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 09:18:38 2021

@author: bouaziz
"""
from hydromt.stats import skills
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import scipy.stats as stats
import pandas as pd
import xarray as xr

from typing import Union, Optional, List

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
        "legend": "Snow water equivalent (mm)",
        "legend_annual": "Snow water equivalent (mm)",
    },
    "glacier": {
        "resample": "mean",
        "legend": "Glacier water equivalent (mm)",
        "legend_annual": "Glacier water equivalent (mm)",
    },
}


def rsquared(x, y):
    """Return R^2 and p_value where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value**2, p_value


def compute_metrics(
    qsim: xr.DataArray,
    qobs: xr.DataArray,
    station_name: str = "station",
    climate_source: str = "climate_source",
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
    climate_source : str, optional
        Climate source used, by default "climate_source"
    Returns
    -------
    pd.DataFrame
        Dataframe with performance metrics for this station.
    """
    ### 1. Calculate performance metrics based on daily and monthly timeseries ###
    # Initialize performance array
    metrics = ["KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE"]
    time_type = ["daily", "monthly"]
    climate_source = [f"{climate_source}"]
    da_perf = xr.DataArray(
        np.zeros((len(metrics), len(time_type), len(climate_source))),
        coords=[metrics, time_type, climate_source],
        dims=["metrics", "time_type", "climate_source"],
    )
    # Find the common period between obs and sim
    start = max(qsim.time.values[0], qobs.time.values[0])
    end = min(qsim.time.values[-1], qobs.time.values[-1])
    # make sure obs and sim have period in common
    if start < end:
        qsim = qsim.sel(time=slice(start, end))
        qobs = qobs.sel(time=slice(start, end))
    else:
        print(
            f"No common period between obs and sim for {station_name}. Skipping signatures."
        )
        return

    # Select data and resample to monthly timeseries as well
    qsim_monthly = qsim.resample(time="ME").mean("time", skipna=False)
    qobs_monthly = qobs.resample(time="ME").mean("time", skipna=False)
    # compute perf metrics
    # nse
    nse = skills.nashsutcliffe(qsim, qobs)
    da_perf.loc[
        dict(metrics="NSE", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = nse
    nse_m = skills.nashsutcliffe(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(metrics="NSE", time_type="monthly", climate_source=f"{climate_source[0]}")
    ] = nse_m

    # nse logq
    nselog = skills.lognashsutcliffe(qsim, qobs)
    da_perf.loc[
        dict(metrics="NSElog", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = nselog
    nselog_m = skills.lognashsutcliffe(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(
            metrics="NSElog", time_type="monthly", climate_source=f"{climate_source[0]}"
        )
    ] = nselog_m

    # kge
    kge = skills.kge(qsim, qobs)
    da_perf.loc[
        dict(metrics="KGE", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = kge["kge"]
    kge_m = skills.kge(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(metrics="KGE", time_type="monthly", climate_source=f"{climate_source[0]}")
    ] = kge_m["kge"]

    # rmse
    rmse = skills.rmse(qsim, qobs)
    da_perf.loc[
        dict(metrics="RMSE", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = rmse
    rmse_m = skills.rmse(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(metrics="RMSE", time_type="monthly", climate_source=f"{climate_source[0]}")
    ] = rmse_m

    # mse
    mse = skills.mse(qsim, qobs)
    da_perf.loc[
        dict(metrics="MSE", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = mse
    mse_m = skills.mse(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(metrics="MSE", time_type="monthly", climate_source=f"{climate_source[0]}")
    ] = mse_m

    # pbias
    pbias = skills.percentual_bias(qsim, qobs)
    da_perf.loc[
        dict(metrics="Pbias", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = pbias
    pbias_m = skills.percentual_bias(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(
            metrics="Pbias", time_type="monthly", climate_source=f"{climate_source[0]}"
        )
    ] = pbias_m

    # ve (volumetric efficiency)
    ve = skills.volumetric_error(qsim, qobs)
    da_perf.loc[
        dict(metrics="VE", time_type="daily", climate_source=f"{climate_source[0]}")
    ] = ve
    ve_m = skills.volumetric_error(qsim_monthly, qobs_monthly)
    da_perf.loc[
        dict(metrics="VE", time_type="monthly", climate_source=f"{climate_source[0]}")
    ] = ve_m

    ### 2. Convert to dataframe ###
    df_perf = da_perf.to_dataframe(name=station_name)

    return df_perf


def plot_signatures(
    qsim: xr.DataArray,
    qobs: xr.DataArray,
    Folder_out: Union[Path, str],
    station_name: str = "station",
    color: dict = {"climate_source": "orange"},
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
        Dataset with simulated streamflow from several runs with different climate sources.

        * Required dimensions: [time]
        * Required attributes: [station_name]
        * Required attributes: [climate_source]
    qobs : xr.DataArray
        Dataset with observed streamflow.

        * Required dimensions: [time]
        * Required attributes: [station_name]
    Folder_out : Union[Path, str]
        Output folder to save plots.
    station_name : str, optional
        Station name, by default "station"
    color : dict, optional
        Color for each climate source of the simulated runs , by default "orange"
    linestyle : str, optional
        Linestyle for the simulated run, by default "-"
    marker : str, optional
        Marker for the simulated run, by default "o"
    lw : float, optional
        Line width, by default 0.8
    fs : int, optional
        Font size, by default 8
    """
    # Drop nans from obs
    qobs = qobs.dropna("time")
    # Find the common period between obs and sim
    start = max(qsim.time.values[0], qobs.time.values[0])
    end = min(qsim.time.values[-1], qobs.time.values[-1])
    # make sure obs and sim have period in common
    if start < end:
        qsim = qsim.sel(time=slice(start, end))
        qobs = qobs.sel(time=slice(start, end))
    else:
        print(
            f"No common period between obs and sim for {station_name}. Skipping signatures."
        )
        return

    # Drop the times in qsim that are not in qobs because of nans
    qsim = qsim.sel(time=qobs.time)

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
    for climate_source in qsim.climate_source.values:
        axes[0].plot(
            qobs,
            qsim.sel(climate_source=climate_source),
            marker="o",
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
            color=color[climate_source],
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
    for climate_source in qsim.climate_source.values:
        r2_score = rsquared(qobs, qsim.sel(climate_source=climate_source))[0]
        text_label += f"R$_2$ {climate_source} = {r2_score:.2f} \n"
    axes[0].text(0.05, 0.7, text_label, transform=axes[0].transAxes, fontsize=fs)

    ### 2. streamflow regime axes[1] ###
    for climate_source in qsim.climate_source.values:
        qsim.sel(climate_source=climate_source).groupby("time.month").mean("time").plot(
            ax=axes[1],
            linewidth=lw,
            label=f"{climate_source}",
            color=color[climate_source],
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
    for climate_source in qsim.climate_source.values:
        axes[2].plot(
            np.arange(0, len(qsim.sel(climate_source=climate_source).time))
            / (len(qsim.sel(climate_source=climate_source).time) + 1),
            qsim.sel(climate_source=climate_source).sortby(
                qsim.sel(climate_source=climate_source), ascending=False
            ),
            color=color[climate_source],
            linestyle=linestyle,
            linewidth=lw,
            label=f"{climate_source}",
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
    for climate_source in qsim.climate_source.values:
        axes[3].plot(
            np.arange(0, len(qsim.sel(climate_source=climate_source).time))
            / (len(qsim.sel(climate_source=climate_source).time) + 1),
            np.log(
                qsim.sel(climate_source=climate_source).sortby(
                    qsim.sel(climate_source=climate_source), ascending=False
                )
            ),
            color=color[climate_source],
            linestyle=linestyle,
            linewidth=lw,
            label=f"{climate_source}",
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
        # Nans can reappear after resampling, drop them
        qobs_max = qobs_max.dropna("time")
        qsim_max = qsim_max.sel(time=qobs_max.time)
    else:
        # Less than a year of data, max over the whole timeseries
        qsim_max = qsim.max("time")
        qobs_max = qobs.max("time")

    for climate_source in qsim.climate_source.values:
        axes[4].plot(
            qobs_max,
            qsim_max.sel(climate_source=climate_source),
            color=color[climate_source],
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
        )
    axes[4].plot(
        [0, max_y * 1.1], [0, max_y * 1.1], color="0.5", linestyle="--", linewidth=1
    )
    axes[4].set_xlim([0, max_y * 1.1])
    axes[4].set_ylim([0, max_y * 1.1])
    # R2 score
    text_label = ""
    for climate_source in qsim.climate_source.values:
        if len(qsim.time) > 365:
            r2_score = rsquared(qobs_max, qsim_max.sel(climate_source=climate_source))[
                0
            ]
            text_label += f"R$_2$ {climate_source} = {r2_score:.2f} \n"
        else:
            text_label = text_label + f"{climate_source}\n"
    axes[4].text(0.05, 0.7, text_label, transform=axes[4].transAxes, fontsize=fs)

    # add MHQ
    if len(qsim.time) > 365:
        mhq = qsim_max.mean("time")
        mhq_obs = qobs_max.mean("time")
    else:
        mhq = qsim_max.copy()
        mhq_obs = qobs_max.copy()
    for climate_source in qsim.climate_source.values:
        axes[4].plot(
            mhq_obs,
            mhq.sel(climate_source=climate_source),
            color="black",
            marker=">",
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
            markersize=6,
        )
    # labels
    axes[4].set_ylabel("Sim. max annual Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[4].set_xlabel("Obs. max annual Q (m$^3$s$^{-1}$)", fontsize=fs)

    ### 6. nm7q axes[5] ###
    qsim_nm7q = qsim.rolling(time=7).mean().resample(time="YE").min("time")
    qobs_nm7q = qobs.rolling(time=7).mean().resample(time="YE").min("time")
    # Nans can reappear after resampling, drop them
    qobs_nm7q = qobs_nm7q.dropna("time")
    qsim_nm7q = qsim_nm7q.sel(time=qobs_nm7q.time)
    max_ylow = max(qsim_nm7q.max().values, qobs_nm7q.max().values)

    for climate_source in qsim.climate_source.values:
        axes[5].plot(
            qobs_nm7q,
            qsim_nm7q.sel(climate_source=climate_source),
            color=color[climate_source],
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
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
    for climate_source in qsim.climate_source.values:
        r2_score = rsquared(qobs_nm7q, qsim_nm7q.sel(climate_source=climate_source))[0]
        text_label += f"R$_2$ {climate_source} = {r2_score:.2f} \n"
    axes[5].text(0.05, 0.7, text_label, transform=axes[5].transAxes, fontsize=fs)
    # labels
    axes[5].set_ylabel("Simulated NM7Q (m$^3$s$^{-1}$)", fontsize=fs)
    axes[5].set_xlabel("Observed NM7Q (m$^3$s$^{-1}$)", fontsize=fs)

    ### 7. cum axes[6] ###
    qobs.cumsum("time").plot(
        ax=axes[6], color="k", linestyle=":", linewidth=lw, label="observed"
    )
    for climate_source in qsim.climate_source.values:
        qsim.sel(climate_source=climate_source).cumsum("time").plot(
            ax=axes[6],
            color=color[climate_source],
            linestyle=linestyle,
            linewidth=lw,
            label=f"{climate_source}",
        )
    axes[6].set_xlabel("")
    axes[6].set_ylabel("Cum. Q (m$^3$s$^{-1}$)", fontsize=fs)

    ### 8. performance measures NS, NSlogQ, KGE, axes[7] ###
    # nse
    for climate_source in qsim.climate_source.values:
        p = axes[7].plot(
            0.8,
            skills.nashsutcliffe(qsim.sel(climate_source=climate_source), qobs).values,
            color=color[climate_source],
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
        )
        if color[climate_source] == None:
            c = p[0].get_color()
        else:
            c = color[climate_source]
        # nselog
        axes[7].plot(
            2.8,
            skills.lognashsutcliffe(
                qsim.sel(climate_source=climate_source), qobs
            ).values,
            color=c,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
        )
        # kge
        axes[7].plot(
            4.8,
            skills.kge(qsim.sel(climate_source=climate_source), qobs)["kge"].values,
            color=c,
            marker=marker,
            linestyle="None",
            linewidth=lw,
            label=f"{climate_source}",
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
        for climate_source in qsim.climate_source.values:
            axes[8].plot(
                gumbel_p1,
                qsim_max.sel(climate_source=climate_source).sortby(
                    qsim_max.sel(climate_source=climate_source)
                ),
                marker=marker,
                color=color[climate_source],
                linestyle="None",
                label=f"{climate_source}",
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
        for climate_source in qsim.climate_source.values:
            axes[9].plot(
                gumbel_p1,
                qsim_nm7q.sel(climate_source=climate_source).sortby(
                    qsim_nm7q.sel(climate_source=climate_source), ascending=False
                ),
                marker=marker,
                color=color[climate_source],
                linestyle="None",
                label=f"{climate_source}",
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
    plt.savefig(os.path.join(Folder_out, f"signatures_{station_name}.png"), dpi=300)
    plt.close()


def plot_hydro(
    qsim: xr.DataArray,
    Folder_out: Union[Path, str],
    qobs: Optional[xr.DataArray] = None,
    color: dict = {"climate_source": "steelblue"},
    station_name: str = "station_1",
    lw: float = 0.8,
    fs: int = 7,
    max_nan_year: int = 60,
    max_nan_month: int = 5,
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
        Simulated streamflow. Coordinate "climate_source" should be present for several runs with different climate sources as forcing.
    Folder_out : Union[Path, str]
        Output folder to save plots.
    qobs : xr.DataArray, optional
        Observed streamflow, by default None
    color : dict, optional
        Color belonging to a climate source run, by default {"climate_source":"steelblue"}
    station_name : str, optional
        Station name, by default "station_1"
    lw : float, optional
        Line width, by default 0.8
    fs : int, optional
        Font size, by default 7
    max_nan_year : int, optional
        Maximum number of missing days per year in the observations data to consider
        the year in the annual hydrograph plot. By default 60.
    max_nan_month : int, optional
        Maximum number of missing days per month in the observations data to consider
        the month in the monthly regime plot. By default 5.
    """
    min_count_year = 365 - max_nan_year
    min_count_month = 30 - max_nan_month

    # Settings for obs
    labobs = "observed"
    colobs = "k"

    # Find the common period between obs and sim
    if qobs is not None:
        start = max(qsim.time.values[0], qobs.time.values[0])
        end = min(qsim.time.values[-1], qobs.time.values[-1])
        # make sure obs and sim have period in common
        if start < end:
            qsim_na = qsim.sel(time=slice(start, end))
            qobs_na = qobs.sel(time=slice(start, end))
        else:
            qobs = None
            qobs_na = None
            qsim_na = qsim
    else:
        qsim_na = qsim

    # Number of years available
    nb_years = np.unique(qsim["time.year"].values).size

    # If less than 3 years, plot a single panel
    if nb_years <= 3:
        nb_panel = 1
        titles = ["Daily time-series"]
        figsize_y = 8
    else:
        nb_panel = 5
        titles = [
            "Daily time-series",
            "Annual time-series",
            "Annual cycle",
            "Wettest year",
            "Driest year",
        ]
        figsize_y = 23
        # Get the wettest and driest year (based on first climate_source)
        if qobs is not None:
            qyr = qobs_na.resample(time="YE").sum(skipna=True, min_count=min_count_year)
            qyr["time"] = qyr["time.year"]
            # Get the year for the minimum as an integer
            year_dry = str(qyr.isel(time=qyr.argmin("time")).time.item())
            year_wet = str(qyr.isel(time=qyr.argmax("time")).time.item())
        else:
            qyr = qsim.resample(time="YE").sum()
            qyr["time"] = qyr["time.year"]
            # Get the year for the minimum as an integer
            year_dry = str(qyr.isel(time=qyr.argmin("time")).time.values[0])
            year_wet = str(qyr.isel(time=qyr.argmax("time")).time.values[0])

    fig, axes = plt.subplots(nb_panel, 1, figsize=(16 / 2.54, figsize_y / 2.54))
    axes = [axes] if nb_panel == 1 else axes

    # 1. long period
    for climate_source in qsim.climate_source.values:
        qsim.sel(climate_source=climate_source).plot(
            ax=axes[0],
            label=f"simulated {climate_source}",
            linewidth=lw,
            color=color[climate_source],
        )
    if qobs is not None:
        qobs_na.plot(
            ax=axes[0], label=labobs, linewidth=lw, color=colobs, linestyle="--"
        )

    if nb_panel == 5:
        # 2. annual Q
        for climate_source in qsim.climate_source.values:
            qsim.sel(climate_source=climate_source).resample(time="YE").sum(
                skipna=True, min_count=min_count_year
            ).plot(
                ax=axes[1],
                label=f"simulated {climate_source}",
                linewidth=lw,
                color=color[climate_source],
            )
        if qobs is not None:
            qobs.resample(time="YE").sum(skipna=True, min_count=min_count_year).plot(
                ax=axes[1], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

        # 3. monthly Q
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        dsqM = qsim_na.resample(time="ME").sum()
        dsqM = dsqM.groupby(dsqM.time.dt.month).mean()
        for climate_source in qsim.climate_source.values:
            dsqM.sel(climate_source=climate_source).plot(
                ax=axes[2],
                label=f"simulated {climate_source}",
                linewidth=lw,
                color=color[climate_source],
            )
        if qobs is not None:
            dsqMo = qobs_na.resample(time="ME").sum(
                skipna=True, min_count=min_count_month
            )
            dsqMo = dsqMo.groupby(dsqMo.time.dt.month).mean()
            dsqMo.plot(ax=axes[2], label=labobs, linewidth=lw, color=colobs)
        axes[2].set_title("Average monthly sum")
        axes[2].set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=5)

        # 4. wettest year
        for climate_source in qsim.climate_source.values:
            qsim.sel(climate_source=climate_source).sel(time=year_wet).plot(
                ax=axes[3],
                label=f"simulated {climate_source}",
                linewidth=lw,
                color=color[climate_source],
            )
        if qobs is not None:
            qobs.sel(time=year_wet).plot(
                ax=axes[3], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

        # 5. driest year
        for climate_source in qsim.climate_source.values:
            qsim.sel(climate_source=climate_source).sel(time=year_dry).plot(
                ax=axes[4],
                label=f"simulated {climate_source}",
                linewidth=lw,
                color=color[climate_source],
            )
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

    plt.savefig(os.path.join(Folder_out, f"hydro_{station_name}.png"), dpi=300)
    plt.close()


def plot_clim(
    ds_clim: xr.DataArray,
    Folder_out: Union[Path, str],
    station_name: str,
    period: str,
    color: dict,
    fs: int = 8,
    skip_precip_sources: List[str] = [],
    skip_temp_pet_sources: List[str] = [],
):
    """
    Plot monthly of annual climatology of precipitation, temperature and potential evaporation.
    Also show trends in mean annual precipitation, temperature and potential evaporation.

    Parameters
    ----------
    ds_clim : xr.DataArray
        Precipitation, temperature and potential evaporation data at model timestep.
        Should include a coordinate named "climate_source"
    Folder_out : Union[Path, str]
        Output folder to save plots.
    station_name : str
        Station name
    period : str
        Either monthly or annual climatology plots
    color : dict
        Color to be used for each climate_source
    fs : int, optional
        Font size, by default 8
    skip_precip_sources : List[str]
        List of climate sources to skip for precipitation plots.
    skip_temp_pet_sources : List[str]
        List of climate sources to skip for temperature and potential evaporation plots.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16 / 2.54, 15 / 2.54), sharex=True
    )

    if period == "year":
        resampleper = "YE"
    else:
        resampleper = "ME"

    # temp
    if period == "month":
        for climate_source in ds_clim.climate_source.values:
            # Check as for some sources only precipitation is used
            # so avoid to plot temperature and evaporation twice
            do_climate_plot = True
            if climate_source in skip_temp_pet_sources:
                do_climate_plot = False

            T_mean_monthly_mean = (
                ds_clim["T_subcatchment"]
                .groupby(f"time.{period}")
                .mean("time")
                .sel(climate_source=climate_source)
            )
            T_mean_monthly_q25 = (
                ds_clim["T_subcatchment"]
                .groupby(f"time.{period}")
                .quantile(0.25, "time")
                .sel(climate_source=climate_source)
            )
            T_mean_monthly_q75 = (
                ds_clim["T_subcatchment"]
                .groupby(f"time.{period}")
                .quantile(0.75, "time")
                .sel(climate_source=climate_source)
            )
            # plot
            # todo: update the workflow to have more control on the forcing and allow the user to select the dataset for precip and for temp+pet and the equation for pet
            # for now only plot temp and evap it climate source is era5 and era5 is in climate source.
            if do_climate_plot:
                p = T_mean_monthly_mean.plot(
                    ax=ax1,
                    color=color[climate_source],
                    label=f"{climate_source} (25%-75%)",
                )
                if color[climate_source] == None:
                    c = p[0].get_color()
                else:
                    c = color[climate_source]
                ax1.fill_between(
                    np.arange(1, 13),
                    T_mean_monthly_q25,
                    T_mean_monthly_q75,
                    color=c,
                    alpha=0.5,
                    # label=f"{climate_source} 25%-75%",
                )
    else:
        for climate_source in ds_clim.climate_source.values:
            do_climate_plot = True
            if climate_source in skip_temp_pet_sources:
                do_climate_plot = False
            T_mean_year = (
                ds_clim["T_subcatchment"]
                .resample(time=resampleper)
                .mean("time")
                .sel(climate_source=climate_source)
            )
            if do_climate_plot:
                x = T_mean_year.time.dt.year
                z = np.polyfit(x, T_mean_year, 1)
                p = np.poly1d(z)
                r2_score, p_value = rsquared(p(x), T_mean_year)

                im = T_mean_year.plot(
                    ax=ax1,
                    color=color[climate_source],
                    label=f"{climate_source} ($R^2$ = {round(r2_score, 3)}, p = {round(p_value, 3)})",
                )
                if color[climate_source] == None:
                    c = im[0].get_color()
                else:
                    c = color[climate_source]
                ax1.plot(
                    T_mean_year.time,
                    p(x),
                    ls="--",
                    color=c,
                    alpha=0.5,
                    # label=f"trend {climate_source} $R^2$ = {round(r2_score, 3)}, p = {round(p_value, 3)}",
                )

    # precip and evap
    for climvar, ax in zip(
        ["P_subcatchment", "EP_subcatchment"],
        [ax2, ax3],
    ):
        for climate_source in ds_clim.climate_source.values:
            do_climate_plot = True
            if climate_source in skip_temp_pet_sources:
                do_climate_plot = False
            # Same for duplicate use of precip sources
            do_precip_plot = True
            if climate_source in skip_precip_sources:
                do_precip_plot = False

            var_sum_monthly = (
                ds_clim[climvar]
                .resample(time=resampleper)
                .sum("time")
                .sel(climate_source=climate_source)
            )

            if period == "month":
                var_sum_monthly_mean = var_sum_monthly.groupby(f"time.{period}").mean(
                    "time"
                )
                var_sum_monthly_q25 = var_sum_monthly.groupby(
                    f"time.{period}"
                ).quantile(0.25, "time")
                var_sum_monthly_q75 = var_sum_monthly.groupby(
                    f"time.{period}"
                ).quantile(0.75, "time")

                if (do_climate_plot and climvar == "EP_subcatchment") | (
                    do_precip_plot and climvar == "P_subcatchment"
                ):
                    p = var_sum_monthly_mean.plot(
                        ax=ax,
                        color=color[climate_source],
                        label=f"{climate_source} (25%-75%)",
                    )
                    if color[climate_source] == None:
                        c = p[0].get_color()
                    else:
                        c = color[climate_source]

                    ax.fill_between(
                        np.arange(1, 13),
                        var_sum_monthly_q25,
                        var_sum_monthly_q75,
                        color=c,
                        alpha=0.5,
                        # label=f"{climate_source} 25%-75%",
                    )
            else:
                x = var_sum_monthly.time.dt.year
                z = np.polyfit(x, var_sum_monthly, 1)
                p = np.poly1d(z)
                r2_score, p_value = rsquared(p(x), var_sum_monthly)

                if (do_climate_plot and climvar == "EP_subcatchment") | (
                    do_precip_plot and climvar == "P_subcatchment"
                ):
                    p = ax.plot(
                        var_sum_monthly.time,
                        p(x),
                        ls="--",
                        alpha=0.5,
                        color=color[climate_source],
                        # label=f"trend {climate_source} $R^2$ = {round(r2_score, 3)}, p = {round(p_value, 3)}",
                    )
                    if color[climate_source] == None:
                        c = p[0].get_color()
                    else:
                        c = color[climate_source]

                    var_sum_monthly.plot(
                        ax=ax,
                        color=c,
                        label=f"{climate_source} ($R^2$ = {round(r2_score, 3)}, p = {round(p_value, 3)})",
                    )

    for ax, title_name, ylab in zip(
        [ax1, ax2, ax3],
        ["Temperature", "Precipitation", "Potential evaporation"],
        [
            "T ($\degree$C)",
            f"P (mm {period}" + "$^{-1}$)",
            f"E$_P$ (mm {period}" + "$^{-1}$)",
        ],
    ):
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_xlabel("", fontsize=fs)
        ax.set_title(title_name)
        ax.grid(alpha=0.5)
        ax.set_ylabel(ylab, fontsize=fs)
        ax.legend(fontsize=fs)

    if period == "month":
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        ax3.set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=fs)

    plt.tight_layout()
    plt.savefig(os.path.join(Folder_out, f"clim_{station_name}_{period}.png"), dpi=300)
    plt.close()


def plot_basavg(
    ds: xr.DataArray,
    Folder_out: Union[Path, str],
    color: dict,
    fs: float = 10,
):
    """
    Subplots of:
    - basin average mean monthly regime and 25-75% uncertainty band for several flux and state variables of the model.
    - basin mean annual trends for several flux and state variables of the model (only if there are 3 or more years of data)

    Parameters
    ----------
    ds : xr.DataArray
        Data of output variables at model timestep.
        Should include a coordinate named "climate_source"
    Folder_out : Union[Path, str]
        Output folder to save plots.
    color : dict
        Color to be used for each climate_source
    fs : int, optional
        Font size, by default 10
    """
    dvars = [dvar for dvar in ds.data_vars]
    n = len(dvars)
    # number of years available
    nb_years = np.unique(ds["time.year"].values).size

    for i in range(n):
        dvar = dvars[i]

        # only plot annual trend if there are more than 3 years of data
        if nb_years < 3:
            fig, (ax1) = plt.subplots(1, 1, figsize=(11, 4))
            axes = [ax1]
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
            axes = [ax1, ax2]
        # axes = [axes] if n == 1 else axes

        if WFLOW_VARS[dvar.split("_")[0]]["resample"] == "sum":
            sum_monthly = ds[dvar].resample(time="ME").sum("time")
            sum_annual = ds[dvar].resample(time="YE").sum("time")
        else:  # assume mean
            sum_monthly = ds[dvar].resample(time="ME").mean("time")
            sum_annual = ds[dvar].resample(time="YE").mean("time")
        sum_monthly_mean = sum_monthly.groupby("time.month").mean("time")
        sum_monthly_q25 = sum_monthly.groupby("time.month").quantile(0.25, "time")
        sum_monthly_q75 = sum_monthly.groupby("time.month").quantile(0.75, "time")

        for climate_source in ds.climate_source.values:
            # plot monthly mean
            p = sum_monthly_mean.sel(climate_source=climate_source).plot(
                ax=ax1, color=color[climate_source], label=f"{climate_source} (mean)"
            )
            if color[climate_source] == None:
                c = p[0].get_color()
            else:
                c = color[climate_source]
            ax1.fill_between(
                np.arange(1, 13),
                sum_monthly_q25.sel(climate_source=climate_source),
                sum_monthly_q75.sel(climate_source=climate_source),
                color=c,
                alpha=0.5,
                label=f"{climate_source} (25%-75%)",
            )

            # plot annual trends if 3 or more years of data:
            if nb_years >= 3:
                var_year = sum_annual.sel(climate_source=climate_source)
                p = var_year.plot(
                    ax=ax2, color=color[climate_source], label=f"{climate_source}"
                )

                if color[climate_source] == None:
                    c = p[0].get_color()
                else:
                    c = color[climate_source]

                x = var_year.time.dt.year
                z = np.polyfit(x, var_year, 1)
                p = np.poly1d(z)
                # if snow 0 each year -- cannot calculate linear regression
                if sum(var_year) > 0:
                    r2_score, p_value = rsquared(p(x), var_year)
                    ax2.plot(
                        var_year.time,
                        p(x),
                        ls="--",
                        color=c,
                        alpha=0.5,
                        label=f"trend {climate_source} $R^2$ = {round(r2_score, 3)}, p = {round(p_value, 3)}",
                    )

                legend_annual = WFLOW_VARS[dvar.split("_")[0]]["legend_annual"]
                ax2.set_ylabel(legend_annual, fontsize=fs)

        legend = WFLOW_VARS[dvar.split("_")[0]]["legend"]
        ax1.set_ylabel(legend, fontsize=fs)

        for ax in axes:
            ax.tick_params(axis="both", labelsize=fs)
            ax.set_xlabel("", fontsize=fs)
            ax.set_title("")
            ax.grid(alpha=0.5)
            ax.legend(fontsize=fs)

        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        ax1.set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=fs)

        plt.tight_layout()
        plt.savefig(os.path.join(Folder_out, f"{dvar}.png"), dpi=300)
        plt.close()
