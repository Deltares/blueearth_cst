# -*- coding: utf-8 -*-
"""Discharge performance metrics and the basin-average output figures.

Created 2021-07-14 (@author: bouaziz).

What this file NO LONGER holds, since 2026-08-10: ``plot_hydro`` and
``plot_signatures``. The four evaluation sheets they became live in
``shared/plot_evaluation.py``, beside the page settings they are drawn under —
this module's own private copy of those settings (``_FIG_WIDTH_MM``,
``_CLIM_RC``, ``_PT_BODY``) went with them, along with the Theil-Sen trend
helpers that served the subcatchment climate plots ADR 0006 retired. All of it
was reachable from nowhere.

What is left is the part that is not about drawing: ``compute_metrics``, which
writes ``performance_metrics.csv`` and which the performance sheet RENDERS
rather than recomputes, and ``plot_basavg``, whose figures are a separate,
config-derivable family with their own declared outputs.
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
from hydromt.stats import skills

from blueearth_cst.shared.snake_utils import save_figure

# %%
# Supported wflow outputs. The resample rule and axis legend live beside the
# code map they are keyed by, in shared/wflow_outputs.py — one table rather than
# two that had to be kept in agreement by hand.
from blueearth_cst.shared.wflow_outputs import (  # noqa: E402
    PLOT_META,
    code_from_variable,
)


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
