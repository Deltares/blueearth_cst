# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

# %%
import os

import hydromt  # noqa: F401 -- registers the xarray .raster accessor (pr/tas.raster.box below)
import matplotlib.pyplot as plt

# S8-08(c): cartopy and matplotlib.colors went with the gridded map
# figures. This script now draws only the eight scalar figures.
import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.shared.snake_utils import log_row


def plot_combination_traces(frame, colors=None, label_prefix="", alpha=0.85):
    """One line per COMBINATION — rulings R3'/R3'' (step 6c).

    This replaces a 5-95% band plus a line labelled "multi-model median". That
    label was the clearest statement of the claim R3' withdraws: a median across
    models asserts an ensemble the design does not construct. Under R3'/R3'' each
    (model, scenario, member) is one data point, so the figure shows the points.

    Every trace is labelled with its combination. A legend naming none of them is
    how "one trace per combination" becomes a hairball, which is the failure this
    step trades the envelope for.

    Returns the number of traces drawn, so a caller can assert it equals the
    number of resolved combinations.
    """
    columns = list(frame.columns)
    for index, column in enumerate(columns):
        plt.plot(
            frame.index,
            frame[column],
            color=colors[index % len(colors)] if colors else None,
            alpha=alpha,
            linewidth=1.0,
            label=f"{label_prefix}{column}",
        )
    return len(columns)


def todatetimeindex_dropvars(ds):
    if "time" in ds.coords:
        if ds.indexes["time"].dtype == "O":
            ds["time"] = ds.indexes["time"].to_datetimeindex()
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    if "height" in ds.coords:
        ds = ds.drop_vars("height")
    return ds


# ===========================================================================
# THE FIGURE INPUTS
#
# Two sources, and the split is the point.
#
# * The MONTHLY change factors are READ from
#   `{clim_project}_change_factors_monthly.csv`, never recomputed here. That
#   table is the authoritative product of `get_change_climate_proj.py`, and it
#   is what WF3 and every downstream reader consume. A figure that recomputed
#   the same quantity would be a second definition of it, free to disagree --
#   and it did: the shipped figures differenced a future month against the
#   historical ANNUAL mean, over the full 2015-2100 series rather than the
#   horizon, so the picture and the table described different quantities under
#   one name. Reading the table makes that class of disagreement structurally
#   impossible rather than merely fixed.
#
# * The ANNUAL OVERVIEWS are computed from `scalar/*.nc`, because they are
#   per-YEAR traces and no table carries a time series. The only arithmetic
#   here is the anomaly against the reference window, and it is differenced
#   against each model's OWN historical run so a future trace is continuous
#   with the historical one it follows.
# ===========================================================================


def normalise_model(name):
    """Drop the institute prefix from a model coordinate.

    ``scalar/*.nc`` carries ``NOAA-GFDL/GFDL-ESM4``; the change-factor tables
    carry ``GFDL-ESM4``. Without this, every join between series-derived and
    table-derived values matches nothing -- and an empty join reads as a broken
    calculation rather than as a spelling difference, which is the expensive
    kind of failure.
    """
    return str(name).split("/")[-1]


def parse_window(text):
    """``"2000-01-01 / 2014-12-01"`` into inclusive ``(start_year, end_year)``.

    The window is read off the artifact that states it rather than recomputed
    from config: the table records what was ACTUALLY differenced, including any
    effective-window override, and that is the only definition the figures can
    be held to.
    """
    try:
        start, end = str(text).split("/")
    except ValueError as exc:
        raise ValueError(f"malformed window {text!r}; expected 'START / END'") from exc
    return int(start.strip()[:4]), int(end.strip()[:4])


def load_scalar_series(paths):
    """Every ``scalar/*.nc`` as one long frame.

    Columns: ``model, scenario, member, year, month, precip, temp``.

    Year and month are read off the time index element-wise rather than through
    a datetime coercion: these series are monthly and declare a ``noleap``
    calendar (``cst_calendar``), so they decode to ``cftime`` objects on which
    ``.resample`` and ``.dt`` behave differently than on a ``DatetimeIndex``.
    Both object types answer ``.year`` and ``.month``, which is all this needs.
    """
    frames = []
    for path in sorted(str(p) for p in paths):
        with xr.open_dataset(path) as ds:
            times = ds.indexes["time"]
            frame = pd.DataFrame(
                {
                    "year": [t.year for t in times],
                    "month": [t.month for t in times],
                    "precip": ds["precip"].squeeze(drop=True).values,
                    "temp": ds["temp"].squeeze(drop=True).values,
                }
            )
            frame["model"] = normalise_model(ds["model"].values.item())
            frame["scenario"] = str(ds["scenario"].values.item())
            frame["member"] = str(ds["member"].values.item())
        frames.append(frame)
    if not frames:
        raise ValueError("no scalar series given; WF2 cannot draw its overviews")
    return pd.concat(frames, ignore_index=True)


def annual_reference(series, reference):
    """Historical mean per ``(model, member)`` over the reference window."""
    hist = series[
        (series["scenario"] == "historical") & series["year"].between(*reference)
    ]
    if hist.empty:
        raise ValueError(
            f"no historical years inside the reference window {reference}; "
            "the overviews would have nothing to difference against"
        )
    return hist.groupby(["model", "member"], as_index=False)[["precip", "temp"]].mean()


def annual_series(series, reference):
    """Annual means per combination per year, plus the anomaly against ``reference``.

    Every trace on both annual overviews comes from here -- historical and
    future alike, each differenced against its OWN model's historical reference
    window, which is what makes a future trace continuous with the historical
    one it follows rather than offset from it by a cross-model mean.
    """
    annual = series.groupby(["model", "scenario", "member", "year"], as_index=False)[
        ["precip", "temp"]
    ].mean()
    merged = annual.merge(
        annual_reference(series, reference),
        on=["model", "member"],
        suffixes=("", "_ref"),
    )
    merged["precip_anomaly"] = (
        (merged["precip"] - merged["precip_ref"]) / merged["precip_ref"] * 100.0
    )
    merged["temp_anomaly"] = merged["temp"] - merged["temp_ref"]
    return merged


def monthly_change_from_table(table, horizon, statistic="mean"):
    """The monthly change factors for one horizon, READ from the table.

    Returns ``model, scenario, member, month, precip_change, temp_change`` --
    the table's own ``relative_value``, pivoted per variable and not touched by
    any arithmetic here. ``relative_units`` is ``%`` for precipitation and
    ``degC`` for temperature, which is what the figure's y-labels state.

    Rows whose ``status`` is not ``ok`` are DROPPED rather than plotted or
    backfilled: a flagged month is one where the near-zero reference made the
    ratio meaningless, and drawing it would put a number on the page that the
    table itself declines to publish.

    Both statistics the table carries beside ``mean`` are available, but the
    figure asks for one: an unfiltered table has three rows per combination and
    a naive read triples every trace.
    """
    wanted = table[
        (table["horizon"] == horizon) & (table["statistic"] == statistic)
    ].copy()
    if wanted.empty:
        raise ValueError(
            f"change-factor table carries no {statistic!r} rows for horizon "
            f"{horizon!r}; have {sorted(table['horizon'].unique())}"
        )
    wanted = wanted[wanted["status"] == "ok"]
    keys = ["model", "scenario", "member", "month"]
    wide = wanted.pivot_table(
        index=keys, columns="variable", values="relative_value", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns={"precip": "precip_change", "temp": "temp_change"})


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            # Snakemake options
            clim_project_dir = sm.params.clim_project_dir
            stats_time_nc_hist = sm.input.stats_time_nc_hist
            stats_time_nc = sm.input.stats_time_nc
            rcps = sm.params.scenarios
            horizons = sm.params.horizons
            # S8-07: figures are named `{proj}_{variable}_{view}_{quantity}.png`.
            # `abs`/`anom` map to `absolute`/`change` -- the same distinction the
            # change-factor tables draw with absolute_value/relative_value.
            clim_project = os.path.basename(clim_project_dir)
            QUANTITY = {"abs": "absolute", "anom": "change"}

            # %% Historical
            log_row("Opening historical gcm timeseries", module="plot")

            # Step 4c: the three "did this file have data?" loops are gone. They
            # removed the dummy empty netCDFs stage A used to write for absent
            # sources; since 4a an unresolved combination never becomes a job, so
            # every path here carries data and dropping any would silently shrink
            # the ensemble (D4).
            # .load() was added for a deadlock that S8-02 has since removed: the
            # merge + to_netcdf round-trip that used to follow read nine netCDFs
            # from dask's thread pool, netCDF4/HDF5 reads take a global lock, and
            # the write DEADLOCKED -- measured twice on win-64 (14 threads parked
            # in Wait/UserRequest, ~6 s CPU over 15 min, output file created and
            # never written), while the same job with DASK_SCHEDULER=synchronous
            # finished in 42 s. RETAINED anyway: every one of these datasets is
            # converted straight to pandas below, so laziness buys nothing here,
            # and the sliced data is a few hundred KB. Same reason
            # get_stats_climate_proj.py loads eagerly after slicing.
            ds_hist = xr.open_mfdataset(
                list(stats_time_nc_hist), preprocess=todatetimeindex_dropvars
            ).load()

            # convert to df and compute anomalies
            log_row("Computing historical gcm timeseries anomalies", module="plot")
            # precip
            gcm_pr = ds_hist["precip"].squeeze(drop=True).transpose().to_pandas()
            # check if gcm_pr_anom is pd.Series or pd.DataFrame
            if isinstance(gcm_pr, pd.Series):
                gcm_pr = gcm_pr.to_frame()
            # %%
            # monthly mean
            gcm_pr_mnmn = gcm_pr.groupby(gcm_pr.index.month).mean()
            q_pr_mnmn = gcm_pr_mnmn  # 6c: per-combination, no cross-model reduction
            gcm_pr_mnref = gcm_pr_mnmn.mean()
            gcm_pr_mnanom = (gcm_pr_mnmn - gcm_pr_mnref) / gcm_pr_mnref * 100
            q_pr_mnanom = gcm_pr_mnanom  # 6c: per-combination, no cross-model reduction
            # annual mean
            gcm_pr_annmn = gcm_pr.resample("YE").mean()
            q_pr_annmn = gcm_pr_annmn  # 6c: per-combination, no cross-model reduction
            gcm_pr_ref = gcm_pr_annmn.mean()
            gcm_pr_anom = (gcm_pr_annmn - gcm_pr_ref) / gcm_pr_ref * 100
            q_pr_anom = gcm_pr_anom  # 6c: per-combination, no cross-model reduction

            # temp
            gcm_tas = ds_hist["temp"].squeeze(drop=True).transpose().to_pandas()
            # check if gcm_pr_anom is pd.Series or pd.DataFrame
            if isinstance(gcm_tas, pd.Series):
                gcm_tas = gcm_tas.to_frame()
            # monthly mean
            gcm_tas_mnmn = gcm_tas.groupby(gcm_tas.index.month).mean()
            q_tas_mnmn = gcm_tas_mnmn  # 6c: per-combination, no cross-model reduction
            gcm_tas_mnref = gcm_tas_mnmn.mean()
            gcm_tas_mnanom = gcm_tas_mnmn - gcm_tas_mnref
            q_tas_mnanom = (
                gcm_tas_mnanom  # 6c: per-combination, no cross-model reduction
            )
            # annual mean
            gcm_tas_annmn = gcm_tas.resample("YE").mean()
            q_tas_annmn = gcm_tas_annmn  # 6c: per-combination, no cross-model reduction
            gcm_tas_ref = gcm_tas_annmn.mean()
            gcm_tas_anom = gcm_tas_annmn - gcm_tas_ref
            q_tas_anom = gcm_tas_anom  # 6c: per-combination, no cross-model reduction

            # %% Future
            fns_future = list(stats_time_nc)

            # Initialise list of future df per rcp
            pr_fut = []
            tas_fut = []
            anom_pr_fut = []
            anom_tas_fut = []
            qanom_pr_fut = []
            qanom_tas_fut = []
            qpr_fut = []
            qtas_fut = []
            qpr_futmonth = []
            qpr_futmonth_sum = []
            qpr_futmonth_anom = []
            qtas_futmonth_anom = []
            qtas_futmonth = []
            qpr_fut_abs = []
            qtas_fut_abs = []
            for i in range(len(rcps)):
                pr_fut.append([])
                tas_fut.append([])
                anom_pr_fut.append([])
                anom_tas_fut.append([])
                qanom_pr_fut.append([])
                qanom_tas_fut.append([])
                qpr_fut.append([])
                qtas_fut.append([])
                qpr_futmonth.append([])
                qpr_futmonth_sum.append([])
                qpr_futmonth_anom.append([])
                qtas_futmonth_anom.append([])
                qtas_futmonth.append([])
                qpr_fut_abs.append([])
                qtas_fut_abs.append([])
            # read files
            for i in range(len(rcps)):
                log_row(
                    f"Opening future gcm timeseries for rcp {rcps[i]}", module="plot"
                )
                fns_rcp = [fn for fn in fns_future if rcps[i] in fn]
                # Eager for the same reason as ds_hist above.
                ds_rcp = xr.open_mfdataset(
                    fns_rcp, preprocess=todatetimeindex_dropvars
                ).load()
                ds_rcp_pr = ds_rcp["precip"].squeeze(drop=True)
                ds_rcp_tas = ds_rcp["temp"].squeeze(drop=True)
                # if len(ds_rcp.horizon) > 1:
                #     hz = ds_rcp.horizon
                #     ds_rcp_pr = xr.merge(
                #         [
                #             ds_rcp_pr.sel({"horizon": hz[0]}, drop=True),
                #             ds_rcp_pr.sel({"horizon": hz[1]}, drop=True),
                #         ]
                #     )
                #     ds_rcp_pr = ds_rcp_pr["precip"]
                #     ds_rcp_tas = xr.merge(
                #         [
                #             ds_rcp_tas.sel({"horizon": hz[0]}, drop=True),
                #             ds_rcp_tas.sel({"horizon": hz[1]}, drop=True),
                #         ]
                #     )
                #     ds_rcp_tas = ds_rcp_tas["temp"]
                # to dataframe
                prfi = ds_rcp_pr.transpose().to_pandas()
                if isinstance(prfi, pd.Series):
                    prfi = prfi.to_frame()
                pr_fut[i] = prfi
                tasfi = ds_rcp_tas.transpose().to_pandas()
                if isinstance(tasfi, pd.Series):
                    tasfi = tasfi.to_frame()
                tas_fut[i] = tasfi

            # compute anomalies
            log_row("Computing future gcm timeseries anomalies", module="plot")
            fut_pr_ref = gcm_pr_annmn.mean()
            fut_tas_ref = gcm_tas_annmn.mean()

            # monthly
            for i in range(len(qpr_futmonth)):
                pr_futmonth = pr_fut[i].groupby(pr_fut[i].index.month).mean()
                qpr_futmonth[i] = pr_futmonth  # 6c: per-combination
                pr_futmonth_anom = (pr_futmonth - fut_pr_ref) / fut_pr_ref * 100
                qpr_futmonth_anom[i] = pr_futmonth_anom.dropna(
                    axis=1, how="all"
                )  # 6c: per-combination

                tas_futmonth = tas_fut[i].groupby(tas_fut[i].index.month).mean()
                qtas_futmonth[i] = tas_futmonth  # 6c: per-combination
                tas_futmonth_anom = tas_futmonth - fut_tas_ref
                qtas_futmonth_anom[i] = tas_futmonth_anom.dropna(
                    axis=1, how="all"
                )  # 6c: per-combination
            # annual
            for i in range(len(anom_pr_fut)):
                qpr_fut[i] = pr_fut[i].resample("YE").mean()  # 6c: per-combination
                anom_pr_fut[i] = (
                    (pr_fut[i].resample("YE").mean() - fut_pr_ref) / fut_pr_ref * 100
                )
                qanom_pr_fut[i] = anom_pr_fut[i]  # 6c: per-combination

                qtas_fut[i] = tas_fut[i].resample("YE").mean()  # 6c: per-combination
                anom_tas_fut[i] = tas_fut[i].resample("YE").mean() - fut_tas_ref
                qanom_tas_fut[i] = anom_tas_fut[i]  # 6c: per-combination

            # S8-02: the merge-and-write of `timeseries/gcm_timeseries.nc` stood
            # here. Removed: nothing read it, and what it wrote was strictly worse
            # than the `scalar/` tier it merged -- rounded to 2 dp (re-imposing the
            # 0.005 mm/day floor step 5c had just removed) and stripped of every
            # `cst_*` attr, so it carried no digest, region fingerprint or calendar
            # and could not be validated. `scalar/*.nc` is the durable timeseries
            # tier; `change_factors/*.csv` is the analysis-ready long form.

            # %% Plots
            if not os.path.exists(os.path.join(clim_project_dir, "plots")):
                os.mkdir(os.path.join(clim_project_dir, "plots"))

            clrs = []
            for s in rcps:
                if s == "ssp126":
                    clrs.append("#003466")
                if s == "ssp245":
                    clrs.append("#f69320")
                if s == "ssp370":
                    clrs.append("#df0000")
                elif s == "ssp585":
                    clrs.append("#980002")
            # precip anomaly and absolute series
            for n in ["abs", "anom"]:
                if n == "abs":
                    # S8-07 (owner ruling): mm/day, not mm/year. The *365 here
                    # made this the ONE artifact reporting a different unit for the
                    # same quantity, so a reader comparing the figure against
                    # `*_change_factors_annual.csv` saw ~2210 beside ~6.05.
                    data_hist = q_pr_annmn
                    data_fut = list(qpr_fut)
                    y_label = "mm/day"
                else:
                    data_hist = q_pr_anom
                    data_fut = qanom_pr_fut
                    y_label = "Anomaly (%)"
                plt.figure(figsize=(8, 6))
                plt.title("Annual precipitation")
                plot_combination_traces(
                    data_hist, colors=["darkgrey"], label_prefix="historical "
                )
                for i in range(len(data_fut)):
                    plot_combination_traces(
                        data_fut[i], colors=[clrs[i]], label_prefix=rcps[i] + " "
                    )
                plt.ylabel(y_label)
                plt.legend()
                plt.grid()
                plt.savefig(
                    os.path.join(
                        clim_project_dir,
                        "plots",
                        f"{clim_project}_precip_annual_{QUANTITY[n]}.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
            # %%
            # temp anomaly
            for n in ["abs", "anom"]:
                if n == "abs":
                    data_hist = q_tas_annmn
                    data_fut = qtas_fut
                    y_label = "degC"
                else:
                    data_hist = q_tas_anom
                    data_fut = qanom_tas_fut
                    y_label = "Anomaly (degC)"
                plt.figure(figsize=(8, 6))
                plt.title("Average annual temperature")
                plot_combination_traces(
                    data_hist, colors=["darkgrey"], label_prefix="historical "
                )
                for i in range(len(data_fut)):
                    plot_combination_traces(
                        data_fut[i], colors=[clrs[i]], label_prefix=rcps[i] + " "
                    )
                plt.ylabel(y_label)
                plt.legend()
                plt.grid()
                plt.savefig(
                    os.path.join(
                        clim_project_dir,
                        "plots",
                        f"{clim_project}_temp_annual_{QUANTITY[n]}.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

            # %%
            # monthly changes precip
            for n in ["abs", "anom"]:
                if n == "abs":
                    qpr = qpr_futmonth
                    qprhist = q_pr_mnmn
                    y_label = "mm/day"
                else:
                    qpr = qpr_futmonth_anom
                    qprhist = q_pr_mnanom
                    y_label = "Anomaly (%)"
                plt.figure(figsize=(8, 6))
                plt.title("Average precipitation")
                plot_combination_traces(
                    qprhist, colors=["k"], label_prefix="historical "
                )

                for i in range(len(qpr)):
                    plot_combination_traces(
                        qpr[i], colors=[clrs[i]], label_prefix=rcps[i] + " "
                    )
                plt.ylabel(y_label)
                plt.xticks(
                    np.arange(1, 13),
                    ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
                )
                plt.legend()
                plt.grid()
                figname = f"{clim_project}_precip_monthly_{QUANTITY[n]}.png"
                plt.savefig(
                    os.path.join(clim_project_dir, "plots", figname),
                    dpi=300,
                    bbox_inches="tight",
                )
            # %%
            # monthly changes temp
            for n in ["abs", "anom"]:
                if n == "abs":
                    qtas = qtas_futmonth
                    qtashist = q_tas_mnmn
                    y_label = "degC"
                else:
                    qtas = qtas_futmonth_anom
                    qtashist = q_tas_mnanom
                    y_label = "Anomaly (degC)"

                plt.figure(figsize=(8, 6))
                plt.title("Average monthly temperature")
                plot_combination_traces(
                    qtashist, colors=["k"], label_prefix="historical "
                )
                plt.ylabel(f"{y_label}")
                plt.xticks(
                    np.arange(1, 13),
                    ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
                )
                for i in range(len(qtas)):
                    plot_combination_traces(
                        qtas[i], colors=[clrs[i]], label_prefix=rcps[i] + " "
                    )
                plt.legend()
                plt.grid()

                plt.savefig(
                    os.path.join(
                        clim_project_dir,
                        "plots",
                        f"{clim_project}_temp_monthly_{QUANTITY[n]}.png",
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )

            # %%
    else:
        raise RuntimeError("plot_proj_timeseries.py runs only as a Snakemake script:")
