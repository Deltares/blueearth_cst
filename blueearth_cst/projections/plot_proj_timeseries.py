# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""
# %%
import hydromt  # noqa: F401 -- registers the xarray .raster accessor (pr/tas.raster.box below)
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import numpy as np

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
            save_grids = sm.params.save_grids
            change_grids_nc = sm.params.change_grids


            # %% Historical
            log_row("Opening historical gcm timeseries", module="plot")


            # Step 4c: the three "did this file have data?" loops are gone. They
            # removed the dummy empty netCDFs stage A used to write for absent
            # sources; since 4a an unresolved combination never becomes a job, so
            # every path here carries data and dropping any would silently shrink
            # the ensemble (D4).
            # .load() is load-bearing, not a tidy-up: open_mfdataset returns a
            # dask-backed lazy dataset, and the merge + to_netcdf round-trip below
            # then reads nine netCDFs from dask's thread pool. netCDF4/HDF5 reads
            # take a global lock, so that write DEADLOCKS -- measured twice on
            # win-64 (14 threads parked in Wait/UserRequest, ~6 s CPU over 15 min,
            # output file created and never written), while the same job with
            # DASK_SCHEDULER=synchronous finishes in 42 s. Loading eagerly here
            # makes every read serial; the sliced data is a few hundred KB, so the
            # memory cost is nil. Same reason get_stats_climate_proj.py loads
            # eagerly after slicing.
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
            q_tas_mnanom = gcm_tas_mnanom  # 6c: per-combination, no cross-model reduction
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
            ds_fut = []
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
                log_row(f"Opening future gcm timeseries for rcp {rcps[i]}", module="plot")
                fns_rcp = [fn for fn in fns_future if rcps[i] in fn]
                # Eager for the same reason as ds_hist above: these datasets are
                # what `xr.merge` + `to_netcdf` write, and a lazy merge of several
                # netCDFs deadlocks dask's thread pool on the HDF5 lock.
                ds_rcp = xr.open_mfdataset(
                    fns_rcp, preprocess=todatetimeindex_dropvars
                ).load()
                ds_fut.append(ds_rcp)
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
                qpr_futmonth_anom[i] = pr_futmonth_anom.dropna(axis=1, how="all")  # 6c: per-combination

                tas_futmonth = tas_fut[i].groupby(tas_fut[i].index.month).mean()
                qtas_futmonth[i] = tas_futmonth  # 6c: per-combination
                tas_futmonth_anom = tas_futmonth - fut_tas_ref
                qtas_futmonth_anom[i] = tas_futmonth_anom.dropna(axis=1, how="all")  # 6c: per-combination
            # annual
            for i in range(len(anom_pr_fut)):
                qpr_fut[i] = pr_fut[i].resample("YE").mean()  # 6c: per-combination
                anom_pr_fut[i] = (pr_fut[i].resample("YE").mean() - fut_pr_ref) / fut_pr_ref * 100
                qanom_pr_fut[i] = anom_pr_fut[i]  # 6c: per-combination

                qtas_fut[i] = tas_fut[i].resample("YE").mean()  # 6c: per-combination
                anom_tas_fut[i] = tas_fut[i].resample("YE").mean() - fut_tas_ref
                qanom_tas_fut[i] = anom_tas_fut[i]  # 6c: per-combination

            # %% Merge and write all timeseries to a single netcdf file
            ds_fut.append(ds_hist)
            ds_all = xr.merge(ds_fut)
            # xarray's merge propagates global attrs from the FIRST dataset, so a
            # merged multi-source product would silently inherit one arbitrary
            # series' cst_* identity -- claiming a single digest, region
            # fingerprint and source pin for a file built from nine series. Strip
            # them: an identity that describes one input is worse than none.
            # (Found by semantic_tree_diff at step 4c; a per-source provenance
            # record is the report stage's job at step 7.)
            for _attr in [k for k in ds_all.attrs if k.startswith("cst_")]:
                del ds_all.attrs[_attr]
            # make sure we have two digits still
            ds_all["precip"] = ds_all["precip"].round(decimals=2)
            ds_all["temp"] = ds_all["temp"].round(decimals=2)
            # write to netcdf
            # R07 B3: the processed timeseries tier
            timeseries_dir = os.path.join(clim_project_dir, "timeseries")
            os.makedirs(timeseries_dir, exist_ok=True)
            ds_all.to_netcdf(os.path.join(timeseries_dir, "gcm_timeseries.nc"))

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
                    data_hist = q_pr_annmn * 365  # q_pr_anom_abs
                    data_fut = [data * 365 for data in qpr_fut]  # qpr_fut_abs
                    y_label = "mm/year"
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
                        clim_project_dir, "plots", f"precipitation_anomaly_projections_{n}"
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
                        clim_project_dir, "plots", f"temperature_anomaly_projections_{n}.png"
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
                    np.arange(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
                )
                plt.legend()
                plt.grid()
                figname = f"precipitation_monthly_projections_{n}.png"
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
                    np.arange(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
                )
                for i in range(len(qtas)):
                    plot_combination_traces(
                        qtas[i], colors=[clrs[i]], label_prefix=rcps[i] + " "
                    )
                plt.legend()
                plt.grid()

                plt.savefig(
                    os.path.join(
                        clim_project_dir, "plots", f"temperature_monthly_projections_{n}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )


            # %%
            # Map plots of gridded change per scenario / horizon
            if save_grids:
                fns_grids = list(change_grids_nc)

                # Loop over rcp and horizon
                for rcp in rcps:
                    for hz in horizons:
                        log_row(f"Preparing change map plots for {rcp} and horizon {hz}", module="plot")
                        fns_rcp_hz = [fn for fn in fns_grids if rcp in fn and hz in fn]
                        ds_rcp_hz = []
                        for fn in fns_rcp_hz:
                            ds = xr.open_dataset(fn)
                            if "time" in ds.coords:
                                if ds.indexes["time"].dtype == "O":
                                    ds["time"] = ds.indexes["time"].to_datetimeindex()
                            ds_rcp_hz.append(ds)
                        ds_rcp_hz = xr.merge(ds_rcp_hz)
                        ds_rcp_hz_med = ds_rcp_hz.median(dim="model").squeeze(drop=True)

                        # Facetplots
                        # precip
                        plt.figure(0)
                        pr = ds_rcp_hz_med["precip"]
                        pr.attrs.update(
                            long_name="Precipitation Change (median over GCMs)", units="%"
                        )
                        g = pr.plot(x="lon", y="lat", col="month", col_wrap=3)
                        g.set_axis_labels("longitude [degree east]", "latitude [degree north]")
                        plt.savefig(
                            os.path.join(
                                clim_project_dir,
                                "plots",
                                f"gridded_monthly_precipitation_change_{rcp}_{hz}-future-horizon.png",
                            )
                        )
                        # temp
                        plt.figure(1)
                        tas = ds_rcp_hz_med["temp"]
                        tas.attrs.update(
                            long_name="Temperature Change (median over GCMs)", units="degC"
                        )
                        g = tas.plot(x="lon", y="lat", col="month", col_wrap=3)
                        g.set_axis_labels("longitude [degree east]", "latitude [degree north]")
                        plt.savefig(
                            os.path.join(
                                clim_project_dir,
                                "plots",
                                f"gridded_monthly_temperature_change_{rcp}_{hz}-future-horizon.png",
                            )
                        )

                        # Average maps
                        grids = ds_rcp_hz_med.mean(dim="month")
                        plt.style.use("seaborn-v0_8-whitegrid")  # set nice style
                        # we assume the model maps are in the geographic CRS EPSG:4326
                        proj = ccrs.PlateCarree()
                        # adjust zoomlevel and figure size to your basis size & aspect
                        zoom_level = 8
                        figsize = (10, 8)

                        # precip
                        pr = grids["precip"]
                        # minmax = max(abs(np.amin(pr.values)), np.amax(pr.values))
                        # divnorm=colors.TwoSlopeNorm(vmin=-minmax, vcenter=0., vmax=minmax)

                        # initialize image with geoaxes
                        fig = plt.figure(figsize=figsize)
                        ax = fig.add_subplot(projection=proj)
                        extent = np.array(pr.raster.box.buffer(0.5).total_bounds)[[0, 2, 1, 3]]
                        ax.set_extent(extent, crs=proj)
                        # add sat background image
                        ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

                        # plot da variables.
                        pr.plot(
                            transform=proj,
                            ax=ax,
                            zorder=1,
                            cbar_kwargs=dict(
                                aspect=30,
                                shrink=0.8,
                                label="Precipitation Change (median over GCMs) [%]",
                            ),
                            cmap="bwr",
                        )  # norm=divnorm) # **kwargs)
                        ax.xaxis.set_visible(True)
                        ax.yaxis.set_visible(True)
                        ax.set_ylabel("latitude [degree north]")
                        ax.set_xlabel("longitude [degree east]")
                        _ = ax.set_title(
                            f"Annual mean precipitation change for {rcp} and time horizon {hz}"
                        )
                        plt.savefig(
                            os.path.join(
                                clim_project_dir,
                                "plots",
                                f"gridded_precipitation_change_{rcp}_{hz}-future-horizon.png",
                            ),
                            dpi=300,
                            bbox_inches="tight",
                        )

                        # temp
                        tas = grids["temp"]
                        minmax = max(abs(np.amin(tas.values)), np.amax(tas.values))
                        divnorm = colors.TwoSlopeNorm(vmin=-minmax, vcenter=0.0, vmax=minmax)

                        # initialize image with geoaxes
                        fig = plt.figure(figsize=figsize)
                        ax = fig.add_subplot(projection=proj)
                        extent = np.array(tas.raster.box.buffer(0.5).total_bounds)[[0, 2, 1, 3]]
                        ax.set_extent(extent, crs=proj)
                        # add sat background image
                        ax.add_image(cimgt.QuadtreeTiles(), zoom_level, alpha=0.5)

                        # plot da variables.
                        tas.plot(
                            transform=proj,
                            ax=ax,
                            zorder=1,
                            cbar_kwargs=dict(
                                aspect=30,
                                shrink=0.8,
                                label="Temperature Change (median over GCMs) [degC]",
                            ),
                            cmap="bwr",
                            norm=divnorm,
                        )  # **kwargs)
                        ax.xaxis.set_visible(True)
                        ax.yaxis.set_visible(True)
                        ax.set_ylabel("latitude [degree north]")
                        ax.set_xlabel("longitude [degree east]")
                        _ = ax.set_title(
                            f"Annual mean temperature change for {rcp} and time horizon {hz}"
                        )
                        plt.savefig(
                            os.path.join(
                                clim_project_dir,
                                "plots",
                                f"gridded_temperature_change_{rcp}_{hz}-future-horizon.png",
                            ),
                            dpi=300,
                            bbox_inches="tight",
                        )
    else:
        raise RuntimeError(
            "plot_proj_timeseries.py runs only as a Snakemake script:"
        )
