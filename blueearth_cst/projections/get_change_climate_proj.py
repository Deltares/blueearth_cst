# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:34:58 2022

@author: bouaziz
"""

import hydromt
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from blueearth_cst.shared.snake_utils import log_row

# %%


def intersection(lst1, lst2):
    """Shared members of two sequences, in a DETERMINISTIC (sorted) order.

    ``sorted`` is load-bearing, not cosmetic. This function drives the variable
    iteration order in ``get_change_annual_clim_proj`` /
    ``get_change_scalar_clim_proj``, which becomes the **column order** of
    ``annual_change_scalar_stats_summary{,_mean}.csv``. Returning a bare
    ``list(set(...) & set(...))`` made that order depend on string hash
    randomization, so the summary CSVs flipped ``temp``/``precip`` between runs
    with no change in values — harmless to consumers, which read by label, but it
    made the two manifested CSV rows unreproducible (``check_baseline``
    fingerprints them by sha256 of the bytes).

    Alphabetical rather than "preserve the first sequence's order": it is
    self-contained, so the guarantee does not silently depend on upstream
    dataset-variable ordering. On the seed config the two coincide anyway —
    ``variables: [precip, temp]``.
    """
    return sorted(set(lst1) & set(lst2))


def _to_datetime_index(ds):
    """Convert an object-dtype (cftime) time index to a proleptic-Gregorian
    DatetimeIndex.

    CMIP6-native calendars (360_day, noleap) decode to an object-dtype
    CFTimeIndex that cannot be sliced with the ``pd.Timestamp`` bounds built in
    ``get_change_annual_clim_proj`` -- the cross-calendar comparison raises
    ``TypeError``. The stats time series here are monthly (``MS``, day-01), which
    has no calendar-invalid dates, so the conversion is lossless for the annual
    resample that follows. Mirrors ``plot_proj_timeseries.py``. (t260720c)
    """
    if "time" in ds.coords and ds.indexes["time"].dtype == "O":
        ds = ds.copy()
        ds["time"] = ds.indexes["time"].to_datetimeindex()
    return ds


def get_change_clim_projections(ds_hist, ds_clim):
    """
    Parameters
    ----------
    ds_hist : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for historical climate simulation.
    ds_clim : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for projected climate data.

    Returns
    -------
    Writes netcdf files with mean monthly (12 maps) change for the grid.
    Also writes scalar mean monthly values averaged over the grid.

    Returns
    -------
    monthly_change_mean_grid : xarray dataset
        mean monthly change over the grid.
    monthly_change_mean_scalar : xarray dataset
        mean monthly change averaged over the grid.

    """
    ds = []
    for var in intersection(ds_hist.data_vars, ds_clim.data_vars):
        if var == "precip":
            # multiplicative for precip
            change = (
                (
                    ds_clim[var]
                    - ds_hist[var].sel(
                        scenario=ds_hist.scenario.values[0],
                    )
                )
                / ds_hist[var].sel(
                    scenario=ds_hist.scenario.values[0],
                )
                * 100
            )
        else:  # for temp
            # additive for temp
            change = ds_clim[var] - ds_hist[var].sel(
                scenario=ds_hist.scenario.values[0]
            )
        ds.append(change.to_dataset())

    monthly_change_mean_grid = xr.merge(ds)

    return monthly_change_mean_grid


def get_change_annual_clim_proj(
    ds_hist_time,
    ds_clim_time,
    stats=["mean", "std", "var", "median", "q_90", "q_75", "q_10", "q_25"],
    start_month_hyd_year="Jan",
):
    """

    Parameters
    ----------
    ds_hist_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over the grid (historical).
    ds_clim_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over the grid (projection).
    stats : list of strings of statistics
        quantiles should be provided as q_xx. The default is ["mean", "std", "var", "median", "q_90", "q_75", "q_10", "q_25"]
    start_month_hyd_year : str, optional
        Month start of hydrological year. The default is "Jan".

    Returns
    -------
    stats_annual_change : xarray dataset
        annual statistics per each models/scenario/horizon.

    """
    # cftime-safe slicing: convert any CMIP6-native cftime index up front so the
    # pd.Timestamp hydrological-year bounds below apply to a pandas-native index
    # (t260720c / D-CAL).
    ds_hist_time = _to_datetime_index(ds_hist_time)
    ds_clim_time = _to_datetime_index(ds_clim_time)

    # Fail loud on asymmetric hist/clim structure rather than silently dropping
    # (t260720d / D-VAR, D-MEM). Otherwise intersection() below quietly discards
    # a configured variable, and xarray's default inner alignment on the change
    # arithmetic quietly discards an unshared member -- both move the response
    # surface with no error.
    hist_vars = set(ds_hist_time.data_vars)
    clim_vars = set(ds_clim_time.data_vars)
    if hist_vars != clim_vars:
        unshared_vars = sorted(hist_vars.symmetric_difference(clim_vars))
        raise ValueError(
            f"asymmetric hist/clim variables: {unshared_vars} present in only "
            f"one dataset (hist={sorted(hist_vars)}, clim={sorted(clim_vars)})"
        )
    if "member" in ds_hist_time.coords and "member" in ds_clim_time.coords:
        hist_mem = set(ds_hist_time["member"].values.tolist())
        clim_mem = set(ds_clim_time["member"].values.tolist())
        if hist_mem != clim_mem:
            unshared_mem = sorted(hist_mem.symmetric_difference(clim_mem))
            raise ValueError(
                f"asymmetric hist/clim members: {unshared_mem} present in only "
                f"one dataset (hist={sorted(hist_mem)}, clim={sorted(clim_mem)})"
            )

    ds = []
    for var in intersection(ds_hist_time.data_vars, ds_clim_time.data_vars):
        # only keep full hydrological years
        start_hyd_year_hist = pd.to_datetime(
            f"{ds_hist_time['time.year'][0].values}-{start_month_hyd_year}"
        )
        end_hyd_year_hist = pd.to_datetime(
            f"{ds_hist_time['time.year'][-1].values}-{start_month_hyd_year}"
        ) - pd.DateOffset(months=1)
        # same for clim
        start_hyd_year_clim = pd.to_datetime(
            f"{ds_clim_time['time.year'][0].values}-{start_month_hyd_year}"
        )
        end_hyd_year_clim = pd.to_datetime(
            f"{ds_clim_time['time.year'][-1].values}-{start_month_hyd_year}"
        ) - pd.DateOffset(months=1)

        if var == "precip":
            # multiplicative for precip
            hist = (
                ds_hist_time[var]
                .sel(time=slice(start_hyd_year_hist, end_hyd_year_hist))
                .resample(time=f"YS-{start_month_hyd_year.upper()[:3]}")
                .sum("time")
                .sel(
                    scenario=ds_hist_time.scenario.values[0],
                )
            )
            clim = (
                ds_clim_time[var]
                .sel(time=slice(start_hyd_year_clim, end_hyd_year_clim))
                .resample(time=f"YS-{start_month_hyd_year.upper()[:3]}")
                .sum("time")
            )
            # change = (ds_clim[var] - ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0])) / ds_hist[var].sel(horizon = ds_hist.horizon.values[0], scenario = ds_hist.scenario.values[0]) * 100
        else:  # for temp
            # additive for temp
            hist = (
                ds_hist_time[var]
                .sel(time=slice(start_hyd_year_hist, end_hyd_year_hist))
                .resample(time=f"YS-{start_month_hyd_year.upper()[:3]}")
                .mean("time")
                .sel(
                    scenario=ds_hist_time.scenario.values[0],
                )
            )
            clim = (
                ds_clim_time[var]
                .sel(time=slice(start_hyd_year_clim, end_hyd_year_clim))
                .resample(time=f"YS-{start_month_hyd_year.upper()[:3]}")
                .mean("time")
            )

        # calc statistics
        for stat_name in stats:  # , stat_props in stats_dic.items():
            if "q_" in stat_name:
                qvalue = int(stat_name.split("_")[1]) / 100
                hist_stat = getattr(hist, "quantile")(qvalue, "time")
                clim_stat = getattr(clim, "quantile")(qvalue, "time")
            else:
                hist_stat = getattr(hist, stat_name)("time")
                clim_stat = getattr(clim, stat_name)("time")

            if var == "precip":
                change = (clim_stat - hist_stat) / hist_stat * 100
            else:
                change = clim_stat - hist_stat
            change = change.assign_coords({"stats": stat_name}).expand_dims("stats")

            if "quantile" in change.coords:
                change = change.drop("quantile")
            ds.append(change.to_dataset())

    stats_annual_change = xr.merge(ds)
    return stats_annual_change


# Time tuples for comparison hist-fut.
# R01 schema delivers these as lists ([1980, 2010]) for both the historical
# window and the future horizons. Pre-R01 configs delivered them as
# comma-separated strings ("1980, 2010"). Accept both.
def _to_str_tuple(value):
    if isinstance(value, str):
        return tuple(map(str, value.split(", ")))
    return tuple(map(str, value))


# %%


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):

            # Snakemake options
            clim_project_dir = sm.params.clim_project_dir
            stats_path_hist = sm.params.stats_path_hist
            stats_path = sm.params.stats_path
            stats_time_nc_hist = sm.input.stats_time_nc_hist
            stats_time_nc = sm.input.stats_time_nc
            start_month_hyd_year = sm.params.start_month_hyd_year
            name_horizon = sm.params.name_horizon
            name_scenario = sm.params.name_scenario
            name_model = sm.params.name_model
            save_grids = sm.params.save_grids

            time_tuple_hist = _to_str_tuple(sm.params.time_horizon_hist)
            time_tuple_fut = _to_str_tuple(sm.params.time_horizon_fut)

            # open datasets and slice times
            ds_hist_time = xr.open_dataset(stats_time_nc_hist)
            ds_clim_time = xr.open_dataset(stats_time_nc)

            # Get names of grids if save_grids
            if save_grids:
                # open datasets
                ds_hist = xr.open_dataset(stats_path_hist)
                ds_clim = xr.open_dataset(stats_path)

            # get lat lon name of data
            XDIMS = ("x", "longitude", "lon", "long")
            YDIMS = ("y", "latitude", "lat")
            for dim in XDIMS:
                if dim in ds_hist_time.coords:
                    x_dim = dim
            for dim in YDIMS:
                if dim in ds_hist_time.coords:
                    y_dim = dim

            # #only calc statistics if netcdf is filled (for snake all the files are made, even dummy when no data)
            # if len(ds_clim_time) > 0:

            #     #if time format is CFTimeIndex, convert to DatetimeIndex
            #     #(NB: This may lead to subtle errors in operations that depend on the length of time between dates. However needed if we want to have the possibility to resample over hydrological years with "AS-Mon")
            #     if ds_hist_time.indexes['time'].dtype == "O":
            #         datetimeindex_hist = ds_hist_time.indexes['time'].to_datetimeindex()
            #         ds_hist_time['time'] = datetimeindex_hist
            #         #same for clim
            #         datetimeindex_clim = ds_clim_time.indexes['time'].to_datetimeindex()
            #         ds_clim_time['time'] = datetimeindex_clim

            #     #calculate change
            #     monthly_change_mean_grid, monthly_change_mean_scalar = get_change_clim_projections(ds_hist, ds_clim)

            #     #write to netcdf files
            #     strings = ["monthly_change_mean_grid", "monthly_change_mean_scalar"]
            #     for i, ds in enumerate([monthly_change_mean_grid, monthly_change_mean_scalar]):
            #         print(f"writing netcdf files {strings[i]}")
            #         dvars = ds.raster.vars
            #         name_model = ds.model.values[0]
            #         name_scenario = ds.scenario.values[0]
            #         name_horizon = ds.horizon.values[0]
            #         name_nc_out = f"{strings[i]}-{name_model}_{name_scenario}_{name_horizon}.nc"
            #         ds.to_netcdf(os.path.join(clim_project_dir, name_nc_out), encoding={k: {"zlib": True} for k in dvars})

            # %% get annual statistics from time series of monthly variables

            # only calc statistics if netcdf is filled (for snake all the files are made, even dummy when no data)
            # create dummy netcdf otherwise as this is the file snake is checking:

            if len(ds_clim_time) > 0:
                ds_hist_time = ds_hist_time.sel(time=slice(*time_tuple_hist))
                ds_clim_time = ds_clim_time.sel(time=slice(*time_tuple_fut))
                # calculate statistics (mean, std, 0.1 0.25 0.50 0.75 0.90 quantiles of annual precip sum and mean temp)
                stats_annual_change = get_change_annual_clim_proj(ds_hist_time, ds_clim_time)
                # add time horizon coords
                stats_annual_change = stats_annual_change.assign_coords(
                    {
                        "horizon": f"{name_horizon}",
                    }
                ).expand_dims(["horizon"])
                # Reorder dims
                stats_annual_change = stats_annual_change.transpose(
                    ..., "clim_project", "model", "scenario", "horizon", "member"
                )

                # write to netcdf files
                dvars = stats_annual_change.raster.vars
                name_model = stats_annual_change.model.values[0]
                name_scenario = stats_annual_change.scenario.values[0]
                name_horizon = stats_annual_change.horizon.values[0]
                name_nc_out = (
                    f"annual_change_scalar_stats-{name_model}_{name_scenario}_{name_horizon}.nc"
                )
                stats_annual_change.to_netcdf(
                    os.path.join(clim_project_dir, name_nc_out),
                    encoding={k: {"zlib": True} for k in dvars},
                )

            else:  # create a dummy netcdf
                name_nc_out = (
                    f"annual_change_scalar_stats-{name_model}_{name_scenario}_{name_horizon}.nc"
                )
                ds_dummy = xr.Dataset()
                ds_dummy.to_netcdf(os.path.join(clim_project_dir, name_nc_out))

            if save_grids:
                if len(ds_clim) > 0:
                    # calculate change
                    monthly_change_mean_grid = get_change_clim_projections(ds_hist, ds_clim)
                    # add time horizon coords
                    monthly_change_mean_grid = monthly_change_mean_grid.assign_coords(
                        {
                            "horizon": f"{name_horizon}",
                        }
                    ).expand_dims(["horizon"])
                    # Reorder dims
                    stats_annual_change = stats_annual_change.transpose(
                        ..., "clim_project", "model", "scenario", "horizon", "member"
                    )

                    # write to netcdf files
                    log_row("writing netcdf files monthly_change_mean_grid", module="change")
                    dvars = monthly_change_mean_grid.raster.vars
                    name_model = monthly_change_mean_grid.model.values[0]
                    name_scenario = monthly_change_mean_grid.scenario.values[0]
                    name_horizon = monthly_change_mean_grid.horizon.values[0]
                    name_nc_out = (
                        f"monthly_change_mean_grid-{name_model}_{name_scenario}_{name_horizon}.nc"
                    )
                    monthly_change_mean_grid.to_netcdf(
                        os.path.join(clim_project_dir, name_nc_out),
                        encoding={k: {"zlib": True} for k in dvars},
                    )
                else:  # create a dummy netcdf
                    name_nc_out = (
                        f"monthly_change_mean_grid-{name_model}_{name_scenario}_{name_horizon}.nc"
                    )
                    ds_dummy = xr.Dataset()
                    ds_dummy.to_netcdf(os.path.join(clim_project_dir, name_nc_out))
    else:
        raise RuntimeError(
            "get_change_climate_proj.py runs only as a Snakemake script:"
        )
