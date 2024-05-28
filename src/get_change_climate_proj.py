"""Compare historical or future climate from a GCM model for a specific time horizon."""

import os
from os.path import join, dirname
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

from typing import List, Tuple, Union


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def get_change_clim_projections(
    ds_hist: xr.Dataset,
    ds_clim: xr.Dataset,
    name_horizon: str = "future",
    drymonth_threshold: float = 3.0,
    drymonth_maxchange: float = 50.0,
):
    """
    Parameters
    ----------
    ds_hist : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for
        historical climate simulation.
    ds_clim : xarray dataset
        Mean monthly values of variables (precip and temp) over the grid (12 maps) for
        projected climate data.
    name_horizon : str, optional
        Name of the horizon to select in ds_clim in case several are available.
        The default is "future".
    drymonth_threshold : float, optional
        Threshold for dry month definition in mm/month. For too dry months, the change
        factors will be limited to ``drymonth_maxchange`` is avoid to avoid too large
        change factors. The default is 3.0 mm/month.
    drymonth_maxchange : float, optional
        Maximum change factor for dry months (% change). The default is +-50.0%.

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
    # Select the horizons
    if "horizon" in ds_hist.dims:
        ds_hist = ds_hist.sel(horizon="historical")
    if "horizon" in ds_clim.dims:
        ds_clim = ds_clim.sel(horizon=name_horizon)
    for var in intersection(ds_hist.data_vars, ds_clim.data_vars):
        if var == "precip" or var == "pet":
            ds_hist_var = ds_hist[var].sel(scenario=ds_hist.scenario.values[0])
            # multiplicative for precip and pet
            change = (ds_clim[var] - ds_hist_var) / ds_hist_var * 100
            # Dry month
            if var == "precip":
                # Add a min limit
                change = change.where(
                    np.invert(
                        np.logical_and(
                            ds_hist_var <= drymonth_threshold,
                            change <= -drymonth_maxchange,
                        )
                    ),
                    -drymonth_maxchange,
                )
                # Add a max limit
                change = change.where(
                    np.invert(
                        np.logical_and(
                            ds_hist_var <= drymonth_threshold,
                            change >= drymonth_maxchange,
                        )
                    ),
                    drymonth_maxchange,
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
    ds_hist_time: xr.Dataset,
    ds_clim_time: xr.Dataset,
    stats: List[str] = ["mean", "std", "var", "median", "q_90", "q_75", "q_10", "q_25"],
    start_month_hyd_year: str = "JAN",
):
    """
    Calculate changes between future and historical climate for several statistics.

    Parameters
    ----------
    ds_hist_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over
        the grid (historical).
    ds_clim_time : xarray dataset
        monthly averages of variables over time horizon period, spatially averaged over
        the grid (projection).
    stats : list of strings of statistics
        quantiles should be provided as q_xx. The default is ["mean", "std", "var",
        "median", "q_90", "q_75", "q_10", "q_25"]
    start_month_hyd_year : str, optional
        Month start of hydrological year. The default is "JAN".

    Returns
    -------
    stats_annual_change : xarray dataset
        annual statistics per each models/scenario/horizon.

    """
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

        if var == "precip" or var == "pet":
            # multiplicative for precip and pet
            hist = (
                ds_hist_time[var]
                .sel(time=slice(start_hyd_year_hist, end_hyd_year_hist))
                .resample(time=f"YS-{start_month_hyd_year}")
                .sum("time")
                .sel(
                    scenario=ds_hist_time.scenario.values[0],
                )
            )
            clim = (
                ds_clim_time[var]
                .sel(time=slice(start_hyd_year_clim, end_hyd_year_clim))
                .resample(time=f"YS-{start_month_hyd_year}")
                .sum("time")
            )
        else:  # for temp
            # additive for temp
            hist = (
                ds_hist_time[var]
                .sel(time=slice(start_hyd_year_hist, end_hyd_year_hist))
                .resample(time=f"YS-{start_month_hyd_year}")
                .mean("time")
                .sel(
                    scenario=ds_hist_time.scenario.values[0],
                )
            )
            clim = (
                ds_clim_time[var]
                .sel(time=slice(start_hyd_year_clim, end_hyd_year_clim))
                .resample(time=f"YS-{start_month_hyd_year}")
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

            if var == "precip" or var == "pet":
                change = (clim_stat - hist_stat) / hist_stat * 100
            else:
                change = clim_stat - hist_stat
            change = change.assign_coords({"stats": stat_name}).expand_dims("stats")

            if "quantile" in change.coords:
                change = change.drop_vars("quantile")
            ds.append(change.to_dataset())

    stats_annual_change = xr.merge(ds)
    return stats_annual_change


def get_expected_change_scalar(
    nc_historical: Union[str, Path],
    nc_future: Union[str, Path],
    path_output: Union[str, Path],
    time_tuple_historical: Tuple[str, str] = ("1990", "2010"),
    time_tuple_future: Tuple[str, str] = ("2040", "2060"),
    start_month_hyd_year: str = "JAN",
    name_horizon: str = "future",
    name_model: str = "model",
    name_scenario: str = "scenario",
):
    """
    Compute the expected change in climate variables from point timeseries.

    Output is a netcdf file with the expected change in annual statistics.

    Parameters
    ----------
    nc_historical : Union[str, Path]
        Path to the historical timeseries netcdf file. Contains monthly timeseries.
        Supported variables: precip, temp.
        Required dimensions: time, model, scenario, member.
    nc_future : Union[str, Path]
        Path to the future timeseries netcdf file. Contains monthly timeseries.
        Supported variables: precip, temp.
        Required dimensions: time, model, scenario, member.
    path_output : Union[str, Path]
        Path to the output directory.
    time_tuple_historical : Tuple[str, str], optional
        Time horizon for historical data to slice in ``nc_historical``. The default is
        ("1990", "2010").
    time_tuple_future : Tuple[str, str], optional
        Time horizon for future data to slice in ``nc_future``. The default is
        ("2040", "2060").
    start_month_hyd_year : str, optional
        Month start of hydrological year. The default is "JAN".
    name_horizon : str, optional
        Name of the horizon. The default is "future". Will be added as an extra
        dimension in the output netcdf file.
    name_model : str, optional
        Name of the model for the output filename. The default is "model".
    name_scenario : str, optional
        Name of the scenario for the output filename. The default is "scenario".
    """
    # Prepare the output filename and directory
    name_nc_out = (
        f"annual_change_scalar_stats-{name_model}_{name_scenario}_{name_horizon}.nc"
    )
    # Create output dir (model name can contain subfolders)
    dir_output = dirname(join(path_output, name_nc_out))
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # open datasets and slice times
    ds_hist = xr.open_dataset(nc_historical, lock=False)
    ds_fut = xr.open_dataset(nc_future, lock=False)

    # get annual statistics from time series of monthly variables

    # only calc statistics if netcdf is filled
    # (for snake all the files are made, even dummy when no data)
    # create dummy netcdf otherwise as this is the file snake is checking:

    if len(ds_fut) > 0:
        ds_hist = ds_hist.sel(time=slice(*time_tuple_historical))
        ds_fut = ds_fut.sel(time=slice(*time_tuple_future))
        # calculate statistics of annual precip sum and mean temp
        stats_annual_change = get_change_annual_clim_proj(
            ds_hist,
            ds_fut,
            stats=["mean", "std", "var", "median", "q_90", "q_75", "q_10", "q_25"],
            start_month_hyd_year=start_month_hyd_year,
        )
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

        # write netcdf
        stats_annual_change.to_netcdf(
            join(path_output, name_nc_out),
            encoding={k: {"zlib": True} for k in stats_annual_change.data_vars},
        )

    else:  # create a dummy netcdf
        ds_dummy = xr.Dataset()
        ds_dummy.to_netcdf(os.path.join(path_output, name_nc_out))


def get_expected_change_grid(
    nc_historical: Union[str, Path],
    nc_future: Union[str, Path],
    path_output: Union[str, Path],
    name_horizon: str = "future",
    name_model: str = "model",
    name_scenario: str = "scenario",
    drymonth_threshold: float = 3.0,
    drymonth_maxchange: float = 50.0,
):
    """
    Compute the expected change in climate variables from gridded timeseries.

    Output is a netcdf file with the expected gridded change in monthly statistics.

    Parameters
    ----------
    nc_historical : Union[str, Path]
        Path to the historical timeseries netcdf file. Contains monthly timeseries.
        Supported variables: precip, temp.
        Required dimensions: lat, lon, time, model, scenario, member.
    nc_future : Union[str, Path]
        Path to the future timeseries netcdf file. Contains monthly timeseries.
        Supported variables: precip, temp.
        Required dimensions: lat, lon, time, model, scenario, member.
    path_output : Union[str, Path]
        Path to the output directory.
    name_horizon : str, optional
        Name of the horizon. The default is "future". Will be added as an extra
        dimension in the output netcdf file.
    name_model : str, optional
        Name of the model for the output filename. The default is "model".
    name_scenario : str, optional
        Name of the scenario for the output filename. The default is "scenario".
    drymonth_threshold : float, optional
        Threshold for dry month definition in mm/month. For too dry months, the change
        factors will be limited to ``drymonth_maxchange`` is avoid to avoid too large
        change factors. The default is 3.0 mm/month.
    drymonth_maxchange : float, optional
        Maximum change factor for dry months (% change). The default is +-50.0%.
    """
    # Prepare the output filename and directory
    name_nc_out = f"{name_model}_{name_scenario}_{name_horizon}.nc"
    # Create output dir (model name can contain subfolders)
    dir_output = dirname(join(path_output, "monthly_change_grid", name_nc_out))
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    # open datasets
    ds_hist = xr.open_dataset(nc_historical, lock=False)
    ds_fut = xr.open_dataset(nc_future, lock=False)

    # Check if the future file is a dummy file
    if len(ds_fut) > 0:
        # calculate change
        monthly_change_mean_grid = get_change_clim_projections(
            ds_hist,
            ds_fut,
            name_horizon,
            drymonth_threshold=drymonth_threshold,
            drymonth_maxchange=drymonth_maxchange,
        )
        # add time horizon coords
        monthly_change_mean_grid = monthly_change_mean_grid.assign_coords(
            {
                "horizon": f"{name_horizon}",
            }
        ).expand_dims(["horizon"])
        # Reorder dims
        monthly_change_mean_grid = monthly_change_mean_grid.transpose(
            ..., "clim_project", "model", "scenario", "horizon", "member"
        )

        # write to netcdf files
        print(f"writing netcdf files monthly_change_grid")
        monthly_change_mean_grid.to_netcdf(
            join(path_output, "monthly_change_grid", name_nc_out),
            encoding={k: {"zlib": True} for k in monthly_change_mean_grid.data_vars},
        )
    else:  # create a dummy netcdf
        ds_dummy = xr.Dataset()
        ds_dummy.to_netcdf(join(path_output, "monthly_change_grid", name_nc_out))


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        # Snakemake options
        save_grids = sm.params.save_grids

        # Time tuples for comparison hist-fut
        time_tuple_hist = sm.params.time_horizon_hist
        time_tuple_hist = tuple(map(str, time_tuple_hist.split(", ")))
        time_tuple_fut = sm.params.time_horizon_fut
        time_tuple_fut = tuple(map(str, time_tuple_fut.split(", ")))

        get_expected_change_scalar(
            nc_historical=sm.input.stats_time_nc_hist,
            nc_future=sm.input.stats_time_nc,
            path_output=sm.params.clim_project_dir,
            time_tuple_historical=time_tuple_hist,
            time_tuple_future=time_tuple_fut,
            start_month_hyd_year=sm.params.start_month_hyd_year,
            name_horizon=sm.params.name_horizon,
            name_model=sm.params.name_model,
            name_scenario=sm.params.name_scenario,
        )

        if save_grids:
            get_expected_change_grid(
                nc_historical=sm.input.stats_grid_nc_hist,
                nc_future=sm.input.stats_grid_nc,
                path_output=sm.params.clim_project_dir,
                name_horizon=sm.params.name_horizon,
                name_model=sm.params.name_model,
                name_scenario=sm.params.name_scenario,
                drymonth_threshold=sm.params.change_drymonth_threshold,
                drymonth_maxchange=sm.params.change_drymonth_maxchange,
            )

    else:
        print("This script should be run from a snakemake environment.")
