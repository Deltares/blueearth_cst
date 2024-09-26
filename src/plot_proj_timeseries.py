# -*- coding: utf-8 -*-
"""
Plots expected change in climate variables based on GCM projections
"""
import os
from os.path import join, dirname
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from hydromt import raster

from typing import List, Union, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_projections import (
        plot_scalar_anomaly,
        plot_gridded_anomaly,
        plot_gridded_anomaly_month,
    )
else:
    from .plot_utils.plot_projections import (
        plot_scalar_anomaly,
        plot_gridded_anomaly,
        plot_gridded_anomaly_month,
    )

CLIM_PLOTS = {
    "precip": {
        "absolute": False,
        "resample": "sum",
        "unit_year": "mm/year",
        "unit_month": "mm/month",
        "unit_anomaly": "%",
        "long_name": "precipitation",
    },
    "temp": {
        "absolute": True,
        "resample": "mean",
        "unit_year": "$\degree$C",
        "unit_month": "$\degree$C",
        "unit_anomaly": "$\degree$C",
        "long_name": "temperature",
    },
    "pet": {
        "absolute": False,
        "resample": "sum",
        "unit_year": "mm/year",
        "unit_month": "mm/month",
        "unit_anomaly": "%",
        "long_name": "potential evapotranspiration",
    },
    "temp_dew": {
        "absolute": True,
        "resample": "mean",
        "unit_year": "$\degree$C",
        "unit_month": "$\degree$C",
        "unit_anomaly": "$\degree$C",
        "long_name": "dew point temperature",
    },
    "kin": {
        "absolute": False,
        "resample": "mean",
        "unit_year": "W/m2",
        "unit_month": "W/m2",
        "unit_anomaly": "%",
        "long_name": "incoming shortwave radiation",
    },
    "wind": {
        "absolute": False,
        "resample": "mean",
        "unit_year": "m/s",
        "unit_month": "m/s",
        "unit_anomaly": "%",
        "long_name": "wind speed",
    },
    "tcc": {
        "absolute": False,
        "resample": "mean",
        "unit_year": "frac",
        "unit_month": "frac",
        "unit_anomaly": "%",
        "long_name": "total cloud cover",
    },
}


# open historical datasets
def todatetimeindex_dropvars(ds):
    if "time" in ds.coords:
        if ds.indexes["time"].dtype == "O":
            ds["time"] = ds.indexes["time"].to_datetimeindex()
    if "spatial_ref" in ds.coords:
        ds = ds.drop_vars("spatial_ref")
    if "height" in ds.coords:
        ds = ds.drop_vars("height")
    return ds


def create_regular_grid(
    bbox: List[float], res: float, align: bool = True
) -> xr.Dataset:
    """
    Create a regular grid based on bounding box and resolution.

    Taken from hydromt.GridModel.setup_grid.
    Replace by HydroMT function when it will be moved to a workflow.
    """
    xmin, ymin, xmax, ymax = bbox

    # align to res
    if align:
        xmin = round(xmin / res) * res
        ymin = round(ymin / res) * res
        xmax = round(xmax / res) * res
        ymax = round(ymax / res) * res
    xcoords = np.linspace(
        xmin + res / 2,
        xmax - res / 2,
        num=round((xmax - xmin) / res),
        endpoint=True,
    )
    ycoords = np.flip(
        np.linspace(
            ymin + res / 2,
            ymax - res / 2,
            num=round((ymax - ymin) / res),
            endpoint=True,
        )
    )
    coords = {"lat": ycoords, "lon": xcoords}
    grid = raster.full(
        coords=coords,
        nodata=1,
        dtype=np.uint8,
        name="mask",
        attrs={},
        crs=4326,
        lazy=False,
    )
    grid = grid.to_dataset()

    return grid


def compute_anomalies(da_hist: xr.DataArray, ds_fut: List[xr.Dataset]):
    """
    Compute anomalies for a given variable and datasets.

    Parameters
    ----------
    da_hist: xr.DataArray
        Historical data array
    ds_fut: List[xr.Dataset]
        List of future datasets

    Returns
    -------
    q_mnmn: List[pd.DataFrame]
        List of quantiles for monthly mean over historical period
    q_mnanom: List[pd.DataFrame]
        List of quantiles for monthly mean anomalies over historical period
    q_annmn: List[pd.DataFrame]
        List of quantiles for annual mean over historical period
    q_anom: List[pd.DataFrame]
        List of quantiles for annual mean anomalies over historical period
    q_futmonth: List[pd.DataFrame]
        List of quantiles for monthly future mean
    q_futmonth_anom: List[pd.DataFrame]
        List of quantiles for monthly future mean anomalies
    q_fut: List[pd.DataFrame]
        List of quantiles for annual future mean
    q_fut_anom: List[pd.DataFrame]
        List of quantiles for annual future mean anomalies
    """
    # Variable
    var = da_hist.name

    # Initialise pandas lists
    df_fut = []
    q_futmonth = []
    q_futmonth_anom = []
    q_fut = []
    q_fut_anom = []

    # Convert to pandas
    gcm = da_hist.squeeze(drop=True).transpose().to_pandas()
    # check if gcm is pd.Series or pd.DataFrame
    if isinstance(gcm, pd.Series):
        gcm = gcm.to_frame()

    # monthly mean
    gcm_mnmn = gcm.groupby(gcm.index.month).mean()
    q_mnmn = gcm_mnmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    gcm_mnref = gcm_mnmn.mean()
    if CLIM_PLOTS[var]["absolute"]:
        gcm_mnanom = gcm_mnmn - gcm_mnref
    else:
        gcm_mnanom = (gcm_mnmn - gcm_mnref) / gcm_mnref * 100
    q_mnanom = gcm_mnanom.quantile([0.05, 0.5, 0.95], axis=1).transpose()

    # annual mean
    if CLIM_PLOTS[var]["resample"] == "sum":
        gcm_annmn = gcm.resample("YE").sum()
    else:
        gcm_annmn = gcm.resample("YE").mean()
    q_annmn = gcm_annmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    gcm_ref = gcm_annmn.mean()
    if CLIM_PLOTS[var]["absolute"]:
        gcm_anom = gcm_annmn - gcm_ref
    else:
        gcm_anom = (gcm_annmn - gcm_ref) / gcm_ref * 100
    q_anom = gcm_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose()

    # Loop over future datasets
    for ds in ds_fut:
        fi = ds[var].squeeze(drop=True).transpose().to_pandas()
        if isinstance(fi, pd.Series):
            fi = fi.to_frame()
        df_fut.append(fi)

    # Compute future anomalies
    fut_ref = gcm_annmn.mean()
    fut_mnref = gcm_mnmn.mean()

    for i in range(len(df_fut)):
        # monthly
        fut_month = df_fut[i].groupby(df_fut[i].index.month).mean()
        q_futmonth.append(fut_month.quantile([0.05, 0.5, 0.95], axis=1).transpose())
        if CLIM_PLOTS[var]["absolute"]:
            fut_month_anom = fut_month - fut_mnref
        else:
            fut_month_anom = (fut_month - fut_mnref) / fut_mnref * 100
        q_futmonth_anom.append(
            fut_month_anom.dropna(axis=1, how="all")
            .quantile([0.05, 0.5, 0.95], axis=1)
            .transpose()
        )

        # annual
        if CLIM_PLOTS[var]["resample"] == "sum":
            fut_year = df_fut[i].resample("YE").sum()
        else:
            fut_year = df_fut[i].resample("YE").mean()
        q_fut.append(fut_year.quantile([0.05, 0.5, 0.95], axis=1).transpose())
        if CLIM_PLOTS[var]["absolute"]:
            fut_anom = fut_year - fut_ref
        else:
            fut_anom = (fut_year - fut_ref) / fut_ref * 100
        q_fut_anom.append(fut_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose())

    return (
        q_mnmn,
        q_mnanom,
        q_annmn,
        q_anom,
        q_futmonth,
        q_futmonth_anom,
        q_fut,
        q_fut_anom,
    )


def plot_climate_projections(
    nc_historical: List[Union[str, Path]],
    nc_future: List[Union[str, Path]],
    path_output_nc: Union[str, Path],
    scenarios: List[str],
    horizons: List[str],
    path_output_plots: Optional[Union[str, Path]] = None,
    nc_grid_projections: Optional[List[Union[str, Path]]] = None,
):
    """
    Plot climate projections from GCMs.

    Output in ``path_output``:
    - gcm_timeseries.nc: all timeseries data
    - plots/precipitation_anomaly_projections_abs.png: precip absolute change
    - plots/precipitation_anomaly_projections_anom.png: precip anomaly change
    - plots/temperature_anomaly_projections_abs.png: temperature absolute change
    - plots/temperature_anomaly_projections_anom.png: temperature anomaly change

    Parameters
    ----------
    nc_historical: List[Union[str, Path]]
        List of historical netcdf scalar timeseries files
    nc_future: List[Union[str, Path]]
        List of future netcdf scalar timeseries files
    path_output_nc: Union[str, Path]
        Path to the output directory to save the netcdf files with combined gcm results.
    path_output_plots: Union[str, Path], optional
        Path to the output directory to save the plots. If None, plots will be saved in
        path_output_nc/plots. By default None.
    scenarios: List[str]
        List of scenarios. Should be part of the nc_future filenames and
        nc_grid_projections for selection in plots.
    horizons: List[str]
        List of horizons. Should be part of the nc_grid_projections for selection in
        plots.
    nc_grid_projections: Optional[Union[str, Path]], optional
        Path to the netcdf files of monthly change grids for plotting. By default None
        for no grid plots to make. Should contain the scenario and horizon in the
        filename.
    """
    # Output directory
    if path_output_plots is None:
        path_output_plots = join(path_output_nc, "plots")

    # 1. Historical timeseries
    print("Opening historical gcm timeseries")
    fns_hist = nc_historical.copy()
    for fn in nc_historical:
        ds = xr.open_dataset(fn, lock=False)
        if len(ds) == 0 or ds is None:
            fns_hist.remove(fn)
        ds.close()
    ds_hist = xr.open_mfdataset(
        fns_hist, preprocess=todatetimeindex_dropvars, lock=False
    )

    # 2. Future timeseries
    # remove files containing empty dataset
    fns_future = nc_future.copy()
    for fn in nc_future:
        ds = xr.open_dataset(fn, lock=False)
        if len(ds) == 0 or ds is None:
            fns_future.remove(fn)
        ds.close()

    # read files
    ds_fut = []
    for i in range(len(scenarios)):
        print(f"Opening future gcm timeseries for rcp {scenarios[i]}")
        fns_rcp = [fn for fn in fns_future if scenarios[i] in fn]
        ds_rcp = xr.open_mfdataset(
            fns_rcp, preprocess=todatetimeindex_dropvars, lock=False
        )
        ds_fut.append(ds_rcp)

    # 3. Merge and write all timeseries to a single netcdf file
    ds_all = xr.merge(ds_fut)
    ds_all = xr.merge([ds_all, ds_hist])
    # make sure we have two digits still
    for var in ds_all.data_vars:
        ds_all[var] = ds_all[var].round(decimals=2)
    # write to netcdf
    if not os.path.exists(path_output_nc):
        os.makedirs(path_output_nc)
    ds_all.to_netcdf(join(path_output_nc, "gcm_timeseries.nc"))

    # 4. Compute anomalies and plots scalar timeseries
    print("Computing anomalies and plotting scalar timeseries")
    for var in ds_hist.data_vars:
        if var not in CLIM_PLOTS:
            print(f"Variable {var} not supported, skipping plots")
            continue

        # Settings
        long_name = CLIM_PLOTS[var]["long_name"]
        unit_year = CLIM_PLOTS[var]["unit_year"]
        unit_month = CLIM_PLOTS[var]["unit_month"]
        unit_anomaly = CLIM_PLOTS[var]["unit_anomaly"]

        # Compute historic and future anomalies
        (
            q_mnmn,
            q_mnanom,
            q_annmn,
            q_anom,
            q_futmonth,
            q_futmonth_anom,
            q_fut,
            q_fut_anom,
        ) = compute_anomalies(ds_hist[var], ds_fut)

        # Absolute change
        plot_scalar_anomaly(
            data_hist=q_annmn,
            data_fut=[data for data in q_fut],
            scenario_names=scenarios,
            title=f"Annual {long_name} anomaly",
            y_label=f"Anomaly ({unit_year})",
            monthly=False,
            figure_filename=join(
                path_output_plots, f"{var}_anomaly_projections_abs.png"
            ),
        )
        # Anomaly change
        plot_scalar_anomaly(
            data_hist=q_anom,
            data_fut=q_fut_anom,
            scenario_names=scenarios,
            title=f"Annual {long_name} anomaly",
            y_label=f"Anomaly ({unit_anomaly})",
            monthly=False,
            figure_filename=join(
                path_output_plots, f"{var}_anomaly_projections_anom.png"
            ),
        )
        # Absolute change monthly
        plot_scalar_anomaly(
            data_hist=q_mnmn,
            data_fut=q_futmonth,
            scenario_names=scenarios,
            title=f"Average monthly {long_name}",
            y_label=f"{unit_month}",
            monthly=True,
            figure_filename=join(
                path_output_plots, f"{var}_monthly_projections_abs.png"
            ),
        )
        # Anomaly change monthly
        plot_scalar_anomaly(
            data_hist=q_mnanom,
            data_fut=q_futmonth_anom,
            scenario_names=scenarios,
            title=f"Average monthly {long_name} anomaly",
            y_label=f"Anomaly ({unit_anomaly})",
            monthly=True,
            figure_filename=join(
                path_output_plots, f"{var}_monthly_projections_anom.png"
            ),
        )

    # 5. Map plots of gridded change per scenario / horizon
    if nc_grid_projections is not None:
        print("Preparing gridded change map plots")
        # Create a regular grid of 0.25 * 0.25 degrees to reproject the models to
        # (most models are defined on their own grid...)
        sc = scenarios[0]
        hz = horizons[0]
        # Read for the first scenario and horizon to find the min / max lat / lon
        fns = [fn for fn in nc_grid_projections if sc in fn and hz in fn]
        ymax, ymin, xmax, xmin = None, None, None, None
        for fn in fns:
            ds = xr.open_dataset(fn, lock=False)
            if len(ds) == 0 or ds is None:
                continue
            lats = ds[ds.raster.y_dim].values
            lons = ds[ds.raster.x_dim].values
            ymin = min(ymin, np.min(lats)) if ymin is not None else np.min(lats)
            ymax = max(ymax, np.max(lats)) if ymax is not None else np.max(lats)
            xmin = min(xmin, np.min(lons)) if xmin is not None else np.min(lons)
            xmax = max(xmax, np.max(lons)) if xmax is not None else np.max(lons)
            ds.close()
        ds_grid = create_regular_grid(
            bbox=[xmin, ymin, xmax, ymax], res=0.25, align=True
        )

        # Loop over rcp and horizon
        ds_rcp_hz = []
        for sc in scenarios:
            for hz in horizons:
                print(f"Preparing change map plots for {sc} and horizon {hz}")
                fns_rcp_hz = [fn for fn in nc_grid_projections if sc in fn and hz in fn]
                for fn in fns_rcp_hz:
                    ds = xr.open_dataset(fn, lock=False)
                    if len(ds) == 0 or ds is None:
                        continue
                    if "time" in ds.coords:
                        if ds.indexes["time"].dtype == "O":
                            ds["time"] = ds.indexes["time"].to_datetimeindex()
                    # Reproject to regular grid
                    # drop extra dimensions for reprojection
                    ds_reproj = ds.squeeze(drop=True)
                    ds_reproj = ds_reproj.raster.reproject_like(
                        ds_grid, method="nearest"
                    )
                    # Re-add the extra dims
                    ds_reproj = ds_reproj.expand_dims(
                        {
                            "clim_project": ds["clim_project"].values,
                            "model": ds["model"].values,
                            "scenario": ds["scenario"].values,
                            "horizon": ds["horizon"].values,
                            "member": ds["member"].values,
                        }
                    )
                    ds_rcp_hz.append(ds_reproj)

        # Merge all datasets to find the total min and max values for color scaling
        ds_rcp_hz = xr.merge(ds_rcp_hz)
        # Compute the median over the models
        ds_rcp_hz_med = ds_rcp_hz.median(dim="model").squeeze(drop=True)
        vmin_m_pr = ds_rcp_hz_med["precip"].min().values
        vmax_m_pr = ds_rcp_hz_med["precip"].max().values
        vmin_m_tas = ds_rcp_hz_med["temp"].min().values
        vmax_m_tas = ds_rcp_hz_med["temp"].max().values
        if "pet" in ds_rcp_hz_med.data_vars:
            vmin_m_pet = ds_rcp_hz_med["pet"].min().values
            vmax_m_pet = ds_rcp_hz_med["pet"].max().values
        # Average maps over the months
        ds_rcp_hz_med_mean = ds_rcp_hz_med.mean(dim="month")
        vmin_pr = ds_rcp_hz_med_mean["precip"].min().values
        vmax_pr = ds_rcp_hz_med_mean["precip"].max().values
        vmin_tas = ds_rcp_hz_med_mean["temp"].min().values
        vmax_tas = ds_rcp_hz_med_mean["temp"].max().values
        if "pet" in ds_rcp_hz_med_mean.data_vars:
            vmin_pet = ds_rcp_hz_med_mean["pet"].min().values
            vmax_pet = ds_rcp_hz_med_mean["pet"].max().values

        # Save the merged factors on the "common" grid
        ds_rcp_hz_med.to_netcdf(join(path_output_nc, "gcm_grid_factors_025.nc"))

        # Reloop over the scenarios and horizons to plot the maps
        for sc in scenarios:
            for hz in horizons:
                # Facetplots
                # precip
                plot_gridded_anomaly_month(
                    da=ds_rcp_hz_med["precip"].sel(scenario=sc, horizon=hz),
                    title="Precipitation Change (median over GCMs)",
                    unit="%",
                    vmin=vmin_m_pr,
                    vmax=vmax_m_pr,
                    cmap="RdYlBu",
                    use_diverging_cmap=True,
                    figure_filename=join(
                        path_output_plots,
                        "grid",
                        f"gridded_monthly_precipitation_change_{sc}_{hz}-future-horizon.png",
                    ),
                )
                # temp
                plot_gridded_anomaly_month(
                    da=ds_rcp_hz_med["temp"].sel(scenario=sc, horizon=hz),
                    title="Temperature Change (median over GCMs)",
                    unit="degC",
                    vmin=vmin_m_tas,
                    vmax=vmax_m_tas,
                    cmap="RdYlBu_r",
                    use_diverging_cmap=True,
                    figure_filename=join(
                        path_output_plots,
                        "grid",
                        f"gridded_monthly_temperature_change_{sc}_{hz}-future-horizon.png",
                    ),
                )
                if "pet" in ds_rcp_hz_med.data_vars:
                    # pet
                    plot_gridded_anomaly_month(
                        da=ds_rcp_hz_med["pet"].sel(scenario=sc, horizon=hz),
                        title="Potential Evapotranspiration Change (median over GCMs)",
                        unit="%",
                        vmin=vmin_m_pet,
                        vmax=vmax_m_pet,
                        cmap="RdYlBu_r",
                        use_diverging_cmap=True,
                        figure_filename=join(
                            path_output_plots,
                            "grid",
                            f"gridded_monthly_pet_change_{sc}_{hz}-future-horizon.png",
                        ),
                    )

                # Average maps
                # precip
                plot_gridded_anomaly(
                    da=ds_rcp_hz_med_mean["precip"].sel(scenario=sc, horizon=hz),
                    title=f"Annual mean precipitation change for {sc} and time horizon {hz}",
                    legend="Precipitation Change (median over GCMs) [%]",
                    vmin=vmin_pr,
                    vmax=vmax_pr,
                    cmap="RdYlBu",
                    use_diverging_cmap=True,
                    figure_filename=join(
                        path_output_plots,
                        "grid",
                        f"gridded_precipitation_change_{sc}_{hz}-future-horizon.png",
                    ),
                )

                # temp
                plot_gridded_anomaly(
                    da=ds_rcp_hz_med_mean["temp"].sel(scenario=sc, horizon=hz),
                    title=f"Annual mean temperature change for {sc} and time horizon {hz}",
                    legend="Temperature Change (median over GCMs) [$\degree$C]",
                    vmin=vmin_tas,
                    vmax=vmax_tas,
                    cmap="RdYlBu_r",
                    use_diverging_cmap=True,
                    figure_filename=join(
                        path_output_plots,
                        "grid",
                        f"gridded_temperature_change_{sc}_{hz}-future-horizon.png",
                    ),
                )

                # pet
                if "pet" in ds_rcp_hz_med_mean.data_vars:
                    plot_gridded_anomaly(
                        da=ds_rcp_hz_med_mean["pet"].sel(scenario=sc, horizon=hz),
                        title=f"Annual mean potential evapotranspiration change for {sc} and time horizon {hz}",
                        legend="Potential Evapotranspiration Change (median over GCMs) [%]",
                        vmin=vmin_pet,
                        vmax=vmax_pet,
                        cmap="RdYlBu_r",
                        use_diverging_cmap=True,
                        figure_filename=join(
                            path_output_plots,
                            "grid",
                            f"gridded_pet_change_{sc}_{hz}-future-horizon.png",
                        ),
                    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        nc_grid_projections = sm.input.monthly_change_mean_grid
        if len(nc_grid_projections) == 0:
            nc_grid_projections = None

        plot_climate_projections(
            nc_historical=sm.input.stats_time_nc_hist,
            nc_future=sm.input.stats_time_nc,
            path_output_nc=dirname(sm.output.timeseries_nc),
            path_output_plots=dirname(sm.output.precip_plt),
            scenarios=sm.params.scenarios,
            horizons=list(sm.params.horizons.keys()),
            nc_grid_projections=nc_grid_projections,
        )

    else:
        print("This script is intended to be run with snakemake.")
