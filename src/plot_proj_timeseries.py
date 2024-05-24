# -*- coding: utf-8 -*-
"""
Plots expected change in climate variables based on GCM projections
"""
import os
from os.path import join
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


def plot_climate_projections(
    nc_historical: List[Union[str, Path]],
    nc_future: List[Union[str, Path]],
    path_output: Union[str, Path],
    scenarios: List[str],
    horizons: List[str],
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
    path_output: Union[str, Path]
        Path to the output directory
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

    # check if pet data is present
    has_pet = "pet" in ds_hist.data_vars

    # convert to df and compute anomalies
    print("Computing historical gcm timeseries anomalies")
    # precip
    gcm_pr = ds_hist["precip"].squeeze(drop=True).transpose().to_pandas()
    # check if gcm_pr_anom is pd.Series or pd.DataFrame
    if isinstance(gcm_pr, pd.Series):
        gcm_pr = gcm_pr.to_frame()

    # monthly mean
    gcm_pr_mnmn = gcm_pr.groupby(gcm_pr.index.month).mean()
    q_pr_mnmn = gcm_pr_mnmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    gcm_pr_mnref = gcm_pr_mnmn.mean()
    gcm_pr_mnanom = (gcm_pr_mnmn - gcm_pr_mnref) / gcm_pr_mnref * 100
    q_pr_mnanom = gcm_pr_mnanom.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    # annual mean
    gcm_pr_annmn = gcm_pr.resample("YE").mean()
    q_pr_annmn = gcm_pr_annmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    gcm_pr_ref = gcm_pr_annmn.mean()
    gcm_pr_anom = (gcm_pr_annmn - gcm_pr_ref) / gcm_pr_ref * 100
    q_pr_anom = gcm_pr_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose()

    # temp
    gcm_tas = ds_hist["temp"].squeeze(drop=True).transpose().to_pandas()
    # check if gcm_pr_anom is pd.Series or pd.DataFrame
    if isinstance(gcm_tas, pd.Series):
        gcm_tas = gcm_tas.to_frame()

    # monthly mean
    gcm_tas_mnmn = gcm_tas.groupby(gcm_tas.index.month).mean()
    q_tas_mnmn = gcm_tas_mnmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    gcm_tas_mnref = gcm_tas_mnmn.mean()
    gcm_tas_mnanom = gcm_tas_mnmn - gcm_tas_mnref
    q_tas_mnanom = gcm_tas_mnanom.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    # annual mean
    gcm_tas_annmn = gcm_tas.resample("YE").mean()
    q_tas_annmn = gcm_tas_annmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
    gcm_tas_ref = gcm_tas_annmn.mean()
    gcm_tas_anom = gcm_tas_annmn - gcm_tas_ref
    q_tas_anom = gcm_tas_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose()

    # pet
    if has_pet:
        gcm_pet = ds_hist["pet"].squeeze(drop=True).transpose().to_pandas()
        # check if gcm_pr_anom is pd.Series or pd.DataFrame
        if isinstance(gcm_pet, pd.Series):
            gcm_pet = gcm_pet.to_frame()

        # monthly mean
        gcm_pet_mnmn = gcm_pet.groupby(gcm_pet.index.month).mean()
        q_pet_mnmn = gcm_pet_mnmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
        gcm_pet_mnref = gcm_pet_mnmn.mean()
        gcm_pet_mnanom = (gcm_pet_mnmn - gcm_pet_mnref) / gcm_pet_mnref * 100
        q_pet_mnanom = gcm_pet_mnanom.quantile([0.05, 0.5, 0.95], axis=1).transpose()
        # annual mean
        gcm_pet_annmn = gcm_pet.resample("YE").mean()
        q_pet_annmn = gcm_pet_annmn.quantile([0.05, 0.5, 0.95], axis=1).transpose()
        gcm_pet_ref = gcm_pet_annmn.mean()
        gcm_pet_anom = (gcm_pet_annmn - gcm_pet_ref) / gcm_pet_ref * 100
        q_pet_anom = gcm_pet_anom.quantile([0.05, 0.5, 0.95], axis=1).transpose()

    # 2. Future timeseries
    # remove files containing empty dataset
    fns_future = nc_future.copy()
    for fn in nc_future:
        ds = xr.open_dataset(fn, lock=False)
        if len(ds) == 0 or ds is None:
            fns_future.remove(fn)
        ds.close()

    # Initialise list of future df per rcp/scenario
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
    for i in range(len(scenarios)):
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
    if has_pet:
        pet_fut = []
        anom_pet_fut = []
        qanom_pet_fut = []
        qpet_fut = []
        qpet_futmonth = []
        qpet_futmonth_sum = []
        qpet_futmonth_anom = []
        qpet_fut_abs = []
        for i in range(len(scenarios)):
            pet_fut.append([])
            anom_pet_fut.append([])
            qanom_pet_fut.append([])
            qpet_fut.append([])
            qpet_futmonth.append([])
            qpet_futmonth_sum.append([])
            qpet_futmonth_anom.append([])
            qpet_fut_abs.append([])

    # read files
    for i in range(len(scenarios)):
        print(f"Opening future gcm timeseries for rcp {scenarios[i]}")
        fns_rcp = [fn for fn in fns_future if scenarios[i] in fn]
        ds_rcp = xr.open_mfdataset(
            fns_rcp, preprocess=todatetimeindex_dropvars, lock=False
        )
        ds_fut.append(ds_rcp)
        ds_rcp_pr = ds_rcp["precip"].squeeze(drop=True)
        ds_rcp_tas = ds_rcp["temp"].squeeze(drop=True)
        if has_pet:
            ds_rcp_pet = ds_rcp["pet"].squeeze(drop=True)

        # to dataframe
        prfi = ds_rcp_pr.transpose().to_pandas()
        if isinstance(prfi, pd.Series):
            prfi = prfi.to_frame()
        pr_fut[i] = prfi
        tasfi = ds_rcp_tas.transpose().to_pandas()
        if isinstance(tasfi, pd.Series):
            tasfi = tasfi.to_frame()
        tas_fut[i] = tasfi
        if has_pet:
            petfi = ds_rcp_pet.transpose().to_pandas()
            if isinstance(petfi, pd.Series):
                petfi = petfi.to_frame()
            pet_fut[i] = petfi

    # compute anomalies
    print("Computing future gcm timeseries anomalies")
    fut_pr_ref = gcm_pr_annmn.mean()
    fut_tas_ref = gcm_tas_annmn.mean()
    if has_pet:
        fut_pet_ref = gcm_pet_annmn.mean()

    # monthly
    for i in range(len(qpr_futmonth)):
        pr_futmonth = pr_fut[i].groupby(pr_fut[i].index.month).mean()
        qpr_futmonth[i] = pr_futmonth.quantile([0.05, 0.5, 0.95], axis=1).transpose()
        pr_futmonth_anom = (pr_futmonth - fut_pr_ref) / fut_pr_ref * 100
        qpr_futmonth_anom[i] = (
            pr_futmonth_anom.dropna(axis=1, how="all")
            .quantile([0.05, 0.5, 0.95], axis=1)
            .transpose()
        )

        tas_futmonth = tas_fut[i].groupby(tas_fut[i].index.month).mean()
        qtas_futmonth[i] = tas_futmonth.quantile([0.05, 0.5, 0.95], axis=1).transpose()
        tas_futmonth_anom = tas_futmonth - fut_tas_ref
        qtas_futmonth_anom[i] = (
            tas_futmonth_anom.dropna(axis=1, how="all")
            .quantile([0.05, 0.5, 0.95], axis=1)
            .transpose()
        )

        if has_pet:
            pet_futmonth = pet_fut[i].groupby(pet_fut[i].index.month).mean()
            qpet_futmonth[i] = pet_futmonth.quantile(
                [0.05, 0.5, 0.95], axis=1
            ).transpose()
            pet_futmonth_anom = (pet_futmonth - fut_pet_ref) / fut_pet_ref * 100
            qpet_futmonth_anom[i] = (
                pet_futmonth_anom.dropna(axis=1, how="all")
                .quantile([0.05, 0.5, 0.95], axis=1)
                .transpose()
            )
    # annual
    for i in range(len(anom_pr_fut)):
        qpr_fut[i] = (
            pr_fut[i]
            .resample("YE")
            .mean()
            .quantile([0.05, 0.5, 0.95], axis=1)
            .transpose()
        )
        anom_pr_fut[i] = (
            (pr_fut[i].resample("YE").mean() - fut_pr_ref) / fut_pr_ref * 100
        )
        qanom_pr_fut[i] = anom_pr_fut[i].quantile([0.05, 0.5, 0.95], axis=1).transpose()

        qtas_fut[i] = (
            tas_fut[i]
            .resample("YE")
            .mean()
            .quantile([0.05, 0.5, 0.95], axis=1)
            .transpose()
        )
        anom_tas_fut[i] = tas_fut[i].resample("YE").mean() - fut_tas_ref
        qanom_tas_fut[i] = (
            anom_tas_fut[i].quantile([0.05, 0.5, 0.95], axis=1).transpose()
        )

        if has_pet:
            qpet_fut[i] = (
                pet_fut[i]
                .resample("YE")
                .mean()
                .quantile([0.05, 0.5, 0.95], axis=1)
                .transpose()
            )
            anom_pet_fut[i] = (
                (pet_fut[i].resample("YE").mean() - fut_pet_ref) / fut_pet_ref * 100
            )
            qanom_pet_fut[i] = (
                anom_pet_fut[i].quantile([0.05, 0.5, 0.95], axis=1).transpose()
            )

    # 3. Merge and write all timeseries to a single netcdf file
    ds_fut.append(ds_hist)
    ds_all = xr.merge(ds_fut)
    # make sure we have two digits still
    for var in ds_all.data_vars:
        ds_all[var] = ds_all[var].round(decimals=2)
    # write to netcdf
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    ds_all.to_netcdf(join(path_output, "gcm_timeseries.nc"))

    # 4. Plots scalar timeseries

    # Precipitation absolute change
    plot_scalar_anomaly(
        data_hist=q_pr_annmn,
        data_fut=[data for data in qpr_fut],
        scenario_names=scenarios,
        title="Annual precipitation anomaly",
        y_label="Anomaly (mm/year)",
        monthly=False,
        figure_filename=join(
            path_output, "plots", "precipitation_anomaly_projections_abs.png"
        ),
    )

    # Precipitation anomaly change
    plot_scalar_anomaly(
        data_hist=q_pr_anom,
        data_fut=qanom_pr_fut,
        scenario_names=scenarios,
        title="Annual precipitation anomaly",
        y_label="Anomaly (%)",
        monthly=False,
        figure_filename=join(
            path_output, "plots", "precipitation_anomaly_projections_anom.png"
        ),
    )

    # Temperature absolute change
    plot_scalar_anomaly(
        data_hist=q_tas_annmn,
        data_fut=qtas_fut,
        scenario_names=scenarios,
        title="Average annual temperature anomaly",
        y_label="Anomaly ($\degree$C)",
        monthly=False,
        figure_filename=join(
            path_output, "plots", "temperature_anomaly_projections_abs.png"
        ),
    )

    # Temperature anomaly change
    plot_scalar_anomaly(
        data_hist=q_tas_anom,
        data_fut=qanom_tas_fut,
        scenario_names=scenarios,
        title="Average annual temperature anomaly",
        y_label="Anomaly ($\degree$C)",
        monthly=False,
        figure_filename=join(
            path_output, "plots", "temperature_anomaly_projections_anom.png"
        ),
    )

    if has_pet:
        # PET absolute change
        plot_scalar_anomaly(
            data_hist=q_pet_annmn,
            data_fut=[data for data in qpet_fut],
            scenario_names=scenarios,
            title="Annual potential evapotranspiration anomaly",
            y_label="Anomaly (mm/year)",
            monthly=False,
            figure_filename=join(
                path_output, "plots", "pet_anomaly_projections_abs.png"
            ),
        )

        # PET anomaly change
        plot_scalar_anomaly(
            data_hist=q_pet_anom,
            data_fut=qanom_pet_fut,
            scenario_names=scenarios,
            title="Annual potential evapotranspiration anomaly",
            y_label="Anomaly (%)",
            monthly=False,
            figure_filename=join(
                path_output, "plots", "pet_anomaly_projections_anom.png"
            ),
        )

    # Precipitation absolute change monthly
    plot_scalar_anomaly(
        data_hist=q_pr_mnmn,
        data_fut=qpr_futmonth,
        scenario_names=scenarios,
        title="Average precipitation",
        y_label="mm/month",
        monthly=True,
        figure_filename=join(
            path_output, "plots", "precipitation_monthly_projections_abs.png"
        ),
    )
    # Precipitation anomaly change monthly
    plot_scalar_anomaly(
        data_hist=q_pr_mnanom,
        data_fut=qpr_futmonth_anom,
        scenario_names=scenarios,
        title="Average precipitation anomaly",
        y_label="Anomaly (%)",
        monthly=True,
        figure_filename=join(
            path_output, "plots", "precipitation_monthly_projections_anom.png"
        ),
    )

    # Temperature absolute change monthly
    plot_scalar_anomaly(
        data_hist=q_tas_mnmn,
        data_fut=qtas_futmonth,
        scenario_names=scenarios,
        title="Average monthly temperature",
        y_label="$\degree$C",
        monthly=True,
        figure_filename=join(
            path_output, "plots", "temperature_monthly_projections_abs.png"
        ),
    )
    # Temperature anomaly change monthly
    plot_scalar_anomaly(
        data_hist=q_tas_mnanom,
        data_fut=qtas_futmonth_anom,
        scenario_names=scenarios,
        title="Average monthly temperature anomaly",
        y_label="Anomaly ($\degree$C)",
        monthly=True,
        figure_filename=join(
            path_output, "plots", "temperature_monthly_projections_anom.png"
        ),
    )

    if has_pet:
        # PET absolute change monthly
        plot_scalar_anomaly(
            data_hist=q_pet_mnmn,
            data_fut=qpet_futmonth,
            scenario_names=scenarios,
            title="Average potential evapotranspiration",
            y_label="mm/month",
            monthly=True,
            figure_filename=join(
                path_output, "plots", "pet_monthly_projections_abs.png"
            ),
        )
        # PET anomaly change monthly
        plot_scalar_anomaly(
            data_hist=q_pet_mnanom,
            data_fut=qpet_futmonth_anom,
            scenario_names=scenarios,
            title="Average potential evapotranspiration anomaly",
            y_label="Anomaly (%)",
            monthly=True,
            figure_filename=join(
                path_output, "plots", "pet_monthly_projections_anom.png"
            ),
        )

    # 5. Map plots of gridded change per scenario / horizon
    if nc_grid_projections is not None:
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
            lats = ds.lat.values
            lons = ds.lon.values
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
        if has_pet:
            vmin_m_pet = ds_rcp_hz_med["pet"].min().values
            vmax_m_pet = ds_rcp_hz_med["pet"].max().values
        # Average maps over the months
        ds_rcp_hz_med_mean = ds_rcp_hz_med.mean(dim="month")
        vmin_pr = ds_rcp_hz_med_mean["precip"].min().values
        vmax_pr = ds_rcp_hz_med_mean["precip"].max().values
        vmin_tas = ds_rcp_hz_med_mean["temp"].min().values
        vmax_tas = ds_rcp_hz_med_mean["temp"].max().values
        if has_pet:
            vmin_pet = ds_rcp_hz_med_mean["pet"].min().values
            vmax_pet = ds_rcp_hz_med_mean["pet"].max().values

        # Save the merged factors on the "common" grid
        ds_rcp_hz_med.to_netcdf(join(path_output, "gcm_grid_factors_025.nc"))

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
                        path_output,
                        "plots",
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
                        path_output,
                        "plots",
                        f"gridded_monthly_temperature_change_{sc}_{hz}-future-horizon.png",
                    ),
                )
                if has_pet:
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
                            path_output,
                            "plots",
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
                        path_output,
                        "plots",
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
                        path_output,
                        "plots",
                        f"gridded_temperature_change_{sc}_{hz}-future-horizon.png",
                    ),
                )

                # pet
                if has_pet:
                    plot_gridded_anomaly(
                        da=ds_rcp_hz_med_mean["pet"].sel(scenario=sc, horizon=hz),
                        title=f"Annual mean potential evapotranspiration change for {sc} and time horizon {hz}",
                        legend="Potential Evapotranspiration Change (median over GCMs) [%]",
                        vmin=vmin_pet,
                        vmax=vmax_pet,
                        cmap="RdYlBu_r",
                        use_diverging_cmap=True,
                        figure_filename=join(
                            path_output,
                            "plots",
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
            path_output=sm.params.clim_project_dir,
            scenarios=sm.params.scenarios,
            horizons=sm.params.horizons.keys(),
            nc_grid_projections=nc_grid_projections,
        )

    else:
        print("This script is intended to be run with snakemake.")
