import matplotlib.pyplot as plt
import numpy as np
import os

from hydromt import flw
import pandas as pd
from hydromt_wflow import WflowModel
import xarray as xr

from typing import Union, List, Optional
from pathlib import Path

__all__ = ["get_upstream_clim_basin", "determine_budyko_curve_terms"]


def get_upstream_clim_basin(
    qobs: xr.DataArray,
    wflow_root: Union[str, Path],
    wflow_config_fn: str = "wflow_sbm.toml",
    timestep: int = 86400,
    start_hydro_year: str = "JAN",
    missing_days_threshold: int = 330,
):
    """
    Returns mean daily and annual precipitation, potential evaporation and specific discharge for the basin upstream of the observation station.

    Parameters
    ----------

    qobs : xr.DataArray
        Geodataframe with discharge observation timeseries.
        Should include index for the station id and geometry for the x and y coordinate.
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    wflow_config_fn : str, optional
        Name of the wflow configuration file, by default "wflow_sbm.toml". Used to read
        the right results files from the wflow model.
    timestep: int, optional
        timestep of the model in s. By default 86400.
    start_hydro_year : str
        The start month (abreviated to the first three letters of the month,
        starting with a capital letter) of the hydrological year.
        The default is "JAN".
    missing_days_threshold: int, optional
        Minimum number of days within a year for that year to be counted in the
        long-term Budyko analysis.

    Returns
    -------

    ds_clim_sub : xr.Dataset
        Dataset containing per subcatchment the daily precipitation,
        potential evaporation and specific discharge sums.

    ds_clim_sub_annual : xr.Dataset
        Dataset containing per subcatchment the annual precipitation,
        potential evaporation and specific discharge sums.

    """
    # open model
    mod = WflowModel(root=wflow_root, mode="r", config_fn=wflow_config_fn)
    ds_grid = mod.grid
    # flwdir (use mod.flwdir directly instead of pyflwdir function!!)
    flwdir = mod.flwdir

    # initiate gdf
    gdf_basins = pd.DataFrame()

    # Loop over basins and get per gauge location a polygon of the upstream
    # area.
    x_all = []
    y_all = []
    for i, id in enumerate(qobs.index.values):
        x, y = qobs.vector.geometry.x.values[i], qobs.vector.geometry.y.values[i]
        ds_basin_single = flw.basin_map(
            ds_grid,
            flwdir,
            ids=id,
            xy=(x, y),
            stream=ds_grid["wflow_river"],
        )[0]
        x_all.append(x)
        y_all.append(y)

        ds_basin_single.name = int(qobs.index[i].values)
        ds_basin_single.raster.set_crs(ds_grid.raster.crs)
        gdf_basin_single = ds_basin_single.raster.vectorize()
        gdf_basins = pd.concat([gdf_basins, gdf_basin_single])
    # Set index to catchment id
    gdf_basins.index = gdf_basins.value.astype("int")

    # add basin area to calculate specific runoff [mm/d]
    # Add the catchment area to gdf_basins and sort in a descending order
    # make sure to also snap to river when retrieving areas
    idxs_gauges = flwdir.snap(xy=(x_all, y_all), mask=ds_grid["wflow_river"].values)[0]
    areas_uparea = ds_grid["wflow_uparea"].values.flat[idxs_gauges]
    df_areas = pd.DataFrame(
        index=qobs.index.values, data=areas_uparea * 1e6, columns=["area"]
    )
    gdf_basins = pd.concat([gdf_basins, df_areas], axis=1)
    gdf_basins = gdf_basins.sort_values(by="area", ascending=False)
    # drop basins where area is NaN.
    gdf_basins = gdf_basins[(gdf_basins["area"] >= 0)]

    # zonal stats:
    ds_clim_grid = xr.merge([mod.forcing["precip"], mod.forcing["pet"]])
    ds_clim_sub = ds_clim_grid.raster.zonal_stats(gdf_basins, stats=["mean"])
    print("Computing zonal statistics for clim, this can take a while")
    ds_clim_sub = ds_clim_sub.compute()

    # add area
    qobs = qobs.assign_coords(
        area=("index", list(gdf_basins["area"][qobs.index.values].values))
    )

    # calc qobs in [mm/d]
    qobs = qobs.to_dataset()
    qobs = qobs.assign(specific_Q=qobs["Q"] / qobs.area * timestep * 1000.0)

    # merge with clim
    ds_clim_sub = xr.merge([qobs, ds_clim_sub])

    # annual sum
    ds_clim_sub_annual = (
        ds_clim_sub[["specific_Q", "precip_mean", "pet_mean"]]
        .resample(time=f"YS-{start_hydro_year}")
        .sum("time", skipna=True, min_count=missing_days_threshold)
        .dropna("time")
    )

    return ds_clim_sub, ds_clim_sub_annual


def determine_budyko_curve_terms(
    ds_sub_annual: xr.Dataset,
):
    """
    Determine the long term average Budyko terms (aridity index, discharge coefficient and evaporation index).

    Parameters
    ----------
    ds_sub_annual : xr.Dataset
        Dataset containing per subcatchment the annual precipitation,
        potential evaporation and specific discharge sums.
        Required variables: ["specific_Q", "precip_mean", "pet_mean"]
    Returns
    -------
    ds_sub_annual : xr.Dataset
        Similar to input, but containing the discharge coefficient, aridity
        index and the evaporative index as long term averages.

    """
    ds_sub_annual["discharge_coeff"] = (
        ds_sub_annual["specific_Q"] / ds_sub_annual["precip_mean"]
    ).mean("time", skipna=True)
    ds_sub_annual["aridity_index"] = (
        ds_sub_annual["pet_mean"] / ds_sub_annual["precip_mean"]
    ).mean("time", skipna=True)
    ds_sub_annual["evap_index"] = 1 - ds_sub_annual["discharge_coeff"]

    return ds_sub_annual


def plot_budyko(
    ds_clim_sub_annual: xr.Dataset,
    Folder_out: Union[Path, str],
    omega: float = 2.0,
    color: dict = {"climate_source": "steelblue"},
    fs: int = 8,
):
    """
    Plots of long-term average discharge coefficient as a function of the long term aridity index in the Budyko framework for different climate sources.
    Budyko is based on the Zhang formula with omega as in Teuling et al 2019 (https://hess.copernicus.org/articles/23/3631/2019/)

    Parameters
    ----------
    ds_clim_sub_annual : xr.DataSet
        Long term aridity_index and discharge_coefficient
    Folder_out : Union[Path, str]
        Output folder to save plots.
    omega : float, optional
        Default omega parameter value in the Zhang formula, by default 2
    color : dict, optional
        Color belonging to a climate source run, by default {"climate_source":"steelblue"}
    fs : int, optional
        Font size, by default 7
    """

    xxx = np.arange(0, 1.4, 0.0001)

    plt.figure(figsize=(12 / 2.54, 12 / 2.54))
    for climate_source in ds_clim_sub_annual.climate_source.values:
        plt.plot(
            ds_clim_sub_annual["aridity_index"].sel(climate_source=climate_source),
            ds_clim_sub_annual["discharge_coeff"].sel(climate_source=climate_source),
            marker="o",
            linestyle="None",
            label=f"{climate_source}",
            color=color[climate_source],
        )
        for i in ds_clim_sub_annual.index.values:
            plt.text(
                ds_clim_sub_annual["aridity_index"].sel(
                    climate_source=climate_source, index=i
                ),
                ds_clim_sub_annual["discharge_coeff"].sel(
                    climate_source=climate_source, index=i
                ),
                f"{i}",
                fontsize=fs,
            )

    # plot budyko Q/P as a function of Ep/P
    plt.plot(
        xxx,
        1 - 1 / (1 + xxx ** (-omega)) ** (1 / omega),
        color="k",
        linewidth=1.2,
        label="Budyko",
    )
    # energy limit
    plt.plot([0, 1], [1, 0], color="red", linestyle="--", label="Energy limit")
    # ganing catchments
    plt.plot([0, 1.4], [1, 1], color="darkblue", linestyle="--", label="Water limit")

    plt.text(0.4, 1.25, "Gaining catchments", fontsize=fs, color="darkblue")
    plt.text(0.4, 0.12, "Leaking catchments", fontsize=fs, color="red")
    plt.text(0.8, 0.5, "'Feasible' domain", fontsize=fs, color="black")

    plt.xlabel("aridity E$_{P}$/P", fontsize=fs)
    plt.ylabel("runoff coefficient Q$_{obs}$/P", fontsize=fs)

    plt.tick_params(axis="both", labelsize=fs)

    plt.legend(fontsize=fs, loc=2)
    plt.xlim([0, 1.5])
    plt.ylim([0, 2.1])

    plt.savefig(os.path.join(Folder_out, f"budyko_qobs.png"), dpi=300)
