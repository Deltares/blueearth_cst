"""Utility functions for wflow models."""

import xarray as xr
import os
from os.path import join, dirname, basename
from pathlib import Path
from hydromt_wflow import WflowModel

from typing import Union, Tuple, Optional, List

__all__ = ["get_wflow_results", "get_wflow_results_delta"]


def get_wflow_results(
    wflow_root: Union[str, Path],
    config_fn: str = "wflow_sbm.toml",
    gauges_locs: Optional[Union[Path, str]] = None,
    remove_warmup: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Get wflow results as xarray.Dataset from the output csv file.

    Results can contain simulated discharges, simulated flux/states as basin averages
    and basin average forcing.

    Parameters
    ----------
    wflow_root : Union[str, Path]
        Path to the wflow model root folder.
    config_fn : str, optional
        Name of the wflow configuration file, by default "wflow_sbm.toml". Used to read
        the right results files from the wflow model.
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    remove_warmup : bool, optional
        Remove the first year of the simulation (warm-up), by default True

    Returns
    -------
    qsim: xr.Dataset
        simulated discharge at wflow basin locations and at observation locations
    ds_clim: xr.Dataset
        basin average precipitation, temperature and potential evaporation.
    ds_basin: xr.Dataset
        basin average flux and state variables

    """
    mod = WflowModel(
        root=wflow_root,
        mode="r",
        config_fn=config_fn,
    )

    # Q at wflow locations
    qsim = mod.results["Q_gauges"].rename("Q")
    # Add station name to the coordinates
    qsim = qsim.assign_coords(
        station_name=(
            "index",
            ["wflow_" + x for x in list(qsim["index"].values.astype(str))],
        )
    )

    # Discharge at the gauges_locs if present
    if gauges_locs is not None and os.path.exists(gauges_locs):
        # Get name of gauges dataset from the gauges locations file
        gauges_output_name = os.path.basename(gauges_locs).split(".")[0]
        if f"Q_gauges_{gauges_output_name}" in mod.results:
            qsim_gauges = mod.results[f"Q_gauges_{gauges_output_name}"].rename("Q")
            # Add station_name > bug for reading geoms if dir_input in toml is not None
            if f"gauges_{gauges_output_name}" not in mod.geoms:
                dir_geoms = dirname(
                    join(
                        mod.root,
                        mod.get_config("dir_input", abs_path=False),
                        mod.get_config("input.path_static", abs_path=False),
                    )
                )
                dir_geoms = join(dir_geoms, "staticgeoms")
                mod.read_geoms(dir_geoms)
            # Add station name
            gdf_gauges = (
                mod.geoms[f"gauges_{gauges_output_name}"]
                .rename(columns={"wflow_id": "index"})
                .set_index("index")
            )
            qsim_gauges = qsim_gauges.assign_coords(
                station_name=(
                    "index",
                    list(gdf_gauges["station_name"][qsim_gauges.index.values].values),
                )
            )
            # merge qsim and qsim_gauges
            qsim = xr.concat([qsim, qsim_gauges], dim="index")

    # Climate data P, EP, T for wflow_subcatch
    if "P_subcatchment" in mod.results:
        ds_clim = xr.merge(
            [
                mod.results["P_subcatchment"],
                mod.results["T_subcatchment"],
                mod.results["EP_subcatchment"],
            ]
        )
    else:
        ds_clim = None

    # Other catchment average outputs
    ds_basin = xr.merge(
        [mod.results[dvar] for dvar in mod.results if "_basavg" in dvar]
    )
    # glacier and other variables may have a different index value introducing nan's in ds_basin
    ds_basin = ds_basin.mean("index")
    ds_basin = ds_basin.squeeze(drop=True)
    # If precipitation, skip as this will be plotted with the other climate data
    if "precipitation_basavg" in ds_basin:
        ds_basin = ds_basin.drop_vars("precipitation_basavg")

    # Remove the first year (model warm-up for historical)
    if remove_warmup:
        qsim = qsim.sel(
            time=slice(
                f"{qsim['time.year'][0].values+1}-{qsim['time.month'][0].values}-{qsim['time.day'][0].values}",
                None,
            )
        )
        if ds_clim is not None:
            ds_clim = ds_clim.sel(
                time=slice(
                    f"{ds_clim['time.year'][0].values+1}-{ds_clim['time.month'][0].values}-{ds_clim['time.day'][0].values}",
                    None,
                )
            )
        ds_basin = ds_basin.sel(
            time=slice(
                f"{ds_basin['time.year'][0].values+1}-{ds_basin['time.month'][0].values}-{ds_basin['time.day'][0].values}",
                None,
            )
        )

    return qsim, ds_clim, ds_basin


def get_wflow_results_delta(
    delta_config_fns: List[str],
    gauges_locs: Optional[Union[Path, str]] = None,
    remove_warmup: bool = True,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Get several wflow results from delta runs and combine per horizon/model/scenario.

    Results can contain simulated discharges, simulated flux/states as basin averages
    and basin average forcing.

    Parameters
    ----------
    delta_config_fns : str, optional
        Name of the wflow configuration files of the delta runs. Used to read
        the right results files from the wflow model.
        The individual run filename should be organised as
        "*_model_scenario_horizon.toml".
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    remove_warmup : bool, optional
        Remove the first year of the simulation (warm-up), by default True

    Returns
    -------
    qsim_delta: xr.Dataset
        simulated discharge at wflow basin locations and at observation locations
    ds_clim_delta: xr.Dataset
        basin average precipitation, temperature and potential evaporation.
    ds_basin_delta: xr.Dataset
        basin average flux and state variables

    """
    qsim_delta = []
    ds_clim_delta = []
    ds_basin_delta = []
    for delta_config in delta_config_fns:
        model = basename(delta_config).split(".")[0].split("_")[-3]
        scenario = basename(delta_config).split(".")[0].split("_")[-2]
        horizon = basename(delta_config).split(".")[0].split("_")[-1]
        root = dirname(delta_config)
        config_fn = basename(delta_config)
        qsim_delta_run, ds_clim_delta_run, ds_basin_delta_run = get_wflow_results(
            root, config_fn, gauges_locs, remove_warmup=remove_warmup
        )
        qsim_delta_run = qsim_delta_run.assign_coords(
            {"horizon": horizon, "model": model, "scenario": scenario}
        ).expand_dims(["horizon", "model", "scenario"])
        ds_clim_delta_run = ds_clim_delta_run.assign_coords(
            {"horizon": horizon, "model": model, "scenario": scenario}
        ).expand_dims(["horizon", "model", "scenario"])
        ds_basin_delta_run = ds_basin_delta_run.assign_coords(
            {"horizon": horizon, "model": model, "scenario": scenario}
        ).expand_dims(["horizon", "model", "scenario"])
        qsim_delta.append(qsim_delta_run)
        ds_clim_delta.append(ds_clim_delta_run)
        ds_basin_delta.append(ds_basin_delta_run)
    qsim_delta = xr.merge(qsim_delta)
    ds_clim_delta = xr.merge(ds_clim_delta)
    ds_basin_delta = xr.merge(ds_basin_delta)

    return qsim_delta, ds_clim_delta, ds_basin_delta
