# -*- coding: utf-8 -*-
"""
Plot wflow results and compare to observations if any
"""

import xarray as xr
import numpy as np
import os
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import hydromt
from hydromt.gis import GeoDataset
from hydromt.readers import open_timeseries_from_table, open_vector
from hydromt_wflow import WflowSbmModel

from typing import Union

from blueearth_cst.shared.func_plot_signature import (
    plot_signatures,
    plot_hydro,
    compute_metrics,
    plot_clim,
    plot_basavg,
)
from blueearth_cst.climate_analysis.subcatchment_climate import climate_forcing_by_subcatchment
from blueearth_cst.model.climate_parity import model_parity_climate
from blueearth_cst.shared.snake_utils import log_row


def _log(message):
    """Emit one standard-format log row (module tag ``plot``) for this rule."""
    log_row(message, module="plot")


def analyse_wflow_historical(
    project_dir: Path,
    observations_fn: Union[Path, str] = None,
    gauges_locs: Union[Path, str] = None,
    climate_nc: Union[Path, str] = None,
    oro_nc: Union[Path, str] = None,
    data_sources: Union[Path, str] = None,
    clim_source: str = "era5",
):
    """
    Analyse and plot wflow model performance for historical run.

    To be read, model should be stored in ``project_dir``/hydrology_model.
    Model results should include the discharge keys Q_outlets and, if gauges are
    provided, Q_gauges_{basename(gauges_locs)}. The climate plots (P/T/EP) are
    derived from the RAW gridded climate extraction (``climate_nc``, rule 1.10)
    brought to model parity (the build's own regrid + corrections + PET method)
    and aggregated per subcatchment — decoupled from the model's stored inmaps
    forcing artifact (P3-2a re-source; supersedes the ADR-0002 interim
    coupling) and not from wflow outputs.


    Outputs:

    - plot of hydrographs at the outlet(s) and gauges_locs if provided. If wflow run is
      three years or less, only the daily hydrograph will be plotted. If wflow run is
      longer than three years, plots will also include the yearly hydrograph, the
      monthly average and hyddrographs for the wettest and driest years. If observations
      are available, they are added as well.
    - plot of signature plots if wflow run is longer than a year and if observations
      are available.
    - plot of climate data per year and per month at the subcatchment level if wflow run
      is longer than a year.
    - plot of basin average outputs (e.g. soil moisture, snow, etc.). The variables to
      include should have the postfix _basavg in the wflow output file.
    - compute performance metrics (daily and monthly KGE, NSE, NSElog, RMSE, MSE, Pbias,
      VE) if observations are available and if wflow run is longer than a year. Metrics
      are saved to a csv file.

    Parameters
    ----------
    project_dir : Path
        path to CST project directory
    observations_fn : Union[Path, str], optional
        Path to observations timeseries file, by default None
        Required columns: time, wflow_id IDs of the locations as in ``gauges_locs``.
        Separator is , and decimal is .
    gauges_locs : Union[Path, str], optional
        Path to gauges/observations locations file, by default None
        Required columns: wflow_id, station_name, x, y.
        Values in wflow_id column should match column names in ``observations_fn``.
        Separator is , and decimal is .
    climate_nc : Union[Path, str], optional
        Path to the wf1 raw climate extraction
        (``climate_historical/wf1_raw/extract_historical.nc``). When absent the
        climate plots are skipped.
    oro_nc : Union[Path, str], optional
        chirps/chirps_global only: the extraction's orography sidecar (the
        declared rule-1.10 ``oro_nc`` output). Mandatory on that branch — it is
        the DEM the parity transform must reference to invert the extraction's
        embedded lapse correction (design ext1-1). None on era5, where the
        orography comes from the data catalog instead.
    data_sources : Union[Path, str], optional
        Data catalog(s) for the era5-branch orography fetch.
    clim_source : str
        ``clim_historical`` source name (era5 / chirps / chirps_global; eobs is
        excluded from this path at DAG-parse time).
    """
    ### 1. Prepare output and plotting options ###

    # Create output folders
    Folder_plots = f"{project_dir}/plots/wflow_model_performance"

    if not os.path.isdir(Folder_plots):
        os.mkdir(Folder_plots)

    # Plotting options
    fs = 7
    lw = 0.8

    # Other plot options
    label = "simulated"  # "observed"
    color = "steelblue"  # "red"
    linestyle = "-"
    marker = "o"

    ### 2. Read the observations ###
    # check if user provided observations
    has_observations = False

    if observations_fn is not None and os.path.exists(observations_fn):
        has_observations = True

        # Read
        gdf_obs = open_vector(gauges_locs, crs=4326, sep=",")
        da_ts_obs = open_timeseries_from_table(
            observations_fn, name="Q", index_dim="wflow_id", sep=";"
        )
        ds_obs = GeoDataset.from_gdf(gdf_obs, da_ts_obs, merge_index="inner")
        # Rename wflow_id to index
        ds_obs = ds_obs.rename({"wflow_id": "index"})
        qobs = ds_obs["Q"].load()

    ### 3. Read the wflow model and results ###
    # Instantiate wflow model
    Folder_run = f"{project_dir}/hydrology_model"
    mod = WflowSbmModel(root=Folder_run, mode="r")
    mod.output_csv.read()
    results = mod.output_csv.data
    geoms = mod.geoms.data

    # Discharge at the outlet(s) (was Q_gauges in 0.x; now Q_outlets).
    qsim = results["Q_outlets"].rename("Q")
    # In hydromt_wflow 1.x outlet ids come from the subcatchment map
    # (e.g. 130000086) instead of the 1..N counter that 0.x used. Keep the
    # 1..N station_name so rule_all and downstream plots stay stable.
    qsim = qsim.assign_coords(
        station_name=(
            "index",
            [f"wflow_{i + 1}" for i in range(qsim["index"].size)],
        )
    )
    # Map subcatchment id -> positional station_name (wflow_1..N) so the climate
    # plots share the discharge station labels instead of raw subcatchment ids.
    station_by_id = dict(zip(qsim["index"].values, qsim["station_name"].values))
    # Discharge at the gauges_locs if present
    if gauges_locs is not None and os.path.exists(gauges_locs):
        gauges_output_name = os.path.basename(gauges_locs).split(".")[0]
        if f"Q_gauges_{gauges_output_name}" in results:
            qsim_gauges = results[f"Q_gauges_{gauges_output_name}"].rename("Q")
            gdf_gauges = (
                geoms[f"gauges_{gauges_output_name}"]
                .rename(columns={"wflow_id": "index"})
                .set_index("index")
            )
            qsim_gauges = qsim_gauges.assign_coords(
                station_name=(
                    "index",
                    list(gdf_gauges["station_name"][qsim_gauges.index.values].values),
                )
            )
            qsim = xr.merge([qsim, qsim_gauges])["Q"]

    # Climate data (P/T/EP) per subcatchment, derived from the RAW gridded
    # climate extracted for the basin over the historical window (rule 1.10),
    # brought to model parity — the build's own regrid + lapse/pressure
    # corrections + PET method — so the plots show the climate that actually
    # drives the model while staying decoupled from the stored inmaps forcing
    # artifact (P3-2a re-source; supersedes the ADR-0002 interim coupling).
    # Model-artifact loading (staticmaps variables) stays model-side here; the
    # aggregation itself takes plain xarray objects (criterion C1). Skip
    # cleanly if the extraction is absent so the hydrograph plot still ships.
    _parity_vars = ("precip", "temp", "press_msl", "kin", "kout")
    if climate_nc is not None and os.path.exists(climate_nc):
        ds_raw = xr.open_dataset(climate_nc)
    else:
        ds_raw = xr.Dataset()
    if ds_raw.data_vars and all(v in ds_raw for v in _parity_vars):
        dem_model = mod.staticmaps.data["land_elevation"]
        # setup_time_horizon.py mapping: eobs -> (eobs_orography, makkink) is
        # excluded from this path at DAG-parse time (ext2-3); every supported
        # source maps to (era5_orography, debruin).
        pet_method = "debruin"
        if oro_nc is not None:
            # chirps/chirps_global: the extraction's sidecar DEM, received as
            # the declared rule input — never discovered as a sibling file
            # (ext2-1). meteo.temp's MSL-shift algebra inverts the correction
            # the extraction embedded against this same DEM (ext1-1).
            dem_forcing = xr.open_dataarray(oro_nc)
        else:
            # era5: fetch the orography source exactly as the forcing build
            # does (wflow_sbm.py setup_temp_pet_forcing).
            data_catalog = hydromt.DataCatalog(data_libs=data_sources)
            dem_forcing = data_catalog.get_rasterdataset(
                "era5_orography",
                geom=ds_raw.raster.box,  # clip dem with forcing bbox for full coverage
                buffer=2,
                variables=["elevtn"],
            ).squeeze()
        ds_parity = model_parity_climate(ds_raw, dem_model, dem_forcing, pet_method)
        ds_clim = climate_forcing_by_subcatchment(
            ds_parity, mod.staticmaps.data["subcatchment"]
        )
    else:
        ds_clim = xr.Dataset()

    basavg_vars = [dvar for dvar in results if "_basavg" in dvar]
    if basavg_vars:
        ds_basin = xr.merge([results[dvar] for dvar in basavg_vars]).squeeze(drop=True)
        if "precipitation_basavg" in ds_basin:
            ds_basin = ds_basin.drop_vars("precipitation_basavg")
    else:
        ds_basin = xr.Dataset()

    ### 4. Plot climate data ###
    # Two distinct skip reasons — keep them separate so the log is honest:
    #  (a) ds_clim empty: no raw climate extraction (or it lacks the parity
    #      input variables), so there is nothing to aggregate. This is NOT a
    #      data-length condition — the discharge run may span many years — so
    #      it must never be reported as "less than 1 year".
    #  (b) climate present but shorter than a year: skip the yearly plots.
    if "time" not in ds_clim.dims:
        _log("No raw climate extraction available; skipping climate plots.")
    elif len(ds_clim.time) < 365:
        _log(
            "Less than 1 year of climate data is available; "
            "no yearly climate plots are made."
        )
    else:
        for index in ds_clim.index.values:
            # Positional station label (wflow_1..N) to match the discharge plots;
            # fall back to the raw id if a subcatchment has no discharge station.
            station = station_by_id.get(index, f"wflow_{index}")
            _log(f"Plot climatic data at {station}")
            ds_clim_i = ds_clim.sel(index=index)
            # Plot per year
            plot_clim(ds_clim_i, Folder_plots, station, "year")
            plt.close()
            # Plot per month
            plot_clim(ds_clim_i, Folder_plots, station, "month")
            plt.close()

    ### 5. Plot other basin average outputs ###
    if ds_basin.data_vars:
        _log("Plot basin average wflow outputs")
        plot_basavg(ds_basin, Folder_plots)
        plt.close()
    else:
        _log("No basin-average outputs configured; skipping plot_basavg.")

    ### 6. Plot hydrographs and compute performance metrics ###
    # Initialise the output performance table
    df_perf_all = pd.DataFrame()
    # Flag for plot signatures
    # (True if wflow run is longer than a year and observations are available)
    do_signatures = False

    # If possible, skip the first year of the wflow run (warm-up period)
    if len(qsim.time) > 365:
        _log("Skipping the first year of the wflow run (warm-up period)")
        qsim = qsim.sel(
            time=slice(
                f"{qsim['time.year'][0].values+1}-{qsim['time.month'][0].values}-{qsim['time.day'][0].values}",
                None,
            )
        )
        if has_observations:
            do_signatures = True
    else:
        _log("Simulation is less than a year so model warm-up period will be plotted.")
    # Sel qsim and qobs so that they have the same time period
    if has_observations:
        start = max(qsim.time.values[0], qobs.time.values[0])
        end = min(qsim.time.values[-1], qobs.time.values[-1])
        qsim = qsim.sel(time=slice(start, end))
        qobs = qobs.sel(time=slice(start, end))

    # Loop over the stations
    for station_id, station_name in zip(qsim.index.values, qsim.station_name.values):
        # Select the station
        qsim_i = qsim.sel(index=station_id)
        qobs_i = None
        if has_observations:
            if station_id in qobs.index.values:
                qobs_i = qobs.sel(index=station_id)

        # a) Plot hydrographs
        _log(f"Plot hydrographs at wflow station {station_name}")
        plot_hydro(
            qsim=qsim_i,
            qobs=qobs_i,
            Folder_out=Folder_plots,
            station_name=station_name,
            label=label,
            color=color,
            lw=lw,
            fs=fs,
        )
        plt.close()
        # b) Signature plot and performance metrics
        if do_signatures and qobs_i is not None:
            _log("observed timeseries are available - making signature plots.")
            # Plot signatures
            plot_signatures(
                qsim=qsim_i,
                qobs=qobs_i,
                Folder_out=Folder_plots,
                station_name=station_name,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                fs=fs,
                lw=lw,
            )
            plt.close()
            # Compute performance metrics
            df_perf = compute_metrics(
                qsim=qsim_i,
                qobs=qobs_i,
                station_name=station_name,
            )
            # Join with other stations
            if df_perf_all.empty:
                df_perf_all = df_perf
            else:
                df_perf_all = df_perf_all.join(df_perf)
        else:
            _log(
                "observed timeseries are not available " "no signature plots are made."
            )

    # Save performance metrics to csv
    df_perf_all.to_csv(os.path.join(Folder_plots, "performance_metrics.csv"))

    ### End of the function ###


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            analyse_wflow_historical(
                project_dir=sm.params.project_dir,
                observations_fn=sm.params.observations_file,
                gauges_locs=sm.params.gauges_output_path,
                climate_nc=sm.input.climate_nc,
                # declared only on the chirps/chirps_global branch (ext2-1)
                oro_nc=getattr(sm.input, "oro_nc", None),
                data_sources=sm.params.data_sources,
                clim_source=sm.params.clim_source,
            )
    else:
        analyse_wflow_historical(
            project_dir=join(os.getcwd(), "test_case", "my_project"),
            observations_fn=None,
            gauges_locs=None,
        )
