"""Compute statistics of delta change runs."""

import xarray as xr
import os
from os.path import join, dirname, basename
from pathlib import Path
import xclim

from typing import Union, List, Optional

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from wflow.wflow_utils import get_wflow_results, get_wflow_results_delta
    from plot_utils.plot_table_statistics import (
        plot_table_statistics_multiindex,
        plot_table_statistics,
    )
else:
    from .wflow.wflow_utils import get_wflow_results, get_wflow_results_delta
    from .plot_utils.plot_table_statistics import (
        plot_table_statistics_multiindex,
        plot_table_statistics,
    )

# Supported wflow outputs
WFLOW_VARS = {
    "overland flow": {
        "resample": "mean",
        "legend": "Overland Flow",
        "units": "m3/s",
    },
    "actual evapotranspiration": {
        "resample": "sum",
        "legend": "Actual Evapotranspiration",
        "units": "mm/yr",
    },
    "groundwater recharge": {
        "resample": "sum",
        "legend": "Groundwater Recharge",
        "units": "mm/yr",
    },
    "snow": {
        "resample": "mean",
        "legend": "Snowpack",
        "units": "mm",
    },
    "glacier": {
        "resample": "mean",
        "legend": "Glacier Volume",
        "units": "m3",
    },
}


def compute_statistics_delta_run(
    wflow_hist_run_config: Path,
    wflow_delta_runs_config: List[Path],
    gauges_locs: Optional[Union[Path, str]] = None,
    plot_dir: Optional[Union[str, Path]] = None,
    precip_peak_threshold: float = 40,
    dry_days_threshold: float = 0.2,
    heat_threshold: float = 25.0,
    rps: List[int] = [5, 10],
    split_plot_per_scenario: bool = False,
    discharge_statistics_locations: Union[str, List[str]] = "all",
):
    """
    Compute climate change flood/drought statistics from delta runs.

    For flow related indices, they can be computed only at the catchment ``outlets``,
    for ``all`` locations (outlets+gauges) or for a specific list of locations.

    Flood indices:
    - Number of days with high rainfall
    - Extreme rainfall statistics
    - Extreme discharge statistics

    Drought indices:
    - Mean annual precipitation
    - Consecutive number of dry days
    - Maximum number of consecutive dry days
    - Number of freezing days
    - Number of days with extreme high temperature
    - 7-day average low flow
    - Average recharge
    - Average snow water equivalent
    - Average glacier water volume
    - Average actual evapotranspiration

    Outputs:
    - A csv file with the absolute values of the different indices for historical and
      future runs
    - A csv file with the relative change of the different indices for future runs
    - Plots for the relative change of the drought indices for near and far future

    Parameters
    ----------
    wflow_hist_run_config : Path
        Path to the historical run configuration file.
    wflow_delta_runs_config : List[Path]
        List of paths to the delta run configuration files. The individual run filename
        should be organised as "*_model_scenario_horizon.toml".
    gauges_locs : Optional[Union[Path, str]], optional
        Path to the gauges locations file, by default None.
    plot_dir : Optional[Union[str, Path]], optional
        Path to the directory where the plots will be saved, by default None.
    precip_peak_threshold : float, optional
        Threshold for high rainfall days, by default 40 mm/day.
    dry_days_threshold : float, optional
        Threshold for dry days, by default 0.2 mm/day.
    heat_threshold : float, optional
        Threshold for heat days, by default 25.0 degC.
    rps : List[int], optional
        Return periods for the extreme statistics, by default [5, 10].
    split_plot_per_scenario : bool, optional
        If True, the plots will also be split per scenario, by default True.
    discharge_statistics_locations : Union[str, List[str]], optional
        Locations where to compute the discharge statistics, by default "outlets".
        Keywords: "outlets", "all" or a list of locations IDs.
    """
    if plot_dir is None:
        wflow_root = dirname(wflow_hist_run_config)
        plot_dir = join(wflow_root, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # If flow stats are computed for outlets only, no need to add the gauges locations
    if discharge_statistics_locations == "outlets":
        gauges_locs = None

    ### Read historical results ###
    # read model results for historical
    root = dirname(wflow_hist_run_config)
    config_fn = basename(wflow_hist_run_config)
    qsim_hist, ds_clim_hist, ds_basin_hist = get_wflow_results(
        root, config_fn, gauges_locs
    )

    ### Read future runs results ###
    # read the model results and merge to single netcdf
    qsim_delta, ds_clim_delta, ds_basin_delta = get_wflow_results_delta(
        wflow_delta_runs_config, gauges_locs
    )

    # Slice historical reference run (may be longer than the future one) before plotting
    qsim_hist = qsim_hist.sel(time=slice(qsim_delta["time"][0], qsim_delta["time"][-1]))
    ds_clim_hist = ds_clim_hist.sel(
        time=slice(ds_clim_delta["time"][0], ds_clim_delta["time"][-1])
    )
    ds_basin_hist = ds_basin_hist.sel(
        time=slice(ds_basin_delta["time"][0], ds_basin_delta["time"][-1])
    )

    # Update units for use with xclim and drop unnecessary coords for the clim
    def simplify_ds_clim(ds):
        # Units
        ds["P_subcatchment"].attrs["units"] = "mm/day"
        ds["T_subcatchment"].attrs["units"] = "degC"
        # Drop index
        ds = ds.squeeze("index")
        # Drop unnecessary other coordinates
        ds = ds.drop_vars(
            ["value", "geometry", "spatial_ref", "index"], errors="ignore"
        )
        return ds

    ds_clim_hist = simplify_ds_clim(ds_clim_hist)
    ds_clim_delta = simplify_ds_clim(ds_clim_delta)
    # For discharge data, keep only the locations of interest
    if (
        discharge_statistics_locations != "all"
        and discharge_statistics_locations != "outlets"
    ):
        qsim_hist = qsim_hist.sel(index=discharge_statistics_locations)
        qsim_delta = qsim_delta.sel(index=discharge_statistics_locations)
    # Update units
    qsim_hist.attrs["units"] = "m3/s"
    qsim_delta["Q"].attrs["units"] = "m3/s"

    # Get the future horizons, scenarios and models
    future_horizons = ds_clim_delta.horizon.values
    scenarios = ds_clim_delta.scenario.values
    models = ds_clim_delta.model.values

    ### Flood indices ###

    # 1. Number of days with high rainfall
    wet_days_hist = (
        xclim.indices.wetdays(
            pr=ds_clim_hist["P_subcatchment"],
            thresh=f"{precip_peak_threshold} mm/day",
            freq="YS",
        )
        .mean()
        .round(0)
    )
    wet_days_delta = (
        xclim.indices.wetdays(
            pr=ds_clim_delta["P_subcatchment"],
            thresh=f"{precip_peak_threshold} mm/day",
            freq="YS",
        )
        .mean(dim="time")
        .round(0)
    )
    wet_name = f"Days with high rainfall\n(P > {precip_peak_threshold} mm/day) [mm/day]"
    absolute_stats_hist = wet_days_hist.astype(int).to_dataset(name=wet_name)
    absolute_stats_delta = wet_days_delta.astype(int).to_dataset(name=wet_name)

    # 2. Extreme rainfall statistics (P5, P10)
    for rp in rps:
        prec_rp_hist = (
            xclim.indices.stats.frequency_analysis(
                ds_clim_hist["P_subcatchment"],
                t=rp,
                dist="genextreme",
                mode="max",
                freq="YS",
            )
            .mean()
            .round(1)
        )
        prec_rp_delta = xclim.indices.stats.frequency_analysis(
            ds_clim_delta["P_subcatchment"],
            t=rp,
            dist="genextreme",
            mode="max",
            freq="YS",
        ).round(1)
        # Drop the unnecessary "return_period" dimension
        prec_rp_delta = prec_rp_delta.squeeze("return_period", drop=True)
        prec_rp_name = f"Extreme precipitation (RP {rp}) [mm/day]"
        absolute_stats_hist[prec_rp_name] = prec_rp_hist
        absolute_stats_delta[prec_rp_name] = prec_rp_delta

    # 3. Extreme discharge statistics (Q5, Q10)
    for loc in qsim_hist.index.values:
        qsim_loc_hist = qsim_hist.sel(index=loc)
        qsim_loc_delta = qsim_delta["Q"].sel(index=loc)
        loc_name = qsim_loc_hist.station_name.item()
        for rp in rps:
            dis_rp_hist = xclim.indices.stats.frequency_analysis(
                qsim_loc_hist,
                t=rp,
                dist="genextreme",
                mode="max",
                freq="YS",
            ).round(1)
            dis_rp_delta = xclim.indices.stats.frequency_analysis(
                qsim_loc_delta,
                t=rp,
                dist="genextreme",
                mode="max",
                freq="YS",
            ).round(1)
            # Drop the unnecessary "return_period" dimension
            dis_rp_hist = dis_rp_hist.squeeze("return_period", drop=True)
            dis_rp_delta = dis_rp_delta.squeeze("return_period", drop=True)
            dis_rp_name = f"Extreme discharge (RP {rp})\nat {loc_name} [m3/s]"
            absolute_stats_hist[dis_rp_name] = dis_rp_hist.values
            absolute_stats_delta[dis_rp_name] = (
                dis_rp_delta.dims,
                dis_rp_delta.values,
            )

    ### Drought indices ###

    # 1. Mean annual precipitation
    prec_yr_hist = ds_clim_hist["P_subcatchment"].resample(time="YS").sum()
    prec_yr_delta = ds_clim_delta["P_subcatchment"].resample(time="YS").sum()
    prec_yr_hist = prec_yr_hist.mean().round(1)
    prec_yr_delta = prec_yr_delta.mean(dim="time").round(1)
    prec_name = "Annual precipitation [mm/year]"
    absolute_stats_hist[prec_name] = prec_yr_hist
    absolute_stats_delta[prec_name] = prec_yr_delta

    # 2.a Consecutive number of dry days
    dry_days_hist = (
        xclim.indices.dry_days(
            ds_clim_hist["P_subcatchment"],
            thresh=f"{dry_days_threshold} mm/day",
            freq="YS",
        )
        .mean()
        .round(0)
    )
    dry_days_delta = (
        xclim.indices.dry_days(
            ds_clim_delta["P_subcatchment"],
            thresh=f"{dry_days_threshold} mm/day",
            freq="YS",
        )
        .mean(dim="time")
        .round(0)
    )
    dry_name = f"Dry days\n(P < {dry_days_threshold}mm/d) [days]"
    absolute_stats_hist[dry_name] = dry_days_hist.astype(int)
    absolute_stats_delta[dry_name] = dry_days_delta.astype(int)

    # 2.b Maximum number of consecutive dry days
    dry_spell_hist = (
        xclim.indices.maximum_consecutive_dry_days(
            ds_clim_hist["P_subcatchment"],
            thresh=f"{dry_days_threshold*4} mm/day",
            freq="YS",
        )
        .mean()
        .round(0)
    )
    dry_spell_delta = (
        xclim.indices.maximum_consecutive_dry_days(
            ds_clim_delta["P_subcatchment"],
            thresh=f"{dry_days_threshold*4} mm/day",
            freq="YS",
        )
        .mean(dim="time")
        .round(0)
    )
    spell_name = f"Longest Dry spell\n(P < {dry_days_threshold*4}mm/d) [days]"
    absolute_stats_hist[spell_name] = dry_spell_hist.astype(int)
    absolute_stats_delta[spell_name] = dry_spell_delta.astype(int)

    # 3. Number of freezing days
    freeze_days_hist = (
        xclim.indices.frost_days(
            ds_clim_hist["T_subcatchment"],
            thresh="0 degC",
            freq="YS",
        )
        .mean()
        .round(0)
    )
    freeze_days_delta = (
        xclim.indices.frost_days(
            ds_clim_delta["T_subcatchment"],
            thresh="0 degC",
            freq="YS",
        )
        .mean(dim="time")
        .round(0)
    )
    freeze_name = "Freezing days\n(T < 0 degC) [degC]"
    absolute_stats_hist[freeze_name] = freeze_days_hist.astype(int)
    absolute_stats_delta[freeze_name] = freeze_days_delta.astype(int)

    # 4. Number of days with extreme high temperature
    hot_days_hist = (
        xclim.indices.tg_days_above(
            tas=ds_clim_hist["T_subcatchment"],
            thresh=f"{heat_threshold} degC",
            freq="YS",
        )
        .mean()
        .round(0)
    )
    hot_days_delta = (
        xclim.indices.tg_days_above(
            tas=ds_clim_delta["T_subcatchment"],
            thresh=f"{heat_threshold} degC",
            freq="YS",
        )
        .mean(dim="time")
        .round(0)
    )
    warm_name = f"Warm days\n(T > {heat_threshold} degC) [degC]"
    absolute_stats_hist[warm_name] = hot_days_hist.astype(int)
    absolute_stats_delta[warm_name] = hot_days_delta.astype(int)

    # 5. 7-day average low flow
    nm7q_hist = (
        qsim_hist.rolling(time=7)
        .mean()
        .resample(time="YS")
        .min("time")
        .min(dim="time")
        .round(1)
    )
    nm7q_delta = (
        qsim_delta["Q"]
        .rolling(time=7)
        .mean()
        .resample(time="YS")
        .min("time")
        .min(dim="time")
        .round(1)
    )
    for loc in nm7q_hist.index.values:
        nm7q_loc_hist = nm7q_hist.sel(index=loc)
        nm7q_loc_delta = nm7q_delta.sel(index=loc)
        loc_name = nm7q_loc_hist.station_name.item()
        nm7q_name = f"Minimum 7-day average flow\nat {loc_name} [m3/s]"
        absolute_stats_hist[nm7q_name] = nm7q_loc_hist.values
        absolute_stats_delta[nm7q_name] = (
            nm7q_loc_delta.dims,
            nm7q_loc_delta.values,
        )

    # 6. Average recharge
    # 7. Average snow water equivalent
    # 8. Average glacier water volume
    # 9. Average actual evapotranspiration
    for dvar in [
        "groundwater recharge",
        "snow",
        "glacier",
        "actual evapotranspiration",
    ]:
        var = f"{dvar}_basavg"
        if var in ds_basin_hist:
            resample_method = WFLOW_VARS[dvar]["resample"]
            name = f"{WFLOW_VARS[dvar]['legend']} [{WFLOW_VARS[dvar]['units']}]"
            if resample_method == "mean":
                var_hist = ds_basin_hist[var].resample(time="YS").mean()
                var_delta = ds_basin_delta[var].resample(time="YS").mean()
            elif resample_method == "sum":
                var_hist = ds_basin_hist[var].resample(time="YS").sum()
                var_delta = ds_basin_delta[var].resample(time="YS").sum()
            else:
                raise ValueError(f"Resample method {resample_method} not supported")
            absolute_stats_hist[name] = var_hist.mean().round(1)
            absolute_stats_delta[name] = var_delta.mean(dim="time").round(1)

    ### Prepare a recap table for the absolute drought indices values
    absolute_stats_df = absolute_stats_hist.expand_dims(
        {"Drought Indices": ["Historical"]}
    ).to_dataframe()
    absolute_stats_df = absolute_stats_df.astype(str).T
    # Loop over the future horizons and scenarios
    for horizon in future_horizons:
        for scenario in scenarios:
            # Add the horizon and scenario to the index
            stats_delta_hz_sc_str = []
            stats_delta_hz_sc = absolute_stats_delta.sel(
                horizon=horizon, scenario=scenario
            )
            for dvar in absolute_stats_delta.data_vars:
                mean_str = stats_delta_hz_sc[dvar].mean().round(1).item()
                min_str = stats_delta_hz_sc[dvar].min().round(1).item()
                max_str = stats_delta_hz_sc[dvar].max().round(1).item()
                # TODO add significant change test with a * in the string if True
                stats_delta_hz_sc_str.append(f"{mean_str} [{min_str}-{max_str}]")
            # Add to the dataframe
            absolute_stats_df[f"{horizon}_{scenario}"] = stats_delta_hz_sc_str
    # Rename the indexes (replace \n by a space)
    absolute_stats_df.index = absolute_stats_df.index.str.replace("\n", " ")
    # Save to csv
    absolute_stats_df.to_csv(
        join(plot_dir, "indices_absolute_values.csv"),
        header=True,
        index=True,
        index_label="Indices",
    )

    ### Prepare a csv / plots for the relative values
    # Compute the relative change
    relative_stats = xr.Dataset()
    for var in absolute_stats_hist.data_vars:
        var_hist = absolute_stats_hist[var].item()
        if var_hist == 0:
            continue
        relative_stats[var] = (
            (absolute_stats_delta[var] - var_hist) / var_hist * 100
        ).round(1)
    # Compute the median relative change
    relative_stats_mean = relative_stats.mean(dim="model").round(1)
    # Add a model dimension to the dataset
    relative_stats_mean = relative_stats_mean.expand_dims({"model": ["MEAN"]})
    relative_stats = xr.concat([relative_stats, relative_stats_mean], dim="model")
    # Remove the units for the variable names
    relative_stats = relative_stats.rename_vars(
        {var: var.split(" [")[0] for var in relative_stats.data_vars}
    )
    # Convert to pandas dataframe
    relative_stats_df = relative_stats.to_dataframe()
    # Rename the columns (replace \n by a space)
    relative_stats_df.columns = relative_stats_df.columns.str.replace("\n", " ")
    # Save to csv
    relative_stats_df.to_csv(join(plot_dir, "indices_relative_change.csv"))

    # Prepare a plot for near and a plot for far future
    for horizon in future_horizons:
        if not split_plot_per_scenario:
            relative_stats_hz = relative_stats.sel(horizon=horizon).to_dataframe()
            # Convert to dataframe and drop the horizon column
            relative_stats_hz = relative_stats_hz.drop(columns="horizon")
            # Put scenario as the first level of the index
            relative_stats_hz = relative_stats_hz.reorder_levels(
                ["scenario", "model"]
            ).sort_index()
            # Plot the heatmap
            plot_table_statistics_multiindex(
                relative_stats_hz,
                output_path=join(plot_dir, f"indices_relative_change_{horizon}.png"),
                x_label="Scenarios",
                y_label="Indices",
                cmap="RdBu",
                cmap_label="Change compared to historical [%]\nNegative values: drier; Positive values: wetter",
                vmin=-100,
                vmax=100,
                invert_cmap_for=["Actual Evapotranspiration"],
                index_separator=":\n",
                bold_keyword="MEAN",
            )
        else:
            for scenario in scenarios:
                relative_stats_hz_sc = relative_stats.sel(
                    horizon=horizon, scenario=scenario
                ).to_dataframe()
                # Convert to dataframe and drop the horizon and scenario columns
                relative_stats_hz_sc = relative_stats_hz_sc.drop(
                    columns=["horizon", "scenario"]
                )
                # Plot the heatmap
                plot_table_statistics(
                    relative_stats_hz_sc,
                    output_path=join(
                        plot_dir,
                        f"indices_relative_change_{horizon}_{scenario}.png",
                    ),
                    x_label="Models",
                    y_label="Indices",
                    cmap="RdBu",
                    cmap_label="Change compared to historical [%]\nNegative values: drier; Positive values: wetter",
                    vmin=-100,
                    vmax=100,
                    invert_cmap_for=["Actual Evapotranspiration"],
                    bold_keyword="MEAN",
                )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        # output folder
        project_dir = sm.params.project_dir
        Folder_plots = f"{project_dir}/plots/model_delta_runs"
        root = f"{project_dir}/hydrology_model"

        compute_statistics_delta_run(
            wflow_hist_run_config=sm.params.wflow_hist_run_config,
            wflow_delta_runs_config=sm.params.wflow_delta_runs_config,
            gauges_locs=sm.params.gauges_locs,
            plot_dir=Folder_plots,
            precip_peak_threshold=sm.params.precip_peak_threshold,
            dry_days_threshold=sm.params.dry_days_threshold,
            heat_threshold=sm.params.heat_threshold,
            rps=sm.params.return_periods,
            split_plot_per_scenario=sm.params.split_plot_per_scenario,
            discharge_statistics_locations=sm.params.discharge_locations,
        )
    else:
        print("run with snakemake please")
