from hydromt.stats import skills
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from typing import Union, Optional

def plot_hydro(
    qsim: xr.DataArray,
    Folder_out: Union[Path, str],
    qobs: Optional[xr.DataArray] = None,
    color: str = "steelblue",
    station_name: str = "station_1",
    lw: float = 0.8,
    fs: int = 7,
    max_nan_year: int = 60,
    max_nan_month: int = 5,
    file_postfix: str = "",
):
    """
    Plot hydrograph for a specific location.

    If observations ``qobs`` are provided, the plot will include the observations.
    If the simulation is less than 3 years, the plot will be a single panel with the
    hydrograph for that year.
    If it is more than 3 years, the plot will be a 5 panel plot with the following:
    - Monthly time-series
    - Annual time-series
    - Annual cycle (monthly average)
    - Wettest year
    - Driest year

    Parameters
    ----------
    qsim : xr.DataArray
        Simulated streamflow.
    Folder_out : Union[Path, str]
        Output folder to save plots.
    qobs : xr.DataArray, optional
        Observed streamflow, by default None
    color : dict, optional
        Color belonging to a climate source run, by default "steelblue"
    station_name : str, optional
        Station name, by default "station_1"
    lw : float, optional
        Line width, by default 0.8
    fs : int, optional
        Font size, by default 7
    max_nan_year : int, optional
        Maximum number of missing days per year in the observations data to consider
        the year in the annual hydrograph plot. By default 60.
    max_nan_month : int, optional
        Maximum number of missing days per month in the observations data to consider
        the month in the monthly regime plot. By default 5.
    """
    min_count_year = 365 - max_nan_year
    min_count_month = 30 - max_nan_month

    # Settings for obs
    labobs = "observed"
    colobs = "k"

    # Find the common period between obs and sim
    if qobs is not None:
        start = max(qsim.time.values[0], qobs.time.values[0])
        end = min(qsim.time.values[-1], qobs.time.values[-1])
        # make sure obs and sim have period in common
        if start < end:
            qsim_na = qsim.sel(time=slice(start, end))
            qobs_na = qobs.sel(time=slice(start, end))
        else:
            qobs = None
            qobs_na = None
            qsim_na = qsim
    else:
        qsim_na = qsim

    # Number of years available
    nb_years = np.unique(qsim["time.year"].values).size

    # If less than 3 years, plot a single panel
    if nb_years <= 3:
        nb_panel = 1
        titles = ["Monthly time-series"]
        figsize_y = 8
    else:
        nb_panel = 5
        titles = [
            "Monthly time-series",
            "Annual time-series",
            "Annual cycle",
            "Wettest year",
            "Driest year",
        ]
        figsize_y = 23
        # Get the wettest and driest year (based on first climate_source)
        if qobs is not None:
            qyr = qobs_na.resample(time="YE").sum(skipna=True, min_count=min_count_year)
            qyr["time"] = qyr["time.year"]
            # Get the year for the minimum as an integer
            year_dry = str(qyr.isel(time=qyr.argmin("time")).time.item())
            year_wet = str(qyr.isel(time=qyr.argmax("time")).time.item())
        else:
            qyr = qsim.resample(time="YE").sum()
            qyr["time"] = qyr["time.year"]
            # Get the year for the minimum as an integer
            year_dry = str(qyr.isel(time=qyr.argmin("time")).time.values[0])
            year_wet = str(qyr.isel(time=qyr.argmax("time")).time.values[0])

    fig, axes = plt.subplots(nb_panel, 1, figsize=(16 / 2.54, figsize_y / 2.54))
    axes = [axes] if nb_panel == 1 else axes

    if qobs is not None:
        # min_count can only be used with sum and not mean
        qobs_na_month = (
            qobs_na.resample(time="ME").sum(skipna=True, min_count=min_count_month) /
            qobs_na.resample(time="ME").count(dim="time")
        )
        qobs_na_year = (
            qobs_na.resample(time="YE").sum(skipna=True, min_count=min_count_year) / 
            qobs_na.resample(time="YE").count(dim="time")
        )

    # 1. Monhtly timeseries
    qsim_na.resample(time="ME").mean().plot(
        ax=axes[0],
        label=f"simulated",
        linewidth=lw,
        color=color,
    )
    if qobs is not None:
        qobs_na_month.plot(
            ax=axes[0], label=labobs, linewidth=lw, color=colobs, linestyle="--"
        )
    if nb_panel == 5:
        # 2. annual Q
        qsim_na.resample(time="YE").mean().plot(
            ax=axes[1],
            label=f"simulated",
            linewidth=lw,
            color=color,
        )
        if qobs is not None:
            qobs_na_year.plot(
                ax=axes[1], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

        # 3. monthly Q
        month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
        dsqM = qsim_na.resample(time="ME").mean()
        dsqM = dsqM.groupby(dsqM.time.dt.month).mean()
        dsqM.plot(
            ax=axes[2],
            label=f"simulated",
            linewidth=lw,
            color=color,
        )
        if qobs is not None:
            dsqMo = qobs_na_month.groupby(qobs_na_month.time.dt.month).mean()
            dsqMo.plot(ax=axes[2], label=labobs, linewidth=lw, color=colobs)
        axes[2].set_title("Average monthly sum")
        axes[2].set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=5)

        # 4. wettest year
        qsim.sel(time=year_wet).plot(
            ax=axes[3],
            label=f"simulated",
            linewidth=lw,
            color=color,
        )
        if qobs is not None:
            qobs.sel(time=year_wet).plot(
                ax=axes[3], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

        # 5. driest year
        qsim.sel(time=year_dry).plot(
            ax=axes[4],
            label=f"simulated",
            linewidth=lw,
            color=color,
        )
        if qobs is not None:
            qobs.sel(time=year_dry).plot(
                ax=axes[4], label=labobs, linewidth=lw, color=colobs, linestyle="--"
            )

    # Axes settings
    for ax, title in zip(axes, titles):
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.set_xlabel("", fontsize=fs)
    axes[0].legend(fontsize=fs)
    # Compute scores
    if qobs is not None:
        # Drop nan values in qobs_na
        qobs_drop = qobs_na.dropna("time")
        # Use the same time for qsim
        qsim_drop = qsim.sel(time=qobs_drop.time)
        # Compute scores
        #common time to avoid value error in join='exact'
        common_time_min = min(qsim_drop.time.min().item(), qobs_drop.time.min().item())
        common_time_max = max(qsim_drop.time.max().item(), qobs_drop.time.max().item())
        nse = skills.nashsutcliffe(qsim_drop.sel(time=slice(common_time_min, common_time_max)), qobs_drop.sel(time=slice(common_time_min, common_time_max)), dim="time")
        nse_monthly = skills.nashsutcliffe(qsim_drop.resample(time="ME").mean().sel(time=slice(common_time_min, common_time_max)), qobs_drop.resample(time="ME").mean().sel(time=slice(common_time_min, common_time_max)), dim="time")
        kge = skills.kge(qsim_drop.sel(time=slice(common_time_min, common_time_max)), qobs_drop.sel(time=slice(common_time_min, common_time_max)), dim="time")
        kge_monthly = skills.kge(qsim_drop.resample(time="ME").mean().sel(time=slice(common_time_min, common_time_max)), qobs_drop.resample(time="ME").mean().sel(time=slice(common_time_min, common_time_max)), dim="time")
        bias = skills.percentual_bias(qsim_drop.sel(time=slice(common_time_min, common_time_max)), qobs_drop.sel(time=slice(common_time_min, common_time_max)), dim="time")
        scores = f"NSE: {nse.values.round(2)} | NSE monthly: {nse_monthly.values.round(2)} | KGE: {kge['kge'].values.round(2)} | KGE monthly: {kge_monthly['kge'].values.round(2)} | Bias: {bias.values.round(2)}%"
    else:
        scores = ""
    # Add a figure title on two lines
    fig.suptitle(f"{file_postfix}\n{scores}", fontsize=fs)
    plt.tight_layout()

    if not os.path.exists(os.path.join(Folder_out, station_name)):
        os.makedirs(os.path.join(Folder_out, station_name))
    plt.savefig(os.path.join(Folder_out, station_name, f"hydro_{file_postfix}.png"), dpi=300)
    plt.close()

def plot_hydro_all_timeseries(
    qsim_uncalibrated: xr.DataArray,
    qobs: xr.DataArray,
    qsim_cals: list[xr.DataArray],
    qsim_cals_names: list[str],
    Folder_out: Union[Path, str],
    lw: float = 0.8,
    fs: int = 7,
    max_nan_month: int = 5,
):
    """Monthly timeseries plot over the different stations."""
    min_count_month = 30 - max_nan_month
    qobs_month = (
        qobs.resample(time="ME").sum(skipna=True, min_count=min_count_month) /
        qobs.resample(time="ME").count(dim="time")
    )

    # Loop per station
    for station_id, station_name in zip(qobs.index.values, qobs.station_name.values):
        print(f"Plotting monthly timeseries at station {station_name}")
        # Start a figure, legend will be outside of the figure on the bottom
        fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
        
        # Drop nan values in qobs_na
        qobs_drop = qobs.sel(index=station_id).dropna("time")

        # 1. Plot the calibrated runs
        for i in range(len(qsim_cals)):
            qsim_id = qsim_cals[i].sel(index=station_id)
            # Find common time range
            common_time = qsim_id.time.where(qsim_id.time.isin(qobs_drop.time), drop=True)
            qsim_common = qsim_id.sel(time=common_time)
            qobs_common = qobs_drop.sel(time=common_time)
            
            nse = skills.nashsutcliffe(qsim_common, qobs_common, dim="time")
            kge = skills.kge(qsim_common, qobs_common, dim="time")
            qsim_id.resample(time="ME").mean().plot(
                ax=ax, 
                label=f"{qsim_cals_names[i]} | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}", 
                linewidth=lw,
                linestyle="-",
            )
        
        # 2. Plot the uncalibrated run (second position)
        qsim_id = qsim_uncalibrated.sel(index=station_id)
        common_time = qsim_id.time.where(qsim_id.time.isin(qobs_drop.time), drop=True)
        qsim_common = qsim_id.sel(time=common_time)
        qobs_common = qobs_drop.sel(time=common_time)
        nse = skills.nashsutcliffe(qsim_common, qobs_common, dim="time")
        kge = skills.kge(qsim_common, qobs_common, dim="time")
        qsim_id.resample(time="ME").mean().plot(
            ax=ax, 
            label=f"uncalibrated | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}", 
            color="gray",
            linestyle="-.", 
            linewidth=lw+0.2,
        )

        # 3. Plot the observations (first position)
        qobs_month.sel(index=station_id).plot(
            ax=ax, 
            label="observed", 
            color="k", 
            linestyle="--", 
            linewidth=lw+0.2,
        )
        
        # Axes settings
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
        ax.set_title("Monthly time-series", fontsize=fs)
        ax.set_xlabel("", fontsize=fs)
        # Add the legend below the graph
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=fs-4)

        plt.tight_layout()
        if not os.path.exists(os.path.join(Folder_out, station_name)):
            os.makedirs(os.path.join(Folder_out, station_name))
        plt.savefig(os.path.join(Folder_out, station_name, "hydro_monthly_timeseries.png"), dpi=300)
        plt.close()

def plot_hydro_all_timeseries_interactive(
    qsim_uncalibrated: xr.DataArray,
    qobs: xr.DataArray,
    qsim_cals: list[xr.DataArray],
    qsim_cals_names: list[str],
    Folder_out: Union[Path, str],
    max_nan_month: int = 5,
):
    """Interactive monthly timeseries plot over the different stations using Plotly."""
    min_count_month = 30 - max_nan_month
    qobs_month = (
        qobs.resample(time="ME").sum(skipna=True, min_count=min_count_month) /
        qobs.resample(time="ME").count(dim="time")
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for station_id, station_name in zip(qobs.index.values, qobs.station_name.values):
        print(f"Plotting interactive monthly timeseries at station {station_name}")
        
        fig = go.Figure()
        
        qobs_drop = qobs.sel(index=station_id).dropna("time")

        for i, qsim_cal in enumerate(qsim_cals):
            qsim_id = qsim_cal.sel(index=station_id)
            nse = skills.nashsutcliffe(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
            kge = skills.kge(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
            
            fig.add_trace(go.Scatter(
                x=qsim_id.resample(time="ME").mean().time,
                y=qsim_id.resample(time="ME").mean(),
                name=f"{qsim_cals_names[i]} | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        
        qsim_id = qsim_uncalibrated.sel(index=station_id)
        nse = skills.nashsutcliffe(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
        kge = skills.kge(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
        
        fig.add_trace(go.Scatter(
            x=qsim_id.resample(time="ME").mean().time,
            y=qsim_id.resample(time="ME").mean(),
            name=f"Uncalibrated | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}",
            line=dict(color='gray', width=3, dash='dot'),
        ))

        fig.add_trace(go.Scatter(
            x=qobs_month.sel(index=station_id).time,
            y=qobs_month.sel(index=station_id),
            name="Observed",
            line=dict(color='black', width=3, dash='dash'),
        ))
        
        # Update layout and save as before
        # ...

def plot_hydro_all_per_year(
    qsim_uncalibrated: xr.DataArray,
    qobs: xr.DataArray,
    qsim_cals: list[xr.DataArray],
    qsim_cals_names: list[str],
    Folder_out: Union[Path, str],
    lw: float = 0.8,
    fs: int = 7,
):
    """Daily timeseries plot per year over the different stations."""
    # Find the available years
    years = np.unique(qobs["time.year"].values)
    
    # Loop per station
    for station_id, station_name in zip(qobs.index.values, qobs.station_name.values):
        print(f"Plotting daily timeseries at station {station_name}")
        # Loop per year
        for year in years:
            # Start a figure, legend will be outside of the figure on the bottom
            fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
            
            # Drop nan values in qobs_na
            qobs_yr = qobs.sel(index=station_id).sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            qobs_drop = qobs_yr.dropna("time")

            # 1. Plot the calibrated runs
            for i in range(len(qsim_cals)):
                qsim_id = qsim_cals[i].sel(index=station_id).sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                nse = skills.nashsutcliffe(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
                kge = skills.kge(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
                qsim_id.plot(
                    ax=ax, 
                    label=f"{qsim_cals_names[i]} | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}", 
                    linewidth=lw,
                    linestyle="-",
                )
            
            # 2. Plot the uncalibrated run (second position)
            qsim_id = qsim_uncalibrated.sel(index=station_id).sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            nse = skills.nashsutcliffe(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
            kge = skills.kge(qsim_id.sel(time=qobs_drop.time), qobs_drop, dim="time")
            qsim_id.plot(
                ax=ax, 
                label=f"uncalibrated | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}", 
                color="gray",
                linestyle="-.", 
                linewidth=lw+0.2,
            )

            # 3. Plot the observations (first position)
            qobs_yr.plot(
                ax=ax, 
                label="observed", 
                color="k", 
                linestyle="--", 
                linewidth=lw+0.2,
            )
            
            # Axes settings
            ax.tick_params(axis="both", labelsize=fs)
            ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
            ax.set_title(f"Daily time-series {year}", fontsize=fs)
            ax.set_xlabel("", fontsize=fs)
            # Add the legend below the graph
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=fs-4)

            plt.tight_layout()
            if not os.path.exists(os.path.join(Folder_out, station_name)):
                os.makedirs(os.path.join(Folder_out, station_name))
            plt.savefig(os.path.join(Folder_out, station_name, f"hydro_daily_timeseries_{year}.png"), dpi=300)
            plt.close()

def plot_hydro_all_per_year_interactive(
    qsim_uncalibrated: xr.DataArray,
    qobs: xr.DataArray,
    qsim_cals: list[xr.DataArray],
    qsim_cals_names: list[str],
    Folder_out: Union[Path, str],
):
    """Interactive daily timeseries plot for each year over the different stations using Plotly."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for station_id, station_name in zip(qobs.index.values, qobs.station_name.values):
        print(f"Plotting interactive daily timeseries at station {station_name}")
        
        # Get unique years
        years = np.unique(qobs.sel(index=station_id).time.dt.year)
        
        for year in years:
            fig = go.Figure()
            
            # Select data for the current year
            qobs_year = qobs.sel(index=station_id).sel(time=str(year))
            qsim_uncal_year = qsim_uncalibrated.sel(index=station_id).sel(time=str(year))
            
            # Find common time range
            common_time = qsim_uncal_year.time.where(qsim_uncal_year.time.isin(qobs_year.time), drop=True)
            
            # Plot calibrated runs
            for i, qsim_cal in enumerate(qsim_cals):
                qsim_cal_year = qsim_cal.sel(index=station_id).sel(time=str(year))
                qsim_cal_common = qsim_cal_year.sel(time=common_time)
                qobs_common = qobs_year.sel(time=common_time)
                
                nse = skills.nashsutcliffe(qsim_cal_common, qobs_common, dim="time")
                kge = skills.kge(qsim_cal_common, qobs_common, dim="time")
                
                fig.add_trace(go.Scatter(
                    x=qsim_cal_year.time,
                    y=qsim_cal_year,
                    name=f"{qsim_cals_names[i]} | NSE: {nse.values.round(2)} | KGE: {kge['kge'].values.round(2)}",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
            
            # Plot uncalibrated run
            qsim_uncal_common = qsim_uncal_year.sel(time=common_time)
            nse_uncal = skills.nashsutcliffe(qsim_uncal_common, qobs_common, dim="time")
            kge_uncal = skills.kge(qsim_uncal_common, qobs_common, dim="time")
            fig.add_trace(go.Scatter(
                x=qsim_uncal_year.time,
                y=qsim_uncal_year,
                name=f"Uncalibrated | NSE: {nse_uncal.values.round(2)} | KGE: {kge_uncal['kge'].values.round(2)}",
                line=dict(color='gray', width=3, dash='dot'),
            ))
            
            # Plot observed data
            fig.add_trace(go.Scatter(
                x=qobs_year.time,
                y=qobs_year,
                name="Observed",
                line=dict(color='black', width=3, dash='dash'),
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Daily Streamflow at {station_name} - {year}",
                xaxis_title="Date",
                yaxis_title="Streamflow (m³/s)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.5,
                    xanchor="center",
                    x=0.5
                ),
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            
            # Save the plot
            if not os.path.exists(os.path.join(Folder_out, station_name)):
                os.makedirs(os.path.join(Folder_out, station_name))
            fig.write_html(os.path.join(Folder_out, station_name, f"hydro_daily_timeseries_{year}_interactive.html"))
            
            # Update layout and save as before
            # ...

def plot_hydro_all_month(
    qsim_uncalibrated: xr.DataArray,
    qobs: xr.DataArray,
    qsim_cals: list[xr.DataArray],
    qsim_cals_names: list[str],
    Folder_out: Union[Path, str],
    lw: float = 0.8,
    fs: int = 7,
    max_nan_month: int = 5,
):
    """Plot the monthly regime over the different stations."""
    min_count_month = 30 - max_nan_month
    month_labels = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    qobs_month = (
        qobs.resample(time="ME").sum(skipna=True, min_count=min_count_month) /
        qobs.resample(time="ME").count(dim="time")
    )

    # Loop per station
    for station_id, station_name in zip(qobs.index.values, qobs.station_name.values):
        print(f"Plotting monthly regime at station {station_name}")
        # Start a figure, legend will be outside of the figure on the bottom
        fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
        
        # 1. Plot the calibrated runs
        for i in range(len(qsim_cals)):
            qsim_id = qsim_cals[i].sel(index=station_id).resample(time="ME").mean()
            qsim_id = qsim_id.groupby(qsim_id.time.dt.month).mean()
            qsim_id.plot(
                ax=ax, 
                label=f"{qsim_cals_names[i]}", 
                linewidth=lw,
                linestyle="-",
            )
        
        # 2. Plot the uncalibrated run (second position)
        qsim_id = qsim_uncalibrated.sel(index=station_id).resample(time="ME").mean()
        qsim_id = qsim_id.groupby(qsim_id.time.dt.month).mean()
        qsim_id.plot(
            ax=ax, 
            label=f"uncalibrated", 
            color="gray",
            linestyle="-.", 
            linewidth=lw+0.2,
        )

        # 3. Plot the observations (first position)
        qobs_id = qobs_month.sel(index=station_id)
        qobs_id = qobs_id.groupby(qobs_id.time.dt.month).mean()
        qobs_id.plot(
            ax=ax, 
            label="observed", 
            color="k", 
            linestyle="--", 
            linewidth=lw+0.2,
        )

        # Axes settings
        ax.tick_params(axis="both", labelsize=fs)
        ax.set_ylabel("Q (m$^3$s$^{-1}$)", fontsize=fs)
        ax.set_title("Monthly regime", fontsize=fs)
        ax.set_xlabel("", fontsize=fs)
        ax.set_xticks(ticks=np.arange(1, 13), labels=month_labels, fontsize=fs-2)
        # Add the legend below the graph
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=fs-4)

        plt.tight_layout()
        if not os.path.exists(os.path.join(Folder_out, station_name)):
            os.makedirs(os.path.join(Folder_out, station_name))
        plt.savefig(os.path.join(Folder_out, station_name, "hydro_monthly_regime.png"), dpi=300)
        plt.close()

def plot_snow_glacier(
    ds_uncalibrated: xr.Dataset,
    ds_cals: xr.Dataset,
    names_cal: list[str],
    Folder_out: Union[Path, str],
    lw: float = 0.8,
    fs: int = 7,
):
    """"Plot snow (and glacier) timeseries."""
    fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
    # Add a secondary y axis for glacier
    if "glacier_basavg" in ds_uncalibrated:
        ax2 = ax.twinx()
    max_storage = 0

    # 1. Plot the calibrated runs
    for i in range(len(ds_cals)):
        ds_id = ds_cals[i]
        ds_id["snow_basavg"].plot(
            ax=ax, 
            label=f"{names_cal[i]}", 
            linewidth=lw,
            linestyle="-",
        )
        max_storage = max(max_storage, ds_id["snow_basavg"].max().values)
        if "glacier_basavg" in ds_id:
            ds_id["glacier_basavg"].plot(
                ax=ax2, 
                label=f"{names_cal[i]}", 
                linewidth=lw,
                linestyle="-",
            )
            max_storage = max(max_storage, ds_id["glacier_basavg"].max().values)
    
    # 2. Plot the uncalibrated run (second position)
    ds_id = ds_uncalibrated
    ds_id["snow_basavg"].plot(
        ax=ax, 
        label="uncalibrated", 
        color="gray",
        linestyle="-.", 
        linewidth=lw+0.2,
    )
    max_storage = max(max_storage, ds_id["snow_basavg"].max().values)
    if "glacier_basavg" in ds_id:
        ds_id["glacier_basavg"].plot(
            ax=ax2, 
            label="uncalibrated", 
            color="gray",
            linestyle="-.", 
            linewidth=lw+0.2,
        )
        max_storage = max(max_storage, ds_id["glacier_basavg"].max().values)
    
    # Axes settings
    ax.tick_params(axis="both", labelsize=fs)
    ax.set_ylabel("Snow Storage (mm)", fontsize=fs)
    ax.set_title("Snow and glacier storage", fontsize=fs)
    ax.set_xlabel("", fontsize=fs)
    ax.set_ylim(0-50, max_storage+50)
    # Add the legend below the graph
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=fs-4)
    if "glacier_basavg" in ds_uncalibrated:
        ax2.tick_params(axis="y", labelsize=fs)
        ax2.set_ylabel("Glacier storage (mm)", fontsize=fs)
        ax2.set_ylim(0-50, max_storage+50)
        #ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=fs-4)
    
    plt.tight_layout()
    if not os.path.exists(os.path.join(Folder_out)):
        os.makedirs(os.path.join(Folder_out))
    plt.savefig(os.path.join(Folder_out, "snow_glacier.png"), dpi=300)
    plt.close()

def plot_snow_glacier_interactive(
    ds_uncalibrated: xr.Dataset,
    ds_cals: list[xr.Dataset],
    names_cal: list[str],
    Folder_out: Union[Path, str],
) -> None:
    """Plot snow (and glacier) timeseries using Plotly."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    max_storage = 0
    has_glacier = "glacier_basavg" in ds_uncalibrated

    # Define a large set of distinct colors
    color_set = [
        # Tableau 20 colors
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896',
        '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7',
        '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        # Additional colors if needed
        '#636363', '#bdbdbd', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
        '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#8dd3c7', '#ffffb3',
        '#bebada', '#fb8072', '#80b1d3', '#fdb462'
    ]

    n_colors = len(ds_cals)
    
    # If we need more colors than available, we'll cycle through the colors
    colors = color_set * (n_colors // len(color_set) + 1)
    
    # Shuffle the colors to make them more random
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(colors)
    
    # Take only the number of colors we need
    colors = colors[:n_colors]

    for i, ds_id in enumerate(ds_cals):
        color = colors[i]
        fig.add_trace(
            go.Scatter(
                x=ds_id["snow_basavg"].time,
                y=ds_id["snow_basavg"],
                name=f"{names_cal[i]} (Snow)",
                line=dict(color=color, width=2),
            ),
            secondary_y=False,
        )
        max_storage = max(max_storage, ds_id["snow_basavg"].max().values)
        
        if has_glacier:
            fig.add_trace(
                go.Scatter(
                    x=ds_id["glacier_basavg"].time,
                    y=ds_id["glacier_basavg"],
                    name=f"{names_cal[i]} (Glacier)",
                    line=dict(color=color, width=2, dash='dot'),
                ),
                secondary_y=True,
            )
            max_storage = max(max_storage, ds_id["glacier_basavg"].max().values)

    # 2. Plot the uncalibrated run
    fig.add_trace(
        go.Scatter(
            x=ds_uncalibrated["snow_basavg"].time,
            y=ds_uncalibrated["snow_basavg"],
            name="Uncalibrated (Snow)",
            line=dict(color='gray', width=3),
        ),
        secondary_y=False,
    )
    max_storage = max(max_storage, ds_uncalibrated["snow_basavg"].max().values)

    if has_glacier:
        fig.add_trace(
            go.Scatter(
                x=ds_uncalibrated["glacier_basavg"].time,
                y=ds_uncalibrated["glacier_basavg"],
                name="Uncalibrated (Glacier)",
                line=dict(color='gray', width=3, dash='dot'),
            ),
            secondary_y=True,
        )
        max_storage = max(max_storage, ds_uncalibrated["glacier_basavg"].max().values)

    # Update layout
    fig.update_layout(
        title="Snow and Glacier Storage",
        xaxis_title="",
        yaxis_title="Snow Storage (mm)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-1.3,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        height=1000,
        margin=dict(l=50, r=50, t=100, b=50),
        autosize=True,
    )

    fig.update_yaxes(title_text="Glacier Storage (mm)", secondary_y=True)
    
    y_max = max_storage + 50
    fig.update_yaxes(range=[0, y_max], secondary_y=False)
    fig.update_yaxes(range=[0, y_max], secondary_y=True)

    # Save the plot
    if not os.path.exists(Folder_out):
        os.makedirs(Folder_out)
    
    # Configure the plot for responsiveness
    config = {
        'responsive': True,
        'displayModeBar': False  # Hide the mode bar for a cleaner look
    }
    
    fig.write_html(
        os.path.join(Folder_out, "snow_glacier_interactive.html"),
        config=config,
        full_html=True,
        include_plotlyjs='cdn',  # Use CDN for plotly.js
    )

