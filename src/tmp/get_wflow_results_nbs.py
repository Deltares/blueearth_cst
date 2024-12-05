#%%

import os
import hydromt
from hydromt_wflow import WflowModel
import xarray as xr 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, dirname, basename
import seaborn as sns
import sys
from hydromt.log import setuplog
logger = setuplog(log_level="INFO")


def get_rel_change(ds):
    """
    Calculate relative change in a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate the relative change for.

    Returns
    -------
    ds_rel : xr.Dataset
        The relative change in the dataset (%).
    """
    # Calculate the mean of the dataset
    mean_ds = ds.mean("time")
    # Calculate the relative change with respect to the 'calib' scenario
    ds_rel = (mean_ds / mean_ds.sel(nbs="Reference") - 1) * 100
    return ds_rel

#%%
if sys.platform == "win32":
    folder_p = r"p:\11210673-fao\14 Subbasins"
else:
    folder_p = r"/p/11210673-fao/14 Subbasins"

savegrids_annual = True

config_fn_list = {
    "wflow_sbm_calib.toml":{
        "short_name": "Reference"
    },
    "wflow_sbm_calib_water_tanks.toml":{
        "short_name": "tanks (urban)"
    },
    "wflow_sbm_calib_pond_gentle_slopes_forest.toml":{
        "short_name": "ponds forest"
    },
    "wflow_sbm_calib_pond_gentle_slopes_forest_0.02.toml":{
        "short_name": "ponds forest"
    },
    "wflow_sbm_calib_pond_grassland_terracing_hr.toml":{
        "short_name": "terracing grassland"
    },
    "wflow_sbm_calib_agroforestry.toml":{
        "short_name": "agroforestry"
    },
    "wflow_sbm_calib_pond_cropland_0.02.toml":{
        "short_name": "hedgerows etc."
    },
    "wflow_sbm_calib_ponding_cropland_reservoirs.toml":{
        "short_name": "ponds (cropland)"
    },
    "wflow_sbm_bunds_0pt3m.toml":{
        "short_name": "bunds (0.3m)"
    },
    "wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.02.toml":{
        "short_name": "all measures (crop 0.02m)"
    },

    "wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.04.toml":{
        "short_name": "all measures (crop 0.04m)"
    },


    "wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.2.toml":{
        "short_name": "all measures (crop 0.2m)"
    },

    "wflow_sbm_calib_bunds_0pt3m_pond_sum_cropland_0.1.toml":{
        "short_name": "all measures (crop 0.1m)"
    },


    # "wflow_sbm_calib_pond_sum.toml",
    # "wflow_sbm_calib_pond_0p5.toml",
    # "wflow_sbm_calib_pond_0p3.toml",

    # "wflow_sbm_calib_pond_sum_1.toml":{
    #     "short_name": "all measures (0.2m)"
    # },
    # "wflow_sbm_calib_pond_sum_1.5.toml":{
    #     "short_name": "all measures (0.3m)"
    # },
    # "wflow_sbm_calib_pond_sum_2.toml":{
    #     "short_name": "all measures (0.4m)"
    # },
    # "wflow_sbm_calib_pond_sum_2.5.toml":{
    #     "short_name": "all measures (0.5m)"
    # },
    # "wflow_sbm_calib_pond_sum_3.toml":{
    #     "short_name": "all measures (0.6m)"
    # },
}

runs_short_name = [config_fn_list[i]["short_name"] for i in config_fn_list]

#plot for gridded output
name_scenario = "all measures (crop 0.02m)"

basins = {
    # "Bhutan": {
    #     "folder": "Bhutan_Damchhu_500m_v3",
    #     "suffix_monitoring_map": "bhutan",
    # },
    # "Nepal": {
    #     "folder": "Nepal_Seti_500m_v3",
    #     "suffix_monitoring_map": "karnali",
    # },
    # "Pakistan": {
    #     "folder": "Pakistan_Swat_500m_v2",
    #     "suffix_monitoring_map": "swat",
    "Afghanistan": {
        "folder": "Afghanistan_Alishing_500m_v2",
        "suffix_monitoring_map": "alishing",
    }
}

gauges_monitoring_prefix = "gauges_monitoring-stations-"
subcatch_monitoring_prefix = "subcatch_monitoring-stations-"

months_dry = [1, 2, 3, 4, 5, 12]  
months_wet = [6,7,8,9,10,11]  

outputs_basin = {
    f"Q_gauges": {"short_name":"Q"},
    f"groundwater recharge_basavg": {"short_name":"Qgwr"},
    f"actual evapotranspiration_basavg": {"short_name":"Ea"},
    f"overland flow_basavg": {"short_name":"Qof"},
}

for basin in basins:
    folder_basin = basins[basin]["folder"]
    gauges_monitoring_suffix = basins[basin]["suffix_monitoring_map"]

    outputs = {
        f"Qnbs_{gauges_monitoring_prefix}{gauges_monitoring_suffix}": {"short_name":"Q"},
        f"gwrecharge_nbs_{subcatch_monitoring_prefix}{gauges_monitoring_suffix}": {"short_name":"Qgwr"},
        f"evap_nbs_{subcatch_monitoring_prefix}{gauges_monitoring_suffix}": {"short_name":"Ea"},
        f"overlandflow_nbs_{subcatch_monitoring_prefix}{gauges_monitoring_suffix}": {"short_name":"Qof"},
        f"satwaterdepth_nbs_{subcatch_monitoring_prefix}{gauges_monitoring_suffix}": {"short_name":"Ss"},
        f"zi_nbs_{subcatch_monitoring_prefix}{gauges_monitoring_suffix}": {"short_name":"zi"},
    }
        

    ds = [] 
    ds_basin = []
    ds_grid = []

    for config_fn in config_fn_list:
        short_name_run = config_fn_list[config_fn]["short_name"]
        root = os.path.join(folder_p, folder_basin , "hydrology_model", "run_nbs")
        folder_plots = os.path.join(folder_p, folder_basin, "plots", "wflow_nbs")
        
        model = WflowModel(root=root, config_fn=config_fn, mode="r", logger=logger)

        #gridded output
        # model.results.keys()
        ds_grid_ = model.results["output"].assign_coords({"nbs":short_name_run}).expand_dims("nbs")
        ds_grid.append(ds_grid_)
        model.read_results()
        print(model.results.keys())

        for output in outputs:
            print(output)
            short_name = outputs[output]["short_name"]
            ds_ = model.results[output].rename(short_name)
            ds_ = ds_.to_dataset()
            ds_ = ds_.drop("geometry")
            ds_ = ds_.assign_coords({"nbs":short_name_run}).expand_dims("nbs")
            ds.append(ds_)

        # #also add groundwater recharge 
        for output in outputs_basin:
            short_name = outputs_basin[output]["short_name"]
            gw_ = model.results[output].rename(short_name)
            gw_["index"] = [0]
            gw_ = gw_.drop(["x", "y", "geometry"], errors="ignore")
            gw_ = gw_.assign_coords({"nbs":short_name_run}).expand_dims("nbs")
            ds_basin.append(gw_)

    ds = xr.merge(ds)
    ds_basin = xr.merge(ds_basin)
    # ds_grid = xr.merge(ds_grid)

    #merge basin outlet (index 0) with other stations 
    ds = xr.merge([ds_basin, ds])

    #remove first year
    ds = ds.sel(time=slice("1981-01-01", None))


    indicators = {
        "Qmin": {
            "var" : ds["Q"].rolling(time = 7).mean().resample(time = 'A'),
            "reducer":np.min,
        },
        "Qmax": {
            "var" : ds["Q"].resample(time = 'A'),
            "reducer":np.max,
        },
        "Qdry": {
            "var" : ds["Q"].sel(time=ds.time.dt.month.isin(months_dry)),
            "reducer":None,
        },
        "Qwet": {
            "var" : ds["Q"].sel(time=ds.time.dt.month.isin(months_wet)),
            "reducer":None,
        },
        "Ea": {
            "var" : ds["Ea"].resample(time = 'A'),
            "reducer":np.sum,
        },
        "Qgwr": {
            "var" : ds["Qgwr"].resample(time = 'A'),
            "reducer":np.mean,
        },  
        "Qof": {
            "var" : ds["Qof"].resample(time = 'A'),
            "reducer":np.mean,
        },        
        # "Ss": {
        #     "var" : ds["Ss"].resample(time = 'A'),
        #     "reducer":np.mean,
        # },        
        
    }

    ds_indicators = []
    for ind in indicators:
        print(ind)
        var = indicators[ind]["var"]
        reducer = indicators[ind]["reducer"]
        if reducer:
            ds_ind = var.reduce(reducer, dim="time")
        else:
            ds_ind = var
        ds_ind_rel = get_rel_change(ds_ind)
        ds_ind_rel.name = ind
        ds_indicators.append(ds_ind_rel)

    ds_indicators = xr.merge(ds_indicators)


    
    for index in ds_indicators.index.values:
        ds_indicators_station = ds_indicators.sel(index=index).to_dataframe().loc[runs_short_name].round(2).drop(columns=["index", "value", "spatial_ref"]).drop(index="Reference")

        #plt heatmap
        fs = 10
        fig, ax = plt.subplots(figsize=(25/2.54, 20/2.54))
        im = sns.heatmap(ds_indicators_station, annot=True, 
                        cmap="RdBu", vmin = -25, vmax = 25, 
                        linewidth=0.5, ax=ax, annot_kws={"fontsize":fs},
                        cbar_kws={
                            'orientation': "horizontal",
                                #    "shrink": 0.8,
                                "pad":0.05,
                                "fraction": 0.02,

                                }
                        #  cbar_kws={"label": "(changement %)"},
                        )
        # y_ticks_labels = ds_indicators_station.index
        # im.set_yticks(np.arange(len(y_ticks_labels))+0.5)
        # im.set_yticklabels(y_ticks_labels, )
        cbar = im.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fs)
        # cbar.set_orientation('horizontal')
        cbar.ax.set_xlabel("change (%)", fontsize=fs)
        ax.xaxis.tick_top()
        plt.tick_params(axis="both", labelsize=fs)
        # plt.xticks(rotation=45, ha="center")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_plots, f"indicators_{index}.png" ), bbox_inches='tight')

#%%
    #gridded outpit plots

    if not os.path.exists(os.path.join(folder_plots, "grids_annual")):
        os.mkdir(os.path.join(folder_plots, "grids_annual"))

    #plot recharge map
    #first save mean annual recharge for each scenario to netcdf
    for i in range(len(ds_grid)):
        print(i)
        name_scenario = ds_grid[i].nbs[0].values
        #save annual grid 
        ds_grid_recharge_a = ds_grid[i]["vertical.recharge"].sel(nbs = name_scenario).sel(time=slice("1981-01-01", None)).resample(time="A").mean("time").mean("time")
        if savegrids_annual:
            ds_grid_recharge_a.to_netcdf(os.path.join(folder_plots, "grids_annual", f"recharge_{name_scenario}.nc"))

        # plt.savefig(os.path.join(folder_plots, f"recharge_{name_scenario}.png" ), bbox_inches='tight')
    
    #plot all measures and reference 
    recharge_ref = xr.open_dataset(os.path.join(folder_plots, "grids_annual", f"recharge_Reference.nc"))
    recharge_scen = xr.open_dataset(os.path.join(folder_plots, "grids_annual", f"recharge_{name_scenario}.nc"))

    recharge_ref = recharge_ref.rename({"vertical.recharge": "recharge (mm/d)"})
    recharge_scen = recharge_scen.rename({"vertical.recharge": "recharge (mm/d)"})
    diff = recharge_scen - recharge_ref

    fs=10
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24 / 2.54, 8 / 2.54), sharex=True, sharey=True)  
    recharge_ref["recharge (mm/d)"].plot(ax=ax1, cmap = "RdBu", vmin = -25, vmax = 25)
    recharge_scen["recharge (mm/d)"].plot(ax=ax2, cmap = "RdBu", vmin = -25, vmax = 25)
    diff["recharge (mm/d)"].plot(ax=ax3, cmap = "RdBu", vmin = -25, vmax = 25)
    ax1.set_title("Reference", fontsize = fs)
    ax2.set_title(f"{name_scenario}", fontsize = fs)
    ax3.set_title("Difference (scen-ref)", fontsize = fs)
    for ax in [ax1,ax2,ax3]:
        ax.tick_params(axis="both", labelsize=fs)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_plots, f"recharge_{name_scenario}_ref.png" ), bbox_inches='tight')


    #quick plot hydrograph 













    





#%%

plt.figure(); ds["Q"].sel(index=0).plot(hue="nbs")

plt.figure(); ds["Q"].sel(index=7).groupby("time.month").mean("time").plot(hue="nbs")

plt.figure(); ds["Ea"].sel(index=7).plot(hue="nbs")
plt.figure(); ds["Ss"].sel(index=7).plot(hue="nbs")

#%% quick plots 

for index in ds.index.values:
    print(index)
    plt.figure(); ds["Qnbs"].sel(index=index).plot(hue="nbs") 
    plt.figure(); ds["Qnbs"].sel(index=index).groupby("time.month").mean("time").plot(hue="nbs") 


runs_selected = ["calib", "calib_pond_sum", "calib_pond_0p3"]
runs_selected = ["calib", "calib_pond_sum_2", "calib_pond_sum_3",]
for index in ds.index.values:
    print(index)
    plt.figure(); ds["Qnbs"].sel(index=index).sel(nbs = runs_selected).plot(hue="nbs") 
    plt.figure(); ds["Qnbs"].sel(index=index).sel(nbs = runs_selected).groupby("time.month").mean("time").plot(hue="nbs") 


# plt.figure(); gw["gwrecharge"].sel(index=0).sel(nbs = runs_selected).plot(hue="nbs") 
# plt.figure(); gw["gwrecharge"].sel(index=0).sel(nbs = runs_selected).groupby("time.month").mean("time").plot(hue="nbs") 

