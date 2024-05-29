import matplotlib.pyplot as plt
import numpy as np
import os

from hydromt import flw
import pandas as pd
from hydromt_wflow import WflowModel
import xarray as xr

from typing import Union, List, Optional
from pathlib import Path

__all__ = ["plot_near_far_abs"]


def plot_near_far_abs(
          qsim_delta_metric: xr.DataArray, 
          q_hist_metric: xr.DataArray,
          index: int,
          plot_dir: str,
          ylabel: str,
          figname_prefix: str,
          cmap: List,
          fs: int = 8,  
):
    """
    todo
    figname_prefix -- xticks 1, 12 
    """
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (16/2.54, 8/2.54), sharex=True, sharey=True)
    for scenario, color in zip(qsim_delta_metric.scenario.values, cmap):
        #first entry just for legend 
        qsim_delta_metric.sel(index=index).sel(horizon="near").sel(scenario=scenario).sel(model=qsim_delta_metric.model[0]).plot(label = f"{scenario}", ax=ax1, color = color)
        qsim_delta_metric.sel(index=index).sel(horizon="far").sel(scenario=scenario).sel(model=qsim_delta_metric.model[0]).plot(label = f"{scenario}", ax=ax2, color = color)

        #plot all lines
        qsim_delta_metric.sel(index=index).sel(horizon="near").sel(scenario=scenario).plot(hue="model", ax=ax1, color = color, add_legend=False)
        qsim_delta_metric.sel(index=index).sel(horizon="far").sel(scenario=scenario).plot(hue="model", ax=ax2, color = color, add_legend=False)
    for ax in [ax1,ax2]:
        q_hist_metric.sel(index=index).plot(label = "hist", color = "k", ax=ax)    
        ax.tick_params(axis="both", labelsize = fs)
        ax.set_xlabel("")
    ax1.set_ylabel(f"{ylabel}", fontsize = fs); ax2.set_ylabel("")
    ax1.set_title("near future", fontsize=fs); ax2.set_title("far future", fontsize=fs)
    if figname_prefix == "mean_monthly_Q":
        ax1.set_xticks(np.arange(1,13)); ax2.set_xticks(np.arange(1,13))
    ax1.legend(fontsize=fs); ax2.legend(fontsize=fs) 
    plt.savefig(os.path.join(plot_dir, f"{figname_prefix}_{index}.png"), dpi=300)