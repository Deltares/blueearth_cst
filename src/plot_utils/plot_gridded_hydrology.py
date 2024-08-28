import os
from os.path import join
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt

from typing import Union, Optional, Dict

__all__ = ["plot_gridded_snow_cover", "plot_gridded_snow_cover_relative"]


def plot_gridded_snow_cover(
    snow_cover: dict,
    plot_dir: Union[str, Path],
    plot_filename: str = "snow_cover.png",
    gdf_basin: Optional[gpd.GeoDataFrame] = None,
    gdf_river: Optional[gpd.GeoDataFrame] = None,
    fs: int = 8,
):
    """
    Plot snow cover for different datasets.

    Parameters
    ----------
    snow_cover : dict
        Dictionary with snow cover for different datasets.
    plot_dir : Union[str, Path]
        Path to the output folder for plots.
    gdf_basin : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame with basin boundaries.
    gdf_river : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame with river network.
    fs : int, optional
        Font size for the plot. Default is 8.
    """
    # Prepare a figure with 2 columns, lines depends on the number of datasets
    fig_width = 8 if len(snow_cover) == 1 else 16
    n_cols = 2 if len(snow_cover) > 1 else 1
    n_rows = int(np.ceil(len(snow_cover) / 2))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width / 2.54, 8 / 2.54 * n_rows),
        sharey=True,
        sharex=True,
        layout="compressed",
    )
    axes = [axes] if (n_rows * n_cols) == 1 else axes.flatten()
    # Reduce the whitespace between the plots
    # plt.subplots_adjust(wspace=0.05)

    # Plot
    for i, (name, data) in enumerate(snow_cover.items()):
        ax = axes[i]
        im = data.plot(ax=ax, cmap="Blues", add_colorbar=False, vmin=0, vmax=100)
        ax.set_title(name, fontsize=fs)
        if gdf_basin is not None:
            gdf_basin.boundary.plot(ax=ax, color="red", alpha=1.0, edgecolor="red")
        if gdf_river is not None:
            gdf_river.plot(ax=ax, color="black", linewidth=gdf_river["strord"] / 2)
        ax.set_xlabel("Longitude [degrees east]", fontsize=fs)
        # Only add y label if first column
        if i % 2 == 0:
            ax.set_ylabel("Latitude [degrees north]", fontsize=fs)
        else:
            ax.set_ylabel("")
        ax.set_title(name.upper(), fontsize=fs + 1, fontweight="bold")
        ax.tick_params(axis="both", labelsize=fs)
    # Mask axes that do not have data
    for i in range(len(axes)):
        if i >= len(snow_cover):
            axes[i].axis("off")

    # Add colorbar to figure
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, aspect=30, label="% of time with snow")
    # Change the fontisize of the colorbar label
    cbar.ax.yaxis.label.set_fontsize(fs + 1)
    # Change the fontsize of the colorbar ticks
    cbar.ax.tick_params(labelsize=fs)

    # Save the figure
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(join(plot_dir, plot_filename), dpi=300, bbox_inches="tight")
    plt.close()

    return


def plot_gridded_snow_cover_relative(
    snow_cover: Dict[str, xr.DataArray],
    reference_snow_cover: str,
    plot_dir: Union[str, Path],
    plot_filename=f"snow_cover_rel.png",
    gdf_basin: Optional[gpd.GeoDataFrame] = None,
    gdf_river: Optional[gpd.GeoDataFrame] = None,
    fs: int = 8,
    use_diverging: bool = False,
):
    """
    Plot relative snow cover for different datasets compared to a reference.

    Parameters
    ----------
    snow_cover : Dict[str, xr.DataArray]
        Dictionary with snow cover for different datasets.
    reference_snow_cover : str
        Name of the reference dataset in snow_cover dict.
    plot_dir : Union[str, Path]
        Path to the output folder for plots.
    plot_filename : str, optional
        Name of the plot file. Default is 'snow_cover_rel.png'.
    gdf_basin : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame with basin boundaries.
    gdf_river : Optional[gpd.GeoDataFrame], optional
        GeoDataFrame with river network.
    fs : int, optional
        Font size for the plot. Default is 8.
    use_diverging : bool, optional
        Use a diverging colormap (red to blue) or negative only (reds). Default is
        False.
    """
    # Extract the reference snow cover
    ref = snow_cover.pop(reference_snow_cover)

    # Prepare a figure with 2 columns, lines depends on the number of datasets
    fig_width = 8 if len(snow_cover) == 1 else 16
    n_cols = 2 if len(snow_cover) > 1 else 1
    n_rows = int(np.ceil(len(snow_cover) / 2))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width / 2.54, 8 / 2.54 * n_rows),
        sharey=True,
        sharex=True,
        layout="compressed",
    )
    axes = [axes] if (n_rows * n_cols) == 1 else axes.flatten()

    # Compute relative snow cover
    for name, data in snow_cover.items():
        snow_cover[name] = 100 * (data - ref) / ref
    # Compute min and max for colorbar
    vmin = min([data.min() for data in snow_cover.values()])
    vmax = max([data.max() for data in snow_cover.values()])
    # Create a red blue colorbar centered around 0
    if use_diverging:
        minmax = max(abs(vmin), abs(vmax))
        cmap = plt.cm.get_cmap("RdBu")
        norm = plt.Normalize(vmin=-minmax, vmax=minmax)
    else:
        cmap = plt.cm.get_cmap("Reds_r")
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Plot
    for i, (name, data) in enumerate(snow_cover.items()):
        ax = axes[i]
        im = data.plot(ax=ax, cmap=cmap, add_colorbar=False, norm=norm)
        ax.set_title(name, fontsize=fs)
        if gdf_basin is not None:
            gdf_basin.boundary.plot(ax=ax, color="red", alpha=1.0, edgecolor="red")
        if gdf_river is not None:
            gdf_river.plot(ax=ax, color="black", linewidth=gdf_river["strord"] / 2)
        ax.set_xlabel("Longitude [degrees east]", fontsize=fs)
        # Only add y label if first column
        if i % 2 == 0:
            ax.set_ylabel("Latitude [degrees north]", fontsize=fs)
        else:
            ax.set_ylabel("")
        ax.set_title(name.upper(), fontsize=fs + 1, fontweight="bold")
        ax.tick_params(axis="both", labelsize=fs)
    # Mask axes that do not have data
    for i in range(len(axes)):
        if i >= len(snow_cover):
            axes[i].axis("off")

    # Add colorbar to figure
    cbar = fig.colorbar(
        im, ax=axes, shrink=0.95, aspect=30, label=f"change in % of time with snow"
    )
    # Change the fontisize of the colorbar label
    cbar.ax.yaxis.label.set_fontsize(fs + 1)
    # Change the fontsize of the colorbar ticks
    cbar.ax.tick_params(labelsize=fs)

    # Save the figure
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(join(plot_dir, plot_filename), dpi=300, bbox_inches="tight")
    plt.close()
