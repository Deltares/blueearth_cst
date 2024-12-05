"""Function to plot the region and location of the data."""

from os.path import basename, dirname, isfile
from pathlib import Path
from typing import Union, Optional, List
import numpy as np
from hydromt import DataCatalog

import matplotlib.pyplot as plt
from matplotlib import colors

# Avoid relative import errors
import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.func_plot_map import plot_map
else:
    from .plot_utils.func_plot_map import plot_map


def plot_region_and_location(
    region_fn: Union[str, Path],
    fn_out: Union[str, Path],
    data_catalog: List[Union[str, Path]] = [],
    subregions_fn: Optional[Union[str, Path]] = None,
    locations_fn: Optional[Union[str, Path]] = None,
    hydrography_fn: Optional[Union[str, Path]] = None,
    rivers_fn: Optional[Union[str, Path]] = None,
    buffer_km: Optional[float] = 2.0,
    legend_loc: str = "lower right",
):
    """
    Plot the region and location of the data.

    Parameters
    ----------
    region_fn : str, Path
        Path to the region vector file.
    fn_out : str, Path
        Path to the output figure file.
    data_catalog : List of str, Path
        Path to the data catalogs yaml file or pre-defined catalogs.
    subregions_fn : str, Path, optional
        Path to the subregions vector file or data catalog entry.
        Optional variables: "name" for the subregion name.
    locations_fn : str, Path, optional
        Path to the locations vector file or data catalog entry.
        Optional variables: "name" for the location name and "elevtn" for the elevation
        of the location.
    hydrography_fn : str, Path, optional
        Path to the hydrography raster file or data catalog entry.
    rivers_fn : str, Path, optional
        Path to the rivers vector file or data catalog entry.
        Optional variables for plotting: "strord".
    buffer_km : float, optional
        Buffer in km around the region to extract the data.
    legend_loc : str, optional
        Location of the legend in the plot. Default is "lower right".
    """

    # Small function to set the index of the geodataframe
    def _update_gdf_index(gdf, legend_column="value"):
        if legend_column in gdf.columns:
            if gdf[legend_column].dtype == float:
                gdf[legend_column] = gdf[legend_column].astype(int)
            # gdf.index = f"{prefix}_" + gdf[legend_column].astype(str)
            gdf.index = gdf[legend_column]

        gdf.index.name = "index"

        return gdf

    # Initialize data catalog
    data_catalog = DataCatalog(data_catalog)

    # Read the region
    print(f"Reading region from {region_fn}")
    region = data_catalog.get_geodataframe(region_fn)
    region = _update_gdf_index(region, legend_column="value")

    # Read the subregions
    if subregions_fn is not None:
        subregions = data_catalog.get_geodataframe(subregions_fn)
        subregions = _update_gdf_index(subregions, legend_column="value")
        subregions = {"subregion": subregions}
    else:
        subregions = {}

    # Read the locations
    if locations_fn is not None:
        # If the locs are a direct file without a crs property, assume 4326
        if isfile(locations_fn):
            crs = 4326
        else:
            crs = None
        print(f"DEBUG: **** locations_fn: {locations_fn}")
        try:    
            locations = data_catalog.get_geodataframe(
                locations_fn,
                crs=crs,
                geom=region,
                buffer=buffer_km * 1000,
            )
        except:
            print(f"DEBUG: **** locations_fn: {locations_fn} failed")
        locations.index.name = "index"
        locations = {"meteorological stations": locations}
    else:
        locations = {}

    # Read the hydrography
    if hydrography_fn is not None:
        hydrography = data_catalog.get_rasterdataset(
            hydrography_fn,
            geom=region,
            variables=["elevtn"],
            single_var_as_array=True,
        )
        hydrography = hydrography.raster.clip_geom(region, mask=True)
        hydrography = hydrography.raster.mask_nodata()
        hydrography.attrs.update(long_name="elevation", units="m")

        # create nice colormap for elevation
        vmin, vmax = (
            hydrography.chunk(
                {hydrography.raster.x_dim: -1, hydrography.raster.y_dim: -1}
            )
            .quantile([0.0, 0.98])
            .compute()
        )
        c_dem = plt.cm.terrain(np.linspace(0.25, 1, 256))
        cmap = colors.LinearSegmentedColormap.from_list("dem", c_dem)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        kwargs = dict(cmap=cmap, norm=norm)
    else:
        hydrography = None
        kwargs = {}

    # Read the rivers
    if rivers_fn is not None:
        rivers = data_catalog.get_geodataframe(rivers_fn, geom=region)
    else:
        rivers = None

    plot_map(
        da=hydrography,
        figname=basename(fn_out).split(".")[0],
        plot_dir=dirname(fn_out),
        basins=region,
        subregions=subregions,
        rivers=rivers,
        gauges=locations,
        buffer_km=buffer_km,
        annotate_regions=True,
        shaded=True if hydrography is not None else False,
        legend_loc=legend_loc,
        **kwargs,
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]

        plot_region_and_location(
            region_fn=sm.input.region_file,
            fn_out=sm.output.region_plot,
            data_catalog=sm.params.data_catalog,
            subregions_fn=sm.params.subregion_file,
            locations_fn=sm.params.location_file,
            hydrography_fn=sm.params.hydrography_fn,
            rivers_fn=sm.params.river_fn,
            buffer_km=sm.params.buffer_km,
            legend_loc=sm.params.legend_loc,
        )

    else:
        print("This script should be run from a snakemake environment")
