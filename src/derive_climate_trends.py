"""Plot historical anomalies at several locations to see if there is a trend."""

from os.path import join, dirname, isfile
from pathlib import Path
from typing import Union, Optional, List

import xarray as xr
import hydromt

import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_anomalies import plot_timeseries_anomalies
else:
    from .plot_utils.plot_anomalies import plot_timeseries_anomalies


def derive_timeseries_trends(
    clim_filenames: List[Union[str, Path]],
    path_output: Union[str, Path],
    split_year: Optional[int] = None,
):
    """
    Plot the historical anomalies for a set of locations.

    Outputs:
    * **timeseries_trends.txt**: a file to indicate that the plots were created.
    * plots of the historical anomalies for each source and per climate
      variable.

    Parameters
    ----------
    clim_filenames : xr.Dataset
        Path to the timeseries geodataset files extracted for specific locations. They
        should contain the climate ``source`` in the coords or dims.
        Supported variables: ["precip", "temp"].
    path_output : str or Path
        Path to the output directory where the plots are stored.
    split_year : int, optional
        Derive additional trends for years before and after this year.
    """
    # Read the different geodataset file and merge them
    geods_list = []
    for climate_file in clim_filenames:
        geods = hydromt.vector.GeoDataset.from_netcdf(climate_file)
        geods_list.append(geods)

    geods = xr.concat(geods_list, dim="source")

    # Derive the anomalies and trends for each climate source
    for source in geods.source.values:
        # Filter the dataset
        geods_source = geods.sel(source=source)

        # Plot the anomalies
        plot_timeseries_anomalies(
            ds=geods_source,
            path_output=path_output,
            split_year=split_year,
            suffix=source,
        )

    if "snakemake" in globals():
        # Write a file when everything is done for snakemake tracking
        text_out = join(path_output, "timeseries_trends.txt")
        with open(text_out, "w") as f:
            f.write("Timeseries anomalies plots were made.\n")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        obs_fn = sm.params.point_observed
        clim_filenames = sm.input.point_climate
        if isfile(obs_fn):
            clim_filenames.append(obs_fn)

        derive_timeseries_trends(
            clim_filenames=clim_filenames,
            path_output=dirname(sm.output.trends_timeseries_done),
            split_year=sm.params.split_year,
        )

    else:
        print("This script should be run from a snakemake environment")
