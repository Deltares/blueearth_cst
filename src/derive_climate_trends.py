"""Plot historical anomalies at several locations to see if there is a trend."""

from os.path import join
from pathlib import Path
from typing import Union, Optional

import xarray as xr

import sys

parent_module = sys.modules[".".join(__name__.split(".")[:-1]) or "__main__"]
if __name__ == "__main__" or parent_module.__name__ == "__main__":
    from plot_utils.plot_anomalies import plot_timeseries_anomalies
else:
    from .plot_utils.plot_anomalies import plot_timeseries_anomalies


def derive_timeseries_trends(
    clim_filename: Union[str, Path],
    path_output: Union[str, Path],
    split_year: Optional[int] = None,
):
    """
    Plot the historical anomalies for a set of locations.

    Outputs:
    * **timeseries_trends.txt**: a file to indicate that the plots were created.
    * **trends**: plots of the historical anomalies for each source and per climate
      variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with the climate data.
        Supported variables: []"precip", "temp"].
    path_output : str or Path
        Path to the output directory where the plots are stored.
    split_year : int, optional
        Derive additional trends for years before and after this year.
    """
    # Read the timeseries
    ds = xr.open_dataset(clim_filename)

    # Derive the anomalies and trends for each climate source
    for source in ds.source.values:
        # Filter the dataset
        ds_source = ds.sel(source=source)

        # Plot the anomalies
        plot_timeseries_anomalies(
            ds=ds_source,
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
        project_dir = sm.params.project_dir

        derive_timeseries_trends(
            clim_filename=sm.input.geods,
            path_output=join(project_dir, "climate_historical", "plots", "trends"),
            split_year=sm.params.split_year,
        )

    else:
        print("This script should be run from a snakemake environment")
