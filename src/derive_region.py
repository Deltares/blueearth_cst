"""Derive region vector file from hydromt region dictionary."""

import os
from os.path import join, dirname
from pathlib import Path
from typing import Union, Optional, List
from ast import literal_eval

from hydromt.workflows import parse_region, get_basin_geometry
from hydromt import DataCatalog

import geopandas as gpd
from shapely.geometry import box


def derive_region(
    region: Union[dict, str],
    path_output: Union[Path, str],
    buffer: Optional[float] = None,
    data_catalog: Optional[Union[List, str, Path]] = None,
    hydrography_fn: str = "merit_hydro",
    basin_index_fn: str = "merit_hydro_index",
):
    """Prepare the `region` of interest of the project.

    Adds the following files in path_output:

    * **region.geojson**: region boundary vector
    * **region_buffer.geojson**: region vector with buffer if requested

    Parameters
    ----------
    region : dict
        Dictionary describing region of interest, e.g.:

        * {'bbox': [xmin, ymin, xmax, ymax]}

        * {'geom': 'path/to/polygon_geometry'}

        * {'basin': [xmin, ymin, xmax, ymax]}

        * {'subbasin': [x, y], '<variable>': threshold}

        For a complete overview of all region options,
        see :py:function:~hydromt.workflows.basin_mask.parse_region
    path_output : str
        Path to output directory where region files are stored.
    buffer : float, optional
        Buffer distance in kilometres to add around the region boundary.
    data_catalog : str, list, optional
        Data catalogs to use. Only required if the `region` is
        based on a (sub)(inter)basins.
    hydrography_fn : str
        Name of data source for hydrography data. Only required if the `region` is
        based on a (sub)(inter)basins without a 'bounds' argument.
    basin_index_fn : str
        Name of data source with basin (bounding box) geometries associated with
        the 'basins' layer of `hydrography_fn`. Only required if the `region` is
        based on a (sub)(inter)basins without a 'bounds' argument.

    See Also
    --------
    hydromt.workflows.basin_mask.parse_region
    """
    # Parse region from string to dict if needed
    if isinstance(region, str):
        region = literal_eval(region)
    # Parse region dictionary
    kind, region = parse_region(region, data_catalog=data_catalog)

    # Derive region geometry
    if kind in ["basin", "subbasin", "interbasin", "outlet"]:
        data_catalog = DataCatalog(data_libs=data_catalog)
        # retrieve global hydrography data (lazy!)
        ds_org = data_catalog.get_rasterdataset(hydrography_fn)
        if "bounds" not in region:
            region.update(basin_index=data_catalog.get_source(basin_index_fn))
        # get basin geometry
        geom, xy = get_basin_geometry(
            ds=ds_org,
            kind=kind,
            **region,
        )
        region.update(xy=xy)
    elif "bbox" in region:
        bbox = region["bbox"]
        geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
    elif "geom" in region:
        geom = region["geom"]
        if geom.crs is None:
            raise ValueError('Model region "geom" has no CRS')
    else:
        raise ValueError(f"model region argument not understood: {region}")

    # Save region to file
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    region_fn = join(path_output, "region.geojson")
    geom.to_file(region_fn, driver="GeoJSON")

    # Add buffer if requested
    if buffer is not None and buffer > 0:
        geom_buffer = geom.copy()
        # make sure geom is projected > buffer in meters!
        if geom_buffer.crs.is_geographic:
            geom_buffer = geom_buffer.to_crs(3857)
        geom_buffer = geom_buffer.buffer(buffer * 1000)
        geom_buffer = geom_buffer.to_crs(geom.crs)
    else:
        geom_buffer = geom

    # Save region buffer to file
    region_buffer_fn = join(path_output, "region_buffer.geojson")
    geom_buffer.to_file(region_buffer_fn, driver="GeoJSON")


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        region_file = sm.output.region_file

        derive_region(
            region=sm.params.hydromt_region,
            path_output=dirname(region_file),
            buffer=sm.params.buffer_km,
            data_catalog=sm.params.data_catalog,
            hydrography_fn=sm.params.hydrography_fn,
            basin_index_fn=sm.params.basin_index_fn,
        )

    else:
        print("This script should be run from a snakemake environment")
