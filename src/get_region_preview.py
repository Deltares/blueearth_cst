""" A command line script for delineating basins and subbasin with river geometries.
    The script expects a region string in the following format:
    -r "{'basin': [x,y]}"

    A data catalog file path:
    -d path/to/datacatalog

    A path to an output directory for the region GeoJSON
    -p path/to/output/dir

    There is also the option to use different hydrography and river files:
    -h <hydrography file name>, by default merit_hydro_ihu
    -n <rivers file name>, by default rivers_atlas_v10

    The resulting (sub)basin is written as a GeoJSON to the given output directory
    in a folder named region    
"""

from json import loads
import argparse
import logging
from typing import List
import os

import hydromt
from hydromt.cli.api import get_region
import geopandas as gpd
import pandas as pd


logger = logging.getLogger(__name__)


def get_basin_preview(
    region: dict, datacatalog_fn: str | List, hydrography_fn: str = "merit_hydro_ihu"
) -> dict | None:
    try:
        region_geojson = get_region(
            region, datacatalog_fn, hydrography_fn=hydrography_fn
        )

        region_geojson = loads(region_geojson)
        region = gpd.GeoDataFrame.from_features(region_geojson, crs=4326)
        return region
    except IndexError as e:
        logger.warning(f"Region out of index, see following error: {e}")
        return None


def get_river_preview(
    region: gpd.GeoDataFrame,
    data_catalog_fn: str | List,
    rivers_fn: str = "rivers_atlas_v10",
) -> dict | None:
    datacatalog = hydromt.DataCatalog(data_libs=data_catalog_fn)
    surface_water_source = datacatalog.get_source(source=rivers_fn)
    try:
        surface_water_data = surface_water_source.get_data(geom=region.geometry)
        return surface_water_data
    except IndexError as e:
        logger.warning(f"River geometry out of index, see following error{e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Get preview of region and rivers for a given geometry"
    )
    parser.add_argument(
        "-r",
        "--region",
        help="Geometry of interest for which a basin/subbasin needs to be delineated.",
    )
    parser.add_argument("-d", "--datacatalog", help="Path to data catalog")
    parser.add_argument("-p", "--path", help="Path to save the region geojson file to")
    parser.add_argument(
        "-f",
        "--hydrography_fn",
        help="hydrography file name for delineating (sub)basins",
        required=False,
        default="merit_hydro_ihu",
    )
    parser.add_argument(
        "-n",
        "--rivers_fn",
        help="file name of rivers dataset to use",
        required=False,
        default="rivers_atlas_v10",
    )
    args = parser.parse_args()
    region_json = args.region.replace("'", '"')
    region = loads(region_json)
    region_geom = get_basin_preview(
        region=region,
        datacatalog_fn=args.datacatalog,
        hydrography_fn=args.hydrography_fn,
    )
    river_geom = get_river_preview(
        region=region_geom,
        data_catalog_fn=args.datacatalog,
        rivers_fn=args.rivers_fn,
    )

    region = gpd.GeoDataFrame(pd.concat([region_geom, river_geom]))
    file_dir = os.path.join(args.path, "region")
    if not os.path.isdir(file_dir):
        os.mkdir(file_dir)
    file_path = os.path.join(file_dir, "region.geojson")
    region_geojson = region.to_file(filename=file_path, driver="GeoJSON")
