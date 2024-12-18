import os
import hydromt
from pathlib import Path
from typing import Union, List


def prepare_clim_data_catalog(
    fns: List[Union[str, Path]],
    data_libs_like: Union[str, Path],
    source_like: str,
    fn_out: Union[str, Path] = None,
):
    """
    Prepares a data catalog for files path listed in fns using the same attributes as source_like
    in data_libs_like.
    If fn_out is provided writes the data catalog to that path.

    Parameters
    ----------
    fns: list(path)
        Path to the new data sources files.
    data_libs_like: str or list(str)
        Path to the existing data catalog where source_like is stored.
    source_like: str
        Data sources with the same attributes as the new sources in fns.
    fn_out: str, Optional
        If provided, writes the new data catalog to the corresponding path.

    Returns
    -------
    climate_data_catalog: hydromt.DataCatalog
        Data catalog of the new sources in fns.
    """

    data_catalog = hydromt.DataCatalog(data_libs=data_libs_like)
    dc_like = data_catalog[source_like].to_dict()

    climate_data_catalog = hydromt.DataCatalog()
    climate_data_dict = dict()

    for fn in fns:
        fn = Path(fn, resolve_path=True)
        name = os.path.basename(fn).split(".")[0]
        dc_fn = dc_like.copy()
        dc_fn["path"] = fn
        dc_fn["driver"] = "netcdf"
        if "driver_kwargs" not in dc_fn:
            dc_fn["driver_kwargs"] = dict()
        dc_fn["driver_kwargs"]["preprocess"] = "transpose_dims"
        dc_fn["driver_kwargs"]["lock"] = False
        if source_like == "chirps" or source_like == "chirps_global":  # precip only
            dc_fn["meta"][
                "processing"
            ] = f"Climate data generated from {source_like} for precipitation and era5 using Deltares/weathergenr"
        else:
            dc_fn["meta"][
                "processing"
            ] = f"Climate data generated from {source_like} using Deltares/weathergenr"
        # remove entries that have already been processed while reading in the data:
        for v in ["unit_mult", "unit_add", "rename"]:
            if v in dc_fn:
                dc_fn.pop(v)
        climate_data_dict[name] = dc_fn

    # Add local orography for chirps resolution
    if source_like == "chirps" or source_like == "chirps_global":
        fn_oro = Path(fns[0], resolve_path=True)
        fn_oro = os.path.join(
            os.path.dirname(fn_oro),
            "..",
            "..",
            "climate_historical",
            "raw_data",
            f"{source_like}_orography.nc",
        )
        fn_oro = Path(fn_oro, resolve_path=True)
        dc_oro = {
            "crs": 4326,
            "data_type": "RasterDataset",
            "driver": "netcdf",
            "kwargs": {
                "chunks": {
                    "latitude": 100,
                    "longitude": 100,
                },
                "lock": False,
            },
            "meta": {
                "category": "topography",
                "processing": f"Resampled DEM from MERIT Hydro to the resolution of {source_like}",
            },
            "path": fn_oro,
        }
        climate_data_dict[f"{source_like}_orography"] = dc_oro

    climate_data_catalog.from_dict(climate_data_dict)
    if fn_out is not None:
        climate_data_catalog.to_yml(fn_out)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        # Read the two list of nc files and combine
        nc_fns = sm.input.cst_nc
        nc_fns2 = sm.input.rlz_nc
        nc_fns.extend(nc_fns2)

        prepare_clim_data_catalog(
            fns=nc_fns,
            data_libs_like=sm.params.data_sources,
            source_like=sm.params.clim_source,
            fn_out=sm.output.clim_data,
        )
    else:
        raise ValueError("This script should be run from a snakemake environment")
