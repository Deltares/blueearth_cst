import os
import yaml
import hydromt
from copy import deepcopy
from pathlib import Path
from typing import Union, List


def prepare_clim_data_catalog(
    fns: List[Union[str, Path]],
    data_libs_like: Union[str, Path],
    source_like: str,
    fn_out: Union[str, Path] = None,
    oro_path: Union[str, Path] = None,
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
    oro_path: str or Path, Optional
        Absolute path to the ``<source_like>_orography.nc`` sidecar produced by
        the historical extraction, under the dataset+window keyed store dir. Used
        only for the chirps/chirps_global branch. Passed explicitly by the caller
        (design §4a) so the path is not reconstructed by fragile ``../..``
        walking from a realization NC — which broke on the store keying and on
        every subsequent move of the realization NC dir (now
        ``experiments/<name>/climate/weathergenr/output/``; R07 B5 moved it,
        R9 P2 moved it again, which is the point). Required
        when ``source_like`` is chirps/chirps_global.

    Returns
    -------
    climate_data_catalog: hydromt.DataCatalog
        Data catalog of the new sources in fns.
    """

    data_catalog = hydromt.DataCatalog(data_libs=data_libs_like)
    dc_like = data_catalog.to_dict()[source_like]

    climate_data_dict = dict()

    for fn in fns:
        fn = Path(fn).resolve()
        name = os.path.basename(fn).split(".")[0]
        dc_fn = deepcopy(dc_like)
        dc_fn["uri"] = str(fn)
        # The R-generated NetCDFs are written one realization per file with
        # variables already in their standard units, so override the inherited
        # driver to the netcdf-friendly raster_xarray with a harmonise pass
        # and drop any unit/rename adapters that came from the source.
        dc_fn["driver"] = {
            "name": "raster_xarray",
            "options": {
                "preprocess": "harmonise_dims",
                "lock": False,
            },
        }
        dc_fn.pop("data_adapter", None)
        dc_fn.pop("root", None)

        metadata = dc_fn.setdefault("metadata", {})
        if source_like in ("chirps", "chirps_global"):
            metadata["processing"] = (
                f"Climate data generated from {source_like} for precipitation "
                "and era5 using Deltares/weathergenr"
            )
        else:
            metadata["processing"] = (
                f"Climate data generated from {source_like} using Deltares/weathergenr"
            )

        climate_data_dict[name] = dc_fn

    # Add local orography for chirps resolution
    if source_like in ("chirps", "chirps_global"):
        # The sidecar lives beside extract_historical.nc under the keyed store
        # dir; the caller passes its absolute path (design §4a). No ../.. walk.
        if oro_path is None:
            raise ValueError(
                f"oro_path is required for source_like={source_like!r} "
                "(the <source>_orography.nc sidecar under the keyed store dir)"
            )
        fn_oro = Path(oro_path).resolve()
        climate_data_dict[f"{source_like}_orography"] = {
            "data_type": "RasterDataset",
            "uri": str(fn_oro),
            "driver": {
                "name": "raster_xarray",
                "options": {
                    "chunks": {"latitude": 100, "longitude": 100},
                    "lock": False,
                },
            },
            "metadata": {
                "category": "topography",
                "processing": (
                    f"Resampled DEM from MERIT Hydro to the resolution of {source_like}"
                ),
                "crs": 4326,
            },
        }

    if fn_out is not None:
        # Dump the dict directly with PyYAML rather than via
        # `DataCatalog().from_dict(...).to_yml(...)` because hydromt 1.3
        # silently strips driver.options.preprocess on `to_dict`/`to_yml`,
        # which would lose the `harmonise_dims` step the R-generated nc
        # files rely on (their dim order is longitude/latitude/time).
        # The downstream reader (DataCatalog(data_libs=...)) parses YAML
        # via from_dict, which DOES preserve preprocess.
        with open(fn_out, "w") as f:
            yaml.safe_dump(climate_data_dict, f, sort_keys=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            # Read the two list of nc files and combine
            nc_fns = sm.input.cst_nc
            nc_fns2 = sm.input.rlz_nc
            nc_fns.extend(nc_fns2)

            prepare_clim_data_catalog(
                fns=nc_fns,
                data_libs_like=sm.params.data_sources,
                source_like=sm.params.clim_source,
                fn_out=sm.output.clim_data,
                oro_path=getattr(sm.params, "oro_path", None),
            )
    else:
        raise ValueError("This script should be run from a snakemake environment")
