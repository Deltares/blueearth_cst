"""Prepare a hydromt build/update workflow YAML for adding forcing to wflow."""

from pathlib import Path
from typing import Optional, Union

import yaml

from hydromt_wflow import WflowSbmModel


def prep_hydromt_update_forcing_config(
    starttime: str,
    endtime: str,
    fn_yml: Union[str, Path] = "wflow_build_forcing_historical.yml",
    precip_source: str = "era5",
    wflow_root: Optional[Union[str, Path]] = None,
    store_source: Optional[str] = None,
):
    """Write a hydromt 1.x `steps:`-format YAML to add forcing to a wflow model.

    ``store_source`` names the catalog entry the forcing is READ from, when that
    differs from the source it came from. Rule 1.10 passes the climate store's
    entry so the forcing is built from the extraction rule 1.04 already made,
    instead of re-reading the global dataset from the catalog a second time.

    It is a separate argument rather than a substituted ``precip_source``
    because the branch below keys on the real source: eobs picks a different
    orography and PET method, and passing the store's name would silently take
    the era5 branch. What is read and what it IS are two different questions.
    """
    if precip_source == "eobs":
        clim_source = "eobs"
        oro_source = "eobs_orography"
        pet_method = "makkink"
    else:
        clim_source = "era5"
        oro_source = "era5_orography"
        pet_method = "debruin"

    if wflow_root is not None:
        mod = WflowSbmModel(root=wflow_root, mode="r")
        size = mod.staticmaps.data.raster.size
        if size > 1e6:
            chunksize = 1
        elif size > 2.5e5:
            chunksize = 30
        elif size > 1e5:
            chunksize = 100
        else:
            chunksize = 365
    else:
        chunksize = 30

    workflow = {
        "modeltype": "wflow_sbm",
        "steps": [
            {
                "setup_config": {
                    "data": {
                        "time.starttime": starttime,
                        "time.endtime": endtime,
                        "time.timestepsecs": 86400,
                        # B2: forcing now lives INSIDE the hydromt model root, so the pointer
                        # is a plain relative path with no ../ hop out of it.
                        "input.path_forcing": "forcing/inmaps_historical.nc",
                    }
                }
            },
            {
                "setup_precip_forcing": {
                    # The store carries precip AND the climate variables in one
                    # file (the extraction merges them for a precip-only source
                    # like chirps), so both steps read the same entry when it is
                    # in play. `dem_forcing_fn` below still resolves from the
                    # main catalog -- the store holds no orography.
                    "precip_fn": store_source or precip_source,
                    "chunksize": chunksize,
                }
            },
            {
                "setup_temp_pet_forcing": {
                    "temp_pet_fn": store_source or clim_source,
                    "press_correction": True,
                    "temp_correction": True,
                    "dem_forcing_fn": oro_source,
                    "pet_method": pet_method,
                    "skip_pet": False,
                    "chunksize": chunksize,
                }
            },
        ],
    }

    fn_yml = Path(fn_yml)
    fn_yml.parent.mkdir(parents=True, exist_ok=True)
    with fn_yml.open("w") as f:
        yaml.safe_dump(workflow, f, sort_keys=False)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import log_row, tee_to_log

        with tee_to_log(sm.log[0]):
            prep_hydromt_update_forcing_config(
                starttime=sm.params.starttime,
                endtime=sm.params.endtime,
                fn_yml=sm.output.forcing_yml,
                precip_source=sm.params.clim_source,
                wflow_root=sm.params.basin_dir,
            )
            log_row(
                f"Prepared forcing config {sm.params.starttime}..{sm.params.endtime} "
                f"(clim_source={sm.params.clim_source}) -> {sm.output.forcing_yml}",
                module="forcing",
            )
    else:
        prep_hydromt_update_forcing_config(
            starttime="2010-01-01T00:00:00",
            endtime="2010-12-31T00:00:00",
            fn_yml="wflow_build_forcing_historical.yml",
            precip_source="era5",
        )
