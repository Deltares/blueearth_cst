"""Canonical climate figures for the wflow FORCING (rule 1.13).

The model-grid half of the pair: ``hydrology_model/forcing/inmaps_historical.nc``
is the same climate as the shared store's extraction, after the build's regrid,
lapse/pressure corrections and PET derivation. Drawing it with the SAME figure
set as the source grid (``climate_analysis.climate_figures``) is what makes
"what did the downscaling change?" answerable by putting two directories side by
side, which is why this module is now a thin caller rather than a plotter.

This module owns only what is model-specific: loading the model, masking to the
basin, and handing over the basin/river geometries as map overlays.

Two things changed with the canonical set (2026-08), both deliberate:

* **No more cartopy basemap tiles.** The previous ``plot_map_model`` called
  ``cartopy.io.img_tiles.QuadtreeTiles``, i.e. a live tile request in the middle
  of WF1. Rule 1.13 now needs no NETWORK. The basin/river context it bought is
  drawn from the model's own geometries instead, and rule 1.12's
  ``basin_area.png`` still provides the full satellite-backed map for anyone who
  wants it (that rule keeps its cartopy dependency).
* **Filenames carry the dataset.** ``precip.png`` became
  ``forcing_precip_map.png`` and friends — see ``climate_figures.figure_names``.
"""

import os
from os.path import join
from pathlib import Path
from typing import Optional, Union

from blueearth_cst.climate_analysis.climate_figures import (
    CLIMATE_VARS,
    plot_climate_figures,
)

#: Rendered on every forcing figure, so it survives the file being copied out.
_CAVEAT = (
    "Wflow forcing on the model grid: the build's regrid, lapse/pressure "
    "corrections and PET. Compare against the source_* figures to see what "
    "downscaling changed."
)


def plot_forcing(
    wflow_root: Union[str, Path],
    plot_dir: Optional[Union[str, Path]] = None,
    gauges_fn: Optional[Union[str, Path]] = None,
):
    """Write the canonical climate figure set for the wflow forcing.

    Parameters
    ----------
    wflow_root : str | Path
        The wflow model root (``hydrology_model/``).
    plot_dir : str | Path, optional
        Destination. Defaults to ``<wflow_root>/plots``; rule 1.13 passes
        ``hydrology_model/forcing/plots`` so the figures sit beside the forcing
        they describe (R07 B10).
    gauges_fn : str | Path, optional
        The config's ``output_locations`` PATH. The staticgeoms layer is
        resolved from the model (``shared.gauges``) rather than derived from
        this name, and a configured file whose layer cannot be found warns
        instead of skipping quietly.
    """
    from hydromt_wflow import WflowSbmModel

    from blueearth_cst.shared.gauges import gauges_layer_name

    mod = WflowSbmModel(str(wflow_root), mode="r")
    if plot_dir is None:
        plot_dir = os.path.join(str(wflow_root), "plots")

    forcing = mod.forcing.data
    staticmaps = mod.staticmaps.data
    geoms = mod.geoms.data

    missing = [var for var in CLIMATE_VARS if var not in forcing]
    if missing:
        raise ValueError(
            f"plot_forcing: {wflow_root} forcing is missing {missing}; the "
            f"canonical climate set needs {list(CLIMATE_VARS)}"
        )

    # Mask to the modelled basin. Cells outside it carry forcing values (the
    # forcing grid is rectangular) that are not part of the model's climate, and
    # including them shifts every domain mean.
    inside = staticmaps["subcatchment"] >= 0
    ds = forcing[list(CLIMATE_VARS)].where(inside)

    overlays = {"basins": mod.basins, "rivers": mod.rivers}
    if "outlets" in geoms:
        overlays["outlets"] = geoms["outlets"]
    gauges_layer = gauges_layer_name(geoms, gauges_fn)
    if gauges_layer is not None:
        overlays["gauges"] = geoms[gauges_layer]

    return plot_climate_figures(
        ds, plot_dir, "forcing", caveat=_CAVEAT, overlays=overlays
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            project_dir = sm.params.project_dir

            # R07 B10: forcing / model-input QA figures live beside the forcing
            # they describe, inside the engine subtree.
            plot_forcing(
                wflow_root=f"{project_dir}/hydrology_model",
                plot_dir=f"{project_dir}/hydrology_model/forcing/plots",
                gauges_fn=sm.params.output_locations,
            )
    else:
        plot_forcing(
            wflow_root=join(os.getcwd(), "test_case", "my_project", "hydrology_model")
        )
