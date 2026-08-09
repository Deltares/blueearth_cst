"""Canonical climate figures for the wflow FORCING (rule 1.13).

The model-grid half of the pair:
``models/hydrology/wflow/forcing/inmaps_historical.nc``
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
  drawn from the model's own geometries instead. Rule 1.12's ``basin_area``
  dropped the same tiles in 2026-08 for the same reason plus two more —
  licence/attribution on a submitted figure, and a basemap that the server can
  re-render out from under a "reproducible" run — so NO rule in WF1 fetches
  tiles any more. It draws its terrain context as a hillshade of the model's own
  DEM (``shared.cartographic_map``), which is offline, reproducible, and finer
  tiles were at the zoom level that rule had hardcoded.
* **Filenames carry the dataset.** ``precip.png`` became
  ``forcing_precip_map.png`` and friends — see ``climate_figures.figure_names``.
"""

import os
from os.path import join
from pathlib import Path
from typing import Optional, Union

from blueearth_cst.climate_analysis.climate_figures import (
    CLIMATE_VARS,
    load_spatial_overlays,
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
    geoms_dir: Optional[Union[str, Path]] = None,
):
    """Write the canonical climate figure set for the wflow forcing.

    Parameters
    ----------
    wflow_root : str | Path
        The wflow model root (``models/hydrology/wflow/``).
    plot_dir : str | Path, optional
        Destination. Defaults to ``<wflow_root>/plots``; rule 1.13 passes
        ``models/hydrology/wflow/forcing/plots`` so the figures sit beside the
        forcing they describe (R07 B10).
    gauges_fn : str | Path, optional
        Retained for the rule's signature; the map overlays no longer come from
        the model's own gauge layer (see ``geoms_dir``).
    geoms_dir : str | Path, optional
        ``data/spatial/geoms/`` — the ENGINE-NEUTRAL vector foundation from rule
        1.03, and the SAME layers the source-grid maps draw. Both climate map
        families read it so the two differ only in the raster underneath, which
        is what makes "what did downscaling change?" answerable by putting the
        two directories side by side.
    """
    from hydromt_wflow import WflowSbmModel

    mod = WflowSbmModel(str(wflow_root), mode="r")
    if plot_dir is None:
        plot_dir = os.path.join(str(wflow_root), "plots")

    forcing = mod.forcing.data
    staticmaps = mod.staticmaps.data

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

    return plot_climate_figures(
        ds,
        plot_dir,
        "forcing",
        caveat=_CAVEAT,
        overlays=load_spatial_overlays(geoms_dir),
    )


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            project_dir = sm.params.project_dir

            # R07 B10: forcing / model-input QA figures live beside the forcing
            # they describe, inside the engine subtree.
            # Rule-owned model root (R9 P2 commit 1): the script no longer
            # rebuilds `{project_dir}/hydrology_model` for itself.
            model_dir = sm.params.model_dir
            plot_forcing(
                wflow_root=model_dir,
                plot_dir=f"{model_dir}/forcing/plots",
                gauges_fn=getattr(sm.input, "output_locations", None),
                geoms_dir=sm.params.geoms_dir,
            )
    else:
        plot_forcing(
            wflow_root=join(os.getcwd(), "test_case", "my_project", "hydrology_model")
        )
