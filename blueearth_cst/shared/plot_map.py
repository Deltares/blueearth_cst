# -*- coding: utf-8 -*-
"""The wflow basin map: read a model off disk and draw its DEM (rule 1.12).

Created 2022-01-13 (@author: bouaziz); refactored in R3 into a guarded
function; rebuilt in 2026-08 as a publication-grade figure; reduced in 2026-08
to what is actually wflow-specific once the cartography moved to
``shared.cartographic_map``.

What is left here is the reading half and one style choice:

* ``load_basin_layers`` opens a model's own files directly — ``xarray`` for
  ``staticmaps.nc``, ``geopandas`` for ``staticgeoms/*.geojson`` — so the whole
  render path imports neither ``hydromt`` nor ``hydromt_wflow``, and anyone
  holding those two artifacts can draw the figure.
* ``plot_basin_map`` calls the template with the elevation style, which is the
  only one that asks for a hillshade and the only one kept on a LINEAR scale:
  elevation is the quantity a reader does arithmetic on.
* ``plot_basin_map_from_model`` resolves a model directory into layers and
  writes ``basin_area.{pdf,png}``. This is what the Snakemake rule runs.

Verified equivalent before the hydromt drop, not assumed: rendering the same
model through ``WflowSbmModel`` and through the files produced byte-identical
images (0 differing pixels of 3,754,080).

What is genuinely given up: ``mod.rivers`` and ``mod.basins`` are FALLBACK
properties (``hydromt_wflow/wflow_base.py``), reconstructing the network from
the flow-direction raster via ``pyflwdir`` when the geojson is absent. A model
whose ``staticgeoms/`` lacks them raises instead of being rebuilt — a
deliberate trade, and the reason ``load_basin_layers`` names the missing layers
rather than failing obscurely downstream.
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from blueearth_cst.shared.cartographic_map import (
    ELEVATION_LABEL,
    RASTER_DPI,
    RASTER_STYLES,
    RasterStyle,
    _class_levels,
    _mask_nodata,
    plot_raster_map,
)
from blueearth_cst.shared.snake_utils import save_figure


# ---------------------------------------------------------------------------
# Model layout on disk. These four names are hydromt_wflow's write conventions,
# not ours -- they are stated here, as constants, precisely BECAUSE this module
# now reads the files itself instead of asking hydromt where they are. If a
# future hydromt_wflow changes a name, this block is the whole blast radius.
# ---------------------------------------------------------------------------
#: Model root, relative to ``project_dir``. R9 P2 commit 1 moved it under
#: ``models/``. Kept as a default only -- the RULE passes ``model_dir``, and
#: this constant serves standalone callers (dev/scripts/preview_basin_map.py)
#: that have no rule to ask.
MODEL_DIRNAME = "models/hydrology/wflow"
#: Gridded model parameters; carries the DEM this figure shades.
STATICMAPS_FILENAME = "staticmaps.nc"
#: Vector layers, one GeoJSON per layer, stem == layer name.
STATICGEOMS_DIRNAME = "staticgeoms"
#: The DEM variable inside ``staticmaps.nc`` (a CSDMS Standard Name).
ELEVATION_VARIABLE = "land_elevation"

# --- the engine-neutral spatial products (ADR 0007) --------------------------
# basin_area depicts ELEVATION, which is data rather than a model result, so it
# is drawn from the shared spatial foundation instead of from the wflow model.

#: Project-relative home of the shared spatial products (rule 1.03 / 1.06).
SPATIAL_DIRNAME = "data/spatial"
#: The hydrography grid stack; carries ``elevation`` on the model's own grid.
HYDROGRAPHY_FILENAME = "hydrography.nc"
#: The DEM variable inside it. Spelled plainly here, not as a CSDMS name:
#: this file is ours, not hydromt_wflow's.
SPATIAL_ELEVATION_VARIABLE = "elevation"
#: Vector layers the figure draws, mapped to the argument each one feeds.
SPATIAL_MAP_LAYERS = {
    "basins": "basins",
    "subbasins": "subbasins",
    "rivers": "rivers",
    "gauges": "locations",
}

#: Layers the figure cannot be drawn without; everything else is optional.
REQUIRED_GEOM_LAYERS = ("rivers", "basins")


def _elevation_levels(dem):
    """Class boundaries for the elevation ramp.

    The template's rule, applied with elevation's own style. Kept as a name
    because the elevation bar is the one this module owns; the arithmetic is
    ``cartographic_map._class_levels`` and is not duplicated here.
    """
    return _class_levels(dem, RASTER_STYLES["elevation"])


def load_basin_layers(model_dir):
    """Read a wflow model's DEM and vector layers straight off disk.

    No hydromt: ``xarray`` for the grid, ``geopandas`` for the geometries. Any
    directory holding ``staticmaps.nc`` and a ``staticgeoms/`` of GeoJSON works,
    so a caller does not need a model object — or the packages to build one.

    Parameters
    ----------
    model_dir : str | Path
        The model root (the folder containing ``staticmaps.nc``).

    Returns
    -------
    (elevation, rivers, basins, geoms)
        ``geoms`` maps layer name to GeoDataFrame for EVERY GeoJSON found, so
        optional layers (waterbodies, gauges) resolve by name exactly as they did
        through ``mod.geoms.data``.
    """
    model_dir = Path(model_dir)
    staticmaps_path = model_dir / STATICMAPS_FILENAME
    staticgeoms_dir = model_dir / STATICGEOMS_DIRNAME
    if not staticmaps_path.is_file():
        raise FileNotFoundError(f"no {STATICMAPS_FILENAME} in {model_dir}")
    if not staticgeoms_dir.is_dir():
        raise FileNotFoundError(f"no {STATICGEOMS_DIRNAME}/ in {model_dir}")

    with xr.open_dataset(staticmaps_path) as dataset:
        if ELEVATION_VARIABLE not in dataset:
            raise KeyError(
                f"{staticmaps_path} has no {ELEVATION_VARIABLE!r}; it holds "
                f"{sorted(dataset.data_vars)}"
            )
        # load() before the file closes -- the render touches values repeatedly.
        elevation = _mask_nodata(dataset[ELEVATION_VARIABLE].load())

    geoms = {
        path.stem: gpd.read_file(path)
        for path in sorted(staticgeoms_dir.glob("*.geojson"))
    }
    missing = [name for name in REQUIRED_GEOM_LAYERS if name not in geoms]
    if missing:
        # Named explicitly: hydromt would have REBUILT these from the flow
        # direction raster, so their absence is the one case where dropping
        # hydromt changes behaviour. Say so here rather than fail downstream.
        raise FileNotFoundError(
            f"{staticgeoms_dir} is missing {missing}; found {sorted(geoms)}. "
            "hydromt_wflow would derive these from staticmaps; this reader does not."
        )
    return elevation, geoms["rivers"], geoms["basins"], geoms


def _basin_outline(gdf_bas):
    """The basin's OUTER boundary, dissolved to a single polygon.

    ``mod.basins`` returns one polygon PER SUBCATCHMENT once gauges are burned
    into the subcatchment map — four for a four-gauge model, one for a model
    with no user gauges. Drawing them all at boundary weight makes an internal
    divide indistinguishable from the basin outline, which is the one line on
    this figure a reader has to be able to trust. Observed 2026-08-03 on a real
    four-gauge project; the single-basin fixture cannot surface it.
    """
    return gdf_bas.dissolve()


def plot_basin_map(dem, rivers, basin, *, elevation_label=ELEVATION_LABEL, **kwargs):
    """Draw the basin's DEM as a shaded-relief map — rule 1.12's figure.

    A thin caller of :func:`plot_raster_map` with the elevation style. It is
    kept as its own name because the elevation map is the one figure that wants
    a hillshade, and because ``plot_basin_map_from_model`` and the Snakemake
    rule have called it since R3.

    Every keyword :func:`plot_raster_map` takes is forwarded, so a caller can
    still override the extent, the label columns, or the style.
    """
    style = kwargs.pop("style", None)
    if style is None:
        style = RASTER_STYLES["elevation"]
    if elevation_label != style.label:
        # The caller named the units; carry that onto the bar rather than
        # letting the style's default contradict it.
        style = RasterStyle(
            label=elevation_label,
            palette=style.palette,
            classification=style.classification,
            clip_quantiles=style.clip_quantiles,
            zero_baseline=style.zero_baseline,
            relief=style.relief,
            interpolation=style.interpolation,
            diverging_center=style.diverging_center,
        )
    return plot_raster_map(dem, rivers, basin, style=style, **kwargs)



def load_spatial_basin_layers(spatial_dir):
    """Read the shared spatial products into the layers the template draws.

    The model-free counterpart of :func:`load_basin_layers`. Everything comes
    from ``data/spatial/`` — the elevation grid from ``hydrography.nc`` and the
    vectors from ``geoms/`` — so this figure needs no wflow model and can be
    drawn before one exists.

    What is NOT here, and is the known cost of the move: waterbodies. Lakes,
    reservoirs and glaciers reach ``staticgeoms/`` from rule 1.08, a MODEL rule,
    and the shared foundation carries none of them. Producing them data-side is
    the fix, tracked separately — see ADR 0007.
    """
    spatial_dir = Path(spatial_dir)
    hydrography = spatial_dir / HYDROGRAPHY_FILENAME
    geoms_dir = spatial_dir / "geoms"
    if not hydrography.is_file():
        raise FileNotFoundError(f"no {HYDROGRAPHY_FILENAME} in {spatial_dir}")
    with xr.open_dataset(hydrography) as dataset:
        if SPATIAL_ELEVATION_VARIABLE not in dataset:
            raise KeyError(
                f"{hydrography} has no {SPATIAL_ELEVATION_VARIABLE!r}; it holds "
                f"{sorted(dataset.data_vars)}"
            )
        elevation = _mask_nodata(dataset[SPATIAL_ELEVATION_VARIABLE].load())

    layers = {}
    for argument, stem in SPATIAL_MAP_LAYERS.items():
        path = geoms_dir / f"{stem}.geojson"
        if path.is_file():
            layers[argument] = gpd.read_file(path)
    missing = [n for n in ("basins", "rivers") if n not in layers]
    if missing:
        raise FileNotFoundError(f"{geoms_dir} is missing {missing}")
    return elevation, layers


def plot_basin_map_from_spatial(spatial_dir, plot_dir=None):
    """Render basin_area.{pdf,png} from the shared spatial products.

    Replaces ``plot_basin_map_from_model`` as rule 1.12's entry point. Reading
    the foundation rather than the model also retires that rule's HDF5 race
    workaround: it no longer opens ``staticmaps.nc``, so it no longer has to be
    ordered behind every writer of that file.
    """
    spatial_dir = Path(spatial_dir)
    if plot_dir is None:
        plot_dir = spatial_dir / "plots"
    elevation, layers = load_spatial_basin_layers(spatial_dir)
    basins = layers["basins"]
    fig, _ = plot_basin_map(
        elevation,
        layers.get("rivers"),
        _basin_outline(basins),
        subbasins=layers.get("subbasins"),
        gauges=layers.get("gauges"),
    )
    save_figure(
        os.path.join(str(plot_dir), "basin_area.pdf"),
        fig=fig,
        metadata={"CreationDate": None},
    )
    save_figure(
        os.path.join(str(plot_dir), "basin_area.png"),
        fig=fig,
        dpi=RASTER_DPI,
        metadata={"Software": None},
    )
    plt.close(fig)


def plot_basin_map_from_model(project_dir, gauges_fn, plot_dir=None, model_dir=None):
    """Render basin_area.{pdf,png} for a wflow model on disk.

    The file-reading half of the figure: it resolves a model directory into the
    layers ``plot_basin_map`` takes, then saves the result. This is what the
    Snakemake rule calls; anything that already holds the layers should call
    ``plot_basin_map`` directly.

    The gauge layer is resolved from the MODEL (``shared.gauges``), not from the
    configured filename: hydromt_wflow renames ``output_locations`` to
    ``output-locations``, and deriving the name here is what silently dropped
    the gauges from this figure (2026-08-01).
    """
    from blueearth_cst.shared.gauges import gauges_layer_name

    # The model root is the RULE's fact when a rule is calling; the constant is
    # the fallback for standalone callers (R9 P2 commit 1).
    root = str(model_dir) if model_dir else f"{project_dir}/{MODEL_DIRNAME}"
    if plot_dir is None:
        # basin_area depicts the MODEL, not its evaluation, so it sits at the
        # model root's plots/ -- not under evaluation/ (P1).
        plot_dir = f"{root}/plots"

    # Read straight off disk -- no model object, no hydromt (see module docstring).
    dem, rivers, basins, geoms = load_basin_layers(root)
    # ``basins`` is one polygon PER SUBCATCHMENT once gauges are burned into the
    # subcatchment map, and a single polygon otherwise. Split it into the two
    # roles the figure draws: a dissolved outline, and the divides — which exist
    # only when there is more than one subcatchment to divide.
    gauges_name = gauges_layer_name(geoms, gauges_fn)
    fig, _ = plot_basin_map(
        dem,
        rivers,
        _basin_outline(basins),
        subbasins=basins if len(basins) > 1 else None,
        # Resolved against what the model actually holds; ``gauges_layer_name``
        # warns loudly (never skips silently) when output_locations is set but
        # no layer matches.
        gauges=geoms.get(gauges_name) if gauges_name is not None else None,
        outlets=geoms.get("outlets"),
        lakes=geoms.get("lakes"),
        reservoirs=geoms.get("reservoirs"),
        glaciers=geoms.get("glaciers"),
    )

    # No bbox_inches="tight": it re-crops to the drawn content, which throws
    # away the declared 180 mm width. Constrained layout already fits the
    # furniture inside that width.
    save_figure(
        os.path.join(plot_dir, "basin_area.pdf"),
        fig=fig,
        # Drop the timestamp so two identical runs produce identical bytes.
        metadata={"CreationDate": None},
    )
    save_figure(
        os.path.join(plot_dir, "basin_area.png"),
        fig=fig,
        dpi=RASTER_DPI,
        # Same reason: the default embeds the matplotlib version, which
        # would move the baseline fingerprint on every env bump.
        metadata={"Software": None},
    )
    plt.close(fig)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        from blueearth_cst.shared.snake_utils import tee_to_log

        with tee_to_log(sm.log[0]):
            plot_basin_map_from_spatial(spatial_dir=sm.params.spatial_dir)
