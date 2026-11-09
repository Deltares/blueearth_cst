"""Default spatial GCM diagnostics, independent of scenario generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from blueearth_cst.plotting.fundamentals import MONTH_ABBR
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.projections.projection_figures import period_folder
from blueearth_cst.projections.grid_weights import midpoint_edges
from blueearth_cst.shared.plot_colors import (
    PROJECTION_CHANGE_COLORS,
)
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.snake_utils import save_figure


def _token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _cmap(variable: str):
    return LinearSegmentedColormap.from_list(
        variable, PROJECTION_CHANGE_COLORS[variable]
    )


def render_gcm_diagnostics(
    tables,
    record,
    root,
    *,
    grids=None,
    geometries=(),
    scenarios=None,
    annotation="none",
) -> dict:
    """Render all available native-grid change maps.

    Map pages preserve model/scenario identity. Members are summarized by a
    cell-wise median within a model/scenario only. Native grids differ across
    models and are never interpolated onto a false common fine grid.
    """
    root = Path(root)
    manifest = {}

    def publish(fig, relative, text, window, **extra):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_figure(
                path,
                fig=fig,
                dpi=600,
                annotation=annotation,
                title="GCM projection changes",
                caption=text,
                facecolor="white",
            )
        finally:
            plt.close(fig)
        manifest[relative] = {
            "title": "GCM projection changes",
            "caption": text,
            "window_id": window["window_id"],
            "horizons": window.get("horizons", []),
            **extra,
        }

    with use_preset(CST_PRESET):
        if grids is not None:
            _spatial_maps(Path(grids), geometries, record, publish, scenarios)
    return manifest


def _spatial_maps(grids, geometries, record, publish, scenarios):
    import geopandas as gpd
    from blueearth_cst.plotting.prepared_maps import (
        MapScale,
        RasterMap,
        draw_raster_matrix,
        draw_raster_panels,
    )
    from blueearth_cst.shared.map_frames import basin_frame, features, overlay_features

    inventory = json.loads((grids / "inventory.json").read_text(encoding="utf-8"))
    layers = {}
    for path in geometries:
        path = Path(path)
        if path.suffix.lower() == ".geojson" and path.stem in {
            "basins",
            "subbasins",
            "rivers",
            "locations",
            "region",
        }:
            layer = gpd.read_file(path)
            if layer.crs is None:
                raise ValueError(f"map geometry requires a CRS: {path}")
            layers[path.stem] = layer.to_crs(4326)
    basin = layers.get("basins", layers.get("region"))
    if basin is None or basin.empty:
        raise ValueError("spatial GCM diagnostics require basin or region geometry")
    layers["basins"] = basin.dissolve()
    frame = basin_frame(layers["basins"])
    overlays = overlay_features(
        {name: features(value) for name, value in layers.items()}
    )
    from matplotlib.collections import QuadMesh
    from matplotlib.path import Path as PlotPath
    from shapely.plotting import patch_from_polygon

    geometry = layers["basins"].geometry.union_all()
    polygons = (
        list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    )
    clip_path = PlotPath.make_compound_path(
        *(patch_from_polygon(polygon).get_path() for polygon in polygons)
    )

    def clip_maps(fig):
        for ax in fig.axes:
            if np.allclose(ax.get_xlim(), frame.extent[:2]) and np.allclose(
                ax.get_ylim(), frame.extent[2:]
            ):
                for artist in ax.collections:
                    if isinstance(artist, QuadMesh):
                        artist.set_clip_path(clip_path, ax.transData)

    loaded = []
    for member in inventory["members"]:
        if scenarios and member["scenario"] not in scenarios:
            continue
        path = grids / member["path"]
        if not path.resolve().is_relative_to(grids.resolve()):
            raise ValueError(
                "spatial diagnostic member path escapes its retained directory"
            )
        if member.get("sha256") and file_sha256(path) != member["sha256"]:
            raise ValueError(f"retained spatial diagnostic is damaged: {path.name}")
        with xr.open_dataset(path) as ds:
            loaded.append((member, ds.load()))

    def raster(field):
        x = next(name for name in ("longitude", "lon", "x") if name in field.dims)
        y = next(name for name in ("latitude", "lat", "y") if name in field.dims)
        return RasterMap(
            field.transpose(y, x).values,
            midpoint_edges(field[x].values),
            midpoint_edges(field[y].values),
        )

    for window in record["windows"]:
        folder = period_folder(window["window_id"])
        members = [
            (identity, ds)
            for identity, ds in loaded
            if identity["window_id"] == window["window_id"]
        ]
        if not members:
            continue
        models = sorted({identity["model"] for identity, _ in members})
        scenario_names = sorted({identity["scenario"] for identity, _ in members})
        grouped = {}
        for identity, ds in members:
            grouped.setdefault((identity["model"], identity["scenario"]), []).append(ds)
        span = f"{window['used']} relative to {window['reference']}"
        for variable in ("temp", "precip"):
            if not any(f"{variable}_annual" in ds for _, ds in members):
                continue
            summaries = {
                key: xr.concat(values, "member_sample", join="exact").median(
                    "member_sample", skipna=True
                )
                for key, values in grouped.items()
            }
            unit = "°C" if variable == "temp" else "%"
            for period in ("annual", "monthly"):
                key = f"{variable}_{period}"
                finite = np.concatenate(
                    [ds[key].values.ravel() for ds in summaries.values() if key in ds]
                )
                finite = finite[np.isfinite(finite)]
                if not len(finite):
                    continue
                limit = max(float(np.abs(finite).max()), 0.01)
                bounds = MaxNLocator(nbins=8, symmetric=True).tick_values(-limit, limit)
                cmap = _cmap(variable)
                scale = MapScale(
                    unit,
                    bounds,
                    bounds,
                    cmap(np.linspace(0, 1, len(bounds) - 1)),
                    cmap(0.0),
                    cmap(1.0),
                    "#dddddd",
                    "available",
                )
                if period == "annual":
                    matrices, titles = [], []
                    for model in models:
                        row, labels = [], []
                        template = next(
                            ds for (m, _), ds in summaries.items() if m == model
                        )
                        for scenario in scenario_names:
                            ds = summaries.get((model, scenario))
                            value = (
                                ds[key]
                                if ds is not None and key in ds
                                else xr.full_like(template[key], np.nan)
                            )
                            row.append(raster(value))
                            labels.append("" if ds is not None else "No members")
                        matrices.append(row)
                        titles.append(labels)
                    figures = draw_raster_matrix(
                        matrices,
                        [m.rsplit("/", 1)[-1] for m in models],
                        scenario_names,
                        titles,
                        frame,
                        scale,
                        overlays,
                        colorbar_title=f"Change ({unit})",
                    )
                    for page, fig in enumerate(figures, 1):
                        clip_maps(fig)
                        counts = {
                            f"{m}/{s}": len(values)
                            for (m, s), values in grouped.items()
                        }
                        publish(
                            fig,
                            f"{folder}/cmip6_{variable}_annual_change_map_models_page{page:02d}_basin.png",
                            f"Annual native-grid GCM changes, {span}. Models are rows and scenarios columns; "
                            "cell-wise median of available paired member changes within each model/scenario. "
                            "Temperature uses day-weighted annual means; precipitation uses annual duration integrals, "
                            "not averaged monthly percentages. Models retain their own grid; no interpolation. "
                            "Cells require complete monthly inputs in both windows. Gray marks unavailable cells or "
                            "model/scenario combinations. Display is clipped to the basin outline without resampling. "
                            "Shared signed color scale uses the UNESCO Figure 13 hues. "
                            f"Page {page}/{len(figures)}; member counts: {counts}.",
                            window,
                            member_counts=counts,
                        )
                else:
                    for (model, scenario), ds in summaries.items():
                        if key not in ds:
                            continue
                        panels = [
                            raster(ds[key].sel(month=month)) for month in range(1, 13)
                        ]
                        figures = draw_raster_panels(
                            panels,
                            MONTH_ABBR,
                            frame,
                            scale,
                            overlays,
                            colorbar_title=f"Change ({unit})",
                        )
                        for page, fig in enumerate(figures, 1):
                            clip_maps(fig)
                            publish(
                                fig,
                                f"{folder}/cmip6_{variable}_monthly_change_map_{_token(model)}_{_token(scenario)}_page{page:02d}_basin.png",
                                f"Monthly native-grid {variable} changes for {model}/{scenario}, {span}; "
                                f"cell-wise member median, n={len(grouped[(model, scenario)])}. "
                                "Months use their own reference means. Temperature is an additive offset (°C); "
                                "precipitation is percent change, 100 × (future/reference − 1). "
                                "Dry-reference or incomplete cells are unavailable, not zero. All months and members "
                                f"are retained in the diagnostic grids; page {page}/{len(figures)}. "
                                "Display is clipped to the basin outline without resampling. "
                                "Shared signed color scale uses the UNESCO Figure 13 hues.",
                                window,
                            )
