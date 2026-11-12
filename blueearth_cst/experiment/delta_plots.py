"""Member-resolved monthly and spatial diagnostics of retained delta factors."""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

from blueearth_cst.experiment.content_identity import read_canonical_json
from blueearth_cst.experiment.delta_scenarios import read_delta_collection
from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference


def _scenario_title(scenario: str) -> str:
    """Format standard ScenarioMIP IDs; preserve unrecognized identifiers."""
    match = re.fullmatch(r"ssp([1-5])([0-9]{2})", scenario)
    return f"SSP{match[1]}-{int(match[2]) / 10:.1f}" if match else scenario


def plot_delta_collection(
    manifest: Path,
    output: Path,
    *,
    model_root: Path | None = None,
    annual_matrix: bool = False,
    member_summary: str = "median",
) -> list[Path]:
    """Render every member without pooling across models or implying probabilities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from blueearth_cst.plotting.export import export_figure
    from blueearth_cst.plotting.fundamentals import MONTH_ABBR, new_fig
    from blueearth_cst.plotting.prepared_maps import (
        MapScale,
        RasterMap,
        draw_raster_matrix,
        draw_raster_panels,
    )
    from blueearth_cst.plotting.preset import use_preset
    from blueearth_cst.shared.map_frames import (
        basin_frame,
        features,
        grid_edges,
        overlay_features,
    )
    from blueearth_cst.shared.plot_colors import (
        CLIMATE_COLORMAPS,
        PRECIP_ANOMALY_COLORMAP,
    )
    from blueearth_cst.shared.plot_preset import CST_PRESET

    manifest = manifest.resolve(strict=True)
    marker = read_delta_collection(manifest)
    from blueearth_cst.experiment.collection_index import marker_project_root
    from blueearth_cst.experiment.delta_scenarios import collection_input_root

    project = marker_project_root(manifest)
    inputs = collection_input_root(project, marker)
    intent = read_canonical_json(
        resolve_file_reference(marker["intent"], {"project_root": project})
    )
    output = output.resolve()
    if output.is_relative_to(project / "scenarios/_engine"):
        raise ValueError("plots must be outside immutable collection storage")
    settings = intent["settings"]
    if settings["schema_version"] == "delta-change/3":
        # Multi-window figures land with the delta showcase (t016 commit 7).
        raise ValueError(
            "delta plots do not yet support WF2-sourced (multi-window) sets"
        )
    if member_summary not in {"median", "mean"}:
        raise ValueError("member_summary must be median or mean")
    with xr.open_dataset(inputs / settings["historical_path"]) as historical:
        if annual_matrix:
            historical.load()
        active = historical.precip.notnull().any("time")
        if settings.get("factor_layout") == "grid":
            active = active.transpose("latitude", "longitude")
        active = active.values
        if settings.get("factor_layout") == "grid" or annual_matrix:
            xedges, yedges = grid_edges(historical.precip.isel(time=0))
    if settings.get("factor_layout") == "grid" or annual_matrix:
        import geopandas as gpd

        geometry_root = (
            model_root or project / "models/hydrology/wflow"
        ) / "staticgeoms"
        basins = gpd.read_file(geometry_root / "basins.geojson")
        if basins.empty or basins.crs is None:
            raise ValueError(
                "basin map requires nonempty geometry with an explicit CRS"
            )
        basins = basins.to_crs(4326).dissolve()
        layers = {"basins": features(basins)}
        missing_layers = []
        empty_features = []
        for layer in ("subbasins", "rivers", "locations"):
            path = geometry_root / f"{layer}.geojson"
            if path.exists():
                loaded = gpd.read_file(path)
                if loaded.crs is None:
                    raise ValueError(f"map layer requires an explicit CRS: {path}")
                empty = loaded.geometry.isna() | loaded.geometry.is_empty
                if empty.any():
                    empty_features.append(f"{layer}: {int(empty.sum())}")
                    loaded = loaded.loc[~empty]
                layers[layer] = features(loaded)
            else:
                missing_layers.append(layer)
        overlays = overlay_features(layers)
        frame = basin_frame(basins)
    groups = defaultdict(list)
    for index, member in enumerate(settings["members"]):
        groups[(member["model"], member["scenario"])].append(index)
    paths, captions = [], []
    variables = [
        ("temp_offset_degC", "Temp.", "°C", 0),
        ("precip_ratio", "Precip.", "%", 1),
    ]
    if intent["factors"] and "pet_ratio" in intent["factors"][0]:
        variables.append(("pet_ratio", "PET", "%", 1))
    output.mkdir(parents=True, exist_ok=True)
    with use_preset(CST_PRESET):
        for group, indices in enumerate(groups.values(), 1):
            fig, axes = new_fig(85, ncols=len(variables), width_mm=190)
            labels = []
            for column, (key, label, unit, neutral) in enumerate(variables):
                ax = axes[0, column]
                for number, index in enumerate(indices):
                    member = settings["members"][index]
                    values = np.asarray(intent["factors"][index][key])
                    if neutral:
                        values = 100 * (values - 1)
                    if values.ndim == 3:
                        values = np.where(active[None, :, :], values, np.nan)
                        low, high = (
                            np.nanmin(values, axis=(1, 2)),
                            np.nanmax(values, axis=(1, 2)),
                        )
                        values = np.nanmean(values, axis=(1, 2))
                    (line,) = ax.plot(
                        range(1, 13),
                        values,
                        marker=("o", "s", "^")[number % 3],
                        linestyle=("-", "--", ":")[number % 3],
                        label=member["member"],
                    )
                    if settings.get("factor_layout") == "grid":
                        ax.fill_between(
                            range(1, 13), low, high, color=line.get_color(), alpha=0.12
                        )
                    if column == 0:
                        labels.append(
                            f"{member['model']} / {member['scenario']} / {member['member']}"
                        )
                ax.axhline(0, color="0.5", linewidth=0.5)
                ax.set_xticks(range(1, 13), MONTH_ABBR, rotation=90)
                ax.set_ylabel(
                    f"Relative to {settings['reference_period']['start']}–{settings['reference_period']['end']} ({unit})"
                )
                ax.set_title(f"{chr(97 + column)}) {label}", loc="left")
                ax.legend(loc="best", frameon=False)
            path = output / f"comparison_{group:04d}_delta_monthly.png"
            export_figure(fig, path, dpi=200)
            plt.close(fig)
            paths.append(path)
            captions.append(
                f"{path.name}: {', '.join(labels)}. Future {settings['future_period']}. "
                "Grid lines are unweighted active-cell means; shaded ranges are spatial minima/maxima, "
                "not confidence intervals. Scalar lines show basin factors. Members are not pooled."
            )
        if settings.get("factor_layout") == "grid":
            for index, member in enumerate(settings["members"]):
                for key, label, unit, neutral in variables:
                    values = np.asarray(intent["factors"][index][key])
                    if neutral:
                        values = 100 * (values - 1)
                    values = np.where(active[None, :, :], values, np.nan)
                    limit = max(
                        max(
                            float(
                                np.nanmax(
                                    np.abs(
                                        (np.asarray(item[key]) - 1) * 100
                                        if neutral
                                        else np.asarray(item[key])
                                    )
                                )
                            )
                            for item in intent["factors"]
                        ),
                        1e-6,
                    )
                    cmap = plt.get_cmap(
                        PRECIP_ANOMALY_COLORMAP
                        if key == "precip_ratio"
                        else CLIMATE_COLORMAPS["temp_crossing_freezing"]
                    )
                    # Round outward to readable whole-number classes, retaining
                    # one symmetric scale for every member and month.
                    limit = max(limit, 1.0)
                    boundaries = MaxNLocator(
                        nbins=10, integer=True, symmetric=True, steps=[1, 2, 5, 10]
                    ).tick_values(-limit, limit)
                    scale = MapScale(
                        unit,
                        boundaries,
                        boundaries,
                        cmap(np.linspace(0.05, 0.95, len(boundaries) - 1)),
                        cmap(0.0),
                        cmap(1.0),
                        "#dddddd",
                        "available",
                    )
                    figures = draw_raster_panels(
                        [
                            RasterMap(values[month], xedges, yedges)
                            for month in range(12)
                        ],
                        MONTH_ABBR,
                        frame,
                        scale,
                        overlays,
                        colorbar_title=f"{label}\nchange ({unit})",
                    )
                    maximum = CST_PRESET.map_panels["max_panels_per_figure"]
                    for page, fig in enumerate(figures, 1):
                        suffix = f"_page_{page:02d}" if len(figures) > 1 else ""
                        path = output / (
                            f"run_{index + 3:0{intent['run_group_id_width']}d}_{key.lower()}"
                            f"_monthly_map_basin_ext{suffix}.png"
                        )
                        export_figure(fig, path, dpi=200)
                        plt.close(fig)
                        paths.append(path)
                        page_months = MONTH_ABBR[(page - 1) * maximum : page * maximum]
                        captions.append(
                            f"{path.name}: {member['model']} / {member['scenario']} / {member['member']}; "
                            f"{', '.join(page_months)}, left to right by row; page {page}/{len(figures)}; "
                            f"future {settings['future_period']}. "
                            "Factors on the historical forcing grid; common scale across members and months. "
                            f"Compact {CST_PRESET.map_panels['columns']}-column panels retain basin outline, subbasin divides, rivers "
                            "and location markers; coordinate and location labels are omitted. "
                            "Basin zoom does not add subgrid climate information. "
                            f"Unavailable optional layers: {', '.join(missing_layers) or 'none'}. "
                            f"Empty geometry records omitted: {', '.join(empty_features) or 'none'}."
                        )
        if annual_matrix:
            import csv

            from blueearth_cst.experiment.delta_annual import (
                annual_member_maps,
                summarize_members,
            )

            fields = annual_member_maps(inputs, settings, historical)
            models = list(
                dict.fromkeys(member["model"] for member in settings["members"])
            )
            scenarios = list(
                dict.fromkeys(member["scenario"] for member in settings["members"])
            )
            rows = []
            for key, groups in fields.items():
                summaries = {
                    group: summarize_members(values, member_summary)
                    for group, values in groups.items()
                }
                limit = max(
                    1.0,
                    max(
                        float(np.nanmax(np.abs(values)))
                        for values in summaries.values()
                    ),
                )
                boundaries = MaxNLocator(
                    nbins=10, integer=True, symmetric=True, steps=[1, 2, 5, 10]
                ).tick_values(-limit, limit)
                label = {
                    "temp_offset_degc": "Temp.",
                    "precip_change_percent": "Precip.",
                    "pet_change_percent": "PET",
                }[key]
                unit = "°C" if key == "temp_offset_degc" else "%"
                cmap = plt.get_cmap(
                    PRECIP_ANOMALY_COLORMAP
                    if key.startswith("precip")
                    else CLIMATE_COLORMAPS["temp_crossing_freezing"]
                )
                scale = MapScale(
                    unit,
                    boundaries,
                    boundaries,
                    cmap(np.linspace(0.05, 0.95, len(boundaries) - 1)),
                    cmap(0.0),
                    cmap(1.0),
                    "#dddddd",
                    "available",
                )
                empty = np.full((len(yedges) - 1, len(xedges) - 1), np.nan)
                rasters, titles = [], []
                for model in models:
                    rasters.append([])
                    titles.append([])
                    for scenario in scenarios:
                        values = summaries.get((model, scenario), empty)
                        count = len(groups.get((model, scenario), []))
                        rasters[-1].append(RasterMap(values, xedges, yedges))
                        titles[-1].append("" if count else "No members")
                        for iy, lat in enumerate(historical.latitude.values):
                            for ix, lon in enumerate(historical.longitude.values):
                                if np.isfinite(values[iy, ix]):
                                    rows.append(
                                        (
                                            model,
                                            scenario,
                                            key,
                                            member_summary,
                                            count,
                                            float(lat),
                                            float(lon),
                                            float(values[iy, ix]),
                                            unit,
                                        )
                                    )
                figures = draw_raster_matrix(
                    rasters,
                    models,
                    [_scenario_title(scenario) for scenario in scenarios],
                    titles,
                    frame,
                    scale,
                    overlays,
                    colorbar_title=f"{label}\nchange ({unit})",
                )
                for page, fig in enumerate(figures, 1):
                    suffix = f"_page_{page:02d}" if len(figures) > 1 else ""
                    path = (
                        output
                        / f"comparison_{key}_annual_{member_summary}_map_matrix{suffix}.png"
                    )
                    export_figure(fig, path, dpi=200)
                    plt.close(fig)
                    paths.append(path)
                    captions.append(
                        f"{path.name}: Annual projection change, {settings['future_period']} versus {settings['reference_period']}; "
                        f"cell-wise {member_summary} of member changes; model rows and scenario columns; page {page}/{len(figures)}. "
                        "Temperature uses day-weighted annual means; precipitation/PET use mean annual totals, not averaged monthly percentages. "
                        "Each input uses its own calendar; annual changes are mapped before member aggregation. "
                        f"Source layout: {settings.get('factor_layout', 'basin_scalar')}; basin-scalar inputs produce spatially uniform maps. "
                        "Paired member counts: "
                        + "; ".join(
                            f"{model} / {_scenario_title(scenario)}: {len(groups.get((model, scenario), []))}"
                            for model in models
                            for scenario in scenarios
                        )
                        + ". Members are equally weighted; no cross-model pooling or independence claim. "
                        "Grey panels labelled No members identify absent combinations. These are projection climatology changes, "
                        "not annual changes of the transformed historical simulation forcing."
                    )
            with (output / "annual_change_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "model",
                        "scenario",
                        "variable",
                        "member_summary",
                        "member_count",
                        "latitude",
                        "longitude",
                        "change",
                        "units",
                    )
                )
                writer.writerows(rows)
    (output / "figure_captions.md").write_text(
        "\n\n".join(captions) + "\n", encoding="utf-8"
    )
    return paths
