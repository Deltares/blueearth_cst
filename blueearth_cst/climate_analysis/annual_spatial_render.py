"""WF0 annual anomaly display preparation; drawing belongs to prepared_maps."""

import json
import re
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from blueearth_cst.climate_analysis.annual_spatial_anomalies import (
    anomaly_paths,
    load_retained,
)
from blueearth_cst.plotting.prepared_maps import (
    MapFrame,
    MapScale,
    RasterMap,
    draw_raster_panels,
)
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.map_frames import basin_frame, map_caption
from blueearth_cst.shared.map_reference import (
    drawing_frame,
    field_edges,
    frame_intersection,
)
from blueearth_cst.shared.plot_colors import (
    CLIMATE_COLORMAPS,
    MAP_FEATURE_COLORS,
    PRECIP_ANOMALY_COLORMAP,
)
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.snake_utils import save_figure


def native_frame(field: xr.DataArray, reference: dict, overlays: dict) -> MapFrame:
    """Match climatology viewport and explicit singleton-cell edge semantics."""
    x, y = field_edges(field)
    frame = drawing_frame(reference)
    if overlays.get("basins"):
        import geopandas as gpd

        frame = basin_frame(
            gpd.GeoDataFrame.from_features(overlays["basins"], crs="EPSG:4326"),
            within=(x.min(), x.max(), y.min(), y.max()),
        )
    return frame


def prepare_display(
    roots: list, reference: dict, overlays: dict, settings: dict
) -> tuple[dict, dict]:
    """Pool only finite displayed comparison cells across enabled native sources."""
    loaded, limits = {}, {}
    for root in roots:
        fields, record = load_retained(root)
        for key in (
            "reference_period",
            "comparison_period",
            "max_missing_days_month",
            "max_missing_days_year",
        ):
            if record["settings"][key] != settings[key]:
                raise ValueError(f"{root}: stale annual spatial anomaly {key}")
        loaded[record["source"]] = (fields, record)
        for variable in record["variables"]:
            field = fields[f"{variable}_anomaly"]
            frame = native_frame(field, reference, overlays)
            support = frame_intersection(field, frame.extent)
            values = field.sel(reporting_year=record["comparison_years"]).values[
                :, support
            ]
            finite = values[np.isfinite(values)]
            if finite.size:
                limits[variable] = max(
                    limits.get(variable, 0.0), float(np.abs(finite).max())
                )
    scales = {}
    for variable in {v for _, record in loaded.values() for v in record["variables"]}:
        maximum = limits.get(variable)
        limit = maximum if maximum else 1.0
        scales[variable] = {
            "maximum_absolute": maximum,
            "limits": [-limit, limit] if maximum is not None else None,
            "all_zero": maximum == 0,
            "fallback": "display-only ±1; all finite anomalies are zero"
            if maximum == 0
            else None,
            "status": "available" if maximum is not None else "unavailable",
            "scope": "finite comparison-year cells intersecting each displayed native frame; all candidate sources",
        }
    return loaded, scales


def render_pages(
    paths: dict,
    settings: dict,
    roots: list,
    reference: dict,
    overlays: dict,
    annotation: str,
) -> dict:
    """Render every declared page and return ordinary title/caption records."""
    loaded, scales = prepare_display(roots, reference, overlays, settings)
    # Source identity is explicit in the source-owned retained record.
    source = json.loads(
        Path(anomaly_paths(paths["root"])["record"]).read_text(encoding="utf-8")
    )["source"]
    fields, record = loaded[source]
    out_root = Path(paths.get("render_root", paths["root"]))
    records = {}
    for form, filename in paths["figures"].items():
        if not form.startswith("annual-map-"):
            continue
        _, _, variable, page = form.split("-")
        years = record["pages"][int(page) - 1]
        field = fields[f"{variable}_anomaly"]
        frame = native_frame(field, reference, overlays)
        support = frame_intersection(field, frame.extent)
        x, y = field_edges(field)
        info = scales[variable]
        cmap = mpl.colormaps[
            PRECIP_ANOMALY_COLORMAP
            if variable == "precip"
            else CLIMATE_COLORMAPS["temp_crossing_freezing"]
        ]
        boundaries = np.linspace(*info["limits"], 257) if info["limits"] else []
        scale = MapScale(
            "mm" if variable == "precip" else "degC",
            boundaries,
            np.linspace(*info["limits"], 5) if info["limits"] else [],
            cmap(np.linspace(0, 1, 256)) if info["limits"] else [],
            cmap(0.0),
            cmap(1.0),
            MAP_FEATURE_COLORS["missing"],
            info["status"],
            extend="neither",
        )
        notes, counts, rasters, details = [], [], [], []
        for year in years:
            values = field.sel(reporting_year=year).values
            count = int((support & np.isfinite(values)).sum())
            reason = (
                "Incomplete/out-of-span year"
                if year in record["comparison_partial"]
                else "No cells in frame"
                if not support.any()
                else "Reference unavailable"
                if not record["variables"][variable]["reference_cells"]
                else "No valid cell-years"
                if not count
                else ""
            )
            notes.append(reason if not count else "")
            counts.append(count)
            rasters.append(RasterMap(values, x, y, cell_edges=False))
            dates = record["year_dates"][str(year)]
            details.append(
                f"{year}: {dates[0]} to {dates[1]}, {count}/{int(support.sum())} cells"
                + (f", {reason.lower()}" if not count else "")
            )
        label = "Precipitation" if variable == "precip" else "Temperature"
        units = "mm" if variable == "precip" else "°C"
        eligible = record["reference_eligible"]
        ref_dates = [
            record["year_dates"][str(eligible[0])][0],
            record["year_dates"][str(eligible[-1])][1],
        ]
        methods_text = map_caption(
            f"Annual {label.lower()} anomalies ({units}) on the {source.upper()} native grid",
            source,
            [
                f"Annual {'totals' if variable == 'precip' else 'daily means'} minus each cell's arithmetic mean over {len(eligible)} eligible reference reporting years ({ref_dates[0]} to {ref_dates[1]}); every eligible reference year is required at each cell",
                f"Requested reference {record['reference_requested'][0]}–{record['reference_requested'][-1]}; excluded incomplete/out-of-span years: {record['reference_excluded_partial'] or 'none'}",
                "One reference year only; this is not a climate normal"
                if record["single_reference_year"]
                else "",
                f"Start-year labels, month {record['water_year_start']}; at most {settings['max_missing_days_month']} missing days per month and {settings['max_missing_days_year']} per year; tolerated gaps use that calendar month's observed mean",
                "Native cells intersecting the frame include cells outside the basin; no polygon mask or regridding. Gray is missing; white is uncovered. Geographic aspect uses midpoint latitude",
                "Symmetric zero-centred scale pooled over displayed cells of all candidate sources and comparison years, without clipping"
                if info["limits"]
                else "No finite anomaly scale is available",
                info["fallback"] or "",
                "; ".join(details),
                "Descriptive departures, not trends or significance",
            ],
        )
        text = (
            f"Annual {label.lower()} anomalies ({units}) from {source.upper()} "
            f"for {years[0]}–{years[-1]}, relative to the {eligible[0]}–{eligible[-1]} mean. "
            + (
                "Red indicates drier and blue wetter conditions."
                if variable == "precip"
                else "Blue indicates cooler and red warmer conditions."
            )
        )
        if any(count < int(support.sum()) for count in counts):
            text += " Gray denotes unavailable estimates."
        title = f"Annual {label.lower()} anomalies — {source.upper()} ({years[0]}–{years[-1]}), reference {eligible[0]}–{eligible[-1]}, page {int(page)}"
        # Annual pages retain their existing three-column, six-year contract.
        preset = replace(
            CST_PRESET,
            map_panels={
                **CST_PRESET.map_panels,
                "columns": 3,
                "max_panels_per_figure": 6,
            },
        )
        with use_preset(preset):
            (fig,) = draw_raster_panels(
                rasters,
                [f"{chr(97 + i)}) {year}" for i, year in enumerate(years)],
                frame,
                scale,
                overlays,
                colorbar_title=f"Anomaly\n({units})",
            )
            for ax, note in zip(fig.axes[: len(rasters)], notes):
                if note:
                    ax.text(
                        0.5,
                        0.5,
                        note,
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=preset.fonts["note"],
                        wrap=True,
                        bbox=preset.label_box,
                    )
            try:
                save_figure(
                    filename,
                    fig=fig,
                    annotation=annotation,
                    title=title,
                    caption=text,
                    dpi=600,
                    facecolor="white",
                )
            finally:
                plt.close(fig)
        relative = Path(filename).relative_to(out_root).as_posix()
        records[relative] = {
            "title": title,
            "caption": text,
            "methods_detail": methods_text,
            "sources": [source],
            "encoding": "annual_spatial_anomalies",
            "variable": variable,
            "period": [years[0], years[-1]],
            "years": years,
            "reference": settings["reference_period"],
            "reference_eligible": eligible,
            "support": "native source cells intersecting displayed frame",
            "display_frame": list(frame.extent),
            "scale": info,
            "displayed_cells": int(support.sum()),
            "finite_cells_by_year": counts,
            "unavailable_reasons": notes,
            "fields_sha256": record["fields_sha256"],
            "availability": "available"
            if any(counts)
            else "unavailable: no finite anomalies",
        }
    return records


def retire_pages(
    out_root: str | Path, previous: dict, current: dict, source: str
) -> None:
    """Retire only exact, previously managed anomaly PNGs; never follow links."""
    root = Path(out_root).resolve()
    pattern = re.compile(
        rf"plots/annual_anomalies/page_[0-9]{{3,}}/{re.escape(source)}_(precip|temp)_annual_anomaly_map_basin(?:_ext)?\.png"
    )
    for relative, record in previous.items():
        if (
            relative in current
            or record.get("encoding") != "annual_spatial_anomalies"
            or not pattern.fullmatch(relative)
        ):
            continue
        path = root / relative
        if any(
            part.is_symlink() or getattr(part, "is_junction", lambda: False)()
            for part in [path, *path.parents]
            if part != root
        ):
            continue
        if path.resolve().is_relative_to(root) and path.is_file():
            path.unlink()


def retained_enabled(root: str | Path) -> bool:
    """A retained numerical file alone does not re-enable a retired figure family."""
    inventory = Path(root) / "plots/_engine/figure_captions.json"
    if not inventory.is_file():
        return False
    records = json.loads(inventory.read_text(encoding="utf-8"))["figures"]
    return any(
        record.get("encoding") == "annual_spatial_anomalies"
        for record in records.values()
    )
