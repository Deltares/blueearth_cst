"""Render retained CSV/NetCDF products and one recoverable caption inventory."""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis import diagnostic_figures as artists
from blueearth_cst.climate_analysis.diagnostic_captions import caption, title
from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.diagnostic_tables import table_renames, write_json
from blueearth_cst.climate_analysis.diagnostics import (
    monthly_reference_diagnostics,
    sen_trend,
)
from blueearth_cst.plotting.export import export_figure
from blueearth_cst.plotting.figure_titles import write_figure_titles
from blueearth_cst.plotting.fundamentals import new_fig
from blueearth_cst.plotting.prepared_maps import RasterMap, draw_raster_map
from blueearth_cst.plotting.preset import use_preset
from blueearth_cst.shared.map_frames import (
    basin_frame,
    map_caption,
    overlay_features,
)
from blueearth_cst.shared.map_reference import (
    drawing_frame,
    drawing_scale,
    field_edges,
    frame_intersection,
    load_reference,
    reference_arguments,
    validate_fields,
)
from blueearth_cst.shared.plot_preset import CST_PRESET
from blueearth_cst.shared.snake_utils import listed, log_row, plural


def load_map_overlays(overlays):
    """Load existing offline GeoJSON features in the adapter, never the core."""
    return overlay_features(
        {
            key: json.loads(Path(path).read_text(encoding="utf-8"))["features"]
            for key, path in (overlays or {}).items()
            if Path(path).is_file()
        }
    )


#: Caption leads of the historical climate maps, with their display units.
MAP_QUANTITIES = {
    "precip": "Mean annual precipitation (mm yr⁻¹)",
    "temp": "Mean annual temperature (°C)",
    "pet": "Mean annual potential evapotranspiration (PET, mm yr⁻¹)",
}


def prepared_map(
    field, counts, reference, variable, overlays, *, inside=None, cell_edges=True
):
    """Draw basin-focused native fields with the declared historical color scale."""
    field = field.transpose("latitude", "longitude")
    counts = counts.transpose("latitude", "longitude")
    mask = None
    if inside is not None:
        field, inside = xr.align(field, inside, join="exact")
        mask = inside.transpose("latitude", "longitude").values
    x, y = field_edges(field)
    frame = drawing_frame(reference)
    if overlays.get("basins"):
        import geopandas as gpd

        frame = basin_frame(
            gpd.GeoDataFrame.from_features(overlays["basins"], crs="EPSG:4326"),
            # Native source grids fill the view; masked model grids keep the
            # padding, which reads as outside the model.
            within=(x.min(), x.max(), y.min(), y.max()) if mask is None else None,
        )
    support = frame_intersection(field, frame.extent)
    if mask is not None:
        support &= mask
    finite = support & np.isfinite(field.values)
    scale = reference["variables"][variable]
    boundaries = scale["boundaries"]
    below = int((finite & (field.values < boundaries[0])).sum()) if boundaries else 0
    above = int((finite & (field.values > boundaries[-1])).sum()) if boundaries else 0
    positive = counts.values[finite]
    eligible = [int(positive.min()), int(positive.max())] if positive.size else None
    label = {"precip": "Precip.", "temp": "Temp.", "pet": "PET"}[variable]
    drawn_scale = drawing_scale(reference, variable)
    # Climate display rules (toolbox-plot-style.md): reference units, a tick on
    # every class boundary and both rectangular extensions.
    if (
        drawn_scale.units not in {"degC", "mm/year"}
        or drawn_scale.extend != "both"
        or list(drawn_scale.ticks) != list(drawn_scale.boundaries)
    ):
        raise ValueError(f"{variable}: historical reference scale breaks climate rules")
    fig = draw_raster_map(
        RasterMap(field.values, x, y, inside=mask, cell_edges=cell_edges),
        frame,
        drawn_scale,
        overlays,
        colorbar_title=f"{label}\n{'(°C)' if variable == 'temp' else '(mm yr⁻¹)'}",
    )
    info = {
        "reference_id": reference["reference_id"],
        "reference_path": "data/climate/historical/_engine/map_reference.json",
        "frame": reference["frame"],
        "display_frame": {
            "extent": list(frame.extent),
            "longitude_ticks": list(frame.longitude_ticks),
            "latitude_ticks": list(frame.latitude_ticks),
            "aspect": frame.aspect,
            "crs": "EPSG:4326",
        },
        "historical_reference": scale,
        "reference_completeness": reference["identity"]["completeness"],
        "displayed_cells": int(support.sum()),
        "finite_cells": int(finite.sum()),
        "unavailable_cells": int((support & ~np.isfinite(field.values)).sum()),
        "eligible_year_counts": eligible,
        "strict_under_cells": below,
        "strict_over_cells": above,
        "scale_status": scale["status"],
    }
    # Caption notes (map_frames.map_caption), only when they apply; support
    # counts, periods and thresholds stay in ``info``.
    notes = []
    if not boundaries:
        notes.append(
            f"No historical colour scale is available ({scale['reason']}); "
            "cells with values are hatched"
        )
    elif below or above:
        notes.append(f"{below} cells fall below and {above} above the colour scale")
    if not info["displayed_cells"]:
        notes.append("No source cells intersect the map")
    return fig, info, notes


def load_result(root, source):
    """Restore artist inputs from explicit plot-ready tables, never raw climate."""
    folder = Path(root) / "tables"
    metadata = json.loads(
        (Path(root) / "_engine" / "diagnostics.json").read_text(encoding="utf-8")
    )
    renames = table_renames(metadata["schema_version"])

    def read(name):
        frame = pd.read_csv(folder / f"{name}.csv").rename(columns=renames)
        return frame[frame.source == source].drop(
            columns=["source", "period_start", "period_end"]
        )

    coverage = read("coverage").pivot(
        index="calendar_month", columns="reporting_year", values="class"
    )
    monthly = read("monthly_values")
    anomaly = monthly.pivot(
        index="calendar_month", columns="reporting_year", values="anomaly"
    )
    indices = read("annual_indices").set_index("reporting_year")
    spi = read("spi")
    spi.date = pd.to_datetime(spi.date)
    checks = read("spi_fit_checks")
    events = read("drought_events")
    events.start = pd.to_datetime(events.start)
    events.end = pd.to_datetime(events.end)
    trends = read("trends").set_index("indicator")
    sensitivity = read("trend_sensitivity")
    temp = read("temp_climatology").set_index("calendar_month")
    pt = read("pt_anomalies")
    spells = read("dry_spell_exceedance")
    valid = read("valid_years")
    reference = tuple(
        metadata["settings"]["reference_period"][k] for k in ("start", "end")
    )
    _, reference_support = monthly_reference_diagnostics(monthly, reference)
    trend_records = trends.to_dict(orient="index")
    # These plots also show trends for indicators absent from the retained trend table.
    for indicator in ("Rx5day", "sdii"):
        if indicator not in trend_records:
            trend_records[indicator] = sen_trend(indices[indicator])
    return {
        "coverage": coverage,
        "valid": valid.loc[valid.valid_precip, "reporting_year"].to_numpy(),
        "clim": read("precip_climatology").set_index("calendar_month"),
        "tclim": temp if not temp.empty else None,
        "timing": read("rainfall_timing")
        .pivot(index="reporting_year", columns="fraction", values="day")
        .reindex(columns=[0.25, 0.5, 0.75]),
        "anomaly": anomaly,
        "anomaly_reference_support": reference_support,
        "spi": {k: spi[spi.scale == k].set_index("date").value for k in (3, 6, 12)},
        "spi_dist": metadata["settings"]["spi_distribution"],
        "spi_checks": {
            k: checks[checks.scale == k].set_index("calendar_month") for k in (3, 6, 12)
        },
        "drought": {k: events[events.scale == k] for k in (3, 12)},
        "extremes": indices.rename(columns={"sdii": "SDII", "wet_days": "Wet days"}),
        "spells": spells.set_index("length_days").probability,
        "n_spells": len(read("dry_spells")),
        "acf": read("anomaly_acf").set_index("lag_months"),
        "annual": indices,
        "trends": trend_records,
        "sensitivity": {
            k: sensitivity[sensitivity.indicator == k]
            .set_index("start_year")
            .drop(columns="indicator")
            for k in ("total", "Rx1day")
        },
        "pt": pt if not pt.empty else None,
        "t_source": pt.temperature_source.iloc[0] if not pt.empty else None,
    }


def render(
    paths,
    settings,
    m0,
    sources,
    pooled_roots,
    overlays=None,
    scope="basin_avg",
    *,
    map_reference=None,
    map_reference_params=None,
    annotation="none",
    reference_record=None,
):
    """Export selected PNG annotations and independent JSON/Markdown inventories.

    ``reference_record`` replaces ``map_reference``/``map_reference_params`` for a
    re-render of retained products (``diagnostic_rerender``): the caller has
    checked its integrity, and current-byte identity drift is expected there.
    Figures and inventories go under ``paths["render_root"]`` when present.
    """
    root = Path(paths["root"])
    out_root = Path(paths.get("render_root", root))
    has_maps = any(form.startswith("map-") for form in paths["figures"])
    arguments = (
        reference_arguments(map_reference_params)
        if map_reference_params is not None
        else None
    )
    reference = None
    retained = None

    # Check the inputs once, then render the loaded snapshot. Exporting another
    # PNG does not require rereading climate inputs or qualifying the environment.
    if reference_record is not None:
        reference = reference_record
        arguments = {"completeness": reference["identity"]["completeness"]}
    elif arguments is not None:
        reference = load_reference(map_reference, **arguments)
    elif has_maps:
        raise ValueError("Source maps require the declared historical reference")
    if has_maps:
        with xr.open_dataset(paths["compute"]["maps"]) as fields:
            retained = fields.load()
        source = sources[0]
        entry = reference["identity"]["source_preparations"][source]
        validate_fields(
            retained,
            entry["preparation_identity"],
            entry,
            source=source,
            target=paths["compute"]["maps"],
        )
    features = load_map_overlays(overlays) if has_maps else {}
    metadata = json.loads(
        Path(paths["compute"]["metadata"]).read_text(encoding="utf-8")
    )
    period = metadata["period"]
    results = {s: load_result(root, s) for s in sources}
    months = artists.month_order(m0)
    calendar = pd.date_range(
        pd.Timestamp(year=period[0], month=m0, day=1),
        pd.Timestamp(year=period[1] + 1, month=m0, day=1),
        freq="MS",
        inclusive="left",
    )
    for result in results.values():
        for key in ("coverage", "anomaly"):
            result[key] = result[key].reindex(
                index=months, columns=range(period[0], period[1] + 1)
            )
        for scale in (3, 6, 12):
            result["spi"][scale] = result["spi"][scale].reindex(calendar)
    pooled = [
        pd.read_csv(Path(p) / "tables" / "monthly_values.csv").anomaly.to_numpy()
        for p in pooled_roots
    ]
    bounds = artists.anomaly_bounds(np.concatenate(pooled))
    view = artists.View(
        sources=sources,
        labels={
            s: "CHIRPS" if s in {"chirps", "chirps_global"} else s.upper()
            for s in sources
        },
        styles=artists.source_styles(sources),
        results=results,
        years=np.arange(period[0], period[1] + 1),
        m0=m0,
        reference=tuple(settings["reference_period"][k] for k in ("start", "end")),
        wet_mm=settings["wet_day_threshold_mm"],
        agreement=pd.DataFrame(),
        common_years=np.array([]),
        support=f"{scope.replace('_', ' ')}, equal-weight cells",
        status="real-data exercised, not scientifically qualified",
        anom_bounds=bounds,
        max_missing_month=settings["max_missing_days_month"],
        max_missing_year=settings["max_missing_days_year"],
    )
    records = {}
    for form, filename in paths["figures"].items():
        availability = "available"
        support = (
            "native source cells intersecting the displayed map frame"
            if form.startswith("map-")
            else view.support
        )
        with use_preset(CST_PRESET):
            if form.startswith("map-"):
                variable = form.removeprefix("map-")
                field = retained[variable]
                counts = retained[f"{variable}_valid_years"]
                fig, map_info, notes = prepared_map(
                    field, counts, reference, variable, features
                )
                map_info["preparation_id"] = retained.attrs["map_preparation_id"]
                map_info["fields_digest"] = retained.attrs["map_fields_digest"]
                map_info["field_completeness"] = arguments["completeness"]
                map_info["field_period"] = [
                    retained.attrs["period_start"],
                    retained.attrs["period_end"],
                ]
                map_info["source_role"] = reference["identity"]["source_preparations"][
                    sources[0]
                ]["preparation_identity"]["role"]
                if (
                    variable == "pet"
                    and map_info["source_role"] == "temperature-carrier"
                ):
                    notes.append(
                        "PET is derived on the source grid (de Bruin) and can differ from model PET"
                    )
                elif (
                    variable in {"temp", "pet"}
                    and map_info["source_role"] == "precipitation-only"
                ):
                    notes.append("This auxiliary field does not set the colour classes")
                text = map_caption(
                    f"{MAP_QUANTITIES[variable]} on the {view.labels[sources[0]]} native grid, "
                    f"{period[0]}–{period[1]},",
                    sources[0],
                    notes,
                )
                if not map_info["displayed_cells"]:
                    availability = "not displayed: native support does not intersect the displayed frame"
                elif not map_info["finite_cells"]:
                    availability = "unavailable: no complete annual map values"
                elif map_info["scale_status"] == "unavailable":
                    availability = "unscaled: historical reference unavailable"
            else:
                available = form in {"coverage", "anomaly"} or any(
                    result["annual"].total.notna().any() for result in results.values()
                )
                if form in {"spi", "drought-events"}:
                    available = any(
                        series.notna().any()
                        for result in results.values()
                        for series in result["spi"].values()
                    )
                if form == "pt":
                    available = any(
                        result["pt"] is not None
                        and result["pt"].t_anom.notna().any()
                        and result["pt"].p_anom_pct.notna().any()
                        for result in results.values()
                    )
                if form == "timing":
                    available = any(
                        result["timing"].notna().any().any()
                        for result in results.values()
                    )
                fig = (
                    artists.CHARTS[form](view)
                    if available
                    else "Unavailable: no usable data or reference fit for this diagnostic."
                )
                if isinstance(fig, str):
                    availability = re.sub("<[^>]*>", "", fig)
                    fig, axs = new_fig(80)
                    axs[0, 0].text(
                        0.5, 0.5, availability, ha="center", va="center", wrap=True
                    )
                    axs[0, 0].set_axis_off()
                text = (
                    caption(form, view)
                    if form not in {"spi-checks", "sensitivity"}
                    else (
                        "Shapiro–Wilk normality checks of reference-period SPI by calendar month and accumulation scale, "
                        "comparing gamma and Pearson III fits. Cells show p-values; p < 0.05 indicates evidence of "
                        "departure from normality, while p ≥ 0.05 indicates insufficient evidence of departure. "
                        "Gray cells indicate unavailable estimates. Tests are exploratory and are not adjusted "
                        "for multiple comparisons."
                        if form == "spi-checks"
                        else "Sensitivity of Theil–Sen trends to the fitting period. Each point is a separate "
                        "fit from the indicated start year to the fixed final year; vertical error bars show "
                        "95% intervals adjusted for lag-1 serial dependence. Filled points indicate a two-sided "
                        "Mann–Kendall p < 0.05 and hollow points p ≥ 0.05. Each fit requires at least 15 valid "
                        "annual values. Dataset positions are offset slightly for readability."
                    )
                )
            if form in {"spi-checks", "sensitivity"}:
                text += f" Sources: {', '.join(view.labels.values())}. {support}."
            if form == "spi-checks":
                details = []
                for source in view.sources:
                    checks = view.results[source]["spi_checks"]
                    for distribution, name in (
                        ("gamma", "gamma"),
                        ("pearson3", "Pearson III"),
                    ):
                        values = np.concatenate(
                            [
                                checks[scale][f"sw_{distribution}"].to_numpy()
                                for scale in artists.SPI_SCALES
                            ]
                        )
                        details.append(
                            f"{view.labels[source]} {name}: {int((values < 0.05).sum())}"
                            f"/{int(np.isfinite(values).sum())} tests with p < 0.05"
                        )
                    hits = sum(
                        int(checks[scale]["p3_bound_hits"].sum())
                        for scale in artists.SPI_SCALES
                    )
                    details.append(
                        f"{view.labels[source]} Pearson III: {hits} monthly values at fitted "
                        "probability bounds, summed across accumulation scales"
                    )
                text += " " + "; ".join(details) + "."
            if availability != "available":
                text += f" Availability: {availability}."
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            # Fixed-aspect maps can leave a wide unused strip on the canvas.
            # Trim to all visible artists, retaining a 2 mm border on every side.
            map_export = (
                {"bbox_inches": "tight", "pad_inches": 2 / 25.4}
                if form.startswith("map-")
                else {}
            )
            contributing = (
                [s for s in sources if results[s]["tclim"] is not None]
                if form == "temperature"
                else sources
            )
            figure_title = title(
                form, [view.labels[s] for s in contributing], period, support
            )
            export_figure(
                fig,
                filename,
                annotation=annotation,
                title=figure_title,
                caption=text,
                dpi=600,
                facecolor="white",
                **map_export,
            )
            plt.close(fig)
            relative = str(Path(filename).relative_to(out_root)).replace("\\", "/")
            records[relative] = {
                "title": title(
                    form, [view.labels[s] for s in contributing], period, support
                ),
                "caption": text,
                "sources": contributing,
                "period": period,
                "reference": settings["reference_period"],
                "support": support,
                "encoding": form,
                "methods": metadata["methods"],
                "availability": availability,
                "captioned_path": None,
                "caveats": [
                    "Not scientifically qualified",
                    "Source agreement is not accuracy",
                    "Input homogeneity unverified",
                ],
            }
            if form.startswith("map-"):
                records[relative]["map_reference"] = map_info
            if form == "anomaly":
                records[relative]["reference_support"] = {
                    s: results[s]["anomaly_reference_support"] for s in sources
                }
    if retained is not None:
        retained.close()
    write_json(
        paths["render"]["captions_json"],
        {
            "schema_version": 1,
            "figures": records,
            "anomaly_bounds": bounds,
            "produced_paths": list(records)
            + [
                "plots/_engine/figure_titles.json",
                "plots/figure_titles.md",
                "plots/_engine/figure_captions.json",
                "plots/figure_captions.md",
            ],
        },
    )
    Path(paths["render"]["captions_md"]).write_text(
        "# Figure captions\n\n"
        + "\n\n".join(
            f"## {key}\n\n{value['caption']}" for key, value in records.items()
        )
        + "\n",
        encoding="utf-8",
    )
    write_figure_titles(out_root / "plots", records)
    unavailable = Counter(
        item["availability"]
        for item in records.values()
        if item["availability"] != "available"
    )
    log_row(
        f"Wrote {plural(len(records), 'diagnostic figure')} for {listed(sources)} "
        "→ plots/_engine/figure_captions.json",
        module="diagnostics",
    )
    for reason, count in unavailable.items():
        log_row(f"{plural(count, 'figure')}: {reason}", module="diagnostics")
    if len(sources) == 1 and not carries_temp(sources[0]):
        log_row(
            f"{sources[0]}: no native temperature; temperature and precipitation–temperature figures not requested",
            module="diagnostics",
        )
    return records
