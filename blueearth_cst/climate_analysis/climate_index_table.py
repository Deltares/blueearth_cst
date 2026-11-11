"""Calculate climate-index tables with explicit spatial and reporting supports."""

import json
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401 -- registers the native raster accessor
import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis.climate_indices import (
    implementation_digest,
    qualified_source_data,
)
from blueearth_cst.indicators.climate.definitions import INDEX_METADATA
from blueearth_cst.indicators.climate.evaluate import evaluate
from blueearth_cst.indicators.climate.periods import annual_periods, seasonal_periods
from blueearth_cst.indicators.climate.reference import fit_reference
from blueearth_cst.shared.grid_cells import LAT, LON, cells_csv_mask, geometry_mask
from blueearth_cst.shared.provenance import file_sha256

INDEX_VARIABLES = {
    **dict.fromkeys(
        (
            "cdd",
            "cwd",
            "r10mm",
            "r20mm",
            "r95p_amount",
            "r95p_fraction",
            "wet_spell_mean",
            "wet_spell_median",
            "wet_spell_p90",
            "rx1day",
            "rx5day",
            "wet_days",
            "prcptot",
            "sdii",
        ),
        "precip",
    ),
    **dict.fromkeys(
        ("txx", "tx90p", "wsdi", "heatwave_count", "heatwave_days", "longest_heatwave"),
        "temp_max",
    ),
    **dict.fromkeys(("tnx", "tn90p"), "temp_min"),
    **dict.fromkeys(
        ("mean_temp_below_zero_days", "mean_temp_above_threshold_days"), "temp"
    ),
}
SCIENTIFIC_SETTINGS = (
    "climate_indices",
    "analysis_window",
    "wet_day_threshold_mm",
    "max_missing_days_month",
    "max_missing_days_year",
)


def _identifier(value: object) -> str:
    """Keep declared string identifiers, normalizing only numeric integral IDs."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value)


def _evaluate_support(
    data: xr.Dataset, settings: dict, periods: tuple
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Fit reference thresholds and evaluate one daily spatial support in memory."""
    options = settings["climate_indices"]
    years = options["reference_period"]
    reference = fit_reference(
        data,
        years["start"],
        years["end"],
        wet_day_threshold_mm=settings["wet_day_threshold_mm"],
        minimum_fraction=options["reference_minimum_fraction"],
    )
    return evaluate(
        data,
        reference,
        periods,
        max_missing_days_month=settings["max_missing_days_month"],
        max_missing_days_period=settings["max_missing_days_year"],
        mean_temperature_threshold_degc=options["mean_temperature_threshold_degc"],
    )


def _table_rows(
    result: xr.Dataset | None,
    periods: tuple,
    support: dict,
    source: str,
    unavailable_variables: dict,
    reason: str = "",
) -> list[dict]:
    """Emit every registered metric, including explicit unavailable values."""
    rows = []
    for period in periods:
        for metric, (label, units) in INDEX_METADATA.items():
            variable = INDEX_VARIABLES[metric]
            unavailable = reason or unavailable_variables.get(variable, "")
            value, valid = np.nan, np.nan
            expected = (period.stop - period.start).days
            if not unavailable:
                if result is None or metric not in result:
                    raise ValueError(
                        f"Climate-index result lacks qualified metric {metric}"
                    )
                value = float(result[metric].sel(period=period.label).item())
                valid = int(
                    result[f"{variable}_valid_days"].sel(period=period.label).item()
                )
                expected = int(
                    result[f"{variable}_expected_days"].sel(period=period.label).item()
                )
                if not np.isfinite(value):
                    eligible = bool(
                        result[f"{variable}_eligible"].sel(period=period.label).item()
                    )
                    unavailable = (
                        "index unavailable under reference or event rules"
                        if eligible
                        else "insufficient period coverage"
                    )
            rows.append(
                {
                    "schema_version": "climate-index-table/1",
                    "source": source,
                    **support,
                    "period_type": "seasonal" if ":" in period.label else "annual",
                    "period": period.label,
                    "period_start": period.start.date().isoformat(),
                    "period_stop": period.stop.date().isoformat(),
                    "index": metric,
                    "label": label,
                    "value": value,
                    "units": units,
                    "status": "unavailable" if unavailable else "available",
                    "reason": unavailable,
                    "valid_days": valid,
                    "expected_days": expected,
                }
            )
    return rows


def _wet_spell_exceedance(events: pd.DataFrame, source: str) -> pd.DataFrame:
    """Summarize complete basin wet events using the existing empirical definition."""
    complete = events[
        events.event_type.eq("wet")
        & ~events.left_censored.astype(bool)
        & ~events.right_censored.astype(bool)
    ]
    rows = [
        {
            "location_id": location,
            "duration": length,
            "exceedance_probability": float((group.duration >= length).mean()),
            "complete_events": len(group),
            "source": source,
            "spatial_operation": "basin",
        }
        for location, group in complete.groupby("location_id")
        for length in sorted(group.duration.unique())
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "location_id",
            "duration",
            "exceedance_probability",
            "complete_events",
            "source",
            "spatial_operation",
        ],
    )


def export_climate_index_table(
    climate_path: str | Path,
    cells_path: str | Path,
    subbasins_path: str | Path,
    locations_path: str | Path,
    output_path: str | Path,
    *,
    source: str,
    settings: dict,
    start_month: int,
    record_path: str | Path,
    spell_path: str | Path,
    catalogs=(),
) -> Path:
    """Calculate area indices after daily averaging, then publish the record last.

    Basin and subbasin means weight available native cells equally before index
    calculation. Locations use their original coordinates and containing native
    cell, with explicit rows for missing or out-of-domain supports. Qualified
    source units, reference fitting, event and completeness rules are unchanged.
    No reference arrays, native-grid results or immutable bundles are retained.
    Input or implementation drift refuses before replacing existing products.
    """
    if set(INDEX_VARIABLES) != set(INDEX_METADATA):
        raise ValueError("Climate-index table variable registry is incomplete")
    climate_path, cells_path, subbasins_path, locations_path = map(
        Path, (climate_path, cells_path, subbasins_path, locations_path)
    )
    output_path, record_path, spell_path = map(
        Path, (output_path, record_path, spell_path)
    )
    root = record_path.parent.parent.resolve()
    catalog_paths = (
        [Path(catalogs)]
        if isinstance(catalogs, (str, Path))
        else list(map(Path, catalogs))
    )
    inputs = {
        "climate": climate_path,
        "cells": cells_path,
        "subbasins": subbasins_path,
        "locations": locations_path,
    }
    hashes = {role: file_sha256(path) for role, path in inputs.items()}
    catalog_hashes = {str(path): file_sha256(path) for path in catalog_paths}
    implementation = implementation_digest()
    scientific = {key: settings[key] for key in SCIENTIFIC_SETTINGS}
    targets = (output_path, spell_path, record_path)
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("Climate-index product targets must differ")
    for path in targets:
        if not path.resolve().is_relative_to(root) or path.resolve() in {
            item.resolve() for item in (*inputs.values(), *catalog_paths)
        }:
            raise ValueError(
                "Climate-index outputs must stay in the source store and preserve inputs"
            )

    data, unavailable = qualified_source_data(climate_path, source, catalog_paths)
    field = data[next(iter(data))]
    native_crs = field.raster.crs
    if native_crs is None:
        raise ValueError(
            "Climate-index spatial supports require an explicit native raster CRS"
        )
    mask = cells_csv_mask(data, cells_path)
    declared = pd.read_csv(cells_path)
    keys = set(zip(declared[LAT].round(6), declared[LON].round(6)))
    available = {
        (round(float(y), 6), round(float(x), 6))
        for y in data[LAT].values
        for x in data[LON].values
    }
    if mask is None or not keys or not keys <= available:
        raise ValueError("Declared basin cells do not match the climate grid")
    window = scientific["analysis_window"]
    periods = annual_periods(window["start"], window["end"], start_month) + tuple(
        period
        for name, months in scientific["climate_indices"]["seasons"].items()
        for period in seasonal_periods(
            window["start"], window["end"], name=name, months=tuple(months)
        )
    )
    empty_ids = {
        "subbasin_id": "",
        "spatial_location_id": "",
        "wflow_id": "",
        "location_code": "",
    }
    basin = {
        "spatial_scope": "basin",
        "spatial_id": "basin",
        "name": "Basin average",
        **empty_ids,
        "x": np.nan,
        "y": np.nan,
        "cell_count": int(mask.sum().item()),
        "spatial_reduction": "equal-weight available-cell daily mean before index",
    }
    result, basin_events = _evaluate_support(
        data.where(mask).mean([LAT, LON], skipna=True, keep_attrs=True),
        scientific,
        periods,
    )
    rows = _table_rows(result, periods, basin, source, unavailable)
    subbasins = gpd.read_file(subbasins_path).to_crs(native_crs)
    if "subbasin_id" not in subbasins:
        raise ValueError("Subbasin layer requires subbasin_id")
    ids = subbasins.subbasin_id.map(_identifier)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Subbasin identifiers must be nonempty and unique")
    for (_, item), identifier in zip(subbasins.iterrows(), ids, strict=True):
        geometry = None if pd.isna(item.geometry) else item.geometry
        mask = geometry_mask(data, geometry)
        support = {
            **basin,
            "spatial_scope": "subbasin",
            "spatial_id": identifier,
            "name": _identifier(item.get("name", identifier)),
            "subbasin_id": identifier,
            "cell_count": int(mask.sum().item()) if mask is not None else 0,
        }
        result = (
            _evaluate_support(
                data.where(mask).mean([LAT, LON], skipna=True, keep_attrs=True),
                scientific,
                periods,
            )[0]
            if mask is not None
            else None
        )
        rows.extend(
            _table_rows(
                result,
                periods,
                support,
                source,
                unavailable,
                "subbasin intersects no native climate cells" if mask is None else "",
            )
        )

    locations = gpd.read_file(locations_path)
    required = {"wflow_id", "original_x", "original_y"}
    if not required <= set(locations.columns):
        raise ValueError(f"Location layer requires {sorted(required)}")
    identifiers = locations.wflow_id.map(_identifier)
    if identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("Location wflow_id values must be nonempty and unique")
    basin_geometry = subbasins.geometry.union_all()
    for (_, item), identifier in zip(locations.iterrows(), identifiers, strict=True):
        x, y = (
            float(v) if pd.notna(v) else np.nan
            for v in (item.original_x, item.original_y)
        )
        support = {
            **basin,
            **{key: _identifier(item.get(key)) for key in empty_ids},
            "spatial_location_id": _identifier(
                item.get("spatial_location_id", item.get("location_id"))
            ),
            "spatial_scope": "location",
            "spatial_id": identifier,
            "name": _identifier(item.get("station_name", identifier)),
            "x": x,
            "y": y,
            "cell_count": 0,
            "spatial_reduction": "containing native cell at original coordinates",
        }
        reason, row, column = "invalid original location coordinates", -1, -1
        if np.isfinite([x, y]).all() and -180 <= x <= 180 and -90 <= y <= 90:
            point = gpd.GeoSeries(gpd.points_from_xy([x], [y]), crs=4326).to_crs(
                native_crs
            )
            rr, cc = field.raster.rowcol(
                point.x.values, point.y.values, mask_outside=True, nodata=-1
            )
            row, column = int(rr[0]), int(cc[0])
            reason = "outside native climate grid" if min(row, column) < 0 else ""
            if not reason and not basin_geometry.covers(point.iloc[0]):
                reason = "outside_basin"
            support["cell_count"] = 0 if reason else 1
        result = (
            _evaluate_support(data.isel({LAT: row, LON: column}), scientific, periods)[
                0
            ]
            if not reason
            else None
        )
        rows.extend(_table_rows(result, periods, support, source, unavailable, reason))

    table = pd.DataFrame(rows)
    table["implementation"] = implementation
    for role, digest in hashes.items():
        table[f"{role}_sha256"] = digest
    table["reference_start"] = scientific["climate_indices"]["reference_period"][
        "start"
    ]
    table["reference_end"] = scientific["climate_indices"]["reference_period"]["end"]
    table["wet_day_threshold_mm"] = scientific["wet_day_threshold_mm"]
    table["mean_temperature_threshold_degc"] = scientific["climate_indices"][
        "mean_temperature_threshold_degc"
    ]
    products = {
        "table": (output_path, table),
        "spells": (spell_path, _wet_spell_exceedance(basin_events, source)),
    }
    staged = {}
    try:
        for key, (target, frame) in products.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".climate-indices-", suffix=".csv", dir=target.parent
            )
            staged[key] = Path(temporary)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                frame.to_csv(stream, index=False)
        record = {
            "schema_version": "climate-index-results/1",
            "source": source,
            "settings": scientific,
            "start_month": start_month,
            "unavailable_variables": unavailable,
            "implementation": implementation,
            "inputs": {**hashes, "catalogs": catalog_hashes},
            "products": {
                key: {
                    "path": target.resolve().relative_to(root).as_posix(),
                    "sha256": file_sha256(staged[key]),
                }
                for key, (target, _) in products.items()
            },
        }
        record_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".climate-indices-", suffix=".json", dir=record_path.parent
        )
        staged["record"] = Path(temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if any(
            file_sha256(inputs[role]) != digest for role, digest in hashes.items()
        ) or any(
            file_sha256(path) != digest for path, digest in catalog_hashes.items()
        ):
            raise ValueError("Climate-index inputs changed during calculation")
        if implementation_digest() != implementation:
            raise ValueError("Climate-index implementation changed during calculation")
        for key, (target, _) in products.items():
            os.replace(staged[key], target)
        os.replace(staged["record"], record_path)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return output_path


def run(sm) -> Path:
    """Run direct calculation through the declared Snakemake script interface."""
    return export_climate_index_table(
        sm.input.climate_nc,
        sm.input.basin_cells,
        sm.input.subbasins,
        sm.input.locations,
        sm.output.table,
        source=sm.params.source,
        settings=dict(sm.params.settings),
        start_month=int(sm.params.m0),
        record_path=sm.output.record,
        spell_path=sm.output.spells,
        catalogs=getattr(sm.input, "catalogs", ()),
    )


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
