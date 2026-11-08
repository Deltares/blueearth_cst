"""Source-owned climate-index publication and verified retained-product loading."""

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

from blueearth_cst.shared.provenance import canonical_sha256, file_sha256


def implementation_digest() -> str:
    """Identify contributing code, including uncommitted implementation edits."""
    package = Path(__file__).resolve().parents[1]
    paths = sorted((package / "indicators" / "climate").glob("*.py"))
    paths += [
        Path(__file__),
        Path(__file__).with_name("source_capabilities.py"),
        Path(__file__).with_name("diagnostic_settings.py"),
        package / "shared" / "grid_cells.py",
    ]
    return canonical_sha256(
        {str(p.relative_to(package)): file_sha256(p) for p in paths}
    )


def load_publication(pointer_path: str | Path) -> tuple[Path, dict]:
    """Resolve and hash-check a complete publication beneath its source store."""
    pointer_path = Path(pointer_path)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    root = pointer_path.parent.parent.resolve()
    manifest_path = (root / pointer["manifest"]).resolve()
    if not manifest_path.is_relative_to(root / "indices"):
        raise ValueError("Climate-index manifest escapes its source store")
    if file_sha256(manifest_path) != pointer["sha256"]:
        raise ValueError("Climate-index manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("Unsupported or incomplete climate-index publication")
    for name, digest in manifest["products"].items():
        path = (manifest_path.parent / name).resolve()
        if not path.is_relative_to(manifest_path.parent) or file_sha256(path) != digest:
            raise ValueError(f"Climate-index product failed integrity check: {name}")
    return manifest_path.parent, manifest


def _write_pointer(path: Path, manifest: Path, root: Path) -> None:
    """Atomically expose a verified completed bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".indices-", suffix=".json", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "manifest": manifest.relative_to(root).as_posix(),
                "sha256": file_sha256(manifest),
            },
            stream,
        )
    os.replace(temporary, path)


def publish(
    climate_path, cells_path, source, store_dir, settings, start_month, catalogs=()
):
    """Calculate basin and native-grid products, retaining immutable evaluations.

    The declared pointer is written last. Numerical payloads are enumerated and
    checksummed in the completion manifest, which every reader verifies. Failed
    staging directories remain visibly incomplete and never become current.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    import yaml

    from blueearth_cst.climate_analysis.source_capabilities import source_climate_vars
    from blueearth_cst.indicators.climate.evaluate import evaluate
    from blueearth_cst.indicators.climate.periods import (
        annual_periods,
        seasonal_periods,
    )
    from blueearth_cst.indicators.climate.reference import fit_reference
    from blueearth_cst.shared.grid_cells import LAT, LON, cells_csv_mask

    root = Path(store_dir)
    options = settings["climate_indices"]
    reference_period = options["reference_period"]
    identity = {
        "source": source,
        "inputs": {
            "climate": file_sha256(climate_path),
            "cells": file_sha256(cells_path),
        },
        "settings": {
            key: settings[key]
            for key in (
                "climate_indices",
                "analysis_window",
                "wet_day_threshold_mm",
                "max_missing_days_month",
                "max_missing_days_year",
            )
        },
        "start_month": start_month,
        "implementation": implementation_digest(),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "xarray", "netCDF4")
        },
    }
    catalog_paths = [catalogs] if isinstance(catalogs, str) else list(catalogs)
    identity["catalogs"] = {str(path): file_sha256(path) for path in catalog_paths}
    evaluation_id = canonical_sha256(identity)
    destination = root / "indices" / evaluation_id
    pointer = root / "_engine" / "climate_indices.json"
    if destination.exists():
        manifest_path = destination / "calculation.json"
        if not manifest_path.is_file():
            raise ValueError(f"Incomplete climate-index evaluation: {destination}")
        # Verify existing payloads before making this evaluation current again.
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("complete") is not True:
            raise ValueError("Existing evaluation identity/completion mismatch")
        for name, digest in manifest["products"].items():
            path = (destination / name).resolve()
            if (
                not path.is_relative_to(destination.resolve())
                or file_sha256(path) != digest
            ):
                raise ValueError(f"Existing climate-index payload changed: {name}")
        _write_pointer(pointer, manifest_path, root)
        return pointer
    with xr.open_dataset(climate_path) as opened:
        original = opened.load()
    mask = cells_csv_mask(original, cells_path)
    cells = pd.read_csv(cells_path)
    declared = set(zip(cells[LAT].round(6), cells[LON].round(6)))
    available = {
        (round(float(a), 6), round(float(b), 6))
        for a in original[LAT].values
        for b in original[LON].values
    }
    if mask is None or not declared or not declared <= available:
        raise ValueError("Declared basin cells do not match the climate grid")
    data = xr.Dataset()
    unavailable = {}
    offsets = {}
    for path in catalog_paths:
        entry = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get(source, {})
        if entry:
            offsets = entry.get("data_adapter", entry).get("unit_add", {})
    for variable in ("precip", "temp", "temp_max", "temp_min"):
        if variable not in original or (
            variable != "precip" and "temp" not in source_climate_vars(source)
        ):
            unavailable[variable] = "not available from this source"
            continue
        field = original[variable].where(mask)
        units = str(field.attrs.get("units", "")).lower()
        if variable == "precip":
            verified = units in {
                "mm",
                "mm/day",
                "mm/d",
                "mm d-1",
                "mm d**-1",
                "mm day-1",
                "mm day**-1",
            }
            canonical = "mm/day"
        else:
            verified = units in {
                "degc",
                "celsius",
                "degree_celsius",
                "degrees_celsius",
                "°c",
            }
            verified |= (
                units in {"k", "kelvin"}
                and offsets.get(variable) == -273.15
                and bool(original.attrs.get("region_geojson_sha256"))
            )
            finite = field.values[np.isfinite(field.values)]
            verified &= (
                bool(len(finite))
                and (finite.min() >= -100 if len(finite) else False)
                and (finite.max() <= 70 if len(finite) else False)
            )
            canonical = "degC"
        if not verified:
            unavailable[variable] = "units or values could not be verified"
            continue
        field.attrs = {"units": canonical, "source": source}
        data[variable] = field
    if not data.data_vars:
        raise ValueError("No source-owned climate variables have verified units")
    window = settings["analysis_window"]
    annual = annual_periods(window["start"], window["end"], start_month)
    seasons = tuple(
        period
        for name, months in options["seasons"].items()
        for period in seasonal_periods(
            window["start"], window["end"], name=name, months=tuple(months)
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".incomplete-", dir=destination.parent))
    products = []
    for spatial_operation, values in (
        ("grid", data),
        ("basin", data.mean([LAT, LON], skipna=True, keep_attrs=True)),
    ):
        reference = fit_reference(
            values,
            reference_period["start"],
            reference_period["end"],
            wet_day_threshold_mm=settings["wet_day_threshold_mm"],
            minimum_fraction=options["reference_minimum_fraction"],
        )
        path = staging / f"reference_{spatial_operation}.nc"
        reference.to_netcdf(path)
        products.append(path.name)
        result, events = evaluate(
            values,
            reference,
            annual,
            max_missing_days_month=settings["max_missing_days_month"],
            max_missing_days_period=settings["max_missing_days_year"],
            mean_temperature_threshold_degc=options["mean_temperature_threshold_degc"],
        )
        result.attrs.update(
            source=source,
            spatial_operation=spatial_operation,
            evaluation_id=evaluation_id,
        )
        path = staging / f"annual_{spatial_operation}.nc"
        result.to_netcdf(path)
        products.append(path.name)
        if seasons:
            seasonal, _ = evaluate(
                values,
                reference,
                seasons,
                max_missing_days_month=settings["max_missing_days_month"],
                max_missing_days_period=settings["max_missing_days_year"],
                mean_temperature_threshold_degc=options[
                    "mean_temperature_threshold_degc"
                ],
            )
            seasonal.attrs.update(
                source=source,
                spatial_operation=spatial_operation,
                evaluation_id=evaluation_id,
            )
            path = staging / f"seasonal_{spatial_operation}.nc"
            seasonal.to_netcdf(path)
            products.append(path.name)
        events["source"] = source
        events["spatial_operation"] = spatial_operation
        path = staging / f"events_{spatial_operation}.csv"
        events.to_csv(path, index=False)
        products.append(path.name)
        complete = events[
            (events.event_type == "wet")
            & ~events.left_censored.astype(bool)
            & ~events.right_censored.astype(bool)
        ]
        rows = []
        for location, group in complete.groupby("location_id"):
            for length in sorted(group.duration.unique()):
                rows.append(
                    {
                        "location_id": location,
                        "duration": length,
                        "exceedance_probability": float(
                            (group.duration >= length).mean()
                        ),
                        "complete_events": len(group),
                    }
                )
        path = staging / f"wet_spell_exceedance_{spatial_operation}.csv"
        pd.DataFrame(
            rows,
            columns=[
                "location_id",
                "duration",
                "exceedance_probability",
                "complete_events",
            ],
        ).to_csv(path, index=False)
        products.append(path.name)
    coverage = data.notnull().sum([LAT, LON])
    coverage.attrs.update(
        spatial_reduction="equal-weight available basin cells",
        declared_cells=len(declared),
    )
    coverage.to_netcdf(staging / "basin_contributing_cells.nc")
    products.append("basin_contributing_cells.nc")
    manifest = {
        "schema_version": 1,
        "complete": True,
        "identity": identity,
        "evaluation_id": evaluation_id,
        "unavailable_variables": unavailable,
        "products": {name: file_sha256(staging / name) for name in products},
        "limitations": [
            "Short references are exploratory; sample coverage is not scientific qualification",
            "Censored observed spell lengths are lower bounds",
            "WSDI is fixed-reference, not bootstrap-adjusted",
        ],
    }
    (staging / "calculation.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if (
        file_sha256(climate_path) != identity["inputs"]["climate"]
        or file_sha256(cells_path) != identity["inputs"]["cells"]
        or any(
            file_sha256(path) != digest for path, digest in identity["catalogs"].items()
        )
    ):
        raise ValueError(
            "Climate inputs changed during index calculation; staging retained"
        )
    staging.rename(destination)
    _write_pointer(pointer, destination / "calculation.json", root)
    return pointer
