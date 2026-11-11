"""Retain native-cell annual P/T departures with fixed reference support."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis import diagnostics as dg
from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp
from blueearth_cst.climate_analysis.diagnostic_tables import daily_series, write_json
from blueearth_cst.shared.map_reference import field_edges


def parse_settings(block: dict | None, window: dict, sources: list) -> dict:
    """Admit the optional product without changing shared climatology code identity."""
    from blueearth_cst.climate_analysis.diagnostic_settings import (
        parse_settings as base_settings,
    )

    if block is not None and not isinstance(block, dict):
        raise ValueError("WF0 diagnostics must be a mapping with registered keys")
    block = dict(block or {})
    enabled = block.pop("annual_spatial_anomalies", False)
    if type(enabled) is not bool:
        raise ValueError("annual_spatial_anomalies must be a boolean")
    return base_settings(block, window, sources) | {"annual_spatial_anomalies": enabled}


def anomaly_paths(root: str | Path) -> dict[str, str]:
    """Numerical products are independent of the shared climatology producer."""
    root = Path(root)
    return {
        "fields": str(root / "tables/annual_spatial_anomalies.nc"),
        "record": str(root / "_engine/annual_spatial_anomalies.json"),
    }


def page_years(settings: dict) -> list[list[int]]:
    """Every requested comparison year, ordered in pages of at most six."""
    period = settings["comparison_period"]
    years = list(range(period["start"], period["end"] + 1))
    return [years[i : i + 6] for i in range(0, len(years), 6)]


def prepare(
    ds: xr.Dataset,
    settings: dict,
    m0: int,
    source: str,
    *,
    temperature_qualified: bool = True,
) -> tuple[xr.Dataset, dict]:
    """Apply the existing annual kernel at each cell, with no spatial reduction.

    Missing dates are reindexed before completeness. Reference means require
    every fully contained requested reference year, independently per variable
    and cell. Partial comparison years remain present as unavailable panels.
    """
    index = pd.DatetimeIndex(ds.time.values)
    if (
        index.empty
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not (index == index.normalize()).all()
    ):
        raise ValueError(f"{source}: expected ordered unique daily midnight dates")
    calendar = pd.date_range(index.min(), index.max(), freq="D")
    full = set(int(y) for y in dg.full_years(pd.Series(0, index=calendar), m0))
    requested = list(
        range(
            settings["reference_period"]["start"],
            settings["reference_period"]["end"] + 1,
        )
    )
    eligible = [y for y in requested if y in full]
    excluded = [y for y in requested if y not in full]
    if not eligible:
        raise ValueError(
            f"{source}: reference period contains no full reporting year in stored daily span"
        )
    if excluded:
        warnings.warn(
            f"{source}: excluded incomplete/out-of-span reference years {excluded}",
            stacklevel=2,
        )
    comparison = [y for page in page_years(settings) for y in page]
    years = sorted(set(requested) | set(comparison))
    tol = dg.Completeness(
        settings["max_missing_days_month"], settings["max_missing_days_year"]
    )
    output = xr.Dataset(coords={"reporting_year": years})
    variables = {}
    for variable in ("precip", "temp") if carries_temp(source) else ("precip",):
        if variable not in ds:
            raise ValueError(f"{source}: missing owned {variable} field")
        field = ds[variable].transpose("time", "latitude", "longitude")
        attrs = {"units": "degC" if variable == "temp" else "mm"}
        for name in ("latitude", "longitude"):
            resolution = field.attrs.get(f"{name}_resolution")
            if len(field[name]) > 1:
                resolution = abs(float(field[name][1] - field[name][0]))
            elif resolution is None and hasattr(ds, "raster"):
                resolution = abs(float(ds.raster.res[0 if name == "longitude" else 1]))
            if resolution is not None:
                attrs[f"{name}_resolution"] = resolution
        template = field.isel(time=0, drop=True).assign_attrs(attrs)
        field_edges(template)  # refuse ambiguous singleton-cell footprints
        data = np.asarray(field.values, dtype=float)
        if np.isinf(data).any() or (variable == "precip" and (data < 0).any()):
            raise ValueError(
                f"{source}.{variable}: unqualified nonfinite or negative source values"
            )
        if variable == "temp" and not temperature_qualified:
            data = np.full(data.shape, np.nan)
        values = np.full((len(years), *data.shape[1:]), np.nan)
        for i, j in np.ndindex(data.shape[1:]):
            daily = pd.Series(data[:, i, j], index=index).reindex(calendar)
            values[:, i, j] = (
                dg.annual(daily, m0, "mean" if variable == "temp" else "sum", tol)
                .reindex(years)
                .to_numpy()
            )
        annual = xr.DataArray(
            values,
            dims=("reporting_year", "latitude", "longitude"),
            coords={
                "reporting_year": years,
                "latitude": field.latitude,
                "longitude": field.longitude,
            },
            attrs=attrs,
        )
        ref = annual.sel(reporting_year=eligible)
        counts = ref.notnull().sum("reporting_year").astype("int32")
        baseline = ref.mean("reporting_year", skipna=False)
        output[f"{variable}_annual"] = annual
        output[f"{variable}_valid"] = annual.notnull()
        output[f"{variable}_reference_count"] = counts
        output[f"{variable}_reference_mean"] = baseline.assign_attrs(attrs)
        output[f"{variable}_reference_valid"] = counts == len(eligible)
        output[f"{variable}_anomaly"] = (annual - baseline).assign_attrs(attrs)
        variables[variable] = {
            "units": attrs["units"],
            "reference_cells": int((counts == len(eligible)).sum()),
            "total_cells": int(counts.size),
            "temperature_qualified": temperature_qualified
            if variable == "temp"
            else None,
        }
    dates = {
        str(y): [
            pd.Timestamp(y, m0, 1).date().isoformat(),
            (pd.Timestamp(y + 1, m0, 1) - pd.Timedelta(days=1)).date().isoformat(),
        ]
        for y in years
    }
    output.attrs.update(
        schema_version=1,
        source=source,
        water_year_start=m0,
        reporting_year_label="start",
    )
    return output, {
        "schema_version": 1,
        "source": source,
        "settings": settings,
        "water_year_start": m0,
        "reporting_year_label": "start",
        "daily_span": [
            calendar.min().date().isoformat(),
            calendar.max().date().isoformat(),
        ],
        "reference_requested": requested,
        "reference_eligible": eligible,
        "reference_excluded_partial": excluded,
        "single_reference_year": len(eligible) == 1,
        "comparison_years": comparison,
        "comparison_partial": [y for y in comparison if y not in full],
        "year_dates": dates,
        "pages": page_years(settings),
        "variables": variables,
        "method": "annual minus per-cell arithmetic reference mean; all eligible reference years required",
        "infill": "tolerated missing days filled with that calendar month's observed mean",
    }


def file_hash(path: str | Path) -> str:
    """Stream a retained/input file hash without loading its bytes at once."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_retained(root: str | Path) -> tuple[xr.Dataset, dict]:
    """Read annual fields without daily inputs; refuse mismatched/corrupt products."""
    paths = anomaly_paths(root)
    record = json.loads(Path(paths["record"]).read_text(encoding="utf-8"))
    if (
        record["schema_version"] != 1
        or file_hash(paths["fields"]) != record["fields_sha256"]
    ):
        raise ValueError(f"{root}: annual spatial anomaly integrity check failed")
    with xr.open_dataset(paths["fields"]) as ds:
        fields = ds.load()
    if fields.attrs["source"] != record["source"]:
        raise ValueError(f"{root}: annual spatial anomaly source mismatch")
    return fields, record


def run(sm) -> None:
    """Optional WF0 numerical producer, independent of map display settings."""
    source = sm.params.source
    from blueearth_cst.shared.forcing_recipe import normalize_forcing_recipe

    recipe = normalize_forcing_recipe(source, getattr(sm.params, "forcing", None))
    data_sources = sm.params.data_sources
    catalogs = [data_sources] if isinstance(data_sources, str) else data_sources
    inputs = [sm.input.climate_nc, sm.input.basin_cells, *catalogs]
    hashes = {str(path): file_hash(path) for path in inputs}
    daily, lineage = daily_series(
        sm.input.climate_nc,
        sm.input.basin_cells,
        source,
        sm.params.data_sources,
        recipe=recipe,
    )
    with xr.open_dataset(sm.input.climate_nc) as raw:
        fields, record = prepare(
            raw,
            dict(sm.params.settings),
            int(sm.params.m0),
            source,
            temperature_qualified=bool(daily.temp.notna().any()),
        )
    if hashes != {str(path): file_hash(path) for path in inputs}:
        raise ValueError(f"{source}: inputs changed during annual anomaly preparation")
    record.update(input_hashes=hashes, lineage=lineage, forcing_recipe=recipe.to_dict())
    from blueearth_cst.climate_analysis.figure_naming import annual_anomaly_filename

    record["figure_pages"] = [
        {
            "years": years,
            "figures": {
                variable: "plots/"
                + annual_anomaly_filename(source, variable, page, len(record["pages"]))
                for variable in record["variables"]
            },
        }
        for page, years in enumerate(record["pages"], 1)
    ]
    record["code_digest"] = dict(sm.params.code_digest)
    record["environment"] = dict(sm.params.environment)
    target = Path(sm.output.fields)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields.to_netcdf(target)
    record["fields_sha256"] = file_hash(target)
    write_json(sm.output.record, record)


if "snakemake" in globals():
    from blueearth_cst.shared.snake_utils import tee_to_log

    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
