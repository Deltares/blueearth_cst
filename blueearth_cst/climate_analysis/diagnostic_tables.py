"""Toolbox-native adapters and retained plot-ready diagnostic tables."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis import diagnostics as dg
from blueearth_cst.shared.grid_cells import LAT, LON, cells_csv_mask

#: Version of the retained WF0 tables and ``_engine/diagnostics.json``. Bump it
#: with any change to a retained table's columns, and add the upgrade below.
#: 2: the trend column "metric" became "indicator" (4faf0580).
SCHEMA_VERSION = 2

#: Column renames that read tables written under an older schema version.
_COLUMN_UPGRADES = {1: {"metric": "indicator"}}


def table_renames(schema_version):
    """Column renames for reading retained tables of ``schema_version``.

    Raises for a version this code neither writes nor knows how to upgrade, so
    a renderer never draws from tables whose columns it misreads.
    """
    if schema_version == SCHEMA_VERSION:
        return {}
    if schema_version in _COLUMN_UPGRADES:
        return _COLUMN_UPGRADES[schema_version]
    raise ValueError(
        f"Unsupported WF0 diagnostics schema {schema_version!r} "
        f"(this toolbox writes {SCHEMA_VERSION}); re-run WF0 compute"
    )


def write_json(path, value):
    """Write strict JSON: unavailable numerical values are null."""

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def daily_series(climate_nc, cells_csv, source, data_sources=None):
    """Exact basin-mask reduction; no buffered extraction fallback."""
    with xr.open_dataset(climate_nc) as ds:
        cells = pd.read_csv(cells_csv)
        mask = cells_csv_mask(ds, cells_csv)
        keys = set(zip(cells[LAT].round(6), cells[LON].round(6)))
        available = {
            (round(float(a), 6), round(float(b), 6))
            for a in ds[LAT].values
            for b in ds[LON].values
        }
        if not keys or not keys <= available or mask is None:
            raise ValueError(f"{source}: basin_cells must match every declared cell")
        index = pd.DatetimeIndex(ds.time.values)
        if (
            index.has_duplicates
            or not index.is_monotonic_increasing
            or (index != index.normalize()).any()
        ):
            raise ValueError(
                f"{source}: climate store requires ordered unique daily dates"
            )
        daily = pd.DataFrame(index=pd.date_range(index.min(), index.max(), freq="D"))
        warning = []
        for var in ("precip", "temp"):
            from blueearth_cst.climate_analysis.diagnostic_settings import carries_temp

            own = var == "precip" or carries_temp(source)
            if own and var in ds:
                field = ds[var].where(mask)
                values = field.mean([LAT, LON], skipna=True).to_series()
                if var == "temp":
                    finite = values.dropna()
                    units = str(ds[var].attrs.get("units", "")).lower()
                    declared_celsius = units in {
                        "degc",
                        "celsius",
                        "degree_celsius",
                        "degrees_celsius",
                        "°c",
                    }
                    catalog_offset = None
                    if data_sources:
                        import yaml

                        catalogs = (
                            [data_sources]
                            if isinstance(data_sources, str)
                            else data_sources
                        )
                        for path in catalogs:
                            spec = yaml.safe_load(
                                Path(path).read_text(encoding="utf-8")
                            ).get(source, {})
                            if spec:
                                adapter = spec.get("data_adapter", spec)
                                catalog_offset = adapter.get("unit_add", {}).get("temp")
                    normalized_legacy = (
                        units in {"k", "kelvin"}
                        and catalog_offset == -273.15
                        and bool(ds.attrs.get("region_geojson_sha256"))
                    )
                    if (
                        not (declared_celsius or normalized_legacy)
                        or finite.empty
                        or finite.min() < -100
                        or finite.max() > 70
                    ):
                        values[:] = np.nan
                        warning.append(
                            f"{source}: temperature diagnostics unavailable because units or values could not be verified"
                        )
                    elif normalized_legacy:
                        warning.append(
                            f"{source}: temperature values are already in °C despite the K label; using as °C"
                        )
                daily[var] = values
                daily[f"{var}_cells"] = field.notnull().sum([LAT, LON]).to_series()
            else:
                daily[var] = np.nan
                daily[f"{var}_cells"] = 0
        daily.index.name = "date"
        lineage = {
            "source": source,
            "cells": len(keys),
            "units_original": {
                v: ds[v].attrs.get("units") for v in ("precip", "temp") if v in ds
            },
            "warnings": warning,
            "extraction_attributes": {
                key: str(value) for key, value in ds.attrs.items()
            },
            "extraction_span": [str(index.min().date()), str(index.max().date())],
            "catalogs": data_sources,
        }
    lineage["inputs"] = {
        str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (climate_nc, cells_csv)
    }
    return daily, lineage


def compute(daily, settings, m0, period, temperature=None, temperature_source=None):
    """Fit references on full records, then compute period-dependent products."""
    p = daily.precip
    t = daily.temp if temperature is None else temperature
    tol = dg.Completeness(
        settings["max_missing_days_month"], settings["max_missing_days_year"]
    )
    ref = tuple(settings["reference_period"][k] for k in ("start", "end"))
    wet = settings["wet_day_threshold_mm"]
    filled = dg.infill(p, tol)
    sel = dg.in_years(p.index, m0, period)
    displayed = p[sel]
    pf = filled[sel]
    valid = dg.valid_years(displayed, m0, tol)
    ws = dg.wet_day_stats(displayed, m0, wet, tol)
    ext = dg.extremes(displayed, m0, wet, tol)
    full_spi = {
        scale: dg.spi(filled, scale, ref, m0, settings["spi_distribution"])
        for scale in dg.SPI_SCALES
    }
    spi = {scale: v[dg.in_years(v.index, m0, period)] for scale, v in full_spi.items()}
    spells = dg.dry_spells(displayed, wet)
    pt = None
    if t.notna().any():
        pt = dg.pt_anomalies(p, t, ref, tol)
        pt = pt[pt.year.between(*period)]
    result = {
        "coverage": dg.coverage_class(displayed, m0, tol),
        "valid": valid,
        "clim": dg.monthly_climatology(pf, m0, valid),
        "tclim": dg.temperature_climatology(
            daily.temp[dg.in_years(daily.index, m0, period)],
            m0,
            dg.valid_years(daily.temp[dg.in_years(daily.index, m0, period)], m0, tol),
            tol,
        )
        if daily.temp.notna().any()
        else None,
        "timing": dg.accumulation_timing(displayed, m0, tol=tol),
        "anomaly": dg.monthly_anomaly(filled, m0, ref),
        "spi": spi,
        "spi_dist": settings["spi_distribution"],
        "spi_checks": {k: dg.spi_fit_checks(filled, k, ref, m0) for k in dg.SPI_SCALES},
        "drought": {k: dg.drought_event_table(spi[k]) for k in dg.DROUGHT_SCALES},
        "extremes": ext,
        "spells": dg.exceedance(spells),
        "n_spells": len(spells),
        "acf": dg.anomaly_acf(pf, ref, m0),
        "annual": ws,
        "trends": {
            "total": dg.sen_trend(ws.total),
            "wet_days": dg.sen_trend(ws.wet_days),
            "Rx1day": dg.sen_trend(ext.Rx1day),
        },
        "sensitivity": {
            "total": dg.start_year_sensitivity(ws.total),
            "Rx1day": dg.start_year_sensitivity(ext.Rx1day),
        },
        "pt": pt,
        "t_source": temperature_source,
    }
    # ACF reference means come from full record; only the pairs are display clipped.
    anomaly_months = dg.monthly_totals(filled)
    reference_mean = (
        anomaly_months[dg.in_years(anomaly_months.index, m0, ref)]
        .groupby(lambda d: d.month)
        .mean()
    )
    anomalies = (
        anomaly_months - reference_mean.reindex(anomaly_months.index.month).to_numpy()
    )
    anomalies = anomalies[dg.in_years(anomalies.index, m0, period)]
    result["acf"] = pd.DataFrame(
        [
            {
                "r": anomalies.corr(anomalies.shift(k))
                if anomalies.dropna().nunique() > 1
                else np.nan,
                "pairs": int((anomalies.notna() & anomalies.shift(k).notna()).sum()),
            }
            for k in range(1, 13)
        ],
        index=range(1, 13),
    )
    tables = {}
    fraction = dg.coverage(displayed, m0)
    missing = dg.missing_days(displayed, m0)
    tables["coverage"] = pd.DataFrame(
        [
            {
                "reporting_year": y,
                "calendar_month": month,
                "fraction": fraction.loc[month, y],
                "missing_days": missing.loc[month, y],
                "class": result["coverage"].loc[month, y],
            }
            for month in result["coverage"].index
            for y in result["coverage"].columns
        ],
        columns=[
            "reporting_year",
            "calendar_month",
            "fraction",
            "missing_days",
            "class",
        ],
    )
    years = sorted(set(dg.reporting_year(displayed.index, m0)))
    full = set(dg.full_years(displayed, m0))
    tv = (
        set(dg.valid_years(daily.temp[dg.in_years(daily.index, m0, period)], m0, tol))
        if daily.temp.notna().any()
        else set()
    )
    tables["valid_years"] = pd.DataFrame(
        [
            {
                "reporting_year": y,
                "full_year": y in full,
                "valid_precip": y in valid,
                "valid_temp": y in tv,
                "missing_days": int(
                    displayed[dg.reporting_year(displayed.index, m0) == y].isna().sum()
                ),
                "reason": "" if y in valid else "partial year or completeness failure",
            }
            for y in years
        ],
        columns=[
            "reporting_year",
            "full_year",
            "valid_precip",
            "valid_temp",
            "missing_days",
            "reason",
        ],
    )
    tables["annual_indices"] = (
        ws.join(ext[["Rx1day", "Rx5day"]]).rename_axis("reporting_year").reset_index()
    )
    months = dg.monthly_totals(filled)
    tm = (
        daily.temp.resample("MS")
        .mean()
        .where(
            daily.temp.resample("MS").size() - daily.temp.resample("MS").count()
            <= tol.max_month
        )
    )
    tables["monthly_values"] = (
        pd.DataFrame(
            {
                "total": months,
                "reference_mean": reference_mean.reindex(months.index.month).to_numpy(),
                "temp_mean": tm,
            }
        )
        .rename_axis("date")
        .reset_index()
    )
    tables["monthly_values"]["anomaly"] = (
        tables["monthly_values"].total - tables["monthly_values"].reference_mean
    )
    tables["monthly_values"]["reporting_year"] = dg.reporting_year(
        pd.DatetimeIndex(tables["monthly_values"].date), m0
    )
    tables["monthly_values"]["calendar_month"] = tables["monthly_values"].date.dt.month
    tables["monthly_values"]["usable_precip"] = tables["monthly_values"].total.notna()
    tables["monthly_values"]["usable_temp"] = tables["monthly_values"].temp_mean.notna()
    tables["precip_climatology"] = (
        result["clim"].rename_axis("calendar_month").reset_index()
    )
    tables["temp_climatology"] = (
        (
            result["tclim"]
            if result["tclim"] is not None
            else pd.DataFrame(columns=["mean", "p10", "p90", "n"])
        )
        .rename_axis("calendar_month")
        .reset_index()
    )
    if "n" not in tables["temp_climatology"]:
        tg = daily.temp.resample("MS")
        usable = tg.mean().where(tg.size() - tg.count() <= tol.max_month)
        usable = usable[np.isin(dg.reporting_year(usable.index, m0), tv)]
        tables["temp_climatology"]["n"] = (
            usable.groupby(usable.index.month)
            .count()
            .reindex(tables["temp_climatology"].calendar_month)
            .to_numpy()
        )
    tables["rainfall_timing"] = (
        result["timing"]
        .rename_axis("reporting_year")
        .rename_axis("fraction", axis=1)
        .stack(future_stack=True)
        .rename("day")
        .reset_index()
    )
    tables["rainfall_timing"]["valid"] = tables["rainfall_timing"].day.notna()
    tables["spi"] = pd.concat(
        [
            v.rename("value")
            .rename_axis("date")
            .reset_index()
            .assign(
                scale=k,
                distribution=settings["spi_distribution"],
                available=v.notna().to_numpy(),
                reason=np.where(
                    v.notna(), "", "insufficient accumulation or reference fit"
                ),
            )
            for k, v in spi.items()
        ],
        ignore_index=True,
    )
    checks = [
        v.reindex(
            columns=[
                "n",
                "zeros",
                "sw_gamma",
                "sw_pearson3",
                "skew",
                "max_abs",
                "mom_diff",
                "p3_bound_hits",
            ]
        )
        .rename_axis("calendar_month")
        .reset_index()
        .assign(
            scale=k,
            reason=np.where(
                v.get("sw_gamma", pd.Series(np.nan, index=v.index)).notna(),
                "",
                "insufficient or degenerate reference fit",
            ),
        )
        for k, v in result["spi_checks"].items()
    ]
    tables["spi_fit_checks"] = pd.concat(checks, ignore_index=True)
    tables["drought_events"] = pd.concat(
        [
            v.reset_index(names="event_id").assign(scale=k)
            for k, v in result["drought"].items()
        ],
        ignore_index=True,
    )
    tables["dry_spells"] = pd.DataFrame(
        {"spell_id": range(len(spells)), "length_days": spells}
    )
    tables["dry_spell_exceedance"] = (
        result["spells"]
        .rename("probability")
        .rename_axis("length_days")
        .reset_index()
        .assign(n_complete_spells=len(spells))
    )
    tables["anomaly_acf"] = result["acf"].rename_axis("lag_months").reset_index()
    tables["trends"] = (
        pd.DataFrame.from_dict(result["trends"], orient="index")
        .rename_axis("indicator")
        .reset_index()
    )
    tables["trends"]["correction"] = "yue_wang"
    tables["trend_sensitivity"] = pd.concat(
        [
            v.rename_axis("start_year").reset_index().assign(indicator=k)
            for k, v in result["sensitivity"].items()
        ],
        ignore_index=True,
    )
    tables["trend_sensitivity"]["end_year"] = period[1]
    tables["pt_anomalies"] = (
        pt
        if pt is not None
        else pd.DataFrame(columns=["season", "year", "p", "t", "p_anom_pct", "t_anom"])
    )
    tables["pt_anomalies"] = tables["pt_anomalies"].assign(
        temperature_source=temperature_source, season_year=tables["pt_anomalies"].year
    )
    rows = []
    for indicator in ("total", "wet_days", "sdii", "Rx1day", "Rx5day"):
        series = tables["annual_indices"].set_index("reporting_year")[indicator]
        good = series.dropna()
        for test, fn in (("pettitt", dg.pettitt), ("snht", dg.snht)):
            data = (
                fn(good)
                if len(good) >= 3 and good.std() > 0
                else {
                    "n": len(good),
                    "stat": np.nan,
                    "p": np.nan,
                    "break_after": np.nan,
                }
            )
            rows.append(
                data
                | {
                    "indicator": indicator,
                    "method": test,
                    "seed": 0,
                    "n_sim": 20000 if test == "snht" else 0,
                    "series_kind": "source",
                    "source_pair": "",
                    "reason": ""
                    if len(good) >= 3 and good.std() > 0
                    else "insufficient or constant samples",
                }
            )
    tables["homogeneity"] = pd.DataFrame(rows)
    return result, tables


def write_tables(paths, tables_by_source, settings, m0, period, lineage):
    """Write fixed source-tagged CSVs plus a retained calculation record."""
    for key, path in paths["compute"].items():
        if key not in next(iter(tables_by_source.values())):
            continue
        frame = pd.concat(
            [
                table[key].assign(
                    source=s, period_start=period[0], period_end=period[1]
                )
                for s, table in tables_by_source.items()
            ],
            ignore_index=True,
        )
        if key == "pt_anomalies":
            frame["precip_source"] = frame.source
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "water_year_start": m0,
        "period": list(period),
        "sources": list(tables_by_source),
        "lineage": lineage,
        "scientific_status": "Real-data exercised on Ntoum; not scientifically qualified",
        "units": {"precip": "mm/day", "temp": "Celsius"},
        "methods": "ETCCDI completeness; gamma/Pearson III SPI; McKee events; Theil–Sen/Yue–Wang; exploratory Pettitt/SNHT",
        "declared_outputs": [
            str(Path(p).relative_to(paths["root"]))
            for section in ("compute", "render")
            for p in paths[section].values()
        ],
        "availability": {
            s: {
                "valid_precip_years": table["valid_years"]
                .loc[table["valid_years"].valid_precip, "reporting_year"]
                .tolist(),
                "spi_reference_fit_failures": int(
                    table["spi_fit_checks"].sw_gamma.isna().sum()
                ),
            }
            for s, table in tables_by_source.items()
        },
    }
    metadata["scientific_status"] = {
        "prototype_tested": True,
        "validation_case": "Gabon Ntoum real data",
        "current_run_qualified": False,
        "scientifically_qualified": False,
    }
    metadata["reporting_conventions"] = {
        "year_label": "start",
        "year_start_month": m0,
        "calendar": "daily Gregorian",
        "missing": "NaN; tolerated amount gaps filled with observed month mean",
        "spatial_reduction": "equal-weight basin cells with skipna; daily contributing-cell counts retained",
    }
    metadata["display_period"] = {
        "requested": list(period),
        "actual": {
            s: [
                int(t["monthly_values"].reporting_year.min()),
                int(t["monthly_values"].reporting_year.max()),
            ]
            if len(t["monthly_values"])
            else None
            for s, t in tables_by_source.items()
        },
    }
    metadata["reference_period"] = {
        "requested": settings["reference_period"],
        "sample_counts": {
            s: t["spi_fit_checks"][["calendar_month", "scale", "n", "reason"]].to_dict(
                orient="records"
            )
            for s, t in tables_by_source.items()
        },
        "qualification": "Fewer than 30 usable reference years is exploratory; fit checks never establish scientific qualification",
    }
    metadata["unavailable_products"] = {
        s: (
            [
                {
                    "product": "temperature and own P–T",
                    "reason": "no genuine usable temperature",
                }
            ]
            if t["temp_climatology"].empty
            else []
        )
        + (
            [
                {
                    "product": "annual diagnostics",
                    "reason": "no complete valid reporting years",
                }
            ]
            if not t["valid_years"].valid_precip.any()
            else []
        )
        for s, t in tables_by_source.items()
    }
    metadata["censoring"] = {
        s: {
            "drought_events": int(t["drought_events"].censored.sum()),
            "complete_dry_spells": len(t["dry_spells"]),
            "dry_spell_policy": "record-edge and gap-touching runs excluded",
        }
        for s, t in tables_by_source.items()
    }
    metadata["column_definitions"] = {
        k: {c: c.replace("_", " ") for c in v.columns}
        for k, v in next(iter(tables_by_source.values())).items()
    }
    metadata["produced_paths"] = [
        str(Path(p).relative_to(paths["root"]))
        for key, p in paths["compute"].items()
        if key == "metadata" or Path(p).is_file()
    ]
    if settings["subbasin_figures"]:
        metadata["produced_paths"] += [
            str(p.relative_to(paths["root"]))
            for p in (Path(paths["root"]) / "tables" / "subbasins").rglob("*.csv")
        ]
    metadata["requested_analysis_window"] = settings["analysis_window"]
    metadata["source_labels"] = {
        s: "CHIRPS v2.0" if s in {"chirps", "chirps_global"} else s.upper()
        for s in tables_by_source
    }
    if len(tables_by_source) > 1:
        metadata["common_valid_years"] = sorted(
            set.intersection(
                *[
                    set(
                        t["valid_years"].loc[
                            t["valid_years"].valid_precip, "reporting_year"
                        ]
                    )
                    for t in tables_by_source.values()
                ]
            )
        )
    metadata["figure_encoding_record"] = (
        "plots/_engine/figure_captions.json (pooled anomaly bounds and rendered availability)"
    )
    metadata["limitations"] = [
        "Input homogeneity unverified; assimilation and gauge density changes can create spurious trends",
        "Source agreement is not accuracy; no independent observations",
        "ERA5 short-scale SPI fits may overstate dry extremes in flagged months",
        "Dry timing is ambiguous in bimodal rainfall regimes",
        "Gridded basin support differs across products; extrema are smoothed",
        "Relative homogeneity tests have weak power; no-break results do not rule out artefacts",
    ]
    write_json(paths["compute"]["metadata"], metadata)
    return metadata
