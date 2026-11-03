"""Prepare WF2 analysis tables from explicitly declared, identity-checked series.

NOTE: Snakemake ``script:`` module; inputs, outputs and settings come from
``snakemake.input/output/params``. No standalone CLI or future-import preamble.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from blueearth_cst.projections import projection_analysis as analysis
from blueearth_cst.projections import projection_trajectories as trajectories
from blueearth_cst.projections import series_identity
from blueearth_cst.projections.get_change_climate_proj import (
    NoCompleteHydrologicalYear,
    hydrological_year_bounds,
)
from blueearth_cst.projections.projection_coverage import coverage_table
from blueearth_cst.projections.projection_figures import figure_key
from blueearth_cst.projections.series_checks import (
    assert_calendar_agreement,
    load_checked_series,
)
from blueearth_cst.shared.snake_utils import listed, log_row, plural, tee_to_log


def distinct_windows(windows):
    """Retain one computation per window and all configured horizon labels."""
    result = {}
    for horizon, supplied in windows.items():
        record = dict(supplied)
        key = record["window_id"]
        if key not in result:
            result[key] = {
                **record,
                "horizons": [],
                "configured": {},
                "horizon_windows": {},
            }
        elif tuple(record["reference"]) != tuple(result[key]["reference"]):
            raise ValueError(f"{key}: shared analysis window has different references")
        result[key]["horizons"].append(horizon)
        result[key]["configured"][horizon] = record["configured"]
        result[key]["horizon_windows"][horizon] = record
    return list(result.values())


def display_null_bands(ratios):
    """Prepare benchmarks for each actual mask at equal-model median CV.

    Several retained-mask strata can occur in one panel. Preserve them rather
    than inventing representative counts or averaging interval bounds.
    Temperature SD ratios are invariant to Gaussian location and scale.
    """
    selected = ratios.loc[
        ratios.variability_eligible
        & ratios.period.isin(["annual", "DJF", "MAM", "JJA", "SON"])
        & (
            ((ratios.variable == "temp") & (ratios.statistic == "std_detrended"))
            | ((ratios.variable == "precip") & (ratios.statistic == "cv_detrended"))
        )
    ]
    keys = ["scenario", "window_id", "period", "variable", "statistic"]
    records = []
    for key, group in selected.groupby(keys, sort=True):
        cv = (
            float(group.groupby("model").reference_value.mean().median())
            if key[3] == "precip"
            else None
        )
        masks = group[["reference_years", "future_years"]].drop_duplicates()
        for mask in masks.itertuples(index=False):
            reference, future = (
                json.loads(mask.reference_years),
                json.loads(mask.future_years),
            )
            band = analysis.null_ratio_band(
                [
                    {
                        "reference_years": reference,
                        "future_years": future,
                        "reference_mean": 1.0 if cv is not None else 0.0,
                        "reference_sd": cv if cv is not None else 1.0,
                    }
                ],
                statistic=key[4],
            )
            records.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    **band,
                    "null_status": band["status"],
                    "reference_cv": cv,
                    "reference_years": reference,
                    "future_years": future,
                    "n_mask_strata": len(masks),
                    "reference_cv_method": "median across model arithmetic means of eligible member reference CVs",
                    "method": "exact retained-mask white-noise benchmark; equal-model median reference CV for precipitation",
                }
            )
    return records


def _model_ratio_summary(models):
    """Equal-model log quantiles, preserving the lab's even-count median."""
    logged = models.assign(value=np.log(models.value.where(models.value > 0)))
    result = analysis.ensemble_summary(logged)
    columns = ["median", "min", "max", "q25", "q75"]
    result[columns] = np.exp(result[columns])
    return result


def prepare_tables(
    points,
    windows,
    *,
    region_fp,
    water_year_start="Jan",
    variable_spec=None,
    min_reference=None,
    combinations=None,
):
    """Compute tables once per distinct window; corrupt identities still raise."""
    windows = distinct_windows(windows)
    if not windows:
        raise ValueError("projection analysis requires at least one future window")
    references = {tuple(window["reference"]) for window in windows}
    if len(references) != 1:
        raise ValueError("full_period trajectories require one analysis reference")
    reference = next(iter(references))
    cache, change_rows, anomaly_rows, inventory, coverage_points = {}, [], [], {}, []
    for point in points:
        datasets = []
        for role, path_key, components_key, series_key in (
            (
                "historical",
                "series_path_hist",
                "digest_components_hist",
                "reference_series_key",
            ),
            (point["scenario"], "series_path", "digest_components_fut", "series_key"),
        ):
            dataset, _ = load_checked_series(
                point[path_key],
                point[components_key],
                region_fp,
                f"{point['model']} {role}",
                cache,
            )
            datasets.append(dataset)
            inventory[point[series_key]] = {
                "series_key": point[series_key],
                "calendar": dataset.attrs.get("cst_calendar", ""),
                "start": str(dataset.time.values[0])[:7],
                "end": str(dataset.time.values[-1])[:7],
            }
        historical, future = datasets
        assert_calendar_agreement(
            historical, future, f"{point['model']} {point['scenario']}"
        )
        configured_reference = next(iter(windows[0]["configured"].values()))[
            "reference"
        ]
        configured_hist = historical.sel(
            time=slice(*(str(year) for year in configured_reference))
        )
        try:
            _, _, reference_count = hydrological_year_bounds(
                configured_hist, water_year_start
            )
        except NoCompleteHydrologicalYear:
            # Analysis rows retain the missing-reference reason; coverage
            # metadata must not stop usable combinations from being prepared.
            reference_count = 0
        coverage_points.append({**point, "n_reference_years": reference_count})
        for window in windows:
            change_rows.append(
                analysis.analysis_change_table(
                    historical,
                    future,
                    identity=point,
                    window_id=window["window_id"],
                    reference_window=window["reference"],
                    future_window=window["used"],
                    water_year_start=water_year_start,
                    variable_spec=variable_spec,
                    min_reference=min_reference,
                )
            )
        anomaly_rows.append(
            trajectories.member_anomalies(
                historical,
                future,
                identity=point,
                reference_window=reference,
                water_year_start=water_year_start,
                min_reference=min_reference,
            )
        )
    changes = (
        pd.concat(change_rows, ignore_index=True)
        if change_rows
        else pd.DataFrame(columns=analysis.CHANGE_COLUMNS)
    )
    anomalies = (
        pd.concat(anomaly_rows, ignore_index=True)
        if anomaly_rows
        else pd.DataFrame(columns=trajectories.MEMBER_COLUMNS)
    )
    coverage = coverage_table(
        coverage_points,
        combinations=combinations,
        inventory=pd.DataFrame(inventory.values()),
    )
    models = analysis.model_changes(changes, statistic=None)
    climate = analysis.climatology(changes)
    ratios = analysis.member_null_bands(analysis.member_ratios(changes))
    model_ratios = analysis.model_ratios(ratios)
    tables = {
        "coverage": coverage,
        "changes": changes,
        "model_changes": models,
        "change_summary": analysis.with_variability_band(
            analysis.ensemble_summary(models), changes
        ),
        "sign_agreement": analysis.sign_agreement(models),
        "climatology": climate,
        "climatology_summary": analysis.climatology_summary(climate),
        "seasonal_values": (values := analysis.seasonal_values(climate, changes)),
        "seasonal_values_summary": analysis.ensemble_summary(values),
        "monthly_change_summary": analysis.with_variability_band(
            analysis.monthly_change(changes)[1], changes
        ),
        "member_anomalies": anomalies,
        "trajectory_range": trajectories.trajectory_range(
            trajectories.model_trajectories(anomalies)
        ),
        "variability_member_ratios": ratios,
        "variability_model_ratios": model_ratios,
        "variability_summary": analysis.ratio_summary(ratios),
    }
    tables["availability"] = analysis.product_availability(
        {
            "coverage": coverage,
            "trajectories": anomalies,
            "models": models,
            "climatology": climate,
            "member_ratios": ratios,
            "model_ratios": model_ratios,
        },
        windows,
    )
    unavailable = changes.loc[
        changes.status == "unavailable",
        ["model", "scenario", "member", "window_id", "reason"],
    ].drop_duplicates()
    record = {
        "schema_version": 1,
        "water_year_start": water_year_start,
        "windows": windows,
        "reference": reference,
        "reference_rule": windows[0]["reference_rule"],
        "methods": {
            "analysis_precision": "float64; aligned kernels and inputs unchanged",
            "dry_rule": "analysis period reference mean; CV also requires wet future mean",
            "aligned_dry_rule": "kernel per-statistic reference threshold",
            "monthly_water_year": "analysis passes configured start; aligned monthly convention retained",
            "season_year": "middle month calendar year; all three months inside slice; calendar-day weighted rates",
            "season_min_years": {
                window["window_id"]: {
                    "reference": analysis.seasonal_min_years(
                        window["reference"][1] - window["reference"][0] + 1
                    ),
                    "future": analysis.seasonal_min_years(
                        window["used"][1] - window["used"][0] + 1
                    ),
                }
                for window in windows
            },
            "season_coverage_rule": "ceil(0.9 * owned window years), minimum 3; separate reference and future thresholds",
            "seasonal_precipitation_parity_exception": "percentages use mean rates; lab uses totals",
            "precipitation_cv": "detrended SD of period integrals divided by integral mean",
            "detrending": "OLS on actual retained year coordinates; residual SD ddof=0",
            "zero_variability_rtol": analysis.ZERO_VARIABILITY_RTOL,
            "degenerate_variability": "raw SD for raw statistics; own detrended SD for ratios; no floors; zero future ratio reported but excluded from logarithmic reductions",
            "weights": "equal model; eligible member weight 1/eligible members in model for pooled log quantiles",
            "iqr_min_models": analysis.MIN_MODELS_IQR,
            "agreement_bands": {
                f"{variable}/{quantity}": value
                for (variable, quantity), value in analysis.AGREEMENT_BANDS.items()
            },
            "agreement_interpretation": "model and family sign counts; bands are display heuristics, not likelihoods",
            "null_band": {
                "level": analysis.NULL_BAND_LEVEL,
                "simulations": analysis.NULL_SIMULATIONS,
                "seed": analysis.NULL_SEED,
                "method": "independent Gaussian union-year draws; separate OLS; exact member/model statistic",
                "caveat": "white-noise benchmark; autocorrelation widens the true null",
                "invalid_draws": "nonpositive CV means excluded and counted; interval conditional on valid draws",
                "pooled_quantiles": "ensemble spread, no null band",
            },
            "trajectory": "unsmoothed annual anomalies; member means within model; ensemble range includes annual variability",
        },
        "models": json.loads(
            coverage.loc[coverage.available]
            .groupby(["model", "family", "family_source"], dropna=False)
            .member.nunique()
            .rename("n_members")
            .reset_index()
            .to_json(orient="records")
        ),
        "unavailable_members": json.loads(unavailable.to_json(orient="records")),
        "availability": json.loads(tables["availability"].to_json(orient="records")),
        "display_null_bands": display_null_bands(ratios),
        "variability_model_summary": json.loads(
            _model_ratio_summary(model_ratios).to_json(orient="records")
        ),
    }
    return tables, record


def skipped_product_messages(availability):
    """Explain skipped products once per product/reason, with affected windows."""
    missing = availability.loc[~availability.available]
    for (filename, reason), group in missing.groupby(
        ["filename", "reason"], sort=False
    ):
        key = figure_key(filename)
        label = {
            "availability.png": "Ensemble coverage",
            "trajectory-temperature.png": "Temperature anomaly trajectories",
            "trajectory-precipitation.png": "Precipitation anomaly trajectories",
            "annual-temperature.png": "Annual temperature overview",
            "annual-precipitation.png": "Annual precipitation overview",
            "change-temperature.png": "Temperature changes",
            "change-precipitation-pct.png": "Percentage precipitation changes",
            "values-precipitation-total.png": "Seasonal precipitation totals",
            "values-temperature-mean.png": "Seasonal mean temperature",
            "seasonal-cycle-temperature.png": "Temperature seasonal cycle",
            "seasonal-cycle-precipitation.png": "Precipitation seasonal cycle",
            "monthly-change-temperature.png": "Monthly temperature changes",
            "monthly-change-precipitation.png": "Monthly precipitation changes",
            "values-temperature-sd.png": "Temperature interannual standard deviation",
            "values-precipitation-sd.png": "Precipitation interannual standard deviation",
            "values-precipitation-cv.png": "Precipitation coefficient of variation",
            "change-factor-cloud.png": "Annual temperature–precipitation change cloud",
            "sign-agreement.png": "Model sign agreement",
            "change-factor-cloud-combined.png": "Combined-window temperature–precipitation change cloud",
        }.get(key, key.removesuffix(".png").replace("-", " "))
        if key == "variability-member-ratios.png":
            label = "Annual variability member-ratio boxplots"
            if reason == "no model with two eligible member ratios":
                reason = "not enough eligible ensemble members"
        windows = list(dict.fromkeys(w for w in group.window_id if w))
        scope = (
            f" ({listed(windows)})"
            if windows and key != "variability-member-ratios.png"
            else ""
        )
        detail = "" if key == "change-factor-cloud-combined.png" else f": {reason}"
        yield f"{label[0].upper() + label[1:]} skipped{scope}{detail}"


def run(sm):
    """Assert declared inputs, write every table, and log bounded summaries."""
    points = [dict(point) for point in sm.params.points]
    declared = {os.path.abspath(str(path)) for path in sm.input.series_nc}
    opened = {
        os.path.abspath(str(point[key]))
        for point in points
        for key in ("series_path_hist", "series_path")
    }
    if declared != opened:
        raise RuntimeError(
            f"analysis series set differs from declared inputs: unused={sorted(declared - opened)}, undeclared={sorted(opened - declared)}"
        )
    tables, record = prepare_tables(
        points,
        sm.params.windows,
        region_fp=series_identity.region_fingerprint(sm.input.region_path),
        water_year_start=sm.params.water_year_start,
        variable_spec=dict(sm.params.variable_spec),
        min_reference=dict(sm.params.min_reference),
        combinations=[dict(item) for item in sm.params.combinations],
    )
    if set(tables) != set(sm.params.table_names):
        raise RuntimeError("analysis table names differ from declared output contract")
    for name, table in tables.items():
        path = Path(sm.output[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, na_rep="")
    record_path = Path(sm.output.analysis_record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    log_row(
        f"Analysis windows: {listed([w['window_id'] for w in record['windows']])}",
        module="analysis",
    )
    available = int(tables["availability"].available.sum())
    moved = [w["window_id"] for w in record["windows"] if w["shifted"]]
    joined = [w["window_id"] for w in record["windows"] if w["joined"]]
    if moved:
        log_row(
            f"Shifted {plural(len(moved), 'window')}: {listed(moved)}; see analysis_record.json",
            module="analysis",
        )
    if joined:
        log_row(
            f"Joined historical data for {plural(len(joined), 'window')}: {listed(joined)}",
            module="analysis",
        )
    missing = tables["availability"].loc[~tables["availability"].available]
    log_row(
        f"Figures: {available} available, {len(missing)} skipped",
        module="analysis",
    )
    for message in skipped_product_messages(tables["availability"]):
        log_row(message, module="analysis")
    if record["unavailable_members"]:
        log_row(
            f"Skipped {plural(len(record['unavailable_members']), 'member–window combination')}; reasons → analysis_record.json",
            module="analysis",
        )


if "snakemake" in globals():
    sm = globals()["snakemake"]
    with tee_to_log(sm.log[0]):
        run(sm)
