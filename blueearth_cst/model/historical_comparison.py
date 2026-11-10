"""Prepare fixed-model historical comparisons on explicitly shared daily support."""

# No `from __future__` import: Snakemake prepends a preamble to `script:` modules.
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from blueearth_cst.indicators.climate.periods import annual_periods
from blueearth_cst.model.historical_runs import read_historical_run, run_directory
from blueearth_cst.shared.func_plot_signature import compute_indicators
from blueearth_cst.shared.provenance import (
    environment_file_hashes,
    file_sha256,
    toolbox_identity,
)
from blueearth_cst.shared.wf3_science import (
    resolve_water_year_start,
    water_year_start_number,
)

METRICS = ("KGE", "NSE", "NSElog", "RMSE", "MSE", "Pbias", "VE")
SCHEMA = "historical-comparison/1"


def _daily_index(index: pd.Index, context: str) -> pd.DatetimeIndex:
    """Reject ambiguous dates instead of shifting or aggregating them."""
    index = pd.DatetimeIndex(index)
    if (
        index.empty
        or index.hasnans
        or index.tz is not None
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.equals(index.normalize())
    ):
        raise ValueError(
            f"{context} requires unique sorted midnight dates without a timezone"
        )
    return index


def evaluation_dates(window: object, water_year_start: str = "Jan") -> pd.DatetimeIndex:
    """Expand inclusive integer start-year labels to complete reporting years."""
    if not isinstance(window, dict) or set(window) != {"start", "end"}:
        raise ValueError("evaluation_window requires only start and end")
    month = water_year_start_number(resolve_water_year_start(water_year_start))
    periods = annual_periods(window["start"], window["end"], month)
    dates = pd.date_range(
        periods[0].start, periods[-1].stop, freq="D", inclusive="left"
    )
    return _daily_index(dates, "evaluation_window")


def complete_reporting_years(
    first: pd.Timestamp, last: pd.Timestamp, water_year_start: str = "Jan"
) -> dict[str, int]:
    """Start-year labels of the reporting years wholly inside ``[first, last]``.

    ``first`` and ``last`` are the first and last daily responses. Wflow labels
    a response at its interval end, so a run started on 1 January has no
    response for that date and its first reporting year is incomplete.
    """
    month = water_year_start_number(resolve_water_year_start(water_year_start))
    first, stop = pd.Timestamp(first), pd.Timestamp(last) + pd.Timedelta(days=1)
    start = first.year if pd.Timestamp(first.year, month, 1) >= first else first.year + 1
    end = stop.year - 1 if pd.Timestamp(stop.year, month, 1) <= stop else stop.year - 2
    if start > end:
        raise ValueError(
            f"no complete reporting year between {first.date()} and {last.date()} "
            f"(water year starts in {water_year_start}); lengthen simulation_window"
        )
    return {"start": start, "end": end}


def read_observations(path: Path, stations: set[str]) -> pd.DataFrame:
    """Read WF1-style semicolon observations while allowing absent stations."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=";"), [])
    if not header or header[0].strip() != "time":
        raise ValueError("observations require a semicolon CSV with first column time")
    labels = []
    for value in header[1:]:
        if not re.fullmatch(r"[0-9]+", value.strip()):
            raise ValueError(f"observation station is not a wflow_id: {value!r}")
        labels.append(str(int(value)))
    if len(labels) != len(set(labels)) or not set(labels) <= stations:
        raise ValueError("duplicate or unknown observation station IDs")
    frame = pd.read_csv(
        path, sep=";", index_col=0, parse_dates=True, float_precision="round_trip"
    )
    frame.columns = labels
    frame.index = _daily_index(frame.index, "observations")
    return frame.apply(pd.to_numeric, errors="raise")


def _metric_reason(value: float, sim: pd.Series, obs: pd.Series, missing: bool) -> str:
    if np.isfinite(value):
        return ""
    if missing:
        return "observations_unavailable"
    if len(obs) == 0:
        return "no_common_days"
    if len(obs) < 2:
        return "insufficient_pairs_or_zero_denominator"
    if float(obs.sum()) == 0:
        return "zero_observed_volume_or_variance"
    if float(obs.std(ddof=0)) == 0:
        return "constant_observations"
    if float(sim.std(ddof=0)) == 0:
        return "constant_simulation"
    return "undefined_metric_denominator"


def prepare_comparison(
    runs: dict[str, tuple[dict, pd.DataFrame, pd.DataFrame]],
    dates: pd.DatetimeIndex,
    *,
    water_year_start: str = "Jan",
    observations: pd.DataFrame | None = None,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Align IDs and calculate both supports; no mutation, infilling or ranking."""
    if len(runs) < 2:
        raise ValueError("select at least two distinct retained runs")
    dates = _daily_index(dates, "evaluation_window")
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
        raise ValueError("evaluation_window must be contiguous daily dates")
    month = water_year_start_number(resolve_water_year_start(water_year_start))
    first_record, first_frame, first_stations = next(iter(runs.values()))
    first_index = _daily_index(first_frame.index, "retained response")
    station_ids = sorted(first_frame.columns)
    if not station_ids:
        raise ValueError("retained runs have no discharge stations")
    identity_columns = sorted(set(first_stations.columns) - {"station_name"})
    reference_stations = first_stations.set_index("wflow_id").sort_index()
    for name, (record, frame, stations) in runs.items():
        if any(
            record.get(key) != first_record.get(key)
            for key in ("model_sha256", "time", "units", "time_label")
        ):
            raise ValueError(
                f"incompatible model, simulation period or initialization: {name}"
            )
        if (
            record.get("units") != "m3 s-1"
            or record.get("time_label") != "interval_end"
        ):
            raise ValueError("unsupported retained discharge units or time label")
        index = _daily_index(frame.index, f"retained run {name}")
        if (
            not index.equals(first_index)
            or not index.equals(pd.date_range(index[0], index[-1], freq="D"))
            or int(record["time"]["timestepsecs"]) != 86400
        ):
            raise ValueError(
                "historical comparisons require identical regular daily responses"
            )
        if not dates.isin(index).all():
            raise ValueError(f"requested evaluation dates unavailable in run {name}")
        if (
            sorted(frame.columns) != station_ids
            or stations.wflow_id.duplicated().any()
            or set(stations.wflow_id) != set(station_ids)
            or sorted(set(stations.columns) - {"station_name"}) != identity_columns
        ):
            raise ValueError(f"incompatible station identities: {name}")
        columns = [key for key in identity_columns if key != "wflow_id"]
        if (
            not stations.set_index("wflow_id")
            .sort_index()[columns]
            .equals(reference_stations[columns])
        ):
            raise ValueError(f"incompatible station registry: {name}")
    if observations is not None:
        _daily_index(observations.index, "observations")
        if observations.columns.duplicated().any() or not set(
            observations.columns
        ) <= set(station_ids):
            raise ValueError("duplicate or unknown observation station IDs")
    warmup_end = (
        first_index[0] + pd.DateOffset(years=1)
        if len(first_index) > 365
        else first_index[0]
    )
    excluded = dates < warmup_end
    rows = {
        key: []
        for key in (
            "coverage",
            "performance",
            "support",
            "daily",
            "monthly",
            "seasonal",
            "flow_duration",
        )
    }
    for station in station_ids:
        simulated = pd.DataFrame(
            {
                name: frame[station].reindex(dates)
                for name, (_, frame, _) in runs.items()
            }
        )
        missing = observations is None or station not in observations
        obs = (
            pd.Series(np.nan, index=dates)
            if missing
            else observations[station].reindex(dates)
        )
        sim_valid = np.isfinite(simulated) & simulated.ge(0)
        obs_valid = np.isfinite(obs) & obs.ge(0)
        sim_common = sim_valid.all(axis=1) & ~excluded
        paired = sim_common & obs_valid
        for day in dates:
            rows["support"].append(
                dict(
                    wflow_id=station,
                    time=day,
                    warmup_excluded=bool(day < warmup_end),
                    simulated_common=bool(sim_common[day]),
                    observation_paired=bool(paired[day]),
                    invalid_simulations=int((~sim_valid.loc[day]).sum()),
                    observation_valid=bool(obs_valid[day]),
                    water_year=day.year - int(day.month < month),
                )
            )
        rows["coverage"].append(
            dict(
                wflow_id=station,
                requested_days=len(dates),
                warmup_days=int(excluded.sum()),
                simulated_common_days=int(sim_common.sum()),
                observation_paired_days=int(paired.sum()),
                simulation_excluded_days=int((~sim_common & ~excluded).sum()),
                observation_excluded_days=int((sim_common & ~obs_valid).sum()),
                observations_available=not missing,
                reason="observations_unavailable" if missing else "",
                simulated_first=dates[sim_common][0] if sim_common.any() else None,
                simulated_last=dates[sim_common][-1] if sim_common.any() else None,
                paired_first=dates[paired][0] if paired.any() else None,
                paired_last=dates[paired][-1] if paired.any() else None,
            )
        )
        paired_obs = obs.where(paired)
        for name in runs:
            sim = simulated[name].where(paired)
            metrics = compute_indicators(
                xr.DataArray(sim, dims="time"), xr.DataArray(paired_obs, dims="time")
            )
            for scale in ("daily", "monthly"):
                metric_sim, metric_obs = sim, paired_obs
                if scale == "monthly":
                    metric_sim, metric_obs = (
                        sim.resample("ME").mean(),
                        paired_obs.resample("ME").mean(),
                    )
                metric_sim, metric_obs = metric_sim.dropna(), metric_obs.dropna()
                for metric in METRICS:
                    value = float(metrics.loc[(metric, scale), "station"])
                    rows["performance"].append(
                        dict(
                            wflow_id=station,
                            run=name,
                            scale=scale,
                            metric=metric,
                            value=value if np.isfinite(value) else np.nan,
                            pairs=len(metric_obs),
                            reason=_metric_reason(
                                value, metric_sim, metric_obs, missing
                            ),
                        )
                    )
        for support, mask in (
            ("simulated_common", sim_common),
            ("observation_paired", paired),
        ):
            if support == "observation_paired" and observations is None:
                continue
            values = simulated.where(mask, axis=0)
            if support == "observation_paired":
                values = values.assign(__observed__=obs.where(mask))
            for name, series in values.items():
                base = dict(wflow_id=station, support=support, series=name)
                rows["daily"].extend(
                    dict(**base, time=day, discharge=value)
                    for day, value in series.items()
                )
                monthly = series.resample("ME").mean()
                counts = series.resample("ME").count()
                rows["monthly"].extend(
                    dict(**base, time=day, discharge=value, days=int(counts[day]))
                    for day, value in monthly.items()
                )
                seasonal = (
                    monthly.groupby(monthly.index.month).mean().reindex(range(1, 13))
                )
                rows["seasonal"].extend(
                    dict(**base, month=m, discharge=value)
                    for m, value in seasonal.items()
                )
                ordered = np.sort(series.dropna().to_numpy())[::-1]
                rows["flow_duration"].extend(
                    dict(
                        **base,
                        exceedance=100 * rank / (len(ordered) + 1),
                        discharge=value,
                    )
                    for rank, value in enumerate(ordered, 1)
                )
    tables = {key: pd.DataFrame(value) for key, value in rows.items()}
    if tables["flow_duration"].empty:
        tables["flow_duration"] = pd.DataFrame(
            columns=["wflow_id", "support", "series", "exceedance", "discharge"]
        )
    tables["stations"] = first_stations.sort_values("wflow_id")
    tables["inventory"] = pd.DataFrame(
        [
            dict(
                run=name,
                model_sha256=record["model_sha256"],
                forcing_sha256=record["forcing_sha256"],
                recipe=json.dumps(record["recipe"], sort_keys=True),
                start=str(frame.index[0]),
                end=str(frame.index[-1]),
                samples=len(frame),
                toolbox=json.dumps(record.get("toolbox", {}), sort_keys=True),
                environment=json.dumps(record.get("environment", {}), sort_keys=True),
            )
            for name, (record, frame, _) in runs.items()
        ]
    )
    metadata = dict(
        schema_version=SCHEMA,
        runs=list(runs),
        requested_start=str(dates[0].date()),
        requested_end=str(dates[-1].date()),
        water_year_start_month=month,
        warmup_exclusive_end=str(warmup_end.date()),
        warmup_applied=len(first_index) > 365,
        warmup_policy="calendar_year_if_more_than_365_samples",
        units="m3 s-1",
        time_label="interval_end",
        observations_supplied=observations is not None,
    )
    return metadata, tables


def compare(config_path: Path) -> Path:
    """Read an explicit selection and publish self-contained numerical tables."""
    config_path = config_path.resolve()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {"project_dir", "runs", "evaluation_window", "output_dir"}
    allowed = required | {"water_year_start", "observations_path"}
    if (
        not isinstance(settings, dict)
        or not required <= settings.keys()
        or settings.keys() - allowed
    ):
        raise ValueError(
            f"comparison requires {sorted(required)}; optional water_year_start, observations_path"
        )
    names = settings["runs"]
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or len(names) < 2
        or len(set(names)) != len(names)
    ):
        raise ValueError("runs must contain at least two distinct retained names")
    for key in ("project_dir", "output_dir", "observations_path"):
        value = settings.get(key)
        if key == "observations_path" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
    project = (config_path.parent / settings["project_dir"]).resolve()
    output = (config_path.parent / settings["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError(
            f"comparison output already exists: {output}; choose a new output_dir"
        )
    retained_hashes = {
        name: file_sha256(run_directory(project, name) / "complete.json")
        for name in names
    }
    runs = {name: read_historical_run(project, name) for name in names}
    observation_path = settings.get("observations_path")
    observations = None
    observation_hash = None
    if observation_path is not None:
        observation_path = (config_path.parent / observation_path).resolve()
        observation_hash = file_sha256(observation_path)
        ids = set(next(iter(runs.values()))[1].columns) if runs else set()
        observations = read_observations(observation_path, ids)
    metadata, tables = prepare_comparison(
        runs,
        evaluation_dates(
            settings["evaluation_window"], settings.get("water_year_start", "Jan")
        ),
        water_year_start=settings.get("water_year_start", "Jan"),
        observations=observations,
    )
    repo = Path(__file__).resolve().parents[2]
    metadata.update(
        config=settings,
        toolbox=toolbox_identity(repo),
        environment=environment_file_hashes(repo),
        retained_runs=retained_hashes,
        observations_sha256=observation_hash,
    )
    if observation_path and file_sha256(observation_path) != observation_hash:
        raise ValueError("observations changed while preparing comparison")
    for name, digest in retained_hashes.items():
        if file_sha256(run_directory(project, name) / "complete.json") != digest:
            raise ValueError(f"retained run changed while preparing comparison: {name}")
    output.mkdir(parents=True)
    (output / "tables").mkdir()
    for key, table in tables.items():
        table.to_csv(
            output / "tables" / f"{key}.csv", index=False, float_format="%.17g"
        )
    metadata["table_hashes"] = {
        path.name: file_sha256(path) for path in sorted((output / "tables").iterdir())
    }
    (output / "comparison.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output


def read_comparison(root: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Verify retained numerical products; never access model or observation inputs."""
    metadata = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    expected = {
        f"{name}.csv"
        for name in (
            "coverage",
            "performance",
            "support",
            "daily",
            "monthly",
            "seasonal",
            "flow_duration",
            "stations",
            "inventory",
        )
    }
    if (
        metadata.get("schema_version") != SCHEMA
        or set(metadata.get("table_hashes", {})) != expected
    ):
        raise ValueError("incomplete or unsupported historical comparison")
    tables = {}
    for name, digest in metadata["table_hashes"].items():
        path = root / "tables" / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError(f"damaged comparison table: {name}")
        tables[path.stem] = pd.read_csv(
            path, dtype={"wflow_id": str}, float_precision="round_trip"
        )
    return metadata, tables


if __name__ == "__main__":
    sm = globals()["snakemake"]
    from blueearth_cst.model.plot_historical_comparison import render
    from blueearth_cst.shared.snake_utils import tee_to_log

    with tee_to_log(sm.log[0]):
        # The settings file is the standalone script's input, so the same
        # comparison can be reproduced or varied with scripts/compare_historical.py.
        settings_path = Path(sm.output.settings)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(
            yaml.safe_dump(dict(sm.params.settings), sort_keys=False), encoding="utf-8"
        )
        render(compare(settings_path), annotation=sm.params.annotation)
