"""Monthly delta-change forcing collections, independent of stochastic WF3."""

from __future__ import annotations

import csv
import importlib.metadata
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import yaml

from blueearth_cst.experiment.content_identity import (
    atomic_record,
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.forcing_descriptor import (
    collection_forcing_descriptor,
    reader_unit_interpretation,
)
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.variable_registry import VARIABLES
from blueearth_cst.shared.workflow_config_snapshot import (
    file_reference,
    resolve_file_reference,
)

SCHEMA = "delta-scenario-collection/1"
ROW_FIELDS = ["run_id", "evaluate", "type", "role", "model", "scenario", "member"]


def _fields(value: Any, names: set[str], label: str) -> dict:
    """Require a closed mapping at an external boundary."""
    if type(value) is not dict or set(value) != names:
        raise ValueError(f"{label}: expected exactly {sorted(names)}")
    return value


def _period(value: Any, label: str) -> dict:
    """Validate inclusive complete calendar years."""
    value = _fields(value, {"start", "end"}, label)
    if any(type(v) is not int or not 1 <= v <= 9999 for v in value.values()):
        raise ValueError(f"{label}: years must be integers in 1..9999")
    if value["start"] > value["end"]:
        raise ValueError(f"{label}: start exceeds end")
    return value


def validate_settings(settings: Any) -> dict:
    """Validate the producer configuration, with no implicit scientific choices."""
    version = settings.get("schema_version") if isinstance(settings, dict) else None
    value = _fields(
        settings,
        {
            "schema_version",
            "historical_path",
            "source_like",
            "reader",
            "reference_period",
            "future_period",
            "simulation_window",
            "members",
            "unit_id_capacity",
        }
        | ({"factor_layout"} if version == "delta-change/2" else set()),
        "delta configuration",
    )
    if version not in {"delta-change/1", "delta-change/2"}:
        raise ValueError("unsupported delta configuration schema")
    if version == "delta-change/2" and value["factor_layout"] not in {
        "grid",
        "basin_scalar",
    }:
        raise ValueError("factor_layout must be grid or basin_scalar")
    if value["source_like"] != "era5":
        raise ValueError(
            "delta-change/1 currently supports extracted ERA5 forcing only"
        )
    for name in ("reference_period", "future_period", "simulation_window"):
        _period(value[name], name)
    units = dict(reader_unit_interpretation(value["reader"]).variables)
    # This profile feeds WF4's reviewed extracted-ERA5 reader without conversion.
    expected_units = {
        "temp": "degC",
        "temp_min": "degC",
        "temp_max": "degC",
        "precip": "mm/day",
        "press_msl": "hPa",
        "kin": "W/m2",
        "kout": "W/m2",
    }
    required = {"temp", "temp_min", "temp_max", "precip"}
    for name, expected in expected_units.items():
        if (name in required or name in units) and units.get(name) != expected:
            raise ValueError(
                f"delta-change/1 requires historical effective units {name}={expected}"
            )
    if type(value["historical_path"]) is not str or not value["historical_path"]:
        raise ValueError("historical_path must be nonempty text")
    members = value["members"]
    if type(members) is not list or not members:
        raise ValueError("members must be a nonempty explicit list")
    identities = []
    for member in members:
        _fields(
            member,
            {"model", "scenario", "member", "reference_path", "future_path"},
            "member",
        )
        if any(
            type(v) is not str or not v.strip() or any(c in v for c in "\r\n")
            for v in member.values()
        ):
            raise ValueError(
                "member identities and paths must be nonempty single-line text"
            )
        identities.append(tuple(member[k] for k in ("model", "scenario", "member")))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate model/scenario/member identity")
    capacity = value["unit_id_capacity"]
    if type(capacity) is not int or capacity < len(members) + 2:
        raise ValueError(
            "unit_id_capacity must accommodate all members and both controls"
        )
    return value


def monthly_factors(
    reference: xr.Dataset,
    future: xr.Dataset,
    reference_period: dict,
    future_period: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive offsets and ratios from complete scalar monthly samples.

    Monthly samples have equal weight, matching the existing WF2/pilot convention.
    Temperature units must agree (K or degC); precipitation units must agree.
    Nonpositive reference precipitation and negative future precipitation are refused.
    """
    means = []
    for ds, period in ((reference, reference_period), (future, future_period)):
        _period(period, "projection period")
        selected = ds.sel(time=slice(str(period["start"]), str(period["end"])))
        pairs = list(
            zip(
                selected.time.dt.year.values.tolist(),
                selected.time.dt.month.values.tolist(),
            )
        )
        expected = [
            (year, month)
            for year in range(period["start"], period["end"] + 1)
            for month in range(1, 13)
        ]
        if pairs != expected:
            raise ValueError(
                "projection requires exactly one ordered sample per requested year/month"
            )
        values = []
        for name in ("temp", "precip"):
            variable = selected[name].squeeze(drop=True)
            if variable.dims != ("time",) or not np.isfinite(variable.values).all():
                raise ValueError(
                    f"{name}: finite basin scalar monthly samples required"
                )
            if name == "precip" and (variable.values < 0).any():
                raise ValueError("projection precipitation cannot be negative")
            values.append(
                np.array(
                    [
                        variable.values[np.array([m for _, m in pairs]) == month]
                        .astype(float)
                        .mean()
                        for month in range(1, 13)
                    ]
                )
            )
        means.append(values)
    for name in ("temp", "precip"):
        units = reference[name].attrs.get("units")
        if not units or units != future[name].attrs.get("units"):
            raise ValueError(
                f"{name}: reference and future units must be explicit and equal"
            )
        if name == "temp" and units not in {
            "K",
            "degC",
            "degree_Celsius",
            "degrees_Celsius",
        }:
            raise ValueError("temperature projection must use K or degC")
    if (means[0][1] <= 0).any():
        raise ValueError("reference monthly precipitation must be strictly positive")
    offsets, ratios = means[1][0] - means[0][0], means[1][1] / means[0][1]
    if not np.isfinite(offsets).all() or not np.isfinite(ratios).all():
        raise ValueError("monthly factors are nonfinite")
    return offsets, ratios


def validate_precipitation_reference(
    reference: xr.Dataset, period: dict, policy: dict
) -> None:
    """Refuse near-zero reference months using a retained daily-rate threshold.

    Monthly totals are normalized using their own calendar before averaging.
    This is admission only: it never clips or changes the calculated factors.
    """
    policy = _fields(policy, {"min_denominator", "units"}, "precipitation policy")
    threshold = policy["min_denominator"]
    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or threshold <= 0
        or policy["units"] != "mm/day"
    ):
        raise ValueError("precipitation policy requires a positive mm/day threshold")
    selected = reference.sel(time=slice(str(period["start"]), str(period["end"])))
    precip = selected.precip.squeeze(drop=True)
    units = precip.attrs.get("units")
    if units == "mm/month":
        precip = precip / selected.time.dt.days_in_month
    elif units != "mm/day":
        raise ValueError("projection precipitation must use mm/day or mm/month")
    means = precip.groupby("time.month").mean("time")
    flagged = means.month.values[np.asarray(means.values) < threshold]
    if len(flagged):
        months = ", ".join(f"{int(month):02d}" for month in flagged)
        raise ValueError(
            f"reference precipitation below {threshold:g} mm/day in months {months}; "
            "relative factors are refused, not clipped"
        )


def apply_factors(
    historical: xr.Dataset, offsets: np.ndarray, ratios: np.ndarray
) -> xr.Dataset:
    """Apply monthly factors by labelled time; preserve dtype, grid and masks."""
    offsets, ratios = np.asarray(offsets), np.asarray(ratios)
    if (
        offsets.shape != ratios.shape
        or offsets.shape
        not in {
            (12,),
            (
                12,
                historical.sizes.get("latitude", 0),
                historical.sizes.get("longitude", 0),
            ),
        }
        or not np.isfinite(offsets).all()
        or not np.isfinite(ratios).all()
        or (ratios < 0).any()
    ):
        raise ValueError("factors require twelve finite offsets and nonnegative ratios")
    result = historical.copy(deep=True)
    months = historical.time.dt.month.values - 1
    for name in ("temp", "temp_min", "temp_max", "precip"):
        source = historical[name]
        if (
            "time" not in source.dims
            or source.dtype.kind != "f"
            or np.isinf(source.values).any()
        ):
            raise ValueError(
                f"{name}: floating time-dependent historical values required; infinity forbidden"
            )
        if name == "precip" and (source.values < 0).any():
            raise ValueError("historical precipitation cannot be negative")
        values = (ratios if name == "precip" else offsets)[months]
        dims = ("time",) if values.ndim == 1 else ("time", "latitude", "longitude")
        factor = xr.DataArray(
            values, dims=dims, coords={key: source[key] for key in dims}
        )
        transformed = (source * factor if name == "precip" else source + factor).astype(
            source.dtype
        )
        if (
            not np.array_equal(np.isnan(transformed), np.isnan(source))
            or np.isinf(transformed.values).any()
        ):
            raise ValueError(
                f"{name}: transformation changed missingness or overflowed"
            )
        result[name].data = transformed.transpose(*source.dims).data
    return result


def gridded_factors(reference, future, historical, settings, policy):
    """Map monthly change fields to the historical grid with HydroMT nearest.

    Projection grids must match each other, with explicit CRS. Missing mapped
    values and dry reference cells in the active historical domain are errors.
    Factors outside that domain are neutral; historical masks remain unchanged.
    """
    import hydromt  # noqa: F401 -- registers the raster accessor

    policy = _fields(policy, {"min_denominator", "units"}, "precipitation policy")
    if (
        type(policy["min_denominator"]) not in (int, float)
        or not np.isfinite(policy["min_denominator"])
        or policy["min_denominator"] <= 0
        or policy["units"] != "mm/day"
    ):
        raise ValueError("precipitation policy requires a positive mm/day threshold")

    target = historical.temp.isel(time=0, drop=True).transpose("latitude", "longitude")
    if historical.raster.crs is None:
        raise ValueError("historical grid requires an explicit CRS")
    target.raster.set_crs(historical.raster.crs)
    active = historical.temp.notnull().any("time").transpose("latitude", "longitude")
    active = active | historical.precip.notnull().any("time").transpose(
        "latitude", "longitude"
    )
    means = []
    for dataset, period in (
        (reference, settings["reference_period"]),
        (future, settings["future_period"]),
    ):
        if dataset.raster.crs is None:
            raise ValueError("projection grid requires an explicit CRS")
        selected = dataset.sel(time=slice(str(period["start"]), str(period["end"])))
        pairs = list(
            zip(
                selected.time.dt.year.values.tolist(),
                selected.time.dt.month.values.tolist(),
            )
        )
        if pairs != [
            (year, month)
            for year in range(period["start"], period["end"] + 1)
            for month in range(1, 13)
        ]:
            raise ValueError(
                "projection requires exactly one ordered sample per requested year/month"
            )
        for name in ("temp", "precip"):
            if set(selected[name].dims) != {
                "time",
                dataset.raster.x_dim,
                dataset.raster.y_dim,
            }:
                raise ValueError(f"{name}: monthly raster required")
            if np.isinf(selected[name]).any() or (
                name == "precip" and (selected[name] < 0).any()
            ):
                raise ValueError(f"{name}: invalid projection values")
        means.append(
            selected[["temp", "precip"]]
            .groupby("time.month")
            .mean("time", skipna=False)
        )
    if reference.raster.crs != future.raster.crs:
        raise ValueError("reference and future projection CRS differ")
    xr.align(means[0], means[1], join="exact")
    for name in ("temp", "precip"):
        units = reference[name].attrs.get("units")
        allowed = (
            {"K", "degC", "degree_Celsius", "degrees_Celsius"}
            if name == "temp"
            else {"mm/day", "mm/month"}
        )
        if units not in allowed or units != future[name].attrs.get("units"):
            raise ValueError(
                f"{name}: explicit equal supported projection units required"
            )
    ref = reference.sel(
        time=slice(
            str(settings["reference_period"]["start"]),
            str(settings["reference_period"]["end"]),
        )
    )
    daily = ref.precip
    if daily.attrs["units"] == "mm/month":
        daily = daily / ref.time.dt.days_in_month
    daily = daily.groupby("time.month").mean("time", skipna=False)
    daily = daily.transpose("month", reference.raster.y_dim, reference.raster.x_dim)
    daily.raster.set_crs(reference.raster.crs)
    mapped_daily = daily.raster.reproject_like(
        target, method="nearest"
    ).raster.mask_nodata()
    if bool((~np.isfinite(mapped_daily) & active).any()):
        raise ValueError(
            "reference precipitation does not cover every active historical grid cell"
        )
    if bool(((mapped_daily < policy["min_denominator"]) & active).any()):
        raise ValueError(
            "reference precipitation below threshold in active grid; relative factors are refused, not clipped"
        )
    fields = (means[1].temp - means[0].temp, means[1].precip / means[0].precip)
    result = []
    for field, neutral in zip(fields, (0.0, 1.0), strict=True):
        field = field.transpose("month", reference.raster.y_dim, reference.raster.x_dim)
        field.raster.set_crs(reference.raster.crs)
        mapped = field.raster.reproject_like(
            target, method="nearest"
        ).raster.mask_nodata()
        if bool((~np.isfinite(mapped) & active).any()):
            raise ValueError(
                "projection factors do not cover every active historical grid cell"
            )
        mapped = mapped.where(active, neutral).transpose(
            "month", "latitude", "longitude"
        )
        result.append(mapped.values)
    return tuple(result)


def derive_factors(reference, future, historical, settings, policy):
    """Dispatch the explicit factor layout without altering legacy arithmetic."""
    if settings.get("factor_layout") == "grid":
        return gridded_factors(reference, future, historical, settings, policy)
    result = monthly_factors(
        reference, future, settings["reference_period"], settings["future_period"]
    )
    if policy is not None:
        validate_precipitation_reference(
            reference, settings["reference_period"], policy
        )
    return result


def scenario_records(settings: dict) -> list[dict[str, str]]:
    """Allocate controls first, then members in explicit configuration order."""
    width = len(str(settings["unit_id_capacity"]))
    rows = [
        dict(role=role, model="", scenario="", member="")
        for role in ("historical", "neutral")
    ]
    rows.extend(
        {
            "role": "projection",
            **{key: member[key] for key in ("model", "scenario", "member")},
        }
        for member in settings["members"]
    )
    return [
        {
            "run_id": f"run_{i:0{width}d}",
            "evaluate": "true",
            "type": "delta-change",
            **row,
        }
        for i, row in enumerate(rows, 1)
    ]


def delta_groups(
    rows: list[dict[str, str]],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Keep controls separate and group only equal model/scenario projections.

    Periods and transformation method are collection-wide invariants. JSON keys
    avoid delimiter collisions in opaque upstream identifiers.
    """
    import json

    groups: dict[str, list[str]] = {}
    references = []
    for row in rows:
        key = json.dumps(
            [row["role"], row["model"], row["scenario"]], separators=(",", ":")
        )
        groups.setdefault(key, []).append(row["run_id"])
        if row["role"] == "historical":
            references.append(row["run_id"])
    return {key: tuple(value) for key, value in groups.items()}, tuple(references)


def generate_collection(config_path: Path, project_root: Path) -> Path:
    """Publish an immutable collection, or verify an identical completed request."""
    config_path, project_root = config_path.resolve(strict=True), project_root.resolve()
    settings = validate_settings(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    paths = [Path(settings["historical_path"])]
    for member in settings["members"]:
        paths.extend(Path(member[key]) for key in ("reference_path", "future_path"))
    paths = [(config_path.parent / path).resolve(strict=True) for path in paths]
    factors = []
    precipitation = VARIABLES["precip"].projections
    factor_policy = {
        "precip": {
            "min_denominator": precipitation.min_denominator,
            "units": precipitation.units,
        }
    }
    for index in range(len(settings["members"])):
        with (
            xr.open_dataset(paths[1 + index * 2]) as reference,
            xr.open_dataset(paths[2 + index * 2]) as future,
            xr.open_dataset(paths[0]) as historical,
        ):
            factors.append(
                derive_factors(
                    reference, future, historical, settings, factor_policy["precip"]
                )
            )
    # Observe the physical source before publishing anything. WF4 uses the same
    # evidenced interpretation when binding the existing preparation catalog.
    descriptor = collection_forcing_descriptor(paths[0], settings["reader"])
    if descriptor["timestep"] != "P1D":
        raise ValueError("delta-change/1 requires daily historical forcing")
    window = settings["simulation_window"]
    if (
        descriptor["start"][:10] > f"{window['start']:04d}-01-01"
        or descriptor["end"][:10] < f"{window['end']:04d}-12-31"
    ):
        raise ValueError("historical forcing does not cover simulation_window")
    resolved = dict(settings)
    resolved["historical_path"] = "inputs/historical.nc"
    resolved["members"] = [
        {
            **member,
            "reference_path": f"inputs/reference_{i}.nc",
            "future_path": f"inputs/future_{i}.nc",
        }
        for i, member in enumerate(settings["members"], 1)
    ]
    relative = [resolved["historical_path"]]
    for member in resolved["members"]:
        relative.extend(member[key] for key in ("reference_path", "future_path"))
    intent = {
        "schema_version": "delta-scenario-intent/2",
        "factor_policy": factor_policy,
        "settings": resolved,
        "scenario_spec": {"simulation_window": dict(window)},
        "input_hashes": [file_sha256(path) for path in paths],
        "input_sizes": [path.stat().st_size for path in paths],
        "provider_revision": file_sha256(Path(__file__)),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xarray", "netCDF4")
            + (
                ("hydromt", "rasterio", "pyproj")
                if settings.get("factor_layout") == "grid"
                else ()
            )
        },
        "historical_descriptor": descriptor,
        "run_group_id_capacity": settings["unit_id_capacity"],
        "run_group_id_width": len(str(settings["unit_id_capacity"])),
        "factors": [
            {"temp_offset_degC": offsets.tolist(), "precip_ratio": ratios.tolist()}
            for offsets, ratios in factors
        ],
    }
    identity = content_sha256(intent)
    engine = project_root / "scenarios/_engine/collections" / identity[:12]
    data = project_root / "scenarios/_engine/data" / identity[:12]
    marker_path = engine / "collection.json"
    if marker_path.exists():
        read_delta_collection(marker_path)
        return marker_path
    engine.mkdir(parents=True, exist_ok=False)
    data.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), data / "producer.py")
    for source, name, digest in zip(
        paths, relative, intent["input_hashes"], strict=True
    ):
        target = data / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(target) != digest:
            raise ValueError(
                "input changed during capture; incomplete collection retained"
            )
    rows = scenario_records(settings)
    atomic_record(engine / "collection_intent.json", intent)
    with (data / "scenario_run_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    series = []
    (data / "series").mkdir()
    with xr.open_dataset(data / relative[0]) as historical:
        historical.load()
        for index, row in enumerate(rows):
            offsets, ratios = (
                factors[index - 2] if index >= 2 else (np.zeros(12), np.ones(12))
            )
            transformed = (
                historical if index == 0 else apply_factors(historical, offsets, ratios)
            )
            path = data / "series" / f"{row['run_id']}.nc"
            transformed.to_netcdf(path)
            series.append(
                {
                    "run_id": row["run_id"],
                    "file": file_reference(path, "project_root", project_root),
                    "descriptor": collection_forcing_descriptor(
                        path, settings["reader"]
                    ),
                }
            )
    marker = {
        "schema_version": SCHEMA,
        "status": "ready",
        "collection_id": identity,
        "data_root": data.relative_to(project_root).as_posix(),
        "intent": file_reference(
            engine / "collection_intent.json", "project_root", project_root
        ),
        "scenario_run_lookup": file_reference(
            data / "scenario_run_lookup.csv", "project_root", project_root
        ),
        "series": series,
    }
    marker["collection_revision"] = content_sha256(marker)
    atomic_record(marker_path, marker)
    read_delta_collection(marker_path)
    return marker_path


def read_delta_collection(marker_path: Path) -> dict:
    """Validate the complete delta collection, including retained physical inputs."""
    marker_path = Path(marker_path)
    if marker_path.is_symlink():
        raise ValueError("collection marker cannot be a symbolic link")
    marker_path = marker_path.resolve(strict=True)
    directory = marker_path.parent
    if (
        marker_path.name != "collection.json"
        or directory.parent.name != "collections"
        or directory.parent.parent.name != "_engine"
        or directory.parent.parent.parent.name != "scenarios"
    ):
        raise ValueError(
            "delta collection must be discovered through its engine marker"
        )
    root = directory.parents[3]
    marker = _fields(
        read_canonical_json(marker_path),
        {
            "schema_version",
            "status",
            "collection_id",
            "collection_revision",
            "data_root",
            "intent",
            "scenario_run_lookup",
            "series",
        },
        "delta marker",
    )
    if marker["schema_version"] != SCHEMA or marker["status"] != "ready":
        raise ValueError("unsupported or unready delta collection")
    anchors = {"project_root": root}
    intent_path = resolve_file_reference(marker["intent"], anchors)
    if intent_path != directory / "collection_intent.json":
        raise ValueError("delta intent must be the sibling frozen record")
    retained_intent = read_canonical_json(intent_path)
    intent_version = (
        retained_intent.get("schema_version") if type(retained_intent) is dict else None
    )
    intent = _fields(
        retained_intent,
        {
            "schema_version",
            "settings",
            "scenario_spec",
            "input_hashes",
            "input_sizes",
            "environment",
            "provider_revision",
            "historical_descriptor",
            "run_group_id_capacity",
            "run_group_id_width",
            "factors",
        }
        | ({"factor_policy"} if intent_version == "delta-scenario-intent/2" else set()),
        "delta intent",
    )
    identity = content_sha256(intent)
    if (
        intent_version not in {"delta-scenario-intent/1", "delta-scenario-intent/2"}
        or identity != marker["collection_id"]
        or directory.name != identity[:12]
        or marker["data_root"] != f"scenarios/_engine/data/{identity[:12]}"
    ):
        raise ValueError("delta identity or location differs")
    if marker["collection_revision"] != content_sha256(
        {key: value for key, value in marker.items() if key != "collection_revision"}
    ):
        raise ValueError("delta collection revision differs")
    settings = validate_settings(intent["settings"])
    if content_sha256(intent["scenario_spec"]) != content_sha256(
        {"simulation_window": settings["simulation_window"]}
    ):
        raise ValueError("delta simulation window differs")
    if intent["run_group_id_capacity"] != settings["unit_id_capacity"] or intent[
        "run_group_id_width"
    ] != len(str(settings["unit_id_capacity"])):
        raise ValueError("delta namespace differs")
    data = root / marker["data_root"]
    if data.is_symlink():
        raise ValueError("delta data root cannot be a symbolic link")
    expected_names = ["inputs/historical.nc"]
    for i in range(1, len(settings["members"]) + 1):
        expected_names.extend([f"inputs/reference_{i}.nc", f"inputs/future_{i}.nc"])
    names = [settings["historical_path"]]
    for member in settings["members"]:
        names.extend(member[key] for key in ("reference_path", "future_path"))
    if (
        names != expected_names
        or len(intent["input_hashes"]) != len(names)
        or len(intent["input_sizes"]) != len(names)
    ):
        raise ValueError("delta input paths/count differ")
    for name, digest, size in zip(
        names, intent["input_hashes"], intent["input_sizes"], strict=True
    ):
        resolve_file_reference(
            {
                "schema_version": "artifact-reference/1",
                "path_base": "project_root",
                "path": (data / name).relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": size,
            },
            anchors,
        )
    if file_sha256(data / "producer.py") != intent["provider_revision"]:
        raise ValueError("retained producer bytes differ")
    if len(intent["factors"]) != len(settings["members"]):
        raise ValueError("delta factor count differs")
    for member, factors in zip(settings["members"], intent["factors"], strict=True):
        _fields(factors, {"temp_offset_degC", "precip_ratio"}, "monthly factors")
        with (
            xr.open_dataset(data / member["reference_path"]) as reference,
            xr.open_dataset(data / member["future_path"]) as future,
            xr.open_dataset(data / names[0]) as historical,
        ):
            precipitation_policy = None
            if intent_version == "delta-scenario-intent/2":
                policy = _fields(intent["factor_policy"], {"precip"}, "factor policy")
                precipitation_policy = policy["precip"]
            offsets, ratios = derive_factors(
                reference, future, historical, settings, precipitation_policy
            )
        if content_sha256(factors) != content_sha256(
            {"temp_offset_degC": offsets.tolist(), "precip_ratio": ratios.tolist()}
        ):
            raise ValueError("retained factors differ from projection inputs")
    if (
        collection_forcing_descriptor(data / names[0], settings["reader"])
        != intent["historical_descriptor"]
    ):
        raise ValueError("historical physical descriptor differs")
    rows = scenario_records(settings)
    lookup = resolve_file_reference(marker["scenario_run_lookup"], anchors)
    if lookup != data / "scenario_run_lookup.csv":
        raise ValueError("delta lookup path differs")
    with lookup.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS or list(reader) != rows:
            raise ValueError("delta lookup differs from declared members")
    if type(marker["series"]) is not list or len(marker["series"]) != len(rows):
        raise ValueError("delta series count differs")
    for row, entry in zip(rows, marker["series"], strict=True):
        _fields(entry, {"run_id", "file", "descriptor"}, "delta series")
        path = resolve_file_reference(entry["file"], anchors)
        if (
            entry["run_id"] != row["run_id"]
            or path != data / "series" / f"{row['run_id']}.nc"
        ):
            raise ValueError("delta series identity/path differs")
        if (
            collection_forcing_descriptor(path, settings["reader"])
            != entry["descriptor"]
        ):
            raise ValueError("delta forcing descriptor differs")
    return marker
