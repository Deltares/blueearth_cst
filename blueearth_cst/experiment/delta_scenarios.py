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
#: delta-change/3 rows also name the WF2 window; windows are never pooled.
WINDOW_ROW_FIELDS = [*ROW_FIELDS, "window_id"]


def row_fields(settings: dict) -> list[str]:
    return (
        WINDOW_ROW_FIELDS
        if settings["schema_version"] == "delta-change/3"
        else ROW_FIELDS
    )


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


#: Authored delta-change/3 keys; members, periods and the historical record are derived.
V3_AUTHORED = {
    "schema_version",
    "name",
    "projections",
    "unit_id_capacity",
    "historical_override",
    "dry_reference",
    "pet",
}
#: Keys the producer adds when it resolves a v3 request into its intent.
V3_RESOLVED_KEYS = {
    "historical_path",
    "source_like",
    "reader",
    "simulation_window",
    "historical_period",
    "factor_layout",
    "members",
    "windows",
    "wf2",
    "historical_recipe",
}
OVERRIDE_KEYS = {"period", "path", "source", "units", "units_note"}
#: Effective units of supported historical sources, supplied by the toolbox
#: (design §3.3). Extraction labels are never trusted for these.
SUPPORTED_SOURCES = {
    "era5": {
        "precip": "mm/day",
        "temp": "degC",
        "temp_min": "degC",
        "temp_max": "degC",
        "press_msl": "hPa",
        "kin": "W/m2",
        "kout": "W/m2",
    }
}
SOURCE_EVIDENCE = (
    "toolbox-supplied effective units of the CST {source} historical extraction "
    "(delta-change/3, design §3.3)"
)


def source_reader(source: str, *, evidence: str | None = None) -> dict:
    """The unit interpretation of a supported historical source."""
    if source not in SUPPORTED_SOURCES:
        raise ValueError(
            f"delta-change/3 supports historical sources {sorted(SUPPORTED_SOURCES)}, "
            f"not {source!r}"
        )
    return {
        "metadata": {
            "cst_unit_interpretation": {
                "revision": "extracted-daily-forcing/1",
                "evidence": evidence or SOURCE_EVIDENCE.format(source=source),
                "variables": [list(item) for item in SUPPORTED_SOURCES[source].items()],
            }
        }
    }


def _validate_override(value: Any) -> dict:
    if type(value) is not dict or not value or not set(value) <= OVERRIDE_KEYS:
        raise ValueError(
            f"historical_override: expected a nonempty subset of {sorted(OVERRIDE_KEYS)}"
        )
    if "period" in value:
        _period(value["period"], "historical_override.period")
    file_keys = OVERRIDE_KEYS - {"period"}
    if file_keys & set(value) and not file_keys <= set(value):
        raise ValueError(
            "historical_override.path requires source, units and units_note"
        )
    if "path" in value:
        for key in ("path", "units_note"):
            if type(value[key]) is not str or not value[key].strip():
                raise ValueError(f"historical_override.{key} must be nonempty text")
        source = value["source"]
        if source not in SUPPORTED_SOURCES:
            raise ValueError(
                f"historical_override.source must be one of {sorted(SUPPORTED_SOURCES)}"
            )
        if value["units"] != SUPPORTED_SOURCES[source]:
            raise ValueError(
                f"historical_override.units must list every variable with the "
                f"{source} effective units {SUPPORTED_SOURCES[source]}"
            )
    return value


def _validate_dry_reference(value: Any) -> dict:
    """``{policy: refuse}`` or ``{policy: cap, max_change: (0, 1]}`` (design §1.2)."""
    if type(value) is not dict or value.get("policy") not in {"refuse", "cap"}:
        raise ValueError("dry_reference.policy must be refuse or cap")
    expected = {"policy", "max_change"} if value["policy"] == "cap" else {"policy"}
    _fields(value, expected, f"dry_reference ({value['policy']})")
    if value["policy"] == "cap":
        change = value["max_change"]
        if type(change) not in (int, float) or not 0 < change <= 1:
            raise ValueError("dry_reference.max_change must be a fraction in (0, 1]")
    return value


def _validate_v3(value: dict) -> dict:
    """Authored or resolved delta-change/3 settings (WF2 projection source)."""
    from blueearth_cst.experiment.collection_index import validate_collection_name
    from blueearth_cst.experiment.delta_wf2 import projection_settings

    if type(value) is not dict:
        raise ValueError("delta-change/3 configuration must be a mapping")
    resolved = "members" in value
    allowed = V3_AUTHORED | (V3_RESOLVED_KEYS if resolved else set())
    unknown = set(value) - allowed
    if unknown or "projections" not in value:
        raise ValueError(
            f"delta-change/3 configuration: unknown keys {sorted(unknown)}; "
            f"allowed {sorted(V3_AUTHORED)} with projections required"
            if unknown
            else "delta-change/3 configuration requires projections"
        )
    if resolved and not V3_RESOLVED_KEYS <= set(value):
        raise ValueError("resolved delta-change/3 settings are incomplete")
    projection_settings(value["projections"])
    if "name" in value:
        validate_collection_name(value["name"])
    if "historical_override" in value:
        _validate_override(value["historical_override"])
    if "dry_reference" in value:
        _validate_dry_reference(value["dry_reference"])
    if "pet" in value:
        pet = value["pet"]
        authored = {"mode"}
        if type(pet) is not dict or not authored <= set(pet) <= authored | {"method"}:
            raise ValueError("pet: expected {mode: projection_ratio}")
        if pet["mode"] != "projection_ratio":
            raise ValueError("pet.mode must be projection_ratio")
        if "method" in pet and not resolved:
            raise ValueError(
                "pet.method is taken from the case's PET formula; do not set it"
            )
        if "path" in value.get("historical_override", {}):
            raise ValueError(
                "pet cannot combine with historical_override.path; PET is derived "
                "from the case's historical record"
            )
    capacity = value.get("unit_id_capacity")
    if capacity is not None and (type(capacity) is not int or capacity < 3):
        raise ValueError("unit_id_capacity must be an integer of at least 3")
    if resolved:
        if value["factor_layout"] != "grid":
            raise ValueError("WF2-derived factors are always gridded")
        _period(value["simulation_window"], "simulation_window")
        if type(value["historical_path"]) is not str or not value["historical_path"]:
            raise ValueError("historical_path must be nonempty text")
        _historical_units(value)
        members = value["members"]
        if type(members) is not list or not members:
            raise ValueError("resolved members must be a nonempty list")
        for member in members:
            _fields(
                member,
                {"model", "scenario", "member", "window_id", "changes_path"},
                "member",
            )
        if capacity is None or capacity < len(members) + 2:
            raise ValueError(
                f"unit_id_capacity must be at least {len(members) + 2} "
                "(WF2 member runs across windows plus both controls)"
            )
    return value


def _historical_units(value: dict) -> None:
    """Bind the historical reader to WF4's extracted-ERA5 units (and PET)."""
    units = dict(reader_unit_interpretation(value["reader"]).variables)
    if "pet" in value and value["schema_version"] == "delta-change/3":
        if units.get("pet") != "mm/day":
            raise ValueError("delta PET requires effective units pet=mm/day")
    elif "pet" in value:
        pet = _fields(value["pet"], {"mode", "method"}, "PET settings")
        if (
            pet["mode"] != "projection_ratio"
            or not isinstance(pet["method"], str)
            or not pet["method"].strip()
        ):
            raise ValueError(
                "PET requires projection_ratio mode and an explicit method"
            )
        if units.get("pet") != "mm/day":
            raise ValueError("historical supplied PET requires effective units mm/day")
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


def validate_settings(settings: Any) -> dict:
    """Validate the producer configuration, with no implicit scientific choices."""
    version = settings.get("schema_version") if isinstance(settings, dict) else None
    if version == "delta-change/3":
        return _validate_v3(settings)
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
        | ({"factor_layout"} if version == "delta-change/2" else set())
        | ({"pet"} if version == "delta-change/2" and "pet" in settings else set()),
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
    # This profile feeds WF4's reviewed extracted-ERA5 reader without conversion.
    _historical_units(value)
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
    historical: xr.Dataset,
    offsets: np.ndarray,
    ratios: np.ndarray,
    pet_ratios: np.ndarray | None = None,
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
    if pet_ratios is not None:
        pet_ratios = np.asarray(pet_ratios)
        if (
            pet_ratios.shape != ratios.shape
            or not np.isfinite(pet_ratios).all()
            or (pet_ratios < 0).any()
        ):
            raise ValueError(
                "PET ratios must match the other factor fields and be finite and nonnegative"
            )
    for name in ("temp", "temp_min", "temp_max", "precip") + (
        ("pet",) if pet_ratios is not None else ()
    ):
        source = historical[name]
        if (
            "time" not in source.dims
            or source.dtype.kind != "f"
            or np.isinf(source.values).any()
        ):
            raise ValueError(
                f"{name}: floating time-dependent historical values required; infinity forbidden"
            )
        if name in {"precip", "pet"} and (source.values < 0).any():
            raise ValueError(f"historical {name} cannot be negative")
        values = (
            pet_ratios if name == "pet" else ratios if name == "precip" else offsets
        )[months]
        dims = ("time",) if values.ndim == 1 else ("time", "latitude", "longitude")
        factor = xr.DataArray(
            values, dims=dims, coords={key: source[key] for key in dims}
        )
        transformed = (
            source * factor if name in {"precip", "pet"} else source + factor
        ).astype(source.dtype)
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

    if policy is not None:
        policy = _fields(policy, {"min_denominator", "units"}, "precipitation policy")
    if policy is not None and (
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
    invalid = (
        mapped_daily <= 0
        if policy is None
        else mapped_daily < policy["min_denominator"]
    )
    if bool((invalid & active).any()):
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


def derive_pet_ratios(reference, future, historical, settings):
    """Require matching PET method provenance, then derive an independent ratio."""
    method = settings["pet"]["method"]
    for label, dataset in (
        ("historical", historical),
        ("reference", reference),
        ("future", future),
    ):
        if "pet" not in dataset or dataset.pet.attrs.get("pet_method") != method:
            raise ValueError(f"{label}: supplied PET must declare pet_method={method}")
        if np.isinf(dataset.pet).any() or (dataset.pet < 0).any():
            raise ValueError(f"{label}: PET cannot be negative or infinite")
    if reference.pet.attrs.get("units") not in {"mm/day", "mm/month"}:
        raise ValueError("projection PET must use mm/day or mm/month")
    # PET uses the same ratio and grid operations, with a strictly positive
    # reference. The precipitation dry-month threshold is not a PET rule.
    datasets = [
        dataset.drop_vars("precip").rename({"pet": "precip"})
        for dataset in (reference, future, historical)
    ]
    try:
        return derive_factors(*datasets, settings, None)[1]
    except ValueError as error:
        raise ValueError(f"PET ratio: {error}") from error


def scenario_records(settings: dict) -> list[dict[str, str]]:
    """Allocate controls first, then members in resolved order (WF2: per window)."""
    width = len(str(settings["unit_id_capacity"]))
    keys = [
        k
        for k in row_fields(settings)
        if k in {"model", "scenario", "member", "window_id"}
    ]
    rows = [
        dict(role=role, **dict.fromkeys(keys, "")) for role in ("historical", "neutral")
    ]
    rows.extend(
        {"role": "projection", **{key: member[key] for key in keys}}
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
        parts = [row["role"], row["model"], row["scenario"]]
        if row.get("window_id"):
            parts.append(row["window_id"])
        key = json.dumps(parts, separators=(",", ":"))
        groups.setdefault(key, []).append(row["run_id"])
        if row["role"] == "historical":
            references.append(row["run_id"])
    return {key: tuple(value) for key, value in groups.items()}, tuple(references)


def input_names(resolved: dict) -> list[str]:
    """Retained input names, historical first, in member order."""
    names = [resolved["historical_path"]]
    keys = (
        ("changes_path",)
        if resolved["schema_version"] == "delta-change/3"
        else ("reference_path", "future_path")
    )
    for member in resolved["members"]:
        names.extend(member[key] for key in keys)
    return names


def _file_policy() -> dict:
    precipitation = VARIABLES["precip"].projections
    return {
        "precip": {
            "min_denominator": precipitation.min_denominator,
            "units": precipitation.units,
        }
    }


def _prepare_files(config_path: Path, settings: dict):
    """Explicit reference/future projection files (delta-change/1 and /2)."""
    paths = [Path(settings["historical_path"])]
    for member in settings["members"]:
        paths.extend(Path(member[key]) for key in ("reference_path", "future_path"))
    paths = [(config_path.parent / path).resolve(strict=True) for path in paths]
    factors, pet_factors = [], []
    factor_policy = _file_policy()
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
            if "pet" in settings:
                pet_factors.append(
                    derive_pet_ratios(reference, future, historical, settings)
                )
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
    return paths, resolved, factors, pet_factors, factor_policy


def _drop_members_without_pet(selection: dict) -> None:
    """Exclude WF2 members without a PET ratio, reporting them per window."""
    kept = []
    for member in selection["members"]:
        with xr.open_dataset(member["source_path"]) as changes:
            has_pet = "pet_monthly_ratio" in changes
        if has_pet:
            kept.append(member)
            continue
        for entry in selection["windows"]:
            if entry["window_id"] == member["window_id"]:
                entry["members"] -= 1
                entry["unavailable"].append(
                    {
                        **{k: member[k] for k in ("model", "scenario", "member")},
                        "reason": "no WF2 PET ratio (missing PET inputs)",
                    }
                )
    empty = [e["window_id"] for e in selection["windows"] if e["members"] == 0]
    if empty:
        raise ValueError(f"no WF2 member has a PET ratio for window(s) {empty}")
    selection["members"] = kept


def _input_location(project_root: Path, path: Path) -> str:
    """Project-relative path of an input referenced in place, or of its snapshot.

    Files inside the project (the climate store, WF2 grids) are referenced in
    place by path and hash; a file outside it gets a content-addressed copy at
    publication, whose location is known from its bytes alone.
    """
    from blueearth_cst.experiment.content_identity import identity_segment

    path = Path(path).resolve(strict=True)
    if path.is_relative_to(project_root):
        return path.relative_to(project_root).as_posix()
    segment = identity_segment(file_sha256(path), "generation_input")
    return f"scenarios/_engine/generation_inputs/{segment}/{path.name}"


def _wf2_policy(changes: xr.Dataset) -> dict:
    threshold = changes["precip_monthly_ratio"].attrs.get("min_reference")
    if type(threshold) is str or threshold is None:
        raise ValueError("WF2 precipitation ratios carry no min_reference threshold")
    return {"precip": {"min_denominator": float(threshold), "units": "mm/day"}}


def _prepare_wf2(project_root: Path, settings: dict, historical: dict):
    """WF2 gridded changes for the selected windows (delta-change/3).

    ``historical`` is ``{"path": Path, "source": str}``: the case's derived
    climate-store extract, or the override file with its declared source.
    """
    from blueearth_cst.experiment.delta_wf2 import member_factors, resolve_windows

    selection = resolve_windows(project_root, settings["projections"])
    override = settings.get("historical_override", {})
    historical_path = Path(historical["path"]).resolve(strict=True)
    reader = source_reader(
        historical["source"],
        evidence=override.get("units_note") if "path" in override else None,
    )
    if "period" in override:
        period = override["period"]
        effective = [f"{period['start']:04d}-01-01", f"{period['end']:04d}-12-31"]
    else:
        effective = list(selection["reference"]["effective"])
    pet_method = None
    if "pet" in settings:
        from blueearth_cst.experiment.delta_wf2 import wf2_pet_method

        pet_method = historical.get("pet_method")
        wf2_method = wf2_pet_method(selection["variable_spec"])
        if wf2_method is None or pet_method != wf2_method:
            raise ValueError(
                f"delta PET needs WF2 PET with the case's formula {pet_method!r}; "
                f"WF2 computed {wf2_method!r}. Set analyze_projections variables.pet."
                "method to match climate forcing PET, or omit pet"
            )
        _drop_members_without_pet(selection)
        reader["metadata"]["cst_unit_interpretation"]["variables"].append(
            ["pet", "mm/day"]
        )
    paths = [historical_path] + [m["source_path"] for m in selection["members"]]
    factors, policies, capped, pet_factors = [], set(), [], []
    dry_reference = settings.get("dry_reference", {"policy": "refuse"})
    with xr.open_dataset(historical_path) as data:
        missing = sorted(
            set(SUPPORTED_SOURCES[historical["source"]]) - set(data.data_vars)
        )
        if missing:
            raise ValueError(
                f"historical record {historical_path} lacks {historical['source']} "
                f"variables {missing}; the delta method needs all of "
                f"{sorted(SUPPORTED_SOURCES[historical['source']])}"
            )
        for member in selection["members"]:
            with xr.open_dataset(member["source_path"]) as changes:
                months: list[int] = []
                label = "/".join(
                    member[k] for k in ("model", "scenario", "member", "window_id")
                )
                try:
                    factors.append(member_factors(changes, data, dry_reference, months))
                    if pet_method is not None:
                        from blueearth_cst.experiment.delta_wf2 import (
                            member_pet_ratios,
                        )

                        pet_factors.append(member_pet_ratios(changes, data))
                except ValueError as error:
                    raise ValueError(f"WF2 member {label}: {error}") from error
                if months:
                    capped.append(
                        {
                            **{
                                k: member[k]
                                for k in ("model", "scenario", "member", "window_id")
                            },
                            "months": months,
                        }
                    )
                factor_policy = _wf2_policy(changes)
                policies.add(content_sha256(factor_policy))
    if len(policies) != 1:
        raise ValueError("WF2 members disagree on the precipitation threshold")
    resolved = {key: value for key, value in settings.items() if key != "name"}
    resolved.setdefault("unit_id_capacity", len(selection["members"]) + 2)
    resolved["dry_reference"] = dry_reference
    if pet_method is not None:
        resolved["pet"] = {"mode": "projection_ratio", "method": pet_method}
    resolved.update(
        historical_path=_input_location(project_root, historical_path),
        source_like=historical["source"],
        historical_recipe=historical.get("recipe"),
        reader=reader,
        simulation_window={
            "start": int(effective[0][:4]),
            "end": int(effective[1][:4]),
        },
        historical_period={
            "effective": effective,
            "wf2_reference": selection["reference"],
            "override": "period" in override,
        },
        factor_layout="grid",
        members=[
            {
                **{key: m[key] for key in ("model", "scenario", "member", "window_id")},
                "changes_path": _input_location(project_root, m["source_path"]),
            }
            for m in selection["members"]
        ],
        windows=selection["windows"],
        wf2={
            "capped": capped,
            "water_year_start": selection["water_year_start"],
            "inventory_sha256": file_sha256(selection["inventory_path"]),
            "analysis_record_sha256": file_sha256(selection["record_path"]),
        },
    )
    validate_settings(resolved)
    # No logging helpers here: their imports would enter the generation code
    # inventory. Membership per window is retained in the intent's windows block.
    for entry in selection["windows"]:
        names = ", ".join(entry["names"]) or entry["window_id"]
        missing = entry["unavailable"]
        listed = [f"{m['model']}/{m['scenario']}/{m['member']}" for m in missing]
        shown = ", ".join(listed[:6]) + (
            f" (+{len(listed) - 6} more)" if len(listed) > 6 else ""
        )
        print(
            f"delta - window {entry['window_id']} ({names}): {entry['members']} "
            f"members"
            + (f"; {len(listed)} unavailable, excluded: {shown}" if listed else ""),
            flush=True,
        )
    if capped:
        shown = "; ".join(
            f"{c['model']}/{c['scenario']}/{c['member']} {c['window_id']} "
            f"months {', '.join(f'{m:02d}' for m in c['months'])}"
            for c in capped[:6]
        ) + (f" (+{len(capped) - 6} more)" if len(capped) > 6 else "")
        print(
            f"WARNING delta - dry reference months capped at "
            f"±{dry_reference['max_change']:.0%}: {shown}",
            flush=True,
        )
    if "period" in override:
        print(
            f"WARNING delta - historical period {effective[0]}..{effective[1]} "
            "overrides the WF2 reference "
            f"{selection['reference']['effective'][0]}..{selection['reference']['effective'][1]}",
            flush=True,
        )
    return paths, resolved, factors, pet_factors, factor_policy


def generate_collection(
    config_path: Path,
    project_root: Path,
    *,
    settings: dict | None = None,
    preview: bool = False,
    expected_input_hashes: list[str] | None = None,
    historical: dict | None = None,
    documents: tuple[tuple[str, bytes], ...] = (),
) -> Path:
    """Publish an immutable collection, or verify an identical completed request.

    delta-change/3 requires ``historical`` (``{"path", "source"}``), resolved
    from the case configuration by :mod:`delta_generation`.
    """
    config_path, project_root = config_path.resolve(strict=True), project_root.resolve()
    settings = validate_settings(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if settings is None
        else settings
    )
    if settings["schema_version"] == "delta-change/3":
        if historical is None:
            raise ValueError(
                "delta-change/3 derives its historical record from the case; "
                "generate through scripts/run_workflows.py or generate_scenarios.py"
            )
        paths, resolved, factors, pet_factors, factor_policy = _prepare_wf2(
            project_root, settings, historical
        )
    else:
        paths, resolved, factors, pet_factors, factor_policy = _prepare_files(
            config_path, settings
        )
    # Observe the physical source before publishing anything. WF4 uses the same
    # evidenced interpretation when binding the existing preparation catalog.
    descriptor = collection_forcing_descriptor(paths[0], _source_reader(resolved))
    if descriptor["timestep"] != "P1D":
        raise ValueError("delta-change/1 requires daily historical forcing")
    window = resolved["simulation_window"]
    first, last = (
        resolved["historical_period"]["effective"]
        if "historical_period" in resolved
        else (f"{window['start']:04d}-01-01", f"{window['end']:04d}-12-31")
    )
    if descriptor["start"][:10] > first or descriptor["end"][:10] < last:
        raise ValueError(
            f"historical forcing {descriptor['start'][:10]}..{descriptor['end'][:10]} "
            f"({paths[0]}) does not cover the required period {first}..{last}"
        )
    if settings["schema_version"] == "delta-change/3":
        return _publish_named(
            project_root,
            settings,
            resolved,
            paths,
            factors,
            factor_policy,
            descriptor,
            pet_factors=pet_factors,
            preview=preview,
            expected_input_hashes=expected_input_hashes,
            documents=documents,
        )
    relative = input_names(resolved)
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
                if resolved.get("factor_layout") == "grid"
                else ()
            )
        },
        "historical_descriptor": descriptor,
        "run_group_id_capacity": settings["unit_id_capacity"],
        "run_group_id_width": len(str(settings["unit_id_capacity"])),
        "factors": [
            {
                "temp_offset_degC": offsets.tolist(),
                "precip_ratio": ratios.tolist(),
                **({"pet_ratio": pet_factors[i].tolist()} if pet_factors else {}),
            }
            for i, (offsets, ratios) in enumerate(factors)
        ],
    }
    # Runner admission captures the authored inputs; WF2 member grids may be
    # produced later in the same invocation and are checked by resolve_members.
    admitted = (
        intent["input_hashes"][:1]
        if settings["schema_version"] == "delta-change/3"
        else intent["input_hashes"]
    )
    if expected_input_hashes is not None and admitted != expected_input_hashes:
        raise ValueError("delta generation inputs changed after admission")
    identity = content_sha256(intent)
    engine = project_root / "scenarios/_engine/collections" / identity[:12]
    data = project_root / "scenarios/_engine/data" / identity[:12]
    marker_path = engine / "collection.json"
    if marker_path.exists():
        read_delta_collection(marker_path)
        return marker_path
    if engine.exists() or data.exists():
        raise ValueError(
            f"delta output is incomplete or claimed by another writer: {engine}; "
            "inspect the retained directories before retrying; no automatic recovery"
        )
    if preview:
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
    rows = scenario_records(resolved)
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
                historical
                if index == 0
                else apply_factors(
                    historical,
                    offsets,
                    ratios,
                    pet_factors[index - 2]
                    if index >= 2 and pet_factors
                    else np.ones(12)
                    if pet_factors
                    else None,
                )
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


#: Named-set delta collections (``scenarios/<name>/``), produced from WF2.
NAMED_SCHEMA = "delta-scenario-collection/2"
NAMED_INTENT = "delta-scenario-intent/3"
FACTOR_FIELDS = ["run_id", "factor_file", "model", "scenario", "member", "window_id"]
_NAMED_MARKER_KEYS = {
    "schema_version",
    "scenario_type",
    "status",
    "collection_id",
    "collection_revision",
    "data_root",
    "intent",
    "scenario_run_lookup",
    "factor_lookup",
    "factors",
    "series",
}
_NAMED_INTENT_KEYS = {
    "schema_version",
    "scenario_type",
    "factor_policy",
    "settings",
    "scenario_spec",
    "inputs",
    "provider_revision",
    "environment",
    "historical_descriptor",
    "run_group_id_capacity",
    "run_group_id_width",
    "factors",
}


def default_delta_name(resolved: dict) -> str:
    """Readable default set name: ensemble and window names (design §5.3)."""
    ensemble = resolved["projections"]["ensemble"]
    import re

    labels = [
        re.sub(r"[^A-Za-z0-9._-]+", "-", w["names"][0]).strip("-._")
        if w["names"]
        else ""
        for w in resolved["windows"]
    ]
    # Horizon names are user text; fall back to the window id when unusable.
    labels = [
        label or w["window_id"]
        for label, w in zip(labels, resolved["windows"], strict=True)
    ]
    name = f"delta-{ensemble}-" + "-".join(labels)
    if len(name) > 64:
        name = f"delta-{ensemble}-{len(labels)}-windows"
    return name[:64].rstrip(".")


def _source_reader(resolved: dict) -> dict:
    """The reader of the historical file itself (delta adds PET to the series)."""
    if "pet" not in resolved or resolved["schema_version"] != "delta-change/3":
        return resolved["reader"]
    import copy

    reader = copy.deepcopy(resolved["reader"])
    binding = reader["metadata"]["cst_unit_interpretation"]
    binding["variables"] = [v for v in binding["variables"] if v[0] != "pet"]
    return reader


def _factor_digest(
    offsets: np.ndarray, ratios: np.ndarray, pet: np.ndarray | None = None
) -> str:
    return content_sha256(
        {"temp_offset_degC": offsets.tolist(), "precip_ratio": ratios.tolist()}
        | ({"pet_ratio": pet.tolist()} if pet is not None else {})
    )


def _factor_dataset(
    offsets: np.ndarray,
    ratios: np.ndarray,
    historical: xr.Dataset,
    pet: np.ndarray | None = None,
) -> xr.Dataset:
    """Monthly factors on the historical grid, with its CRS when it has one."""
    dims = ("month", "latitude", "longitude")
    coords = {
        "month": np.arange(1, 13),
        "latitude": historical["latitude"].values,
        "longitude": historical["longitude"].values,
    }
    dataset = xr.Dataset(
        {
            "temp_offset": (dims, offsets, {"units": "degC"}),
            "precip_ratio": (dims, ratios, {"units": "1"}),
        }
        | ({"pet_ratio": (dims, pet, {"units": "1"})} if pet is not None else {}),
        coords=coords,
    )
    if "spatial_ref" in historical.variables:
        dataset = dataset.assign_coords(spatial_ref=historical["spatial_ref"])
    return dataset


def _publish_named(
    project_root: Path,
    settings: dict,
    resolved: dict,
    paths: list[Path],
    factors: list,
    factor_policy: dict,
    descriptor: dict,
    *,
    preview: bool,
    expected_input_hashes: list[str] | None,
    documents: tuple[tuple[str, bytes], ...] = (),
    pet_factors: list | None = None,
) -> Path:
    """Publish a WF2-sourced collection as ``scenarios/<name>/``.

    Large inputs (climate store, WF2 grids) are referenced in place by path and
    hash; the reader never re-reads them, so a rebuilt store leaves the set
    readable. Factors are retained per run as NetCDF; the intent keeps their
    content digests, so identity follows the factor values.
    """
    from blueearth_cst.experiment.collection_index import (
        allocate_collection_name,
        find_ready_marker,
    )

    inputs = [
        {
            "role": role,
            "path": location,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for role, location, path in zip(
            ["historical"] + ["changes"] * len(resolved["members"]),
            [resolved["historical_path"]]
            + [m["changes_path"] for m in resolved["members"]],
            paths,
            strict=True,
        )
    ]
    if (
        expected_input_hashes is not None
        and [inputs[0]["sha256"]] != expected_input_hashes
    ):
        raise ValueError("delta generation inputs changed after admission")
    intent = {
        "schema_version": NAMED_INTENT,
        "scenario_type": "delta_change",
        "factor_policy": factor_policy,
        "settings": resolved,
        "scenario_spec": {
            "simulation_window": dict(resolved["simulation_window"]),
            "historical_period": resolved["historical_period"]["effective"],
            "windows": [
                {"window_id": w["window_id"], "members": w["members"]}
                for w in resolved["windows"]
            ],
        },
        "inputs": inputs,
        "provider_revision": file_sha256(Path(__file__)),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xarray", "netCDF4", "hydromt", "rasterio", "pyproj")
        },
        "historical_descriptor": descriptor,
        "run_group_id_capacity": resolved["unit_id_capacity"],
        "run_group_id_width": len(str(resolved["unit_id_capacity"])),
        "factors": [
            _factor_digest(o, r, pet_factors[i] if pet_factors else None)
            for i, (o, r) in enumerate(factors)
        ],
    }
    identity = content_sha256(intent)
    existing = find_ready_marker(project_root, identity)
    if existing is not None:
        read_delta_collection(existing)
        return existing
    name = allocate_collection_name(
        project_root, settings.get("name") or default_delta_name(resolved)
    )
    folder = project_root / "scenarios" / name
    engine = folder / "_engine"
    marker_path = engine / "collection.json"
    if preview:
        return marker_path
    # Outside-project inputs are snapshotted by the runner (delta_generation)
    # before publication; importing the snapshot helper here would pull
    # generation planning into every WF4 indicator code inventory.
    for entry, source in zip(inputs, paths, strict=True):
        target = project_root / entry["path"]
        if not source.resolve().is_relative_to(project_root) and (
            not target.is_file() or file_sha256(target) != entry["sha256"]
        ):
            raise ValueError(
                f"delta input outside the project has no snapshot at {target}; "
                "generate through scripts/run_workflows.py or generate_scenarios.py"
            )
    engine.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), engine / "producer.py")
    # The authored settings that produced the set, for the reader (design §5.4);
    # display only, never identity.
    if documents:
        (folder / "config").mkdir()
        for name, data in documents:
            (folder / "config" / Path(name).name).write_bytes(data)
    atomic_record(engine / "collection_intent.json", intent)
    rows = scenario_records(resolved)
    with (folder / "scenario_run_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=row_fields(resolved))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "series").mkdir()
    (folder / "delta" / "factors").mkdir(parents=True)
    factor_rows, factor_refs, series = [], [], []
    historical_path = project_root / resolved["historical_path"]
    first, last = resolved["historical_period"]["effective"]
    with xr.open_dataset(historical_path) as source:
        historical = source.sel(time=slice(first, last)).load()
        if "pet" in resolved:
            from blueearth_cst.experiment.delta_wf2 import historical_pet

            historical["pet"] = historical_pet(historical, resolved["pet"]["method"])
        neutral_pet = np.ones(12) if "pet" in resolved else None
        for index, row in enumerate(rows):
            if index >= 2:
                offsets, ratios = factors[index - 2]
                pet = pet_factors[index - 2] if pet_factors else None
                factor_path = folder / "delta" / "factors" / f"{row['run_id']}.nc"
                _factor_dataset(offsets, ratios, historical, pet).to_netcdf(factor_path)
                factor_rows.append(
                    {
                        "run_id": row["run_id"],
                        "factor_file": factor_path.relative_to(folder).as_posix(),
                        **{
                            k: row[k]
                            for k in ("model", "scenario", "member", "window_id")
                        },
                    }
                )
                factor_refs.append(
                    file_reference(factor_path, "project_root", project_root)
                )
                transformed = apply_factors(historical, offsets, ratios, pet)
            elif index == 1:
                transformed = apply_factors(
                    historical, np.zeros(12), np.ones(12), neutral_pet
                )
            else:
                transformed = historical
            path = folder / "series" / f"{row['run_id']}.nc"
            transformed.to_netcdf(path)
            series.append(
                {
                    "run_id": row["run_id"],
                    "file": file_reference(path, "project_root", project_root),
                    "descriptor": collection_forcing_descriptor(
                        path, resolved["reader"]
                    ),
                }
            )
    with (folder / "factor_lookup.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FACTOR_FIELDS)
        writer.writeheader()
        writer.writerows(factor_rows)
    if file_sha256(historical_path) != inputs[0]["sha256"]:
        raise ValueError(
            "historical input changed during generation; incomplete set retained "
            f"at {folder}"
        )
    marker = {
        "schema_version": NAMED_SCHEMA,
        "scenario_type": "delta_change",
        "status": "ready",
        "collection_id": identity,
        "data_root": folder.relative_to(project_root).as_posix(),
        "intent": file_reference(
            engine / "collection_intent.json", "project_root", project_root
        ),
        "scenario_run_lookup": file_reference(
            folder / "scenario_run_lookup.csv", "project_root", project_root
        ),
        "factor_lookup": file_reference(
            folder / "factor_lookup.csv", "project_root", project_root
        ),
        "factors": factor_refs,
        "series": series,
    }
    marker["collection_revision"] = content_sha256(marker)
    atomic_record(marker_path, marker)
    read_delta_collection(marker_path)
    return marker_path


def _read_named_delta_collection(marker_path: Path) -> dict:
    """Validate a named-set delta collection without re-reading its inputs."""
    engine = marker_path.parent
    folder = engine.parent
    if engine.name != "_engine" or folder.parent.name != "scenarios":
        raise ValueError("delta collection must be discovered through its set marker")
    root = folder.parent.parent
    anchors = {"project_root": root}
    marker = _fields(read_canonical_json(marker_path), _NAMED_MARKER_KEYS, "marker")
    if (
        marker["schema_version"] != NAMED_SCHEMA
        or marker["scenario_type"] != "delta_change"
        or marker["status"] != "ready"
        or marker["data_root"] != f"scenarios/{folder.name}"
    ):
        raise ValueError("unsupported, unready or relocated delta collection")
    if marker["collection_revision"] != content_sha256(
        {key: value for key, value in marker.items() if key != "collection_revision"}
    ):
        raise ValueError("delta collection revision differs")
    intent_path = resolve_file_reference(marker["intent"], anchors)
    if intent_path != engine / "collection_intent.json":
        raise ValueError("delta intent must be the sibling frozen record")
    intent = _fields(read_canonical_json(intent_path), _NAMED_INTENT_KEYS, "intent")
    if (
        intent["schema_version"] != NAMED_INTENT
        or intent["scenario_type"] != marker["scenario_type"]
        or content_sha256(intent) != marker["collection_id"]
    ):
        raise ValueError("delta identity differs")
    settings = validate_settings(intent["settings"])
    if intent["run_group_id_capacity"] != settings["unit_id_capacity"] or intent[
        "run_group_id_width"
    ] != len(str(settings["unit_id_capacity"])):
        raise ValueError("delta namespace differs")
    if file_sha256(engine / "producer.py") != intent["provider_revision"]:
        raise ValueError("retained producer bytes differ")
    expected_inputs = [settings["historical_path"]] + [
        m["changes_path"] for m in settings["members"]
    ]
    if [item.get("path") for item in intent["inputs"]] != expected_inputs:
        raise ValueError("delta input references differ from the settings")
    rows = scenario_records(settings)
    lookup = resolve_file_reference(marker["scenario_run_lookup"], anchors)
    if lookup != folder / "scenario_run_lookup.csv":
        raise ValueError("delta lookup path differs")
    with lookup.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != row_fields(settings) or list(reader) != rows:
            raise ValueError("delta lookup differs from declared members")
    projections = rows[2:]
    if not (
        len(projections)
        == len(intent["factors"])
        == len(marker["factors"])
        == len(settings["members"])
    ):
        raise ValueError("delta factor count differs")
    factor_lookup = resolve_file_reference(marker["factor_lookup"], anchors)
    if factor_lookup != folder / "factor_lookup.csv":
        raise ValueError("delta factor lookup path differs")
    with factor_lookup.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            {
                "run_id": row["run_id"],
                "factor_file": f"delta/factors/{row['run_id']}.nc",
                **{k: row[k] for k in ("model", "scenario", "member", "window_id")},
            }
            for row in projections
        ]
        if reader.fieldnames != FACTOR_FIELDS or list(reader) != expected:
            raise ValueError("delta factor lookup differs from declared members")
    for item, reference, digest in zip(
        expected, marker["factors"], intent["factors"], strict=True
    ):
        path = resolve_file_reference(reference, anchors)
        if path != folder / item["factor_file"]:
            raise ValueError("delta factor file path differs")
        with xr.open_dataset(path) as factors:
            if (
                _factor_digest(
                    factors["temp_offset"].values,
                    factors["precip_ratio"].values,
                    factors["pet_ratio"].values if "pet_ratio" in factors else None,
                )
                != digest
            ):
                raise ValueError("retained factor values differ from the intent")
    if type(marker["series"]) is not list or len(marker["series"]) != len(rows):
        raise ValueError("delta series count differs")
    for row, entry in zip(rows, marker["series"], strict=True):
        _fields(entry, {"run_id", "file", "descriptor"}, "delta series")
        path = resolve_file_reference(entry["file"], anchors)
        if (
            entry["run_id"] != row["run_id"]
            or path != folder / "series" / f"{row['run_id']}.nc"
        ):
            raise ValueError("delta series identity/path differs")
        if (
            collection_forcing_descriptor(path, settings["reader"])
            != entry["descriptor"]
        ):
            raise ValueError("delta forcing descriptor differs")
    return marker


def collection_input_root(project_root: Path, marker: dict) -> Path:
    """Base for intent input paths: the project for named sets, else the data root."""
    if marker["schema_version"] == NAMED_SCHEMA:
        return Path(project_root)
    return Path(project_root) / marker["data_root"]


def read_delta_collection(marker_path: Path) -> dict:
    """Validate the complete delta collection, including retained physical inputs."""
    marker_path = Path(marker_path)
    if marker_path.is_symlink():
        raise ValueError("collection marker cannot be a symbolic link")
    marker_path = marker_path.resolve(strict=True)
    if marker_path.parent.name == "_engine":
        return _read_named_delta_collection(marker_path)
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
    wf2 = settings["schema_version"] == "delta-change/3"
    expected_names = ["inputs/historical.nc"]
    for i in range(1, len(settings["members"]) + 1):
        expected_names.extend(
            [f"inputs/changes_{i}.nc"]
            if wf2
            else [f"inputs/reference_{i}.nc", f"inputs/future_{i}.nc"]
        )
    names = input_names(settings)
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
        _fields(
            factors,
            {"temp_offset_degC", "precip_ratio"}
            | ({"pet_ratio"} if "pet" in settings else set()),
            "monthly factors",
        )
        if wf2:
            from blueearth_cst.experiment.delta_wf2 import member_factors

            with (
                xr.open_dataset(data / member["changes_path"]) as changes,
                xr.open_dataset(data / names[0]) as historical,
            ):
                offsets, ratios = member_factors(changes, historical)
                if content_sha256(_wf2_policy(changes)) != content_sha256(
                    intent["factor_policy"]
                ):
                    raise ValueError("retained WF2 threshold differs from the intent")
            if content_sha256(factors) != content_sha256(
                {"temp_offset_degC": offsets.tolist(), "precip_ratio": ratios.tolist()}
            ):
                raise ValueError("retained factors differ from WF2 member grids")
            continue
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
            pet_ratio = (
                derive_pet_ratios(reference, future, historical, settings)
                if "pet" in settings
                else None
            )
        if content_sha256(factors) != content_sha256(
            {
                "temp_offset_degC": offsets.tolist(),
                "precip_ratio": ratios.tolist(),
                **({"pet_ratio": pet_ratio.tolist()} if pet_ratio is not None else {}),
            }
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
