"""WF2 native-grid changes as the projection source of a delta collection.

The GCM delta method is the projection-conditioned ("top-down") alternative
to the stochastic stress test, so it may consume WF2 products; stochastic WF3
never does. WF2's monthly kernels own the change arithmetic; this module only
selects one analysis window's members, checks their identity and maps their
factors to the historical forcing grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from blueearth_cst.shared.provenance import file_sha256

#: Minimum gridded_changes inventory schema: 2 adds monthly reference means.
INVENTORY_SCHEMA = 2


def projection_settings(value: Any) -> dict:
    """Validate the closed authored ``projections`` mapping."""
    if (
        type(value) is not dict
        or "ensemble" not in value
        or not set(value)
        <= {
            "ensemble",
            "future_windows",
        }
    ):
        raise ValueError(
            "projections: expected 'ensemble' and optionally 'future_windows'"
        )
    ensemble = value["ensemble"]
    if (
        type(ensemble) is not str
        or not ensemble.strip()
        or any(c in ensemble for c in "/\\")
    ):
        raise ValueError("projections.ensemble must be nonempty text without slashes")
    if "future_windows" in value:
        names = value["future_windows"]
        if (
            type(names) is not list
            or not names
            or any(type(n) is not str or not n.strip() for n in names)
        ):
            raise ValueError("projections.future_windows must be a nonempty name list")
        if len(set(names)) != len(names):
            raise ValueError(f"projections.future_windows repeats a name: {names}")
    return value


def ensemble_dir(project_root: Path, projections: dict) -> Path:
    """WF2's product directory for the selected ensemble."""
    return Path(project_root) / "data/climate/projections" / projections["ensemble"]


def _select_windows(record: dict, projections: dict) -> list[dict]:
    """WF2 windows named by ``future_windows`` (all by default), in record order."""
    windows = record.get("windows")
    if type(windows) is not list or not windows:
        raise ValueError("WF2 analysis record lists no windows")
    if "future_windows" not in projections:
        return windows
    by_name = {name: w for w in windows for name in w.get("horizons", [])}
    requested = projections["future_windows"]
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(
            f"unknown WF2 future window(s) {unknown}; available: {sorted(by_name)}"
        )
    ids = [by_name[name]["window_id"] for name in requested]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"future_windows {requested} select the same WF2 window more than once"
        )
    return [w for w in windows if w["window_id"] in set(ids)]


def _period_entry(window: dict, part: str, labels: list) -> dict:
    configured = next(iter(window["configured"].values()))[part]
    effective = window["effective"][part]
    return {
        "configured": [int(v) for v in configured],
        "effective": [effective["start"], effective["end"]],
        "labels": [int(v) for v in labels],
    }


def resolve_windows(project_root: Path, projections: dict) -> dict:
    """Select WF2 windows and their members, refusing stale or inconsistent products.

    Returns structured window entries (design §2.1), the members per window with
    their grid files, WF2's single reference period and the record/inventory
    paths. Unavailable members are retained per window, never dropped silently.
    """
    root = ensemble_dir(project_root, projections)
    grids = root / "analysis/gridded_changes"
    # `spatial_diagnostics.inventory_path`, kept local to stay out of WF2 code.
    inventory_path = root / "_engine/gridded_changes_inventory.json"
    if not inventory_path.is_file() and (grids / "inventory.json").is_file():
        inventory_path = grids / "inventory.json"
    record_path = root / "_engine/analysis_record.json"
    for path in (inventory_path, record_path):
        if not path.is_file():
            raise ValueError(
                f"WF2 product missing: {path}. Enable analyze_projections or run it "
                "first; the delta method reads its published gridded changes."
            )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema_version", 1) < INVENTORY_SCHEMA:
        raise ValueError(
            f"WF2 gridded changes at {grids} predate schema {INVENTORY_SCHEMA}; "
            "re-run analyze_projections to publish monthly reference means"
        )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    entries, members = [], []
    for window in _select_windows(record, projections):
        window_id = window["window_id"]
        entry = {
            "window_id": window_id,
            "names": list(window.get("horizons", [])),
            "reference": _period_entry(window, "reference", window["reference"]),
            "future": _period_entry(window, "future", window["used"]),
        }
        selected = [m for m in inventory["members"] if m["window_id"] == window_id]
        if not selected:
            raise ValueError(f"WF2 has no available members for window {window_id!r}")
        for member in selected:
            path = (grids / member["path"]).resolve(strict=True)
            if not path.is_relative_to(grids.resolve()):
                raise ValueError(
                    "WF2 member path escapes its gridded_changes directory"
                )
            if file_sha256(path) != member["sha256"]:
                raise ValueError(f"WF2 member grid differs from its inventory: {path}")
            with xr.open_dataset(path) as ds:
                periods = (
                    json.loads(ds.attrs["reference_period"]),
                    json.loads(ds.attrs["future_period"]),
                )
            if [int(v) for v in periods[0]] != entry["reference"]["labels"] or [
                int(v) for v in periods[1]
            ] != entry["future"]["labels"]:
                raise ValueError(
                    f"WF2 member grid {path.name} periods {periods} disagree with "
                    f"the analysis record for window {window_id}"
                )
            members.append(
                {key: member[key] for key in ("model", "scenario", "member")}
                | {"window_id": window_id, "source_path": path}
            )
        entry["members"] = len(selected)
        entry["unavailable"] = [
            {key: item[key] for key in ("model", "scenario", "member", "reason")}
            for item in inventory["unavailable"]
            if item["window_id"] == window_id
        ]
        entries.append(entry)
    references = {json.dumps(entry["reference"]) for entry in entries}
    if len(references) != 1:
        raise ValueError("selected WF2 windows disagree on the reference period")
    return {
        "windows": entries,
        "members": members,
        "reference": entries[0]["reference"],
        "water_year_start": record.get("water_year_start", "Jan"),
        "variable_spec": record.get("variable_spec", {}),
        "inventory_path": inventory_path,
        "record_path": record_path,
    }


def _years(bounds: list) -> dict:
    start, end = (int(value) for value in bounds)
    return {"start": start, "end": end}


def historical_target(historical: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """The historical forcing grid and its active (ever-valid) cells."""
    import hydromt  # noqa: F401 -- registers the raster accessor

    target = historical.temp.isel(time=0, drop=True).transpose("latitude", "longitude")
    if historical.raster.crs is None:
        raise ValueError("historical grid requires an explicit CRS")
    target.raster.set_crs(historical.raster.crs)
    active = historical.temp.notnull().any("time").transpose("latitude", "longitude")
    active = active | historical.precip.notnull().any("time").transpose(
        "latitude", "longitude"
    )
    return target, active


def map_monthly(
    field: xr.DataArray,
    crs,
    target: xr.DataArray,
    active: xr.DataArray,
    *,
    neutral: float | None,
    label: str,
) -> np.ndarray:
    """Nearest-neighbour map a (month, y, x) field onto the historical grid.

    Missing values in the active domain are errors; outside it the field is
    ``neutral`` (or left as mapped when ``neutral`` is None).
    """
    field = field.transpose("month", field.raster.y_dim, field.raster.x_dim)
    field.raster.set_crs(crs)
    mapped = field.raster.reproject_like(target, method="nearest").raster.mask_nodata()
    if bool((~np.isfinite(mapped) & active).any()):
        raise ValueError(f"{label} does not cover every active historical grid cell")
    if neutral is not None:
        mapped = mapped.where(active, neutral)
    return mapped.transpose("month", "latitude", "longitude")


def _active_months(flags, crs, target, active, label) -> list[int]:
    """Months in which any active historical cell carries a source flag."""
    mapped = map_monthly(
        flags.astype(float), crs, target, active, neutral=None, label=label
    )
    hit = (mapped.where(active, 0) > 0).any(("latitude", "longitude")).values
    return [int(m) for m in mapped.month.values[hit]]


def _listed(months: list[int]) -> str:
    return ", ".join(f"{m:02d}" for m in months)


def member_factors(
    changes: xr.Dataset,
    historical: xr.Dataset,
    dry_reference: dict | None = None,
    capped: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """WF2 monthly offsets and precipitation ratios on the historical grid.

    WF2 withholds the ratio where the monthly reference mean is below its
    ``min_denominator`` (a dry cell/month). ``dry_reference`` (design §1.2)
    decides: ``refuse`` (default) stops with the months named; ``cap`` uses
    future / reference limited to [1 - max_change, 1 + max_change], still
    refusing a zero or missing reference or a missing future mean. The capped
    months are appended to ``capped``.
    """
    import hydromt  # noqa: F401

    if changes.raster.crs is None:
        raise ValueError("WF2 member grid requires an explicit CRS")
    if changes["temp_monthly"].attrs.get("units") not in {"degC", "K"}:
        raise ValueError("WF2 temperature change requires degC or K units")
    target, active = historical_target(historical)
    crs = changes.raster.crs
    offsets = map_monthly(
        changes["temp_monthly"],
        crs,
        target,
        active,
        neutral=0.0,
        label="WF2 temperature change",
    )
    policy = dry_reference or {"policy": "refuse"}
    ratio = changes["precip_monthly_ratio"]
    reference = changes["precip_monthly_reference"]
    dry = ratio.isnull() & reference.notnull()
    if bool(dry.any()):
        months = _active_months(dry, crs, target, active, "WF2 dry-reference flags")
        threshold = ratio.attrs.get("min_reference", "the WF2 threshold")
        if months and policy["policy"] == "refuse":
            raise ValueError(
                f"reference precipitation below {threshold} mm/day in active grid "
                f"cells in months {_listed(months)}; relative factors are refused "
                "(set dry_reference: {policy: cap, max_change: ...} to cap them)"
            )
        if months:
            future = changes["precip_monthly_future"]
            for flags, problem in (
                (dry & (reference <= 0), "is zero, so no ratio exists"),
                (dry & future.isnull(), "has no future mean"),
            ):
                bad = _active_months(flags, crs, target, active, "WF2 dry flags")
                if bad:
                    raise ValueError(
                        f"reference precipitation {problem} in active grid cells in "
                        f"months {_listed(bad)}; capping cannot apply"
                    )
            change = policy["max_change"]
            bounded = (future / reference).clip(1 - change, 1 + change)
            ratio = ratio.where(~dry, bounded)
            if capped is not None:
                capped.extend(months)
        ratio = ratio.fillna(1.0).where(reference.notnull())
    ratios = map_monthly(
        ratio, crs, target, active, neutral=1.0, label="WF2 precipitation ratio"
    )
    return offsets.values.astype(float), ratios.values.astype(float)


def wf2_pet_method(variable_spec: dict) -> str | None:
    """WF2's PET formula from its recorded variable spec (None without PET)."""
    # Positional VariableSpec record (method is the seventh field); importing
    # the WF2 module would add it to the generation code inventory.
    value = variable_spec.get("pet")
    return value[6] if value and len(value) > 6 else None


def member_pet_ratios(changes: xr.Dataset, historical: xr.Dataset) -> np.ndarray:
    """WF2 monthly PET ratios on the historical grid (neutral outside it)."""
    import hydromt  # noqa: F401

    target, active = historical_target(historical)
    ratios = map_monthly(
        changes["pet_monthly_ratio"],
        changes.raster.crs,
        target,
        active,
        neutral=1.0,
        label="WF2 PET ratio",
    )
    return ratios.values.astype(float)


def historical_pet(historical: xr.Dataset, method: str) -> xr.DataArray:
    """Daily source-grid PET with the case's formula (HydroMT meteo).

    Uses mean-sea-level pressure without elevation correction: no orography
    enters the delta set, unlike WF4's model-grid PET. All runs of the set,
    controls included, share this PET, so their differences carry the WF2
    PET signal only.
    """
    from hydromt.model.processes import meteo

    press = historical["press_msl"]
    if method == "makkink":
        pet = meteo.pet_makkink(
            historical["temp"], press, historical["kin"], timestep=86400
        )
    elif method == "debruin":
        pet = meteo.pet_debruin(
            historical["temp"],
            press,
            historical["kin"],
            historical["kout"],
            timestep=86400,
        )
    else:
        raise ValueError(f"delta PET supports debruin or makkink, not {method!r}")
    pet = pet.where(historical["temp"].notnull()).astype(historical["temp"].dtype)
    pet.attrs.update(
        units="mm/day",
        pet_method=method,
        cst_pet_corrections="mean-sea-level pressure; no elevation correction",
    )
    return pet.rename("pet")
