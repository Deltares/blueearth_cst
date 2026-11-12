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
    if type(value) is not dict or set(value) != {"ensemble", "window_id"}:
        raise ValueError("projections: expected exactly ['ensemble', 'window_id']")
    for key in ("ensemble", "window_id"):
        item = value[key]
        if type(item) is not str or not item.strip() or any(c in item for c in "/\\"):
            raise ValueError(f"projections.{key} must be nonempty text without slashes")
    return value


def ensemble_dir(project_root: Path, projections: dict) -> Path:
    """WF2's product directory for the selected ensemble."""
    return Path(project_root) / "data/climate/projections" / projections["ensemble"]


def resolve_members(project_root: Path, projections: dict) -> dict:
    """Select one WF2 window's members, refusing stale or incomplete products.

    Return the member files, the window's periods, the WF2 record path and the
    members WF2 reported unavailable for this window (never silently dropped:
    they are retained in the collection intent and reported by the caller).
    """
    root = ensemble_dir(project_root, projections)
    grids = root / "analysis/gridded_changes"
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
    window = projections["window_id"]
    selected = [m for m in inventory["members"] if m["window_id"] == window]
    unavailable = [m for m in inventory["unavailable"] if m["window_id"] == window]
    if not selected:
        windows = sorted({m["window_id"] for m in inventory["members"]})
        raise ValueError(
            f"WF2 has no members for window {window!r}; available: {windows}"
        )
    members, periods = [], set()
    for member in selected:
        path = (grids / member["path"]).resolve(strict=True)
        if not path.is_relative_to(grids.resolve()):
            raise ValueError("WF2 member path escapes its gridded_changes directory")
        if file_sha256(path) != member["sha256"]:
            raise ValueError(f"WF2 member grid differs from its inventory: {path}")
        with xr.open_dataset(path) as ds:
            periods.add((ds.attrs["reference_period"], ds.attrs["future_period"]))
        members.append(
            {key: member[key] for key in ("model", "scenario", "member")}
            | {"source_path": path}
        )
    if len(periods) != 1:
        raise ValueError(f"WF2 window {window!r} members disagree on periods")
    (reference, future), *_ = periods
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return {
        "members": members,
        "reference_period": _years(json.loads(reference)),
        "future_period": _years(json.loads(future)),
        "water_year_start": record.get("water_year_start", "Jan"),
        "unavailable": [
            {key: item[key] for key in ("model", "scenario", "member", "reason")}
            for item in unavailable
        ],
        "inventory_path": inventory_path,
        "record_path": record_path,
        "variable_spec": record.get("variable_spec", {}),
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


def member_factors(
    changes: xr.Dataset, historical: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """WF2 monthly offsets and precipitation ratios on the historical grid.

    A missing ratio in the active domain means WF2 withheld it (reference below
    ``min_denominator``) or the source cell is incomplete; both are refused here
    with the affected months named.
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
    ratio = changes["precip_monthly_ratio"]
    reference = changes["precip_monthly_reference"]
    dry = ratio.isnull() & reference.notnull()
    if bool(dry.any()):
        mapped_dry = map_monthly(
            dry.astype(float),
            crs,
            target,
            active,
            neutral=None,
            label="WF2 dry-reference flags",
        )
        months = mapped_dry.month.values[
            (mapped_dry.where(active, 0) > 0).any(("latitude", "longitude")).values
        ]
        if len(months):
            threshold = ratio.attrs.get("min_reference", "the WF2 threshold")
            listed = ", ".join(f"{int(m):02d}" for m in months)
            raise ValueError(
                f"reference precipitation below {threshold} mm/day in active grid "
                f"cells in months {listed}; relative factors are refused, not clipped"
            )
        ratio = ratio.fillna(1.0).where(reference.notnull())
    ratios = map_monthly(
        ratio, crs, target, active, neutral=1.0, label="WF2 precipitation ratio"
    )
    return offsets.values.astype(float), ratios.values.astype(float)
