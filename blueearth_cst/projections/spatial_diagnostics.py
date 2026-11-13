"""Prepare native-grid GCM changes independently of scenario generation.

Raw slices are identity checked before use. Existing WF2 annual/monthly kernels
own aggregation and change arithmetic. No interpolation or pooling across
different climate models occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.projections import series_identity
from blueearth_cst.projections.change_factor_table import COMPANION_SEP
from blueearth_cst.projections.get_change_climate_proj import (
    AnalysisWindowUnavailable,
    NoCompleteHydrologicalYear,
    analysis_slice,
    get_change_annual_clim_proj,
    get_change_monthly_clim_proj,
)
from blueearth_cst.projections.series_checks import assert_calendar_agreement
from blueearth_cst.projections.variable_spec import VariableSpec, source_names
from blueearth_cst.shared.provenance import file_sha256


def _fields(raw: xr.Dataset, spec: dict[str, VariableSpec], source: str) -> xr.Dataset:
    """Canonical fields on the source grid, using the existing derivation recipes."""
    from blueearth_cst.projections.derived_pet import calculate_pet
    from blueearth_cst.projections.derived_states import calculate_state

    fields = {}
    for name, value in spec.items():
        if value.method and name in {"tdew", "wind"}:
            field = calculate_state(raw, name, value.method, source=source)
        elif value.method:
            field = calculate_pet(raw, value.method, source=source)
        else:
            field = raw[value.source].rename(name)
        fields[name] = field
    return xr.Dataset(fields, attrs=raw.attrs)


#: Relative to the ensemble folder.
INVENTORY = "_engine/gridded_changes_inventory.json"


def inventory_path(grids) -> Path:
    """The gridded-changes record of ``<ensemble>/analysis/gridded_changes``.

    It lives in the ensemble's ``_engine/`` (layout 3, R10); earlier layouts kept
    ``inventory.json`` inside the grid folder, which readers still accept.
    """
    grids = Path(grids)
    current = grids.parents[1] / INVENTORY
    legacy = grids / "inventory.json"
    return legacy if not current.is_file() and legacy.is_file() else current


def prepare_spatial_changes(
    points: list[dict],
    windows: list[dict],
    output: str | Path,
    *,
    variable_spec: dict,
    water_year_start: str = "Jan",
    min_reference: dict | None = None,
    region_fp: str = "",
) -> dict:
    """Publish member/window grids and an explicit inventory, including omissions.

    Each source cell must have every monthly value in both windows. This strict
    spatial diagnostic mask prevents incomplete cells acquiring false annual
    totals through a skip-NaN sum. Basin-average diagnostics retain WF2's own
    availability rules. Unavailable windows are recorded, never substituted.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        k: v if isinstance(v, VariableSpec) else VariableSpec(*v)
        for k, v in variable_spec.items()
    }
    inventory = {
        # 2: relative variables add <name>_monthly_reference/_future means.
        "schema_version": 2,
        "members": [],
        "unavailable": [],
        "methods": {
            "aggregation": "existing WF2 annual/monthly kernels",
            "spatial_support": "native source grid; no regridding",
            "cell_coverage": "every monthly value in both complete windows",
            "member_summary": "cell-wise median within model/scenario",
            "water_year_start": water_year_start,
        },
    }
    for point in points:
        loaded = []
        for role, scenario in (("hist", "historical"), ("fut", point["scenario"])):
            path = Path(point[f"raw_path_{role}"])
            components = point.get(f"digest_components_{role}")
            digest = (
                series_identity.raw_digest(components, region_fp)
                if components
                else point[f"raw_digest_{role}"]
            )
            series_identity.assert_raw_identity(path, digest, str(path))
            with xr.open_dataset(path) as raw:
                series_identity.assert_raw_coverage(
                    raw,
                    series_identity.acquisition_window(scenario),
                    source_names(spec),
                    str(path),
                )
                fields = _fields(raw.load(), spec, str(path))
            fields = fields.expand_dims(scenario=[scenario])
            loaded.append(fields)
        historical, future = loaded
        calendar = assert_calendar_agreement(historical, future, point["model"])
        # Comparisons must be on the identical native grid within each member.
        xr.align(historical, future, join="exact", exclude={"time", "scenario"})
        for window in windows:
            identity = {key: point[key] for key in ("model", "scenario", "member")}
            identity["window_id"] = window["window_id"]
            try:
                reference = analysis_slice(
                    historical, window["reference"], water_year_start
                )
                projected = analysis_slice(
                    future, window["used"], water_year_start, historical=historical
                )
                # Ensure internal missing timestamps cannot be mistaken for complete years.
                for selected, bounds in (
                    (reference, window["reference"]),
                    (projected, window["used"]),
                ):
                    if selected.sizes["time"] != 12 * (bounds[1] - bounds[0] + 1):
                        raise AnalysisWindowUnavailable(
                            "missing internal monthly timestamps"
                        )
                annual = get_change_annual_clim_proj(
                    reference,
                    projected,
                    stats=["mean"],
                    start_month_hyd_year=water_year_start,
                    calendar=calendar,
                    variable_spec=spec,
                    min_reference=min_reference,
                )
                monthly = get_change_monthly_clim_proj(
                    reference,
                    projected,
                    stats=["mean"],
                    start_month_hyd_year=water_year_start,
                    variable_spec=spec,
                    min_reference=min_reference,
                )
            except (AnalysisWindowUnavailable, NoCompleteHydrologicalYear) as error:
                inventory["unavailable"].append(identity | {"reason": str(error)})
                continue
            result = xr.Dataset()
            for name in spec:
                valid = np.isfinite(reference[name]).all("time").isel(
                    scenario=0, drop=True
                ) & np.isfinite(projected[name]).all("time").isel(scenario=0, drop=True)
                if spec[name].canonical == "rate":
                    valid = valid & (reference[name] >= 0).all("time").isel(
                        scenario=0, drop=True
                    )
                    valid = valid & (projected[name] >= 0).all("time").isel(
                        scenario=0, drop=True
                    )
                for period, changes in (("annual", annual), ("monthly", monthly)):
                    field = (
                        changes[name].isel(stats=0, scenario=0, drop=True).where(valid)
                    )
                    if spec[name].change == "relative":
                        # The annual kernel's precipitation ratio has no dry guard;
                        # apply the configured rate threshold consistently to maps.
                        baseline = annual[f"{name}{COMPANION_SEP}reference"].isel(
                            stats=0, scenario=0, drop=True
                        )
                        if period == "annual":
                            field = field.where(
                                baseline >= (min_reference or {}).get(name, 0)
                            )
                        field = field.where(np.isfinite(field))
                    result[f"{name}_{period}"] = field
                    units = "%" if spec[name].change == "relative" else spec[name].units
                    if (
                        period == "annual"
                        and spec[name].canonical == "rate"
                        and spec[name].change == "absolute"
                    ):
                        units = (
                            units.removesuffix("/day")
                            if units.endswith("/day")
                            else f"{units} day"
                        )
                    result[f"{name}_{period}"].attrs["units"] = units
                if spec[name].change == "relative":
                    result[f"{name}_monthly_ratio"] = (
                        1 + result[f"{name}_monthly"] / 100
                    )
                    # Monthly reference/future means let consumers (the delta
                    # producer) report and treat dry reference cells; the ratio
                    # above stays NaN where the reference is below threshold.
                    for suffix, label in (
                        ("reference", "reference"),
                        ("level", "future"),
                    ):
                        level = monthly[f"{name}{COMPANION_SEP}{suffix}"].isel(
                            stats=0, scenario=0, drop=True
                        )
                        result[f"{name}_monthly_{label}"] = level.where(valid)
                        result[f"{name}_monthly_{label}"].attrs["units"] = spec[
                            name
                        ].units
                    threshold = (min_reference or {}).get(name)
                    result[f"{name}_monthly_ratio"].attrs["min_reference"] = (
                        "none" if threshold is None else float(threshold)
                    )
            result.attrs.update(
                **identity,
                reference_period=json.dumps(window["reference"]),
                future_period=json.dumps(window["used"]),
                cst_calendar=calendar,
            )
            name = f"member_{len(inventory['members']) + 1:04d}.nc"
            result.to_netcdf(output / name)
            inventory["members"].append(
                identity
                | {
                    "path": name,
                    "sha256": file_sha256(output / name),
                    "raw_historical": point["raw_path_hist"],
                    "raw_future": point["raw_path_fut"],
                }
            )
    record = Path(output).parents[1] / INVENTORY
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    # The readable view of the same members (rule R2): one row per grid file.
    pd.DataFrame(inventory["members"]).to_csv(output / "members.csv", index=False)
    return inventory


def prepare_retained_spatial_changes(ensemble_dir, tables, record, output):
    """Use a finished run's declared series inventory and local raw identities."""
    ensemble_dir = Path(ensemble_dir)
    inventory = {item["series_key"]: item for item in record["series"]}
    points = []
    for row in (
        tables["changes"][["model", "scenario", "member"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        point = {"model": row.model, "scenario": row.scenario, "member": row.member}
        for role, scenario in (("hist", "historical"), ("fut", row.scenario)):
            key = f"{ensemble_dir.name}_{row.model.replace('/', '_')}_{scenario}_{row.member}"
            item = inventory[key]
            series_path = ensemble_dir / "scalar" / Path(item["path"]).name
            with xr.open_dataset(series_path) as source:
                digest = source.attrs.get("cst_raw_digest")
            if not digest:
                raise ValueError(f"retained series has no raw identity: {series_path}")
            point[f"raw_path_{role}"] = str(ensemble_dir / "raw" / series_path.name)
            point[f"raw_digest_{role}"] = digest
        points.append(point)
    spec = {k: VariableSpec(*value) for k, value in record["variable_spec"].items()}
    return prepare_spatial_changes(
        points,
        record["windows"],
        output,
        variable_spec=spec,
        water_year_start=record["water_year_start"],
        min_reference={
            k: value.min_denominator
            for k, value in spec.items()
            if value.min_denominator is not None
        },
    )
