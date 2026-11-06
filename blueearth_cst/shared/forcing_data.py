"""Bounded admission checks for recipe inputs after HydroMT catalog adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from geopandas import GeoDataFrame

    from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
    from blueearth_cst.shared.forcing_recipe import ForcingRecipe

CANONICAL_UNITS = {
    "precip": "mm/day",
    "temp": "degC",
    "temp_min": "degC",
    "temp_max": "degC",
    "press_msl": "hPa",
    "kin": "W/m2",
    "kout": "W/m2",
    "pet": "mm/day",
}
_UNIT_LABELS = {
    "mm/day": {"mm/day", "mm d-1", "mm d**-1", "mm day-1"},
    "degC": {"degC", "degree_Celsius", "degrees_Celsius", "degree C", "degree C.", "C"},
    "hPa": {"hPa", "mbar"},
    "W/m2": {"W/m2", "W m-2", "W m**-2"},
}
_DAILY_BINDING = {
    "unit_add": {"temp": -273.15, "temp_min": -273.15, "temp_max": -273.15},
    "unit_mult": {"kin": 0.000277778, "kout": 0.000277778, "press_msl": 0.01},
    "rename": {
        "msl": "press_msl",
        "ssrd": "kin",
        "t2m": "temp",
        "tisr": "kout",
        "tmax": "temp_max",
        "tmin": "temp_min",
        "tp": "precip",
    },
}


def departs_from_reviewed_binding(adapter: Mapping[str, Any]) -> bool:
    """Check the reviewed ERA5 daily arithmetic, allowing unrelated variables."""
    variables = set(_DAILY_BINDING["rename"].values())
    for key in ("unit_add", "unit_mult"):
        required, found = _DAILY_BINDING[key], adapter.get(key, {})
        if any(found.get(name) != value for name, value in required.items()):
            return True
        if any(name in variables for name in found if name not in required):
            return True
    required, found = _DAILY_BINDING["rename"], adapter.get("rename", {})
    if any(found.get(name) != value for name, value in required.items()):
        return True
    return any(
        value in variables for name, value in found.items() if name not in required
    )


def validate_daily_fields(
    dataset: xr.Dataset,
    variables: Sequence[str],
    *,
    source: str,
    domain: xr.DataArray | GeoDataFrame | None = None,
) -> None:
    """Require complete daily fields; allow masked cells outside the domain.

    ``domain`` is a boolean grid mask or a GeoDataFrame in the raster's CRS.
    With no domain supplied every extracted cell is required. No missing dates,
    duplicates, unsupported calendars or supplied-PET negatives are repaired.
    """
    missing = set(variables) - set(dataset.data_vars)
    if missing:
        raise ValueError(f"{source}: missing required variables {sorted(missing)}")
    if "time" not in dataset.coords:
        raise ValueError(f"{source}: missing daily time axis")
    calendar = str(dataset.time.dt.calendar)
    if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
        raise ValueError(f"{source}: unsupported historical calendar {calendar}")
    dates = pd.DatetimeIndex(dataset.time.values)
    if (
        len(dates) < 2
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not (dates == dates.normalize()).all()
        or not (np.diff(dates.values) == np.timedelta64(1, "D")).all()
    ):
        raise ValueError(f"{source}: expected unique, contiguous daily midnight dates")
    if domain is not None and not isinstance(domain, xr.DataArray):
        domain = dataset.raster.geometry_mask(domain, all_touched=True)
    if domain is not None and not bool(domain.any()):
        raise ValueError(f"{source}: required domain contains no climate cells")
    for name in variables:
        values = dataset[name]
        if "time" not in values.dims:
            raise ValueError(f"{source}.{name}: missing time dimension")
        invalid = ~np.isfinite(values)
        if name == "pet":
            invalid = invalid | (values < 0)
        if domain is not None:
            invalid = invalid & domain
        if bool(invalid.any().compute()):
            raise ValueError(
                f"{source}.{name}: invalid values over required domain/time"
            )


def validate_canonical_units(
    dataset: xr.Dataset, variables: Sequence[str], *, source: str
) -> None:
    """Check post-adapter labels under the admitted interval-end daily contract.

    PET in mm is a daily total, equivalent to mm/day on this one-day axis.
    Source convention and actual daily spacing are checked separately.
    """
    for name in variables:
        attrs = dataset[name].attrs
        labels = [attrs[key] for key in ("units", "unit") if key in attrs]
        accepted = _UNIT_LABELS[CANONICAL_UNITS[name]]
        if name == "pet":
            accepted = accepted | {"mm"}
        if not labels or any(label not in accepted for label in labels):
            raise ValueError(
                f"UnverifiedForcingUnits: {source}.{name} requires "
                f"{CANONICAL_UNITS[name]}, observed {labels}"
            )


def source_unit_profile(
    entry: Mapping[str, Any], variables: Sequence[str], *, source: str
) -> str:
    """Select reviewed arithmetic or identity arithmetic with checked labels.

    The ERA5 profile is recognized by its reviewed adapter, never by its name.
    Other sources must already be in canonical units; this does not approve a
    new conversion profile or infer units from magnitudes.
    """
    adapter = entry.get("data_adapter", {})
    if "pet" not in variables and not departs_from_reviewed_binding(adapter):
        return "reviewed-era5-daily/2"
    if source in {"chirps", "chirps_global"} and tuple(variables) == ("precip",):
        if (
            adapter.get("rename") == {"precipitation": "precip"}
            and adapter.get("unit_add") == {"time": 86400}
            and not adapter.get("unit_mult", {})
        ):
            return "reviewed-chirps-daily/2"
    for field, identity in (("unit_add", 0), ("unit_mult", 1)):
        values = adapter.get(field, {})
        if any(values.get(name, identity) != identity for name in variables):
            raise ValueError(
                f"UnverifiedForcingUnits: {source} has unreviewed {field} arithmetic"
            )
    if entry.get("metadata", {}).get("time_label") != "interval_end":
        raise ValueError(
            f"UnverifiedForcingUnits: {source} canonical daily input requires "
            "catalog metadata.time_label: interval_end"
        )
    return "canonical-daily/1"


def recipe_unit_profiles(
    entries: Mapping[str, Mapping[str, Any]], recipe: ForcingRecipe
) -> dict[str, dict[str, Any]]:
    """Keep role-specific variable admission even when roles share one source."""
    requirements = {
        "precipitation": (recipe.precip_source, ("precip",)),
        "companion": (recipe.temp_pet_source, recipe.companion_variables),
    }
    if recipe.pet_source:
        requirements["pet"] = (recipe.pet_source, ("pet",))
    return {
        role: {
            "source": source,
            "variables": list(variables),
            "profile": source_unit_profile(entries[source], variables, source=source),
        }
        for role, (source, variables) in requirements.items()
    }


def recipe_unit_interpretation(
    catalogs: str | Path | Sequence[str | Path],
    recipe: ForcingRecipe,
    climate_nc: str | Path,
) -> UnitInterpretation:
    """Bind a non-default extraction to its reviewed/validated unit evidence."""
    import json

    import hydromt

    from blueearth_cst.experiment.forcing_descriptor import UnitInterpretation
    from blueearth_cst.shared.provenance import file_sha256

    paths = [catalogs] if isinstance(catalogs, (str, Path)) else list(catalogs)
    entries = hydromt.DataCatalog(data_libs=paths).to_dict()
    profiles = recipe_unit_profiles(entries, recipe)
    with xr.open_dataset(climate_nc) as dataset:
        if (
            dataset.attrs.get("forcing_extraction_sha256")
            != recipe.extraction_fingerprint
        ):
            raise ValueError(
                "forcing extraction identity differs from the selected recipe"
            )
        if json.loads(dataset.attrs.get("forcing_unit_profiles", "{}")) != profiles:
            raise ValueError(
                "forcing extraction unit profiles differ from current catalogs"
            )
        if set(dataset.data_vars) - {"spatial_ref"} != set(recipe.required_variables):
            raise ValueError(
                "forcing extraction variables differ from the selected recipe"
            )
        for evidence in profiles.values():
            if evidence["profile"] == "canonical-daily/1":
                validate_canonical_units(
                    dataset, evidence["variables"], source=evidence["source"]
                )
    return UnitInterpretation(
        revision="recipe-daily/1",
        evidence=f"profiles={profiles}; extraction={recipe.extraction_fingerprint}; "
        f"catalogs={[(str(path), file_sha256(path)) for path in paths]}",
        variables=tuple(
            (name, CANONICAL_UNITS[name]) for name in recipe.required_variables
        ),
    )
