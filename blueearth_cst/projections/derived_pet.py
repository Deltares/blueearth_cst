"""Native-cell PET estimates from admitted monthly CMIP6 meteorology."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

from blueearth_cst.projections.variable_spec import PET_INPUTS

# CMIP6 Amon native units; conversions remain owned by HydroMT's adapter.
SOURCE_UNITS = {
    "tas": "K",
    "ps": "Pa",
    "rsds": "W m-2",
    "rsdt": "W m-2",
}
CANONICAL_UNITS = {"temp": "degC", "press": "hPa", "kin": "W m-2", "kout": "W m-2"}
INPUT_CONTRACT = "cmip6_amon_pet_v1"


def admit_monthly_sources(dataset: xr.Dataset) -> xr.Dataset:
    """Validate native metadata before HydroMT changes names, values or dates.

    Amon and time-mean metadata establish interval semantics; timestamps alone
    cannot establish that a field contains monthly mean meteorology.
    """
    from blueearth_cst.projections.fetch_gcm_raw import harmonise_dims_wide_time

    calendar = str(
        dataset.time.encoding.get("calendar")
        or dataset.time.attrs.get("calendar")
        or ""
    )
    if not calendar:
        raise ValueError(
            "PET source has no declared calendar; supply qualified CMIP6 Amon inputs"
        )
    for name, array in dataset.data_vars.items():
        if "time" not in array.dims:
            continue
        units = str(array.attrs.get("units", ""))
        expected = SOURCE_UNITS.get(name)
        if expected is not None and units != expected:
            raise ValueError(
                f"PET source {name}: units {units!r}, expected {expected!r}; correct the source/adapter"
            )
        methods = str(array.attrs.get("cell_methods", ""))
        table = str(dataset.attrs.get("table_id", ""))
        if expected is not None and (
            table != "Amon" or not re.search(r"\btime:\s*mean\b", methods)
        ):
            raise ValueError(
                f"PET source {name}: requires table_id=Amon and cell_methods time: mean"
            )
        array.attrs.update(
            cst_source_units=units,
            cst_source_calendar=calendar,
            cst_source_cell_methods=methods,
            cst_monthly_admission=INPUT_CONTRACT,
            cst_source_model=f"{dataset.attrs.get('institution_id', '')}/{dataset.attrs.get('source_id', '')}",
            cst_source_member=str(dataset.attrs.get("variant_label", "")),
            cst_source_experiment=str(dataset.attrs.get("experiment_id", "")),
            cst_source_grid=str(dataset.attrs.get("grid_label", "")),
        )
    return harmonise_dims_wide_time(dataset)


def register_pet_preprocess() -> str:
    """Register the source admission hook through HydroMT's supported seam."""
    from hydromt.data_catalog.drivers import preprocessing

    preprocessing.PREPROCESSORS.setdefault(INPUT_CONTRACT, admit_monthly_sources)
    return INPUT_CONTRACT


def validate_pet_adapter(entry: dict, variables: list[str]) -> None:
    """Require the catalog's declared conversions to match admitted native units."""
    adapter = entry.get("data_adapter") or {}
    conversions = {
        "temp": ("tas", -273.15, 1),
        "press": ("ps", 0, 0.01),
        "kin": ("rsds", 0, 1),
        "kout": ("rsdt", 0, 1),
    }
    for target in set(variables) & set(conversions):
        native, add, mult = conversions[target]
        if (
            (adapter.get("rename") or {}).get(native) != target
            or (adapter.get("unit_add") or {}).get(target, 0) != add
            or (adapter.get("unit_mult") or {}).get(target, 1) != mult
        ):
            raise ValueError(
                f"PET input {native}/{target}: catalog adapter must rename and convert admitted native units (add={add}, mult={mult}); rebuild the catalog with its generator"
            )


def calculate_pet(data: xr.Dataset, method: str, *, source: str = "") -> xr.DataArray:
    """Evaluate upstream PET cell-wise, preserving the joint required mask.

    The complete selected buffered bounding box is required in every month.
    Missing values fail rather than silently changing the weighted area.
    """
    import hydromt
    import numpy as np
    import xarray as xr
    from hydromt.model.processes import meteo

    if method not in PET_INPUTS:
        raise ValueError(f"unknown PET method {method!r}")
    names = sorted(PET_INPUTS[method])
    missing = set(names) - set(data.data_vars)
    if missing:
        raise ValueError(
            f"{source}: missing PET inputs {sorted(missing)}; supply matching monthly sources or disable PET"
        )
    arrays = xr.align(*(data[name] for name in names), join="exact")
    calendars = set()
    for name, array in zip(names, arrays, strict=True):
        if set(array.dims) != set(data["temp"].dims):
            raise ValueError(f"{source}: {name} PET dimensions differ from temperature")
        if array.attrs.get("units") != CANONICAL_UNITS[name]:
            raise ValueError(
                f"{source}: {name} PET canonical units must be {CANONICAL_UNITS[name]}"
            )
        if array.attrs.get("cst_monthly_admission") != INPUT_CONTRACT:
            raise ValueError(
                f"{source}: {name} lacks monthly source admission; re-acquire qualified inputs"
            )
        calendars.add(array.attrs.get("cst_source_calendar", ""))
    if len(calendars) != 1 or "" in calendars or "unknown" in calendars:
        raise ValueError(
            f"{source}: PET dependency calendars differ or are unknown: {calendars}"
        )
    months = data.time.dt.year.values * 12 + data.time.dt.month.values
    if not len(months) or np.any(np.diff(months) != 1):
        raise ValueError(f"{source}: PET needs consecutive unique monthly intervals")
    joint = np.isfinite(arrays[0])
    for array in arrays[1:]:
        joint = joint & np.isfinite(array)
    if not joint.size or not bool(joint.all()):
        raise ValueError(
            f"{source}: missing/nonfinite PET inputs in selected buffered bbox; repair input coverage"
        )
    invalid = (data.press <= 0) | (data.kin < 0) | (data.temp <= -243.5)
    if method == "debruin":
        invalid = invalid | (data.kout < 0)
    if bool(invalid.any()):
        raise ValueError(
            f"{source}: invalid PET pressure, temperature or radiation on basin support"
        )
    if method == "makkink":
        result = meteo.pet_makkink(data.temp, data.press, data.kin, timestep=86400)
    elif method == "debruin":
        result = meteo.pet_debruin(
            data.temp, data.press, data.kin, data.kout, timestep=86400
        )
    else:
        raise ValueError(f"unknown PET method {method!r}")
    result = result.where(joint).rename("pet")
    if not bool(np.isfinite(result).all()):
        raise ValueError(
            f"{source}: upstream PET returned nonfinite values on basin support"
        )
    result.attrs.update(
        units="mm/day",
        cst_pet_method=method,
        cst_pet_library=f"hydromt {hydromt.__version__}",
        cst_pet_approximation="native-cell function of monthly mean meteorology; not mean daily-derived PET",
        cst_pet_corrections="none; native surface pressure; no elevation correction",
        cst_spatial_support="entire selected buffered bounding box; spherical midpoint-edge cell areas; no polygon mask",
        cst_pet_inputs=json.dumps(
            {name: dict(data[name].attrs) for name in names},
            sort_keys=True,
            default=str,
        ),
    )
    return result
