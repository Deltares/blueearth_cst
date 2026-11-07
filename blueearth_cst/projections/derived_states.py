"""Monthly native-cell liquid-water dew point and explicit 10 m wind statistics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from blueearth_cst.projections.derived_pet import INPUT_CONTRACT
from blueearth_cst.projections.variable_spec import STATE_INPUTS

if TYPE_CHECKING:
    import xarray as xr


def admit_state_sources(dataset: xr.Dataset) -> xr.Dataset:
    """Admit additional native fields before HydroMT removes scalar heights."""
    from blueearth_cst.projections.derived_pet import admit_monthly_sources

    return admit_monthly_sources(dataset, admit_states=True)


def register_state_preprocess() -> str:
    """Register the extended monthly admission hook without altering PET-only reads."""
    from hydromt.data_catalog.drivers import preprocessing

    name = "cmip6_amon_states_v1"
    preprocessing.PREPROCESSORS.setdefault(name, admit_state_sources)
    return name


def calculate_state(
    data: xr.Dataset, variable: str, method: str, *, source: str = ""
) -> xr.DataArray:
    """Derive each cell/month before spatial or temporal means; require fixed support."""
    import numpy as np
    import xarray as xr

    if variable not in STATE_INPUTS or method not in STATE_INPUTS[variable]:
        raise ValueError(f"{source}: unknown {variable} method {method!r}")
    names = STATE_INPUTS[variable][method]
    label = f"{source} {variable}/{method}"
    if missing := set(names) - set(data.data_vars):
        raise ValueError(f"{label}: missing dependencies {sorted(missing)}")
    arrays = xr.align(*(data[name] for name in names), join="exact")
    calendars = set()
    for name, array in zip(names, arrays, strict=True):
        units = {"sh": "1", "press": "hPa"}.get(name, "m/s")
        if set(array.dims) != set(arrays[0].dims) or array.attrs.get("units") != units:
            raise ValueError(
                f"{label}: {name} requires matching dimensions and canonical units {units}"
            )
        if array.attrs.get("cst_monthly_admission") != INPUT_CONTRACT:
            raise ValueError(f"{label}: {name} lacks monthly source admission")
        if variable == "wind" and array.attrs.get("cst_source_height_m") != 10:
            raise ValueError(f"{label}: {name} requires admitted 10 m height")
        calendars.add(array.attrs.get("cst_source_calendar", ""))
        if not array.size or not bool(np.isfinite(array).all()):
            raise ValueError(
                f"{label}: missing/nonfinite {name} on selected buffered bbox support"
            )
    if len(calendars) != 1 or calendars & {"", "unknown"}:
        raise ValueError(f"{label}: dependency calendars differ or are unknown")
    months = data.time.dt.year.values * 12 + data.time.dt.month.values
    if not len(months) or np.any(np.diff(months) != 1):
        raise ValueError(f"{label}: requires consecutive unique monthly intervals")
    if variable == "tdew":
        q, p = arrays
        if bool(((q <= 0) | (q >= 1) | (p <= 0)).any()):
            raise ValueError(
                f"{label}: requires 0 < specific humidity < 1 and pressure > 0"
            )
        vapour = q * p / (0.622 + (1 - 0.622) * q)
        logarithm = np.log10(vapour / 6.11)
        denominator = 7.5 - logarithm
        if bool((denominator <= 0).any()):
            raise ValueError(f"{label}: singular/out-of-domain liquid-water inversion")
        result = 237.3 * logarithm / denominator
        statistic = (
            "liquid-water dew point from monthly specific humidity and surface pressure"
        )
        approximation = "monthly-input inversion, not monthly mean of subdaily dew points; surface pressure approximates humidity-height pressure; liquid water below freezing, not frost point"
    elif method == "sfcwind":
        result = arrays[0].copy()
        if bool((result < 0).any()):
            raise ValueError(f"{label}: scalar wind speed must be nonnegative")
        statistic = "monthly mean wind-speed magnitude"
        approximation = "supplied monthly scalar speed; no height conversion"
    else:
        result = np.hypot(*arrays)
        statistic = "magnitude of monthly mean wind vector"
        approximation = "not monthly mean wind speed; magnitudes derived per cell before area weighting; no height conversion"
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{label}: nonfinite derived result")
    result = result.rename(variable)
    result.attrs = {
        "units": "degC" if variable == "tdew" else "m/s",
        "cst_derived_method": method,
        "cst_statistic": statistic,
        "cst_approximation": approximation,
        "cst_spatial_support": "entire selected buffered bounding box; spherical midpoint-edge cell areas; no polygon mask",
        "cst_derived_inputs": json.dumps(
            {name: dict(data[name].attrs) for name in names},
            sort_keys=True,
            default=str,
        ),
    }
    if variable == "wind":
        result.attrs["cst_height_m"] = 10.0
    return result
