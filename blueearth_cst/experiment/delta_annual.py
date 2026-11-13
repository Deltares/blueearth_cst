"""Annual projection changes for member-resolved delta diagnostics."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr


def annual_climatology(
    field: xr.DataArray, period: dict, *, temperature: bool
) -> xr.DataArray:
    """Mean of complete annual values, respecting each input's own calendar.

    Temperature is a day-weighted annual mean; water fluxes are annual totals.
    Missing monthly values propagate rather than shortening a year silently.
    """
    selected = field.sel(time=slice(str(period["start"]), str(period["end"])))
    pairs = list(
        zip(selected.time.dt.year.values, selected.time.dt.month.values, strict=True)
    )
    expected = [
        (year, month)
        for year in range(period["start"], period["end"] + 1)
        for month in range(1, 13)
    ]
    if pairs != expected:
        raise ValueError(
            "annual maps require one ordered sample per requested year/month"
        )
    days = selected.time.dt.days_in_month
    units = field.attrs.get("units")
    if temperature:
        if units not in {"K", "degC", "degree_Celsius", "degrees_Celsius"}:
            raise ValueError("annual temperature requires explicit K or degC units")
        annual = (selected * days).groupby("time.year").sum("time", skipna=False)
        annual = annual / days.groupby("time.year").sum("time")
    else:
        if units not in {"mm/day", "mm/month"}:
            raise ValueError("annual water flux requires mm/day or mm/month units")
        if bool((selected < 0).any()):
            raise ValueError("annual water flux cannot be negative")
        totals = selected * days if units == "mm/day" else selected
        annual = totals.groupby("time.year").sum("time", skipna=False)
    return annual.mean("year", skipna=False)


def annual_member_maps(data: Path, settings: dict, historical: xr.Dataset) -> dict:
    """Annual changes from paired projection climatologies on the historical grid.

    Return per-variable, per-model/scenario lists of member fields. Aggregation
    is deliberately after each member's annual change and spatial mapping.
    """
    import hydromt  # noqa: F401 -- raster accessor

    target = historical.temp.isel(time=0, drop=True).transpose("latitude", "longitude")
    active = historical.precip.notnull().any("time").transpose("latitude", "longitude")
    variables = {"temp": "temp_offset_degc", "precip": "precip_change_percent"}
    if "pet" in settings:
        variables["pet"] = "pet_change_percent"
    output = {key: defaultdict(list) for key in variables.values()}
    for member in settings["members"]:
        if "changes_path" in member:
            if historical.raster.crs is None:
                raise ValueError("annual map target requires an explicit CRS")
            target.raster.set_crs(historical.raster.crs)
            # WF2 members already hold annual changes from WF2's own kernel
            # (day-weighted temperature; precipitation/PET as % of the reference).
            with xr.open_dataset(data / member["changes_path"]) as changes:
                for variable, key in variables.items():
                    change = changes[f"{variable}_annual"]
                    change = change.transpose(
                        changes.raster.y_dim, changes.raster.x_dim
                    )
                    change.raster.set_crs(changes.raster.crs)
                    mapped = (
                        change.raster.reproject_like(target, method="nearest")
                        .raster.mask_nodata()
                        .transpose("latitude", "longitude")
                    )
                    if bool((~np.isfinite(mapped) & active).any()):
                        raise ValueError(
                            "annual projection changes do not cover the active historical grid"
                        )
                    output[key][(member["model"], member["scenario"])].append(
                        mapped.where(active).values
                    )
            continue
        with (
            xr.open_dataset(data / member["reference_path"]) as reference,
            xr.open_dataset(data / member["future_path"]) as future,
        ):
            for variable, key in variables.items():
                if reference[variable].attrs.get("units") != future[variable].attrs.get(
                    "units"
                ):
                    raise ValueError("paired annual projection units must match")
                ref = annual_climatology(
                    reference[variable],
                    settings["reference_period"],
                    temperature=variable == "temp",
                )
                fut = annual_climatology(
                    future[variable],
                    settings["future_period"],
                    temperature=variable == "temp",
                )
                if settings.get("factor_layout") != "grid":
                    # Legacy scalar archives retain singleton model/scenario/member
                    # axes; historical and future scenario labels differ by design.
                    ref, fut = ref.squeeze(drop=True), fut.squeeze(drop=True)
                    if ref.ndim or fut.ndim:
                        raise ValueError(
                            "basin-scalar annual maps require scalar projection inputs"
                        )
                ref, fut = xr.align(ref, fut, join="exact")
                change = (
                    fut - ref
                    if variable == "temp"
                    else 100 * (fut / ref.where(ref > 0) - 1)
                )
                if settings.get("factor_layout") == "grid":
                    if (
                        reference.raster.crs is None
                        or reference.raster.crs != future.raster.crs
                    ):
                        raise ValueError(
                            "paired annual projection grids require matching explicit CRS"
                        )
                    if historical.raster.crs is None:
                        raise ValueError("annual map target requires an explicit CRS")
                    target.raster.set_crs(historical.raster.crs)
                    change = change.transpose(
                        reference.raster.y_dim, reference.raster.x_dim
                    )
                    change.raster.set_crs(reference.raster.crs)
                    mapped = change.raster.reproject_like(
                        target, method="nearest"
                    ).raster.mask_nodata()
                else:
                    scalar = change.squeeze(drop=True)
                    if scalar.ndim:
                        raise ValueError(
                            "basin-scalar annual maps require scalar projection inputs"
                        )
                    mapped = xr.full_like(target, float(scalar), dtype=float)
                mapped = mapped.transpose("latitude", "longitude")
                if bool((~np.isfinite(mapped) & active).any()):
                    raise ValueError(
                        "annual projection changes do not cover the active historical grid"
                    )
                output[key][(member["model"], member["scenario"])].append(
                    mapped.where(active).values
                )
    return output


def summarize_members(fields: list[np.ndarray], method: str = "median") -> np.ndarray:
    """Equal-member cell-wise summary; never pool models or skip missing members."""
    if method not in {"median", "mean"} or not fields:
        raise ValueError("member summary requires fields and median or mean")
    values = np.stack(fields)
    return np.median(values, axis=0) if method == "median" else np.mean(values, axis=0)
