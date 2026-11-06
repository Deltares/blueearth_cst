"""Normalize daily Gregorian climate arrays without converting physical units."""

import numpy as np
import pandas as pd
import xarray as xr


def daily_array(data: xr.DataArray, *, units: str) -> xr.DataArray:
    """Validate canonical units and insert missing daily timestamps as NaN.

    Any non-time dimensions are preserved. Dates must be unique, increasing,
    midnight Gregorian timestamps. Infinite values are invalid; NaN represents
    unavailable observations. Conversion from source units is an adapter duty.
    The returned array owns its data and does not mutate the caller's array.
    """
    if "time" not in data.dims or data.sizes["time"] == 0:
        raise ValueError("Daily climate data require a nonempty time dimension")
    if not isinstance(data.indexes.get("time"), pd.DatetimeIndex):
        raise ValueError("Only Gregorian DatetimeIndex inputs are qualified")
    dates = data.indexes["time"]
    if (
        dates.hasnans
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or dates.tz is not None
        or not dates.equals(dates.normalize())
    ):
        raise ValueError("Dates must be unique, increasing, timezone-free daily dates")
    if data.attrs.get("units") != units:
        raise ValueError(f"Expected normalized units {units!r}")
    if not np.issubdtype(data.dtype, np.number) or np.issubdtype(
        data.dtype, np.complexfloating
    ):
        raise ValueError("Climate values must be real numbers")
    if bool(np.isinf(data).any()):
        raise ValueError("Infinite climate values are invalid")
    return data.reindex(time=pd.date_range(dates[0], dates[-1], freq="D")).copy(
        deep=True
    )
