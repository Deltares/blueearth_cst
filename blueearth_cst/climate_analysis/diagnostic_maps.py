"""Native-grid complete reporting-year fields for scientific preparation."""

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.climate_analysis import diagnostics as dg


def annual_fields(ds, m0, settings):
    """Apply amount/temperature completeness cellwise without interpolating."""
    output = xr.Dataset()
    tol = dg.Completeness(
        settings["max_missing_days_month"], settings["max_missing_days_year"]
    )
    for variable in ("precip", "temp", "pet"):
        if variable not in ds:
            continue
        field = ds[variable].transpose("time", "latitude", "longitude")
        data = field.values
        shape = field.shape[1:]
        means = np.full(shape, np.nan)
        counts = np.zeros(shape, dtype=np.int32)
        index = pd.DatetimeIndex(field.time.values)
        calendar = pd.date_range(index.min(), index.max(), freq="D")
        for i in range(shape[0]):
            for j in range(shape[1]):
                values = pd.Series(data[:, i, j], index=index).reindex(calendar)
                yearly = dg.annual(
                    values, m0, "mean" if variable == "temp" else "sum", tol
                )
                counts[i, j] = yearly.notna().sum()
                means[i, j] = yearly.mean()
        coords = {key: field[key] for key in ("latitude", "longitude")}
        attrs = {"units": "degC" if variable == "temp" else "mm/year"}
        for name in ("latitude", "longitude"):
            if len(field[name]) > 1:
                attrs[f"{name}_resolution"] = abs(
                    float(field[name][1] - field[name][0])
                )
            elif hasattr(ds, "raster"):
                attrs[f"{name}_resolution"] = abs(
                    float(ds.raster.res[0 if name == "longitude" else 1])
                )
        output[variable] = xr.DataArray(
            means, dims=("latitude", "longitude"), coords=coords, attrs=attrs
        )
        output[f"{variable}_valid_years"] = xr.DataArray(
            counts, dims=("latitude", "longitude"), coords=coords
        )
        output[f"{variable}_unavailable"] = xr.DataArray(
            ~np.isfinite(means), dims=("latitude", "longitude"), coords=coords
        )
    return output
