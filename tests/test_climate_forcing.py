"""Tests for the per-subcatchment forcing aggregation (ADR 0002).

Synthetic 2x2 grids keep these OS- and data-independent (no hydromt/wflow).
"""

import numpy as np
import xarray as xr

from blueearth_cst.climate_analysis.subcatchment_climate import climate_forcing_by_subcatchment  # noqa: E402


def _grid(values_2d, time=3):
    """A (time, lat, lon) DataArray whose spatial slice is constant over time."""
    arr = np.broadcast_to(np.asarray(values_2d, float), (time, 2, 2))
    return xr.DataArray(
        arr,
        dims=("time", "latitude", "longitude"),
        coords={"time": np.arange(time), "latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )


def _forcing():
    # precip varies per cell so the per-subcatchment mean is checkable
    return xr.Dataset(
        {
            "precip": _grid([[1.0, 3.0], [10.0, 99.0]]),
            "temp": _grid([[20.0, 20.0], [30.0, 99.0]]),
            "pet": _grid([[2.0, 2.0], [2.0, 99.0]]),
        }
    )


def _subcatchment():
    # ids 5 and 7; cell (1,1) is nodata (0) and must be excluded
    sub = xr.DataArray(
        [[5, 5], [7, 0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    sub.attrs["_FillValue"] = 0
    return sub


def test_renames_to_plot_clim_keys():
    ds = climate_forcing_by_subcatchment(_forcing(), _subcatchment())
    assert set(ds.data_vars) == {"P_subcatchment", "T_subcatchment", "EP_subcatchment"}


def test_per_subcatchment_mean_and_nodata_excluded():
    ds = climate_forcing_by_subcatchment(_forcing(), _subcatchment())
    # subcatchment 5 = cells 1.0 and 3.0 -> mean 2.0
    assert float(ds["P_subcatchment"].sel(index=5).isel(time=0)) == 2.0
    # subcatchment 7 = only cell 10.0 (99.0 is nodata, excluded) -> 10.0
    assert float(ds["P_subcatchment"].sel(index=7).isel(time=0)) == 10.0


def test_integer_index_and_dims():
    ds = climate_forcing_by_subcatchment(_forcing(), _subcatchment())
    assert ds["index"].dtype.kind == "i"  # integer ids, not float
    assert list(ds["index"].values) == [5, 7]
    assert set(ds["P_subcatchment"].dims) == {"index", "time"}


def test_default_nodata_is_zero_when_no_fillvalue_attr():
    sub = _subcatchment()
    del sub.attrs["_FillValue"]  # helper must default nodata to 0
    ds = climate_forcing_by_subcatchment(_forcing(), sub)
    assert list(ds["index"].values) == [5, 7]  # the 0 cell still excluded
