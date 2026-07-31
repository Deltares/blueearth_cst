"""Monthly change-factor tests for step 6a-ii (design §5.6). Falsifier M5."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.projections.get_change_climate_proj import (
    get_change_annual_clim_proj,
    get_change_monthly_clim_proj,
)


def _seasonal(first, last, base, amplitude=5.0, name="precip"):
    t = pd.date_range(first, last, freq="MS")
    v = base + amplitude * np.sin(2 * np.pi * (t.month - 1) / 12)
    return xr.DataArray(v, dims="time", coords={"time": t}).to_dataset(name=name)


def _flat(first, last, value, name="precip"):
    t = pd.date_range(first, last, freq="MS")
    return xr.DataArray(
        np.full(len(t), value), dims="time", coords={"time": t}
    ).to_dataset(name=name)


# --- M5: a real computation, not the annual value repeated --------------------


def test_M5_monthly_changes_are_not_all_equal():
    """A seasonal shift is precisely what the annual figure averages away."""
    res = get_change_monthly_clim_proj(
        _seasonal("1990-01", "2010-12", 20.0), _seasonal("2070-01", "2090-12", 24.0),
        stats=["mean"],
    )
    values = res["precip"].sel(stats="mean").values.ravel()
    assert len(values) == 12
    assert not np.allclose(values, values[0])


def test_M5_monthly_differs_from_the_annual_value():
    # The annual path selects a `scenario` coordinate unconditionally, so the
    # comparison fixture must carry one. The monthly path guards that selection --
    # a difference worth keeping rather than papering over, since the annual
    # function is the older and stricter of the two.
    def with_scenario(ds):
        return ds.expand_dims(scenario=["ssp245"])

    hist = with_scenario(_seasonal("1990-01", "2010-12", 20.0))
    clim = with_scenario(_seasonal("2070-01", "2090-12", 24.0))
    monthly = get_change_monthly_clim_proj(hist, clim, stats=["mean"])["precip"]
    annual = get_change_annual_clim_proj(hist, clim, stats=["mean"])["precip"]
    assert not np.allclose(
        monthly.sel(stats="mean").values.ravel(), float(annual.values.ravel()[0])
    )


def test_M5_a_uniform_shift_gives_the_same_change_every_month():
    """The converse: no seasonality in, no seasonality out. Guards against the
    computation inventing structure."""
    res = get_change_monthly_clim_proj(
        _flat("1990-01", "2010-12", 10.0), _flat("2070-01", "2090-12", 12.0),
        stats=["mean"],
    )
    values = res["precip"].sel(stats="mean").values.ravel()
    np.testing.assert_allclose(values, 20.0)  # +20% everywhere


def test_M5_all_twelve_months_are_present_and_ordered():
    res = get_change_monthly_clim_proj(
        _seasonal("1990-01", "2010-12", 20.0), _seasonal("2070-01", "2090-12", 24.0),
        stats=["mean"],
    )
    assert list(res["month"].values) == list(range(1, 13))


# --- every month must draw on the same years ----------------------------------


def test_every_month_uses_the_same_complete_hydrological_years():
    """Slicing by the raw window alone would give January one more sample than
    December whenever the window starts mid-year, and a seasonal pattern built
    from unequal samples is not a pattern."""
    hist = _seasonal("1990-03", "2011-06", 20.0)   # partial years at BOTH ends
    clim = _seasonal("2070-01", "2090-12", 24.0)
    res = get_change_monthly_clim_proj(hist, clim, stats=["mean"])
    assert res["precip"].sel(stats="mean").notnull().all()


# --- change semantics come from the spec, as everywhere else ------------------


def test_absolute_variables_are_differenced_not_ratioed():
    from blueearth_cst.projections.variable_spec import parse

    spec = parse({
        "temp": {"source": "temp", "canonical": "state", "units": "degC", "change": "absolute"},
    })
    res = get_change_monthly_clim_proj(
        _flat("1990-01", "2010-12", 10.0, name="temp"),
        _flat("2070-01", "2090-12", 12.0, name="temp"),
        stats=["mean"], variable_spec=spec,
    )
    np.testing.assert_allclose(res["temp"].sel(stats="mean").values.ravel(), 2.0)
