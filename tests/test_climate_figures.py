"""The canonical climate figure set (blueearth_cst/climate_analysis/climate_figures).

Unit-level, with no model and no snakemake: the module takes a plain
``xr.Dataset`` precisely so it can be tested this way, and that seam is what
keeps the source side model-free (the P4 property
``tests/test_plot_climate_source.py`` pins in the real Snakemake DAG).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

from blueearth_cst.climate_analysis import climate_figures as cf  # noqa: E402


def _dataset(start="2000-01-01", end="2004-12-31", chunk=False) -> xr.Dataset:
    """A small gridded climate carrying the canonical variables."""
    time = pd.date_range(start, end, freq="D")
    ys, xs = np.arange(0.0, 1.01, 0.25), np.arange(9.0, 10.01, 0.25)
    season = np.sin(2 * np.pi * time.dayofyear.values / 365.25).astype("float32")
    ones = np.ones((time.size, ys.size, xs.size), dtype="float32")

    def _var(base, amp):
        return ("time", "y", "x"), (base + amp * season[:, None, None] * ones).astype(
            "float32"
        )

    ds = xr.Dataset(
        {"precip": _var(4.0, 3.0), "temp": _var(24.0, 3.0), "pet": _var(3.5, 1.0)},
        coords={"time": time, "y": ys, "x": xs},
    )
    return ds.chunk({"time": 365}) if chunk else ds


# --- the declared name set -------------------------------------------------


def test_figure_names_is_the_full_cross_product():
    names = cf.figure_names("source")
    assert len(names) == len(cf.CLIMATE_VARS) * len(cf.FIGURE_KINDS)
    assert len(set(names)) == len(names)
    assert all(name.startswith("source_") and name.endswith(".png") for name in names)


def test_every_dataset_gets_its_own_prefix():
    """The prefix is what makes a figure self-identifying once copied out of
    its directory, and what makes the two directories comparable."""
    source, forcing = cf.figure_names("source"), cf.figure_names("forcing")
    assert not set(source) & set(forcing)
    assert [n.replace("source_", "", 1) for n in source] == [
        n.replace("forcing_", "", 1) for n in forcing
    ]


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        cf.figure_names("wflow")


# --- the per-source subset -------------------------------------------------


@pytest.mark.parametrize("source", ["chirps", "chirps_global"])
def test_a_precip_only_source_draws_precipitation_only(source):
    """Its temp/PET fields in the store are era5's, borrowed to force the model.

    Drawing them under this source's name would answer the source comparison
    with a panel that cannot differ (owner ruling 2026-08-16).
    """
    assert cf.source_climate_vars(source) == ("precip",)

    names = cf.figure_names("source", variables=cf.source_climate_vars(source))
    assert len(names) == len(cf.FIGURE_KINDS)
    assert all("precip" in name for name in names)
    assert not any("temp" in name or "pet" in name for name in names)


def test_a_full_source_still_draws_the_whole_set():
    assert cf.source_climate_vars("era5") == tuple(cf.CLIMATE_VARS)
    assert cf.figure_names("source", variables=cf.source_climate_vars("era5")) == (
        cf.figure_names("source")
    )


def test_variable_order_follows_the_canonical_set_not_the_caller():
    """Two callers naming the same variables must declare the same filenames."""
    assert cf.figure_names("source", variables=["pet", "precip"]) == cf.figure_names(
        "source", variables=["precip", "pet"]
    )


def test_an_unknown_variable_is_rejected():
    with pytest.raises(ValueError, match="unknown climate variables"):
        cf.figure_names("source", variables=["precip", "humidity"])


def test_an_empty_variable_set_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        cf.figure_names("source", variables=[])


@pytest.mark.slow
def test_a_narrowed_set_writes_only_those_figures(tmp_path):
    """The declaration and the drawing must agree, or the rule fails.

    Narrow only the declaration and the extra files are undeclared outputs;
    narrow only the drawing and Snakemake raises MissingOutputException.
    """
    written = cf.plot_climate_figures(
        _dataset(), tmp_path, "source", variables=("precip",)
    )

    expected = cf.figure_names("source", variables=("precip",))
    assert [p.name for p in written] == expected
    assert sorted(p.name for p in tmp_path.glob("*.png")) == sorted(expected)


@pytest.mark.slow
def test_a_narrowed_set_ignores_a_variable_absent_from_the_dataset(tmp_path):
    """A precip-only draw must not require the variables it does not draw."""
    ds = _dataset().drop_vars(["temp", "pet"])

    written = cf.plot_climate_figures(ds, tmp_path, "source", variables=("precip",))

    assert len(written) == len(cf.FIGURE_KINDS)


# --- writing the set -------------------------------------------------------


@pytest.mark.slow
def test_writes_exactly_the_declared_names(tmp_path):
    written = cf.plot_climate_figures(_dataset(), tmp_path, "source")
    assert [p.name for p in written] == cf.figure_names("source")
    on_disk = sorted(p.name for p in tmp_path.glob("*.png"))
    assert on_disk == sorted(cf.figure_names("source"))
    assert all(p.stat().st_size > 0 for p in written)


@pytest.mark.slow
def test_a_dask_backed_dataset_works(tmp_path):
    """The regression this module shipped with: PET arrives dask-backed from the
    meteo workflow while precip and temp come straight off the netCDF, and
    `where(..., drop=True)` refuses to index with a boolean DASK array. It
    presented as six figures written and then a KeyError -- so a
    numpy-only fixture would not have caught it.
    """
    written = cf.plot_climate_figures(_dataset(chunk=True), tmp_path, "forcing")
    assert len(written) == len(cf.figure_names("forcing"))


def test_a_missing_variable_is_loud(tmp_path):
    """The rules declare these figures, so a silent skip would resurface as an
    opaque MissingOutputException at the end of the job."""
    ds = _dataset().drop_vars("pet")
    with pytest.raises(ValueError, match="missing \\['pet'\\]"):
        cf.plot_climate_figures(ds, tmp_path, "source")


@pytest.mark.slow
def test_overlays_are_optional_and_absent_entries_are_skipped(tmp_path):
    """A caller with no model passes nothing; a caller with a partial set passes
    what it has."""
    written = cf.plot_climate_figures(
        _dataset(), tmp_path, "source", overlays={"basins": None, "rivers": None}
    )
    assert len(written) == len(cf.figure_names("source"))


# --- the aggregation rules -------------------------------------------------


def test_flux_and_state_aggregate_differently():
    """`sum` vs `mean` is not cosmetic: a summed temperature is meaningless and
    a meaned rainfall understates by ~365x."""
    assert cf.CLIMATE_VARS["precip"]["how"] == "sum"
    assert cf.CLIMATE_VARS["pet"]["how"] == "sum"
    assert cf.CLIMATE_VARS["temp"]["how"] == "mean"


def test_incomplete_years_are_dropped_from_a_total():
    """A window starting mid-year would otherwise draw a first-year dip that
    looks like climate and is calendar."""
    ds = _dataset(start="2000-07-01", end="2004-12-31")
    series = ds["precip"].mean(dim=("y", "x"))
    years = cf._yearly(series, "sum")["time"].dt.year.values
    assert 2000 not in years, "the half year should have been dropped"
    assert list(years) == [2001, 2002, 2003, 2004]


def test_a_mean_keeps_every_year():
    """A mean over a partial year is still a valid mean of what was observed,
    so the completeness filter must not touch it."""
    ds = _dataset(start="2000-07-01", end="2004-12-31")
    series = ds["temp"].mean(dim=("y", "x"))
    years = cf._yearly(series, "mean")["time"].dt.year.values
    assert list(years) == [2000, 2001, 2002, 2003, 2004]
