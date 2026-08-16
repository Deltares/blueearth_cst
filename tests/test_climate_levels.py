"""Shared plotting scales across candidate sources (climate_analysis/climate_levels).

Unit-level: the module takes stores on disk and returns a mapping, so nothing
here needs snakemake or a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from blueearth_cst.climate_analysis import climate_figures as cf
from blueearth_cst.climate_analysis import climate_levels as cl


def _store(tmp_path, name, *, precip_base=4.0, variables=("precip", "temp", "pet")):
    """A small store on disk, carrying only the named variables."""
    time = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    ys, xs = np.arange(0.0, 1.01, 0.25), np.arange(9.0, 10.01, 0.25)
    season = np.sin(2 * np.pi * time.dayofyear.values / 365.25).astype("float32")
    # Varying in SPACE as well as time: a spatially uniform field has a
    # degenerate range, which exercises the widening branch rather than the
    # ordinary one the map scale is really for.
    gradient = np.add.outer(ys, xs).astype("float32") / 10.0
    field = np.broadcast_to(gradient, (time.size, ys.size, xs.size))

    def _var(base, amp):
        return ("time", "latitude", "longitude"), (
            base + amp * season[:, None, None] + field
        ).astype("float32")

    data = {
        "precip": _var(precip_base, 3.0),
        "temp": _var(24.0, 3.0),
        "pet": _var(3.5, 1.0),
    }
    ds = xr.Dataset(
        {k: v for k, v in data.items() if k in variables},
        coords={"time": time, "latitude": ys, "longitude": xs},
    )
    path = tmp_path / f"{name}.nc"
    ds.to_netcdf(path)
    return path


def test_a_scale_is_produced_for_every_kind(tmp_path):
    levels = cl.compute_climate_levels({"era5": _store(tmp_path, "era5")}, ["precip"])
    assert set(levels["precip"]) == set(cf.FIGURE_KINDS)
    # The map scale is a boundary LADDER; the series scales are a pair.
    assert len(levels["precip"]["map"]) > 2
    assert len(levels["precip"]["annual"]) == 2
    assert len(levels["precip"]["monthly"]) == 2


def test_the_scale_spans_both_sources_not_just_one(tmp_path):
    """The whole point: a shared scale must contain BOTH datasets' values."""
    dry = _store(tmp_path, "dry", precip_base=2.0)
    wet = _store(tmp_path, "wet", precip_base=9.0)

    alone = cl.compute_climate_levels({"dry": dry}, ["precip"])
    shared = cl.compute_climate_levels({"dry": dry, "wet": wet}, ["precip"])

    assert shared["precip"]["annual"][1] > alone["precip"]["annual"][1]
    assert max(shared["precip"]["map"]) > max(alone["precip"]["map"])


def test_a_variable_is_pooled_only_over_the_stores_that_carry_it(tmp_path):
    """A precipitation-only source must not contribute to a temperature scale.

    Its `temp` in the store is another dataset's, borrowed to force the model —
    the same reason it draws no temperature figure.
    """
    full = _store(tmp_path, "era5")
    precip_only = _store(tmp_path, "chirps", variables=("precip",))

    levels = cl.compute_climate_levels(
        {"era5": full, "chirps": precip_only}, ["precip", "temp"]
    )

    assert "temp" in levels  # era5 still supplies it
    only_full = cl.compute_climate_levels({"era5": full}, ["temp"])
    assert levels["temp"] == only_full["temp"]


def test_a_variable_no_store_carries_is_omitted(tmp_path):
    """Omitted, not defaulted: the consumer must fall back to its own data."""
    levels = cl.compute_climate_levels(
        {"chirps": _store(tmp_path, "chirps", variables=("precip",))},
        ["precip", "temp"],
    )
    assert "temp" not in levels


def test_round_trip_through_json(tmp_path):
    levels = cl.compute_climate_levels({"era5": _store(tmp_path, "era5")}, ["precip"])
    path = cl.write_climate_levels(levels, tmp_path / "climate_levels.json")

    assert cl.read_climate_levels(path) == levels


@pytest.mark.parametrize("missing", [None, "absent.json"])
def test_an_absent_levels_file_degrades_to_no_shared_scale(tmp_path, missing):
    """WF1 draws ONE source and has nothing to share with — it must not raise."""
    path = None if missing is None else tmp_path / missing
    assert cl.read_climate_levels(path) == {}


@pytest.mark.slow
def test_two_sources_render_against_the_same_scale(tmp_path):
    """End to end: the figures of two datasets carry one y-range."""
    import matplotlib

    matplotlib.use("Agg")

    dry, wet = (
        _store(tmp_path, "dry", precip_base=2.0),
        _store(tmp_path, "wet", precip_base=9.0),
    )
    levels = cl.compute_climate_levels({"dry": dry, "wet": wet}, ["precip"])

    ylims = []
    for name, path in (("dry", dry), ("wet", wet)):
        out = tmp_path / name
        with xr.open_dataset(path) as ds:
            cf.plot_climate_figures(
                ds, out, "source", variables=("precip",), levels=levels
            )
        assert (out / "source_precip_annual.png").stat().st_size > 0
        ylims.append(tuple(levels["precip"]["annual"]))

    assert ylims[0] == ylims[1]
