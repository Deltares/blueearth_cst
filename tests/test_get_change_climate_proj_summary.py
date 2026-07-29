"""Tests for get_change_climate_proj_summary.py (workflow-2 summary helper).

Covers:
- ``preprocess_coords`` unit behaviour (drops ``height``, leaves others).
- Row D of the R4 §7 audit-evidence matrix: the dummy-skip merge path of
  ``summary_climate_proj`` excludes empty netCDFs and keeps non-empty ones.

Heavy deps (``open_mfdataset``, seaborn ``JointGrid``/``savefig``) run in the
real pixi env per the M02c discipline (dask cannot be stubbed at module level).
"""
from os.path import join, dirname, realpath

import numpy as np
import xarray as xr

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

from blueearth_cst.projections.get_change_climate_proj_summary import (  # noqa: E402
    filter_nonempty,
    preprocess_coords,
)


# --------------------------------------------------------------------------- #
# Unit: preprocess_coords
# --------------------------------------------------------------------------- #
def test_preprocess_coords_drops_height():
    """``height`` is a scalar coord that must be stripped before merge."""
    ds = xr.Dataset(
        {"precip": ("x", [1.0, 2.0])},
        coords={"x": [0, 1], "height": 2.0},
    )
    assert "height" in ds.coords
    out = preprocess_coords(ds)
    assert "height" not in out.coords


def test_preprocess_coords_leaves_other_coords():
    """Coords other than ``height`` are untouched."""
    ds = xr.Dataset(
        {"precip": ("x", [1.0, 2.0])},
        coords={"x": [0, 1], "lat": 10.0},
    )
    out = preprocess_coords(ds)
    assert "lat" in out.coords
    assert "x" in out.coords
    # data preserved
    np.testing.assert_array_equal(out["precip"].values, [1.0, 2.0])


def test_preprocess_coords_noop_when_no_height():
    """No ``height`` coord -> returned dataset is equivalent to input."""
    ds = xr.Dataset({"temp": ("x", [3.0])}, coords={"x": [0]})
    out = preprocess_coords(ds)
    xr.testing.assert_identical(ds, out)


# --------------------------------------------------------------------------- #
# Row D (PASS): the dummy-skip merge decision, via the filter_nonempty helper
# --------------------------------------------------------------------------- #
# The design's §7 row D targets the "empty datasets are dropped from the merge"
# behaviour. That logic lives in ``filter_nonempty`` — the small output-neutral
# extract the design named as the fallback when driving the full
# ``summary_climate_proj`` is too heavy (it pulls in the hydromt ``.raster``
# accessor + seaborn, which collide with sibling test files' sys.modules stubs,
# M02c). Unit-testing the extract exercises the exact merge decision under test
# without that pollution surface.
def _write_change_nc(path, model, precip_change, temp_change):
    """Write a minimal *non-empty* annual_change_scalar_stats-style netCDF."""
    ds = xr.Dataset(
        {
            "precip": (("stats", "horizon", "scenario", "model"), [[[[precip_change]]]]),
            "temp": (("stats", "horizon", "scenario", "model"), [[[[temp_change]]]]),
        },
        coords={
            "stats": ["mean"],
            "horizon": ["near"],
            "scenario": ["ssp245"],
            "model": [model],
        },
    )
    ds.to_netcdf(path)


def test_filter_nonempty_excludes_empty_keeps_nonempty(tmp_path):
    """Row D: the dummy (empty) netCDF is dropped; the non-empty one is kept."""
    empty_file = tmp_path / "annual_change_scalar_stats-EMPTY_ssp245_near.nc"
    full_file = tmp_path / "annual_change_scalar_stats-GOOD_ssp245_near.nc"

    # dummy/empty dataset exactly as the producer writes it (xr.Dataset())
    xr.Dataset().to_netcdf(empty_file)
    _write_change_nc(full_file, "GOOD", precip_change=20.0, temp_change=2.0)

    kept = filter_nonempty([str(empty_file), str(full_file)])

    assert kept == [str(full_file)], f"empty file not excluded; kept={kept}"


def test_filter_nonempty_all_empty_returns_empty_list(tmp_path):
    """All-dummy input → nothing survives the filter."""
    f1 = tmp_path / "annual_change_scalar_stats-A_ssp245_near.nc"
    f2 = tmp_path / "annual_change_scalar_stats-B_ssp245_near.nc"
    xr.Dataset().to_netcdf(f1)
    xr.Dataset().to_netcdf(f2)

    assert filter_nonempty([str(f1), str(f2)]) == []
