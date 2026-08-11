"""Tests for get_change_climate_proj_summary.py (workflow-2 summary helper).

Covers:
- ``preprocess_coords`` unit behaviour (drops ``height``, leaves others).
- Row D of the R4 §7 audit-evidence matrix: the dummy-skip merge path of
  ``summary_climate_proj`` excludes empty netCDFs and keeps non-empty ones.

Heavy deps (``open_mfdataset``, seaborn ``JointGrid``/``savefig``) run in the
real pixi env per the M02c discipline (dask cannot be stubbed at module level).
"""

from os.path import dirname, join, realpath

import numpy as np
import xarray as xr

TESTDIR = dirname(realpath(__file__))
SNAKEDIR = join(TESTDIR, "..")

from blueearth_cst.projections.get_change_climate_proj_summary import (  # noqa: E402
    plot_change_factor_cloud,
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
# Row D (SUPERSEDED at step 4c): the dummy-skip merge decision is GONE
# --------------------------------------------------------------------------- #
# The design's §7 row D targeted "empty datasets are dropped from the merge",
# implemented by ``filter_nonempty``. That helper existed only because stage A
# wrote an EMPTY netCDF for a source absent from the catalog, which then had to be
# filtered back out at merge.
#
# Since migration step 4a, resolution decides membership at DAG BUILD, so an
# unresolved combination never becomes a job and no placeholder is ever written.
# Step 4c therefore deletes the helper: with no dummies to drop, a filter at merge
# could only silently shrink the ensemble, which is exactly what D4 forbids.
#
# These tests replace the two that pinned the old behaviour. They assert the NEW
# contract -- every file handed to the merge carries data, and the merge does not
# quietly discard anything.
def _write_change_nc(path, model, precip_change, temp_change):
    """Write a minimal *non-empty* annual_change_scalar_stats-style netCDF."""
    ds = xr.Dataset(
        {
            "precip": (
                ("stats", "horizon", "scenario", "model"),
                [[[[precip_change]]]],
            ),
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


def test_filter_nonempty_is_gone():
    """The helper must not come back: a filter at merge can only shrink silently.

    Asserted rather than merely deleted, so a future refactor reintroducing a
    dummy-skip fails here instead of quietly restoring the silent-shrink path.
    """
    import blueearth_cst.projections.get_change_climate_proj_summary as mod

    assert not hasattr(mod, "filter_nonempty"), (
        "filter_nonempty is back. Since step 4a an unresolved combination never "
        "becomes a job, so there are no dummy netCDFs to drop and a filter at "
        "merge would silently shrink the ensemble (D4)."
    )


def test_merge_consumes_every_file_it_is_given(tmp_path):
    """No file is discarded: two models in, two models out."""
    from blueearth_cst.projections.get_change_climate_proj_summary import (
        preprocess_coords,
    )

    a = tmp_path / "annual_change_scalar_stats-A_ssp245_near.nc"
    b = tmp_path / "annual_change_scalar_stats-B_ssp245_near.nc"
    _write_change_nc(a, "A", precip_change=20.0, temp_change=2.0)
    _write_change_nc(b, "B", precip_change=-5.0, temp_change=3.0)

    ds = xr.open_mfdataset(
        [str(a), str(b)], coords="minimal", preprocess=preprocess_coords
    )
    assert sorted(str(m) for m in ds.model.values) == ["A", "B"]


def test_change_factor_cloud_facets_horizons_and_keeps_every_point():
    """Two horizons become two equal-scale panels, with no marginal axes."""
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "model": model,
                "scenario": scenario,
                "member": member,
                "horizon": horizon,
                "precip": precip,
                "temp": temp,
            }
            for horizon, precip, temp in (("near", 10.0, 1.0), ("far", 30.0, 3.0))
            for model, scenario, member in (
                ("A", "ssp245", "r1"),
                ("B", "ssp585", "r2"),
            )
        ]
    )

    fig = plot_change_factor_cloud(
        frame,
        horizons={"near": [2030, 2060], "far": [2070, 2090]},
        scenarios=["ssp245", "ssp585"],
    )
    try:
        assert len(fig.axes) == 2
        assert [
            sum(len(collection.get_offsets()) for collection in ax.collections)
            for ax in fig.axes
        ] == [2, 2]
        assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()
        assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "SSP2-4.5",
            "SSP5-8.5",
        ]
    finally:
        plt.close(fig)
