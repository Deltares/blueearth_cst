"""Rule 1.11: outlet and user-gauge discharge series share ONE id namespace.

A gauge that sits on the basin outlet appears in BOTH ``Q_outlets`` and
``Q_gauges_<name>`` under the SAME ``index`` — the outlet's subcatchment id is
that gauge's ``wflow_id`` whenever the model was delineated as a subbasin at
the gauge, which is the normal way to build one. The two series then disagree
about ``station_name`` (synthetic ``wflow_1`` vs the user's own name) and
``xr.merge`` raises ``MergeError: conflicting values for variable
'station_name'``. Observed 2026-08-02 on a real basin (gauge 101, "outlet"),
where it killed rule 1.11 outright after the 2026-08-01 fix to
``blueearth_cst/shared/gauges.py`` made the gauge series resolvable in the
first place — the collision had been masked by the gauges never being found.

The outlet label must WIN the collision: rule 1.11 declares
``hydro_wflow_1.png`` and ``clim_wflow_1_{month,year}.png`` as Snakemake
outputs and every figure is written as ``<kind>_{station_name}.png``, so a
user name winning on the first outlet trades the MergeError for a
MissingOutputException.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from blueearth_cst.model.plot_results import merge_outlet_and_gauge_series


def _series(index, station_names, seed=0.0):
    """Discharge on dim ``index`` with a ``station_name`` coordinate."""
    time = pd.date_range("2000-01-01", periods=4, freq="D")
    data = seed + np.arange(len(time) * len(index), dtype=float).reshape(
        len(time), len(index)
    )
    return xr.DataArray(
        data,
        dims=("time", "index"),
        coords={
            "time": time,
            "index": np.asarray(index),
            "station_name": ("index", list(station_names)),
        },
        name="Q",
    )


def test_disjoint_gauges_are_added_to_the_outlet_stations():
    """The ordinary case: no collision, every station survives with its name."""
    outlets = _series([101], ["wflow_1"])
    gauges = _series([102, 103], ["mainstem_mid", "north_tributary"])

    merged = merge_outlet_and_gauge_series(outlets, gauges, log=lambda msg: None)

    assert list(merged["index"].values) == [101, 102, 103]
    assert dict(zip(merged["index"].values, merged["station_name"].values)) == {
        101: "wflow_1",
        102: "mainstem_mid",
        103: "north_tributary",
    }


def test_gauge_on_the_outlet_keeps_the_outlet_label():
    """The 2026-08-02 crash: gauge 101 IS the outlet. No raise, no rename."""
    outlets = _series([101], ["wflow_1"])
    gauges = _series(
        [101, 102, 103, 104],
        ["outlet", "mainstem_mid", "north_tributary", "south_tributary"],
    )

    merged = merge_outlet_and_gauge_series(outlets, gauges, log=lambda msg: None)

    assert list(merged["index"].values) == [101, 102, 103, 104]
    # The declared output hydro_wflow_1.png is named after THIS label.
    assert merged["station_name"].sel(index=101).item() == "wflow_1"
    assert merged["station_name"].sel(index=104).item() == "south_tributary"
    # The outlet series is the one that survives, values included.
    xr.testing.assert_allclose(
        merged.sel(index=101, drop=True), outlets.sel(index=101, drop=True)
    )


def test_gauge_on_the_outlet_is_reported_not_dropped_in_silence():
    """A station the user named vanishes from the filenames; say so."""
    outlets = _series([101], ["wflow_1"])
    gauges = _series([101], ["outlet"])
    messages = []

    merged = merge_outlet_and_gauge_series(outlets, gauges, log=messages.append)

    # Sole gauge collides: nothing left to merge, and the outlets pass through.
    assert list(merged["index"].values) == [101]
    assert merged["station_name"].sel(index=101).item() == "wflow_1"
    # Exact, not substring: the ids and names come off numpy arrays, whose
    # scalars stringify as np.int32(101) / np.str_('outlet') if they reach the
    # message unconverted. A substring check passes on that noise.
    assert messages == [
        "Gauge(s) [101] (outlet) sit on a model outlet and are already in "
        "Q_outlets; plotted under the outlet label(s) ['wflow_1']."
    ]
