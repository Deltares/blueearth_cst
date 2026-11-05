"""Neutral response metadata and bundle compatibility refusals."""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from blueearth_cst.experiment.response_series import (
    ResponseSeries,
    compatible_bundle,
    response_frame,
    validate_responses,
)


def sample(run="01", location="a", ordinal=0):
    return ResponseSeries(
        run,
        "q",
        location,
        ordinal,
        pd.date_range("2046-01-02", periods=3),
        "proleptic_gregorian",
        timedelta(days=1),
        "interval_end",
        "m3/s",
        np.array([1.0, 2.0, np.nan]),
        np.array([False, False, True]),
    )


def test_neutral_series_preserves_missingness_and_native_order():
    a, b = sample(), sample(location="b", ordinal=1)
    validate_responses([b, a], evaluated_runs={"01"}, required_keys={a.key, b.key})
    frame = response_frame([b, a])
    assert list(frame) == ["a", "b"]
    assert frame.iloc[-1].isna().all()
    compatible_bundle([a, sample("02")])


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "unevaluated",
        "missing_response",
        "ordinal",
        "time",
        "calendar",
        "units",
        "mask",
        "infinite",
        "length",
    ],
)
def test_response_contract_refuses(defect):
    a = sample()
    items, evaluated, required = [a], {"01"}, {a.key}
    if defect == "duplicate":
        items.append(a)
    elif defect == "unevaluated":
        evaluated = set()
    elif defect == "missing_response":
        required.add(("01", "q", "absent"))
    elif defect == "ordinal":
        items = [replace(a, location_ordinal=1)]
    elif defect == "time":
        items = [replace(a, time=a.time[[0, 0, 2]])]
    elif defect == "calendar":
        items = [replace(a, calendar="noleap")]
    elif defect == "units":
        items = [replace(a, units="")]
    elif defect == "mask":
        items = [replace(a, missing=np.zeros(3, dtype=bool))]
    elif defect == "infinite":
        items = [replace(a, values=np.array([1.0, np.inf, np.nan]))]
    else:
        items = [replace(a, values=np.ones(2))]
    with pytest.raises(ValueError):
        validate_responses(items, evaluated_runs=evaluated, required_keys=required)


@pytest.mark.parametrize(
    "change",
    [
        {"units": "mm"},
        {"calendar": "gregorian"},
        {"time_label": "instant"},
        {"time": pd.date_range("2047-01-02", periods=3)},
    ],
)
def test_bundle_refuses_unreconciled_metadata(change):
    with pytest.raises(ValueError, match="incompatible"):
        compatible_bundle([sample(), replace(sample("02"), **change)])
