"""Retained responses are rounded once, verified, and readable without raw output."""

import json

import numpy as np
import pytest

from blueearth_cst.experiment.wflow_response_reader import ResponseRequest
from blueearth_cst.shared.csv_output import normalize_csv_output
from blueearth_cst.shared.response_tables import (
    open_table_responses,
    publish_response_tables,
    read_table_record,
)
from tests.test_wflow_response_reader import native as native


@pytest.mark.parametrize("places", [0, 5, 8, 15])
def test_compact_response_precision_missing_and_native_order(native, tmp_path, places):
    text = native.csv_path.read_text().replace("1,2,3,5", "1.123456789,0.000000019,3,5")
    native.csv_path.write_text(text, encoding="utf-8")
    record_path = publish_response_tables(
        native, tmp_path / "retained", decimal_places=places
    )
    record = read_table_record(record_path)
    native.csv_path.unlink()
    native.toml_path.unlink()
    responses = open_table_responses("run", record_path, ResponseRequest(("q", "gwr")))
    assert [(r.variable, r.location_id, r.location_ordinal) for r in responses] == [
        ("q", "9", 0),
        ("q", "2", 1),
        ("gwr", "9", 0),
    ]
    assert responses[0].values[0] == round(1.123456789, places)
    assert responses[1].values[0] == round(0.000000019, places)
    assert responses[1].missing.tolist() == [False, True]
    assert record["decimal_places"] == places
    assert all(np.isfinite(r.values[~r.missing]).all() for r in responses)


def test_compact_response_corruption_refuses(native, tmp_path):
    path = publish_response_tables(native, tmp_path / "retained")
    record = json.loads(path.read_text())
    table = path.parent.parent / record["variables"]["q"]["path"]
    table.write_text("time,9\n2046-01-02,999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        open_table_responses("run", path, ResponseRequest(("q",)))
    assert native.csv_path.is_file()


@pytest.mark.parametrize("value", [True, -1, 16, 5.0, "5", None])
def test_csv_precision_requires_bounded_integer(value):
    with pytest.raises(ValueError, match="integer"):
        normalize_csv_output({"csv": {"decimal_places": value}})


def test_csv_defaults_and_closed_fields():
    assert normalize_csv_output(None) == {"decimal_places": 5}
    with pytest.raises(ValueError, match="only decimal_places"):
        normalize_csv_output({"csv": {"retain_native": True}})
