"""Method-revision pins: every closure edit is recorded as a bump or as no numerical change."""

import json

import pytest

from blueearth_cst.experiment import method_revisions
from blueearth_cst.experiment.content_identity import content_sha256
from blueearth_cst.experiment.method_revisions import (
    PIN_PATH,
    PRODUCERS,
    MethodPinMismatch,
    closure_drift,
    drift_message,
    producer_closure,
    read_pins,
    record_pin,
)


@pytest.mark.parametrize("producer", PRODUCERS)
def test_pin_matches_code(producer):
    pin = read_pins()["producers"][producer]
    files = producer_closure(producer)
    assert content_sha256(files) == pin["closure_sha256"], drift_message(
        producer, closure_drift(pin, files)
    )
    assert pin["last_record"]["kind"] in ("bump", "no-numerical-change")


def test_pin_file_is_in_no_closure():
    relative = PIN_PATH.relative_to(method_revisions.REPO_ROOT).as_posix()
    for producer in PRODUCERS:
        assert relative not in {item["path"] for item in producer_closure(producer)}


def _pins_with_stale(tmp_path, producer):
    document = read_pins()
    document["producers"][producer]["files"][0]["sha256"] = "0" * 64
    document["producers"][producer]["closure_sha256"] = content_sha256(
        document["producers"][producer]["files"]
    )
    path = tmp_path / "method_revisions.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document["producers"][producer]["files"][0]["path"]


def test_mismatch_names_file_and_command(tmp_path, monkeypatch):
    path, stale = _pins_with_stale(tmp_path, "delta-collection")
    monkeypatch.setattr(method_revisions, "PIN_PATH", path)
    method_revisions.method_revision.cache_clear()
    try:
        with pytest.raises(MethodPinMismatch) as error:
            method_revisions.method_revision.__wrapped__("delta-collection")
    finally:
        method_revisions.method_revision.cache_clear()
    assert f"changed: {stale}" in str(error.value)
    assert "record_method_pins.py --producer delta-collection" in str(error.value)


def test_record_keeps_or_bumps_revision(tmp_path):
    path, _ = _pins_with_stale(tmp_path, "delta-collection")
    kept = record_pin("delta-collection", "no-numerical-change", "wording", path)
    revision = read_pins()["producers"]["delta-collection"]["revision"]
    assert kept["revision"] == revision
    assert kept["closure_sha256"] == content_sha256(
        producer_closure("delta-collection")
    )
    bumped = record_pin("delta-collection", "bump", "new factor rule", path)
    assert bumped["revision"] == revision + 1
    assert bumped["last_record"] == {"kind": "bump", "reason": "new factor rule"}
    with pytest.raises(ValueError, match="reason"):
        record_pin("delta-collection", "bump", " ", path)
