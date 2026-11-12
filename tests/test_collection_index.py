"""Scenario collection names, identity lookup and the rebuildable index."""

import csv

import pytest

from blueearth_cst.experiment import collection_index as ci
from blueearth_cst.experiment.content_identity import atomic_record


@pytest.mark.parametrize(
    "name",
    [
        "",
        "_engine",
        "collections.csv",
        "CON",
        "lpt1.x",
        "a/b",
        "trail.",
        "deadbeef00",
        "-x",
    ],
)
def test_unsafe_names_are_refused(name):
    with pytest.raises(ValueError):
        ci.validate_collection_name(name)


def test_name_allocation_suffixes_past_any_existing_entry(tmp_path):
    scenarios = tmp_path / "scenarios"
    (scenarios / "baseline").mkdir(parents=True)
    (scenarios / "BASELINE-2").write_text("leftover", encoding="utf-8")
    assert ci.allocate_collection_name(tmp_path, "baseline") == "baseline-3"
    assert ci.allocate_collection_name(tmp_path, "other") == "other"


def _marker(folder, identity):
    (folder / "_engine").mkdir(parents=True)
    atomic_record(folder / "_engine" / "collection.json", {"collection_id": identity})


def test_lookup_is_by_identity_across_layouts(tmp_path):
    scenarios = tmp_path / "scenarios"
    _marker(scenarios / "named", "a" * 64)
    legacy = scenarios / "_engine" / "collections" / ("b" * 12)
    legacy.mkdir(parents=True)
    atomic_record(legacy / "collection.json", {"collection_id": "b" * 64})
    assert ci.find_ready_marker(tmp_path, "a" * 64).parent.parent.name == "named"
    assert ci.find_ready_marker(tmp_path, "b" * 64).parent == legacy
    assert ci.find_ready_marker(tmp_path, "c" * 64) is None


def test_index_rebuilds_from_markers_and_keeps_aliases(tmp_path):
    scenarios = tmp_path / "scenarios"
    _marker(scenarios / "first", "a" * 64)
    (scenarios / "broken" / "_engine").mkdir(parents=True)
    index = ci.rebuild_collection_index(tmp_path, alias=("a" * 64, "renamed"))
    rows = {row["name"]: row for row in csv.DictReader(index.open(encoding="utf-8"))}
    assert rows["first"]["aliases"] == "renamed"
    assert rows["first"]["status"] == "invalid"  # bare test marker fails the reader
    assert rows["broken"]["status"] == "incomplete"
    index.unlink()
    ci.rebuild_collection_index(tmp_path)
    assert (scenarios / "collections.csv").is_file()
    # A configured alias also blocks a new set from taking that name.
    ci.rebuild_collection_index(tmp_path, alias=("a" * 64, "renamed"))
    assert ci.allocate_collection_name(tmp_path, "renamed") == "renamed-2"
