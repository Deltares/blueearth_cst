"""Stable scenario copies preserve immutable collections and recover failed writes."""

import json

import pytest

from blueearth_cst.experiment import scenario_export
from blueearth_cst.experiment.content_identity import canonical_json_bytes
from tests._v2_experiment import build_v2_experiment, copy_v2_experiment


@pytest.fixture(scope="module", params=[False, True], ids=["engine-data", "legacy"])
def built(tmp_path_factory, request):
    return build_v2_experiment(
        tmp_path_factory.mktemp("scenario-export"), legacy_storage=request.param
    )


@pytest.fixture
def case(built, tmp_path):
    return copy_v2_experiment(built, tmp_path)


def _export(case):
    scenario_export.export_scenario_collection(case.project, case.collection_marker)


def _snapshot(root):
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_export_copies_entire_collection_and_reuse_does_not_rewrite(case):
    marker = json.loads(case.collection_marker.read_text())
    source = case.project / marker["data_root"]
    # Include a product outside the explicit ready-marker lists, as a full mirror must.
    extra = source / "weathergenr/evaluation/plots/additional.png"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"diagnostic")
    (source / "weathergenr/empty").mkdir()
    before = _snapshot(source)
    _export(case)
    destination = case.project / "scenarios"
    if marker["data_root"].startswith("scenarios/_engine/data/"):
        assert not (destination / case.collection_marker.parent.name).exists()
    record = json.loads((destination / "_engine/current.json").read_text())
    assert set(record["files"]) == {p.relative_to(source).as_posix() for p in before}
    assert {
        "scenario_run_lookup.csv",
        "perturbation_lookup.csv",
        "series/run_1.nc",
        "config/run_record.yml",
        "weathergenr/evaluation/plots/additional.png",
    } <= set(record["files"])
    for name in record["files"]:
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert {
        p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_dir()
    } == set(record["directories"])
    assert (destination / "weathergenr/empty").is_dir()
    assert before == _snapshot(source)
    exported = _snapshot(destination)
    _export(case)
    assert exported == _snapshot(destination)
    (destination / "series/run_1.nc").unlink()
    _export(case)
    assert (destination / "series/run_1.nc").read_bytes() == (
        source / "series/run_1.nc"
    ).read_bytes()


def test_only_previously_managed_obsolete_files_are_removed(case):
    _export(case)
    root = case.project / "scenarios"
    tracking = root / "_engine/current.json"
    record = json.loads(tracking.read_text())
    (root / "series/run_99.nc").write_bytes(b"old managed export")
    record["files"]["series/run_99.nc"] = "0" * 64
    tracking.write_bytes(canonical_json_bytes(record))
    (root / "series/custom.nc").write_bytes(b"user series")
    _export(case)
    assert not (root / "series/run_99.nc").exists()
    assert (root / "series/custom.nc").read_bytes() == b"user series"


@pytest.mark.parametrize(
    "failure", ["source", "copy", "replace", "metadata", "unowned", "traversal"]
)
def test_export_failure_preserves_previous_outputs(case, monkeypatch, failure):
    _export(case)
    root = case.project / "scenarios"
    tracking = root / "_engine/current.json"
    (root / "series/run_1.nc").write_bytes(b"previous export")
    if failure == "unowned":
        tracking.unlink()
    elif failure == "traversal":
        record = json.loads(tracking.read_text())
        record["files"]["../unrelated.txt"] = "0" * 64
        tracking.write_bytes(canonical_json_bytes(record))
    paths = [
        root / "scenario_run_lookup.csv",
        root / "perturbation_lookup.csv",
        tracking,
        *sorted((root / "series").iterdir()),
    ]
    before = {p: p.read_bytes() for p in paths if p.exists()}

    def fail(*args, **kwargs):
        raise OSError("injected export failure")

    if failure == "source":
        marker = json.loads(case.collection_marker.read_text())
        (case.project / marker["data_root"] / "series/run_1.nc").write_bytes(
            b"corrupt source"
        )
    elif failure == "copy":
        monkeypatch.setattr(scenario_export.shutil, "copyfile", fail)
    elif failure == "metadata":
        monkeypatch.setattr(scenario_export, "atomic_record", fail)
    elif failure == "replace":
        replace = scenario_export.os.replace
        count = 0

        def fail_second(source, target):
            nonlocal count
            count += 1
            if count == 2:
                fail()
            return replace(source, target)

        monkeypatch.setattr(scenario_export.os, "replace", fail_second)
    with pytest.raises((OSError, ValueError)):
        _export(case)
    assert before == {p: p.read_bytes() for p in before}
    assert not list((root / "_engine").glob(".export-*"))
