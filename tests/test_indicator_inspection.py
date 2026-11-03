"""Saved integrity and fresh numerical evidence have different prerequisites."""

import json
import shutil
import subprocess
import sys

import pytest

from blueearth_cst.experiment import indicator_plan
from blueearth_cst.experiment.content_identity import atomic_record, content_sha256
from blueearth_cst.experiment.indicator_artifacts import ImmutableIndicatorSetError
from blueearth_cst.experiment.indicator_inspection import inspect_indicator_set
from blueearth_cst.experiment.indicator_sets import default_calculations
from blueearth_cst.shared.provenance import file_sha256
from tests._v2_experiment import (
    FIXTURE_ENVIRONMENT,
    build_v2_experiment,
    copy_v2_experiment,
)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    experiment = build_v2_experiment(tmp_path_factory.mktemp("inspection"))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
        )
        request = indicator_plan.current_indicator_request(
            experiment.root, default_calculations(["gwr"]), "YE-DEC"
        )
        plan = indicator_plan.build_indicator_plan(experiment.root, request)
        indicator_plan.publish_indicator_set(experiment.root, plan)
    return experiment, plan["indicator_set_id"][:12]


@pytest.fixture
def publication(saved, tmp_path, monkeypatch):
    original, short = saved
    experiment = copy_v2_experiment(original, tmp_path)
    monkeypatch.setattr(
        indicator_plan, "resolve_indicator_environment", lambda: FIXTURE_ENVIRONMENT
    )
    marker = experiment.root / "_engine/indicator_sets" / short / "indicators.json"
    return experiment.root, marker


def test_integrity_import_does_not_load_indicator_registries():
    code = (
        "import sys; import blueearth_cst.experiment.indicator_inspection; "
        "assert 'blueearth_cst.experiment.indicator_registry' not in sys.modules; "
        "assert 'blueearth_cst.indicators.loader' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_pre_rename_publication_is_checked_without_rewriting(publication):
    """Original identities and old CSV headers are verified before translation."""
    root, marker = publication
    original = json.loads(marker.read_text(encoding="utf-8"))
    fields = {
        "indicator": "metric",
        "indicator_set_id": "metric_set_id",
        "indicators_manifest_sha256": "metrics_manifest_sha256",
        "indicator_definition": "metric_definition",
        "indicator_definition_sha256": "metric_definition_sha256",
        "indicator_environment": "metric_environment",
        "indicator_environment_sha256": "metric_environment_sha256",
    }

    def old_record(value):
        if isinstance(value, list):
            return [old_record(item) for item in value]
        if isinstance(value, dict):
            return {
                fields.get(k, k): v.replace("indicator", "metric")
                if k == "schema_version"
                else old_record(v)
                for k, v in value.items()
            }
        return value

    legacy = old_record(original)
    legacy["identity_projection"]["metric_definition_sha256"] = content_sha256(
        legacy["execution_context"]["metric_definition"]
    )
    legacy["metric_set_id"] = content_sha256(legacy["identity_projection"])
    engine_relative = "_engine/metric_sets/" + legacy["metric_set_id"][:12]
    engine = root / engine_relative
    shutil.copytree(marker.parent, engine)
    legacy["engine_root"] = engine_relative
    legacy["result_root"] = engine_relative + "/data"
    for name in ("group_members", "group_metadata"):
        legacy[name]["path"] = legacy["result_root"] + "/" + name + ".csv"
    for table in legacy["tables"]:
        path = engine / "data" / table["file"]["path"].split("/")[-1]
        path.write_bytes(path.read_bytes().replace(b"indicator,", b"metric,", 1))
        table["file"] = {
            "path": legacy["result_root"] + "/" + path.name,
            "sha256": file_sha256(path),
        }
    for key, filename in (
        ("environment", "metric_environment.json"),
        ("diagnostics", "metric_diagnostics.json"),
    ):
        (engine / legacy[key]["path"]).rename(engine / filename)
        legacy[key]["path"] = filename
    legacy["metrics_manifest_sha256"] = content_sha256(
        {k: v for k, v in legacy.items() if k != "metrics_manifest_sha256"}
    )
    old_marker = engine / "metrics.json"
    (engine / "indicators.json").unlink()
    atomic_record(old_marker, legacy)
    before = {p: p.read_bytes() for p in engine.rglob("*") if p.is_file()}
    report = inspect_indicator_set(root, old_marker, recheck=True)
    assert report["integrity"]["status"] == "passed"
    assert report["publication"]["indicator_set_id"] == legacy["metric_set_id"]
    assert report["numerical"]["status"] == "unavailable"
    assert before == {p: p.read_bytes() for p in engine.rglob("*") if p.is_file()}
    damaged = root / legacy["tables"][0]["file"]["path"]
    damaged.write_bytes(damaged.read_bytes() + b"x")
    with pytest.raises(ImmutableIndicatorSetError, match="table differs"):
        inspect_indicator_set(root, old_marker)


@pytest.mark.parametrize("debug", [False, True])
def test_inspection_cli_missing_completion_has_nonzero_exit(tmp_path, debug):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo / "scripts/inspect_indicators.py"),
        "--experiment",
        str(tmp_path),
        "--manifest",
        str(tmp_path / "indicators.json"),
    ]
    if debug:
        command.append("--traceback")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "missing completion receipt" in result.stderr
    assert ("Traceback" in result.stderr) is debug


def test_fresh_recheck_and_historical_binding(publication):
    root, marker = publication
    result = inspect_indicator_set(root, marker, recheck=True)
    assert result["numerical"]["status"] == "passed", result
    history = inspect_indicator_set(root, marker)["historical_checks"]
    assert len(history) == 1 and history[0]["label"] == "historical"
    unrelated = dict(history[0]["receipt"], publication={"indicator_set_id": "other"})
    atomic_record(marker.parent / "verification/unrelated.json", unrelated)
    comparison = dict(
        history[0]["receipt"], schema_version="comparison/1", kind="comparison"
    )
    atomic_record(marker.parent / "verification/comparison.json", comparison)
    assert len(inspect_indicator_set(root, marker)["historical_checks"]) == 1


def test_corruption_and_missing_completion_fail_integrity(publication):
    root, marker = publication
    artifact = marker.parent / "data/gwr_run_indicators.csv"
    artifact.write_bytes(artifact.read_bytes() + b"x")
    with pytest.raises(ImmutableIndicatorSetError, match="table differs"):
        inspect_indicator_set(root, marker)
    marker.unlink()
    with pytest.raises(ImmutableIndicatorSetError, match="missing completion receipt"):
        inspect_indicator_set(root, marker)


@pytest.mark.parametrize(
    "change", ["missing-inventory", "missing-input", "changed-input"]
)
def test_missing_input_keeps_integrity_and_blocks_requested_recheck(
    publication, change
):
    root, marker = publication
    if change == "missing-inventory":
        (root / "_engine/response_inventory.json").unlink()
    else:
        from tests._v2_experiment import native_runs

        native = next(iter(native_runs(root).values())).csv_path
        if change == "missing-input":
            native.unlink()
        else:
            native.write_bytes(native.read_bytes() + b"\n")
    assert inspect_indicator_set(root, marker)["integrity"]["status"] == "passed"
    result = inspect_indicator_set(root, marker, recheck=True)
    assert result["numerical"]["status"] == "unavailable"


def test_dependency_drift_keeps_integrity_and_blocks_recheck(publication, monkeypatch):
    root, marker = publication
    monkeypatch.setattr(
        indicator_plan,
        "resolve_indicator_environment",
        lambda: {"packages": {"numpy": "changed"}},
    )
    result = inspect_indicator_set(root, marker, recheck=True)
    assert result["integrity"]["status"] == "passed"
    assert "dependency versions differ" in result["numerical"]["reason"]


def test_checker_sources_do_not_enter_execution_identity(publication):
    _, marker = publication
    saved = json.loads(marker.read_text(encoding="utf-8"))
    assert (
        "configured water-year anchor"
        in saved["execution_context"]["set_assumptions"]["default"]["annual_blocks"]
    )
    paths = {
        row["path"] for row in saved["execution_context"]["request"]["code_inventory"]
    }
    assert any(path.endswith("sets/default/calculations.py") for path in paths)
    assert not any(path.endswith(("checks.py", "checking.py")) for path in paths)


def test_failed_numerical_check_is_distinct_from_ineligibility(
    publication, monkeypatch
):
    from blueearth_cst.indicators import loader

    root, marker = publication

    def wrong_answer(*args, **kwargs):
        raise ValueError(
            "default/gwr_annual_total/location=9: deliberate numerical mismatch"
        )

    monkeypatch.setattr(loader, "check", wrong_answer)
    result = inspect_indicator_set(root, marker, recheck=True)
    assert result["integrity"]["status"] == "passed"
    assert result["numerical"]["status"] == "failed"
    with pytest.raises(ValueError, match="deliberate numerical mismatch"):
        inspect_indicator_set(root, marker, recheck=True, traceback=True)


def test_changed_execution_source_blocks_numerical_only(
    publication, tmp_path, monkeypatch
):
    from pathlib import Path

    from blueearth_cst.experiment import indicator_inspection

    root, marker = publication
    document = json.loads(marker.read_text(encoding="utf-8"))
    repo = Path(indicator_inspection.__file__).resolve().parents[2]
    copied_repo = tmp_path / "execution-copy"
    for source in document["execution_context"]["request"]["code_inventory"]:
        destination = copied_repo / source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / source["path"], destination)
    changed = copied_repo / "blueearth_cst/indicators/sets/default/calculations.py"
    changed.write_bytes(changed.read_bytes() + b"\n# deliberate execution edit\n")
    monkeypatch.setattr(
        indicator_inspection,
        "__file__",
        str(copied_repo / "blueearth_cst/experiment/indicator_inspection.py"),
    )
    result = inspect_indicator_set(root, marker, recheck=True)
    assert result["integrity"]["status"] == "passed"
    assert "execution source changed" in result["numerical"]["reason"]


def test_changed_checker_gets_fresh_attribution(publication, monkeypatch):
    from pathlib import Path

    from blueearth_cst.indicators.sets.default import checks

    root, marker = publication
    first = inspect_indicator_set(root, marker, recheck=True)
    assert first["numerical"]["status"] == "passed", first
    checker = Path(checks.__file__)
    original_read = Path.read_bytes

    def changed_bytes(path):
        data = original_read(path)
        return data + b"\n# changed checker evidence\n" if path == checker else data

    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    second = inspect_indicator_set(root, marker, recheck=True)
    assert second["numerical"]["status"] == "passed", second
    before = json.loads(Path(first["numerical"]["receipt"]).read_text(encoding="utf-8"))
    after = json.loads(Path(second["numerical"]["receipt"]).read_text(encoding="utf-8"))
    assert before["checker_sources"] != after["checker_sources"]


def test_old_publication_inspection_without_rewriting_inputs(publication):
    root, marker = publication
    document = json.loads(marker.read_text(encoding="utf-8"))
    document.pop("execution_context")
    for field in ("unavailable_calculations", "unavailable_results"):
        document.pop(field)
    document["schema_version"] = "indicator-set/4"
    document["identity_projection"]["schema_version"] = "indicator-set-identity/4"
    document["indicator_set_id"] = content_sha256(document["identity_projection"])
    old_engine = marker.parent.with_name(document["indicator_set_id"][:12])
    old_engine_relative = old_engine.relative_to(root).as_posix()
    original_root = document["engine_root"]
    # Re-identify only this disposable fixture to exercise the genuine old schema.
    document = json.loads(
        json.dumps(document).replace(original_root, old_engine_relative)
    )
    marker.parent.rename(old_engine)
    # Sets before indicator-set/6 carry four-column tables.
    for item in document["tables"]:
        path = root / item["file"]["path"]
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(
            "".join(",".join(line.split(",")[:4]) + "\n" for line in lines),
            encoding="utf-8",
        )
        item["file"]["sha256"] = file_sha256(path)
    document.pop("indicators_manifest_sha256")
    document["indicators_manifest_sha256"] = content_sha256(document)
    old_marker = old_engine / "indicators.json"
    atomic_record(old_marker, document, replace=True)
    before = old_marker.read_bytes()
    report = inspect_indicator_set(root, old_marker, recheck=True)
    assert report["integrity"]["status"] == "passed"
    assert "legacy publication" in report["numerical"]["reason"]
    assert old_marker.read_bytes() == before
