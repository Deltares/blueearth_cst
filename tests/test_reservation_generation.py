"""Bound WF3 plans remain immutable across continuation and incomplete generation."""

from copy import deepcopy

import pytest

from blueearth_cst.experiment import (
    generation_plan,
    reservation,
    reservation_generation,
)
from blueearth_cst.experiment.content_identity import content_sha256, identity_segment


@pytest.fixture
def bound(tmp_path, monkeypatch):
    plan = {
        "request": {"seed": 123},
        "candidate_identity_digests": {"source_inventory": "source"},
        "intent": {"creator": "retained"},
        "outputs": {
            "record_root": "records/collection",
            "data_root": "data/collection",
        },
        "decision": "create",
        "collection_id": "collection",
        "collection_revision": None,
    }
    record = {
        "seed": {
            "state": "bound",
            "plan": {"path": "plans/original.json", "sha256": "digest"},
            "seed_resolution": {"resolved": 123},
        },
        "resolution": None,
        "request": {"selection": {"mode": "project-generation"}, "simulation": {}},
    }
    candidate = {
        "identity_digests": deepcopy(plan["candidate_identity_digests"]),
        "documents": {"generation_config": {"seed_resolution": {"resolved": 123}}},
    }
    events = []
    monkeypatch.setattr(reservation, "read_reservation", lambda *args: record)
    monkeypatch.setattr(
        reservation_generation,
        "generation_configuration",
        lambda *args: {"request": plan["request"]},
    )
    monkeypatch.setattr(
        reservation_generation, "build_candidate_intent", lambda *args: candidate
    )
    monkeypatch.setattr(reservation_generation, "read_pinned_plan", lambda *args: plan)
    monkeypatch.setattr(
        reservation_generation,
        "publish_plan_pointer",
        lambda *args: events.append("pointer"),
    )
    monkeypatch.setattr(
        reservation_generation,
        "select_generation_plan",
        lambda *args: pytest.fail("must not select a new plan"),
    )
    return tmp_path, plan, record, candidate, events


def test_bound_create_without_initialization_reuses_original_plan(bound):
    root, plan, _, _, events = bound
    result = reservation_generation.select_reserved_plan_owned(
        "config", "repo", root, "stress_test_01"
    )
    assert result[1] is plan
    assert result[3:] == ("digest", False)
    assert events == ["pointer"]


@pytest.mark.parametrize("namespace", ["record_root", "data_root"])
def test_partial_namespace_refuses_without_pointer_repair(bound, namespace):
    root, plan, _, _, events = bound
    (root / plan["outputs"][namespace]).mkdir(parents=True)
    with pytest.raises(ValueError, match="PartialWF3Initialization"):
        reservation_generation.select_reserved_plan_owned(
            "config", "repo", root, "stress_test_01"
        )
    assert events == []


def test_initialization_receipt_distinguishes_incomplete_generation(bound):
    root, _, _, _, events = bound
    receipt = root / "plans/initializations/creator.json"
    receipt.parent.mkdir(parents=True)
    generation_plan.atomic_record(receipt, {"plan_sha256": "digest"})
    with pytest.raises(ValueError, match="IncompleteWF3:"):
        reservation_generation.select_reserved_plan_owned(
            "config", "repo", root, "stress_test_01"
        )
    assert events == []


def test_bound_source_drift_refuses_before_pointer_write(bound):
    root, _, _, candidate, events = bound
    candidate["identity_digests"]["source_inventory"] = "changed"
    with pytest.raises(ValueError):
        reservation_generation.select_reserved_plan_owned(
            "config", "repo", root, "stress_test_01"
        )
    assert events == []


def test_bound_missing_resolved_collection_is_never_recreated(bound):
    root, _, record, _, events = bound
    record["resolution"] = {"collection_id": "collection"}
    with pytest.raises(ValueError, match="MissingWF3Collection"):
        reservation_generation.select_reserved_plan_owned(
            "config", "repo", root, "stress_test_01"
        )
    assert events == []


def test_ready_bound_create_bypasses_generation_and_keeps_original_plan(
    bound, monkeypatch
):
    root, plan, _, _, events = bound
    marker_path = root / "records/collection/collection.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        reservation_generation,
        "read_collection_v2",
        lambda path: {"collection_id": "collection", "collection_revision": "revision"},
    )
    monkeypatch.setattr(
        reservation_generation, "read_canonical_json", lambda path: plan["intent"]
    )
    monkeypatch.setattr(
        reservation_generation,
        "validate_installed_generator",
        lambda *args, **kwargs: events.append("validated"),
    )
    monkeypatch.setattr(
        reservation_generation,
        "bind_ready_collection_owned",
        lambda *args: events.append("bound"),
    )
    result = reservation_generation.select_reserved_plan_owned(
        "config", "repo", root, "stress_test_01"
    )
    assert result[1] is plan and result[4] is True
    assert plan["decision"] == "create"
    assert events == ["validated", "bound", "pointer"]


@pytest.mark.parametrize("field", ["collection_id", "collection_revision"])
def test_ready_binding_rejects_plan_for_another_collection(bound, monkeypatch, field):
    root, plan, _, _, events = bound
    marker_path = root / "records/collection/collection.json"
    marker_path.parent.mkdir(parents=True)
    marker_path.write_bytes(b"{}\n")
    marker = {
        "collection_id": "collection",
        "collection_revision": "revision",
        "series": [{"run_id": "run_1"}],
    }
    plan[field] = "different"
    monkeypatch.setattr(reservation_generation, "read_collection_v2", lambda _: marker)
    monkeypatch.setattr(reservation_generation, "file_reference", lambda *args: {})
    monkeypatch.setattr(
        reservation,
        "bind_reservation_owned",
        lambda *args, **kwargs: events.append("bound"),
    )
    with pytest.raises(
        ValueError, match="ready collection differs from selected generation plan"
    ):
        reservation_generation.bind_ready_collection_owned(
            root,
            "stress_test_01",
            marker_path,
            generation_plan={"path": "plans/original.json", "sha256": "digest"},
        )
    assert events == []


@pytest.mark.parametrize(
    "state",
    ["ready", "changed candidate", "changed seed", "changed request", "unready"],
)
def test_relocated_request_reuses_only_exact_ready_bound_collection(
    bound, monkeypatch, state
):
    root, plan, record, candidate, events = bound
    plan["request"] = {
        "template": "old/template.yml",
        "settings": {"weathergen_config": "old/template.yml", "seed": 123},
        "source": {"catalogs": ["old/catalog.yml"], "store": "same/store"},
    }
    requested = deepcopy(plan["request"])
    requested["template"] = "new/template.yml"
    requested["settings"]["weathergen_config"] = "new/template.yml"
    requested["source"]["catalogs"] = ["new/catalog.yml"]
    if state == "changed candidate":
        candidate["identity_digests"]["source_inventory"] = "different"
    elif state == "changed seed":
        candidate["documents"]["generation_config"]["seed_resolution"]["resolved"] = 456
    elif state == "changed request":
        requested["source"]["store"] = "different/store"
    marker_path = root / "records/collection/collection.json"
    if state != "unready":
        marker_path.parent.mkdir(parents=True)
        marker_path.write_text("{}", encoding="utf-8")
    request_id = content_sha256(requested)
    pointer = (
        root
        / "requests"
        / identity_segment(request_id, "generation_request_id")
        / "request.json"
    )
    settings = {"request": requested, "request_path": str(pointer)}
    monkeypatch.setattr(
        reservation_generation, "generation_configuration", lambda *args: settings
    )
    monkeypatch.setattr(
        reservation_generation,
        "read_collection_v2",
        lambda path: {"collection_id": "collection", "collection_revision": "revision"},
    )
    monkeypatch.setattr(
        reservation_generation, "read_canonical_json", lambda path: plan["intent"]
    )
    monkeypatch.setattr(
        reservation_generation,
        "validate_installed_generator",
        lambda *args, **kwargs: events.append("validated"),
    )
    monkeypatch.setattr(
        reservation_generation,
        "bind_ready_collection_owned",
        lambda *args: events.append("bound"),
    )

    def reuse(actual_settings, actual_candidate, actual_marker):
        assert actual_settings is settings
        assert actual_candidate is candidate
        assert actual_marker == marker_path
        events.append("reuse exact marker")
        return {
            **deepcopy(plan),
            "request": requested,
            "decision": "reuse_ready",
            "generation_request_id": request_id,
            "collection_revision": "revision",
        }

    monkeypatch.setattr(reservation_generation, "build_reuse_plan", reuse)
    monkeypatch.setattr(
        reservation_generation,
        "publish_plan_pointer",
        generation_plan.publish_plan_pointer,
    )
    original_record, original_plan = deepcopy(record), deepcopy(plan)
    if state == "ready":
        result = reservation_generation.select_reserved_plan_owned(
            "config", "repo", root, "stress_test_01"
        )
        assert result[1]["decision"] == "reuse_ready"
        assert result[2].is_file() and result[4] is True
        assert result[3] == content_sha256(result[1])
        assert result[1]["request"] == requested
        assert pointer.is_file()
        assert events == ["validated", "bound", "reuse exact marker"]
    else:
        with pytest.raises(ValueError):
            reservation_generation.select_reserved_plan_owned(
                "config", "repo", root, "stress_test_01"
            )
        assert not pointer.exists()
        assert events == []
    assert record == original_record
    assert plan == original_plan
