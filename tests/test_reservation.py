"""Synthetic allocation and publication acceptance checks."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from blueearth_cst.experiment.content_identity import atomic_record, read_canonical_json
from blueearth_cst.shared.workflow_config_snapshot import _replace, archive_lock


@pytest.fixture(scope="session")
def api():
    from blueearth_cst.experiment import reservation

    return reservation


@pytest.fixture(scope="session")
def helper():
    from tests import _v2_experiment

    return _v2_experiment


@pytest.fixture
def prepared(api, helper, tmp_path):
    intent = helper.collection_intent_v2()
    material = intent["documents"]["generation_config"]["seed_resolution"]["projection"]
    generation = {
        "schema_version": "reservation-generation-request/1",
        **{
            key: material[key]
            for key in (
                "n_realizations",
                "simulation_window",
                "climate_perturbations",
                "water_year_start",
                "generator_settings",
            )
        },
        "requested_seed": "auto",
        "run_group_id_capacity": 2,
        "provider_code": intent["documents"]["provider_code"],
        "source_request": {
            "region": {"bbox": [0, 0, 1, 1]},
            "climate_source": "era5",
            "historical_window": {"start": 1980, "end": 2016},
            "hydrography": "merit",
            "basin_index": "basins",
            "local_inputs": [],
        },
    }
    request = {
        "schema_version": "experiment-request/1",
        "generation": generation,
        "selection": {"mode": "project-generation", "manifest": None},
        "input_catalogs": [],
        "simulation": {
            "climate_source": "era5",
            "model_root": "model/wflow",
            "outvars": ["gwr"],
            "model_request": None,
        },
    }
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = []
    for role in (
        "project",
        "simulate_system",
        "generate_scenarios",
        "generator_template",
    ):
        path = source_dir / f"{role}.yml"
        path.write_bytes(f"role: {role}\n".encode())
        sources.append(api.CapturedSource(role, path, path.read_bytes()))
    return tmp_path / "project", request, sources


_WORKER = """# Independent fault/contended creator; never a workflow producer.

from blueearth_cst.experiment import reservation as api
import json
import os
import sys
import time
import uuid
from pathlib import Path

from blueearth_cst.shared.workflow_config_snapshot import archive_lock


def main():
    root, input_path, mode, fault = sys.argv[1:]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(input_path).read_bytes())
    sources = [
        api.CapturedSource(s["role"], Path(s["path"]), s["data"].encode())
        for s in payload["sources"]
    ]

    # Prepare the scientific request before the contender barrier, as the
    # runner does. Cold scientific imports must not consume the Windows
    # lock-acquisition retry window; admission still revalidates while owned.
    api.validate_request(payload["request"], root)

    if mode in {"automatic", "same-pin"}:
        ready = Path(input_path).parent / f"creator-ready-{os.getpid()}"
        ready.write_bytes(b"ready")
        deadline = time.monotonic() + 20
        while len(list(ready.parent.glob("creator-ready-*"))) < 2:
            if time.monotonic() > deadline:
                raise RuntimeError("second independent creator never became ready")
            time.sleep(0.02)

    def failpoint(boundary):
        if boundary == fault:
            os._exit(71)
        if boundary == "transaction_written" and mode in {"automatic", "same-pin"}:
            time.sleep(0.4)

    try:
        with archive_lock(root, "project-execution"):
            selected = api.admit_experiment_owned(
                root,
                "pinned_37" if mode == "same-pin" else None,
                request=payload["request"],
                sources=sources,
                creator_invocation_id=uuid.uuid4().hex,
                new_experiment=mode == "automatic",
                failpoint=failpoint,
            )
    except PermissionError as error:
        if error.filename is not None:
            raise
        print(json.dumps({"error": "ownership-busy"}))
        return 75
    print(
        json.dumps(
            {
                "action": selected.action,
                "name": selected.name,
                "reservation_id": selected.reservation_id,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def worker_script(tmp_path):
    path = tmp_path / "reservation_worker.py"
    path.write_text(_WORKER, encoding="utf-8", newline="\n")
    return path


def admit(api, prepared, name=None, **options):
    root, request, sources = prepared
    root.mkdir(exist_ok=True)
    with archive_lock(root, "project-execution"):
        return api.admit_experiment_owned(
            root,
            name,
            request=request,
            sources=sources,
            creator_invocation_id=uuid.uuid4().hex,
            **options,
        )


def inventory(root):
    # Windows byte-range ownership prevents reading an active lock byte even
    # in the owning process. Its path/size remain observed; all authoritative
    # records and payloads are byte-hashed.
    return {
        p.relative_to(root).as_posix(): f"lock-size:{p.stat().st_size}"
        if p.suffix == ".lock"
        else hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def worker_env():
    return {
        **os.environ,
        "PYTHONPATH": str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


@pytest.mark.parametrize(
    "name,new,expected",
    [
        (None, False, "stress_test_01"),
        ("custom", False, "custom_01"),
        ("custom_37", False, "custom_37"),
        ("custom_99", False, "custom_99"),
    ],
)
def test_preview_selects_without_creating(api, tmp_path, name, new, expected):
    before = list(tmp_path.rglob("*"))
    selected = api.preview_experiment(tmp_path, name, new_experiment=new)
    assert selected.action == "create"
    assert selected.name == expected
    assert selected.advisory is True
    assert list(tmp_path.rglob("*")) == before


@pytest.mark.parametrize(
    "name", ["", " ", "Upper", "../x", "con", "x_0", "x_1", "x_001"]
)
def test_preview_rejects_invalid_names(api, tmp_path, name):
    assert api.preview_experiment(tmp_path, name).action == "refuse"


@pytest.mark.parametrize(
    "flags",
    [
        {"new_experiment": True, "requested_name": "x_37"},
        {"note": "no new"},
        {"indicators_only": True, "new_experiment": True},
        {"recover": True, "new_experiment": True},
        {"wf4_enabled": False, "new_experiment": True},
        {"wf4_enabled": False, "recover": True},
    ],
)
def test_invalid_flags_are_pure_refusals(api, tmp_path, flags):
    assert api.preview_experiment(tmp_path, **flags).action == "refuse"
    assert not list(tmp_path.iterdir())


def test_growth_exact_pins_no_gap_fill_and_completed_status_is_not_allocation(
    api, prepared
):
    root, _, _ = prepared
    assert admit(api, prepared, "wet_99").name == "wet_99"
    before = inventory(root)
    assert api.preview_experiment(root, "wet").name == "wet_99"
    assert inventory(root) == before
    assert (
        admit(api, prepared, "wet", new_experiment=True, note="once").name == "wet_100"
    )
    assert admit(api, prepared, "wet_37").name == "wet_37"
    assert admit(api, prepared, "wet").name == "wet_100"
    assert admit(api, prepared, "wet", new_experiment=True).name == "wet_101"
    ledger = read_canonical_json(root / "experiments/_engine/allocation.json")
    assert [a["number"] for a in ledger["families"][0]["allocations"]] == [
        37,
        99,
        100,
        101,
    ]
    assert ledger["families"][0]["high_water"] == 101


def test_chosen_roots_allocate_independently_including_active(api, prepared):
    root, request, sources = prepared
    other = (root.parent / "active", request, sources)
    first = admit(api, prepared)
    second = admit(api, other)
    assert first.name == second.name == "stress_test_01"
    assert first.reservation_id != second.reservation_id
    before = inventory(other[0])
    assert admit(api, prepared, new_experiment=True).name == "stress_test_02"
    assert api.preview_experiment(other[0]).name == "stress_test_01"
    assert inventory(other[0]) == before


def test_shared_engine_is_not_an_experiment_and_creation_uses_new_layout(api, prepared):
    root, _, _ = prepared
    selected = admit(api, prepared)
    engine = root / "experiments/_engine"
    assert (engine / "allocation.json").is_file()
    transaction = engine / "transactions" / selected.reservation_id
    assert (transaction / "transaction.json").is_file()
    assert (transaction / "sources/project.bin").is_file()
    assert (root / f"experiments/{selected.name}/_engine/reservation.json").is_file()
    assert not (root / "config/experiments").exists()
    assert not (root / "config/runs").exists()
    assert api.preview_experiment(root).name == selected.name
    assert admit(api, prepared, new_experiment=True).name == "stress_test_02"


def test_engine_file_is_refused_without_mutation(api, prepared):
    root, _, _ = prepared
    (root / "experiments").mkdir(parents=True)
    (root / "experiments/_engine").write_bytes(b"foreign")
    before = inventory(root)
    result = api.preview_experiment(root)
    assert result.action == "refuse"
    assert "engine root is not a directory" in result.reason
    assert inventory(root) == before


def test_deletion_is_permanent_and_preview_cannot_tombstone(api, prepared):
    root, _, _ = prepared
    admit(api, prepared, "wet_99")
    admit(api, prepared, "wet", new_experiment=True)
    shutil.rmtree(root / "experiments/wet_100")
    before = inventory(root)
    selected = api.preview_experiment(root, "wet")
    assert selected.action == "missing"
    assert selected.name == "wet_100"
    assert inventory(root) == before
    with pytest.raises(ValueError, match="tombstoned"):
        admit(api, prepared, "wet")
    ledger = read_canonical_json(root / "experiments/_engine/allocation.json")
    assert ledger["families"][0]["allocations"][-1]["presence"] == "tombstone"
    (root / "experiments/wet_100").mkdir()
    assert api.preview_experiment(root, "wet_100").action == "missing"
    assert admit(api, prepared, "wet", new_experiment=True).name == "wet_101"


@pytest.mark.parametrize("occupied", ["empty", "file", "invalid intent"])
def test_unknown_numeric_occupants_are_never_adopted(api, prepared, occupied):
    root, _, _ = prepared
    path = root / "experiments/wet_01"
    path.parent.mkdir(parents=True)
    if occupied == "file":
        path.write_bytes(b"foreign")
    else:
        path.mkdir()
        if occupied == "invalid intent":
            (path / "_engine").mkdir()
            (path / "_engine/simulation_intent.json").write_bytes(b"{}\n")
    before = inventory(root)
    assert api.preview_experiment(root, "wet").action == "refuse"
    with pytest.raises(ValueError):
        admit(api, prepared, "wet")
    assert inventory(root) == before | {
        "config/_engine/archive-transactions/project-execution.lock": "lock-size:1"
    }
    assert not (root / "experiments/_engine/allocation.json").exists()


def test_new_unledgered_owner_refuses_after_adoption(api, prepared):
    root, _, _ = prepared
    admit(api, prepared)
    (root / "experiments/wet_07").mkdir()
    before = inventory(root)
    assert api.preview_experiment(root).action == "refuse"
    with pytest.raises(ValueError, match="unledgered"):
        admit(api, prepared)
    assert inventory(root) == before


@pytest.mark.parametrize(
    "mutation", ["changed source", "missing source", "duplicate role", "extra role"]
)
def test_capture_validation_precedes_allocation(api, prepared, mutation):
    root, request, sources = prepared
    if mutation == "changed source":
        sources[0].original_path.write_bytes(b"changed")
    elif mutation == "missing source":
        sources.pop()
    elif mutation == "duplicate role":
        sources.append(sources[0])
    else:
        sources.append(
            api.CapturedSource(
                "unclassified", sources[0].original_path, sources[0].data
            )
        )
    with pytest.raises(ValueError):
        admit(api, prepared)
    assert not (root / "experiments/_engine").exists()


def test_changed_request_reports_nested_field_and_preserves_authority(api, prepared):
    root, request, _ = prepared
    selected = admit(api, prepared)
    before = inventory(root)
    request["generation"]["n_realizations"] = 2
    with pytest.raises(ValueError, match=r"request.generation.n_realizations"):
        admit(api, prepared)
    assert inventory(root) == before
    assert (
        api.read_reservation(root, selected.name)["request"]["generation"][
            "n_realizations"
        ]
        == 1
    )
    with pytest.raises(ValueError):
        api.compare_request({"nested": {"value": 1}}, {"nested": {"value": True}})


@pytest.mark.parametrize("value", [1, "JAN", [1, 2], {"nested": [True, 1]}])
def test_request_comparison_accepts_matching_subfields(api, value):
    api.compare_request(value, copy.deepcopy(value))


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("ledger", "revision", True),
        ("reservation", "revision", True),
        ("reservation", "unclassified", "field"),
        ("reservation", "state", "simulation_bound"),
        ("metadata", "note", 42),
    ],
)
def test_closed_records_reject_schema_mutation(api, prepared, record, field, value):
    root, _, _ = prepared
    selected = admit(api, prepared)
    paths = {
        "ledger": root / "experiments/_engine/allocation.json",
        "reservation": root
        / "experiments"
        / selected.name
        / "_engine/reservation.json",
        "metadata": root / "experiments" / selected.name / "experiment.yml",
    }
    path = paths[record]
    if record == "metadata":
        import yaml

        document = yaml.safe_load(path.read_bytes())
        document[field] = value
        path.write_bytes(yaml.safe_dump(document).encode())
    else:
        document = read_canonical_json(path)
        document[field] = value
        atomic_record(path, document, replace=True)
    before = inventory(root)
    assert api.preview_experiment(root).action == "refuse"
    assert inventory(root) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "source_captured:generate_scenarios",
        "source_captured:generator_template",
        "source_captured:project",
        "source_captured:simulate_system",
        "sources_captured",
        "metadata_written",
        "reservation_written",
        "transaction_written",
        "ledger_prepared",
        "directory_published",
        "ledger_published",
    ],
)
@pytest.mark.process_isolation
def test_process_death_at_every_creation_boundary_recovers_twice(
    api, prepared, boundary, tmp_path
):
    root, request, sources = prepared
    payload = tmp_path / "worker-input.json"
    payload.write_text(
        json.dumps(
            {
                "request": request,
                "sources": [
                    {
                        "role": s.role,
                        "path": str(s.original_path),
                        "data": s.data.decode(),
                    }
                    for s in sources
                ],
            }
        ),
        encoding="utf-8",
    )
    worker = worker_script(tmp_path)
    result = subprocess.run(
        [sys.executable, str(worker), str(root), str(payload), "create", boundary],
        capture_output=True,
        text=True,
        timeout=90,
        env=worker_env(),
    )
    assert result.returncode == 71, result.stdout + result.stderr
    ledger_path = root / "experiments/_engine/allocation.json"
    allocated = boundary in {
        "ledger_prepared",
        "directory_published",
        "ledger_published",
    }
    assert ledger_path.exists() is allocated
    if not allocated:
        selected = api.preview_experiment(root)
        assert selected.name == "stress_test_01"
        assert selected.action == "create"
        assert "orphan transaction evidence" in selected.reason
        with (
            archive_lock(root, "project-execution"),
            pytest.raises(ValueError, match="no committed"),
        ):
            api.recover_experiment_owned(root, "stress_test_01")
        return
    ledger = read_canonical_json(ledger_path)
    assert ledger["families"][0]["high_water"] == 1
    if boundary != "ledger_published":
        assert api.preview_experiment(root).action == "refuse"
    with archive_lock(root, "project-execution"):
        api.recover_experiment_owned(root, "stress_test_01")
        authoritative = inventory(root)
        api.recover_experiment_owned(root, "stress_test_01")
    assert inventory(root) == authoritative
    assert api.read_reservation(root, "stress_test_01")["revision"] == 0


@pytest.mark.parametrize(
    "damage", ["neither", "both", "unknown final", "partial payload", "captured bytes"]
)
def test_recovery_refuses_without_verified_exact_recipe(api, prepared, damage):
    root, _, _ = prepared

    def stop(boundary):
        if boundary == "ledger_prepared":
            raise RuntimeError("fault")

    with pytest.raises(RuntimeError):
        admit(api, prepared, failpoint=stop)
    ledger = read_canonical_json(root / "experiments/_engine/allocation.json")
    allocation = ledger["families"][0]["allocations"][0]
    transaction = (
        root / "experiments/_engine/transactions" / allocation["reservation_id"]
    )
    staged = transaction / "experiment"
    final = root / "experiments/stress_test_01"
    if damage == "neither":
        shutil.rmtree(staged)
    elif damage == "both":
        final.mkdir(parents=True)
    elif damage == "unknown final":
        # Fixture damage must tolerate the same transient Windows directory
        # handle as publication; keep the unknown-occupant assertions intact.
        _replace(staged, transaction / "preserved-stage")
        final.mkdir(parents=True)
        (final / "foreign").write_bytes(b"foreign")
    elif damage == "partial payload":
        (staged / "_engine/reservation.json").write_bytes(b'{"partial":')
    else:
        next((transaction / "sources").iterdir()).write_bytes(b"tampered")
    before = inventory(root)
    with archive_lock(root, "project-execution"), pytest.raises((ValueError, OSError)):
        api.recover_experiment_owned(root, "stress_test_01")
    assert inventory(root) == before
    assert api.preview_experiment(root, new_experiment=True).name == "stress_test_02"


@pytest.mark.parametrize("mode", ["automatic", "same-pin"])
@pytest.mark.process_isolation
def test_independent_creators_hold_project_ownership(api, prepared, tmp_path, mode):
    root, request, sources = prepared
    payload = tmp_path / "worker-input.json"
    payload.write_text(
        json.dumps(
            {
                "request": request,
                "sources": [
                    {
                        "role": s.role,
                        "path": str(s.original_path),
                        "data": s.data.decode(),
                    }
                    for s in sources
                ],
            }
        ),
        encoding="utf-8",
    )
    worker = worker_script(tmp_path)
    processes = [
        subprocess.Popen(
            [sys.executable, str(worker), str(root), str(payload), mode, ""],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=worker_env(),
        )
        for _ in range(2)
    ]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=70)
        assert process.returncode == 0, stdout + stderr
        outputs.append(json.loads(stdout.splitlines()[-1]))
    ledger = read_canonical_json(root / "experiments/_engine/allocation.json")
    if mode == "automatic":
        assert sorted(item["name"] for item in outputs) == [
            "stress_test_01",
            "stress_test_02",
        ]
        assert ledger["families"][0]["high_water"] == 2
        assert len({item["reservation_id"] for item in outputs}) == 2
    else:
        assert {item["name"] for item in outputs} == {"pinned_37"}
        assert {item["action"] for item in outputs} == {"create", "continue"}
        assert len({item["reservation_id"] for item in outputs}) == 1
        assert ledger["families"][0]["high_water"] == 37


@pytest.mark.parametrize(
    "name,expected,number",
    [
        ("legacy", "legacy", None),
        ("legacy_20261010", "legacy_20261010", None),
        ("legacy_20261010_v2", "legacy_20261010_v2", None),
        ("legacy_99", "legacy_99", 99),
        ("legacy_100", "legacy_100", 100),
    ],
)
def test_owned_legacy_precedence_and_numbered_adoption(
    api, helper, tmp_path, monkeypatch, name, expected, number
):
    monkeypatch.setattr(helper, "EXPERIMENT", name)
    fixture = helper.build_v2_experiment(tmp_path)
    before = inventory(fixture.project)
    selected = api.preview_experiment(fixture.project, name)
    assert selected.action == "legacy"
    assert selected.name == expected
    assert selected.number == number
    assert inventory(fixture.project) == before
    if number:
        assert api.preview_experiment(fixture.project, "legacy").name == expected
        assert (
            api.preview_experiment(fixture.project, "legacy", new_experiment=True).name
            == f"legacy_{number + 1}"
        )
    elif name == "legacy":
        assert (
            api.preview_experiment(fixture.project, name, new_experiment=True).name
            == "legacy_01"
        )
    else:
        assert (
            api.preview_experiment(fixture.project, name, new_experiment=True).action
            == "refuse"
        )
        assert api.preview_experiment(fixture.project).name == "stress_test_01"
    assert not (fixture.root / "experiment.yml").exists()


def test_legacy_requires_paired_archive_and_readable_complete_records(
    api, helper, tmp_path
):
    fixture = helper.build_v2_experiment(tmp_path)
    (fixture.root / "_engine/simulation.json").unlink()
    (fixture.root / "config/run_record.yml").unlink()
    assert api.preview_experiment(fixture.project, helper.EXPERIMENT).action == "refuse"


@pytest.mark.parametrize(
    "failure", [None, "plan digest", "plan capacity", "seed request"]
)
def test_wf4_only_binds_existing_plan_seed_atomically(
    api, prepared, helper, monkeypatch, failure
):
    from blueearth_cst.experiment import reservation_context
    from blueearth_cst.experiment.content_identity import content_sha256
    from blueearth_cst.experiment.generation_plan import build_create_plan
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    root, request, _ = prepared
    selected = admit(api, prepared)
    fixture = helper.build_v2_experiment(root.parent)
    marker = read_collection_v2(fixture.collection_marker)
    intent = read_canonical_json(
        fixture.collection_marker.with_name("collection_intent.json")
    )
    plan = build_create_plan(
        {
            "request": {"fixture": "wf4-only"},
            "n_realizations": 1,
            "n_design_points": 1,
            "capacity": 2,
        },
        intent,
    )
    plan_path = (
        root
        / "scenarios/_engine/requests"
        / plan["generation_request_id"][:12]
        / f"{content_sha256(plan)}.json"
    )
    plan_path.parent.mkdir(parents=True)
    atomic_record(plan_path, plan)
    reference = file_reference(plan_path, "project_root", root)
    if failure == "plan digest":
        reference["sha256"] = "0" * 64
    elif failure == "plan capacity":
        other = helper.collection_intent_v2(capacity=3)
        other_plan = build_create_plan(
            {
                "request": {"fixture": "other"},
                "n_realizations": 1,
                "n_design_points": 1,
                "capacity": 3,
            },
            other,
        )
        other_path = (
            root
            / "scenarios/_engine/requests"
            / other_plan["generation_request_id"][:12]
            / f"{content_sha256(other_plan)}.json"
        )
        other_path.parent.mkdir(parents=True)
        atomic_record(other_path, other_plan)
        reference = file_reference(other_path, "project_root", root)
    elif failure == "seed request":
        request = copy.deepcopy(request)
        request["generation"]["requested_seed"] = 42
        selected = admit(api, (root, request, prepared[2]), new_experiment=True)
    context = reservation_context.ReservationContext(
        root,
        fixture.config,
        root.parent,
        selected.name,
        selected.reservation_id,
        False,
        False,
        request,
    )
    monkeypatch.setattr(
        reservation_context, "capture_request", lambda *args, **kwargs: (request, [])
    )
    monkeypatch.setattr(
        reservation_context,
        "resolve_collection_with_plan",
        lambda *args: (
            {
                "resolution_mode": "project-generation",
                "manifest_path": str(fixture.collection_marker),
            },
            marker,
            reference,
        ),
    )
    before = inventory(root)
    with archive_lock(root, "project-execution"):
        if failure:
            with pytest.raises(ValueError):
                context.bind_selection()
            assert inventory(root) == before
            return
        from blueearth_cst.experiment.reservation_generation import (
            bind_ready_collection_owned,
        )

        # Reproduce the old WF4 handoff: a collection alone cannot bind a pending seed.
        with pytest.raises(
            ValueError, match="resolution cannot precede generation seed binding"
        ):
            bind_ready_collection_owned(root, selected.name, fixture.collection_marker)
        assert inventory(root) == before
        context.bind_selection()
        bound = api.read_reservation(root, selected.name)
        after = inventory(root)
        context.bind_selection()
    assert bound["state"] == "resolved"
    assert bound["seed"]["state"] == "bound"
    assert bound["seed"]["plan"] == reference
    assert (
        bound["seed"]["resolved"]
        == intent["documents"]["generation_config"]["seed"]["resolved"]
    )
    assert bound["resolution"]["collection_id"] == marker["collection_id"]
    assert bound["revision"] == 1
    assert inventory(root) == after


def test_seed_binding_is_monotonic_plan_verified_and_retry_is_byte_identical(
    api, prepared, helper
):
    from blueearth_cst.experiment.content_identity import content_sha256
    from blueearth_cst.experiment.generation_plan import build_create_plan
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    root, request, _ = prepared
    selected = admit(api, prepared)
    intent = helper.collection_intent_v2()
    (root / "data").mkdir()
    for role in ("basin_cells", "historical_climate"):
        (root / "data" / f"{role}.dat").write_bytes(b"x")
    plan_request = {"schema_version": "generation-request/2", "settings": {}}
    plan = build_create_plan(
        {
            "request": plan_request,
            "n_realizations": 1,
            "n_design_points": 1,
            "capacity": 2,
        },
        intent,
    )
    digest = content_sha256(plan)
    path = (
        root
        / "scenarios/_engine/requests"
        / content_sha256(plan_request)[:12]
        / f"{digest}.json"
    )
    path.parent.mkdir(parents=True)
    atomic_record(path, plan)
    seed = {
        "state": "bound",
        "requested": "auto",
        "resolved": intent["documents"]["generation_config"]["seed"]["resolved"],
        "seed_resolution": intent["documents"]["generation_config"]["seed_resolution"],
        "plan": file_reference(path, "project_root", root),
        "source_identity_sha256": intent["identity_digests"]["source_inventory"],
    }
    with archive_lock(root, "project-execution"):
        bound = api.bind_reservation_owned(root, selected.name, seed=seed)
        before = inventory(root)
        repeated = api.bind_reservation_owned(root, selected.name, seed=seed)
    assert bound["state"] == "pending"
    assert repeated["revision"] == bound["revision"] == 1
    assert inventory(root) == before
    changed = copy.deepcopy(seed)
    changed["resolved"] += 1
    with (
        archive_lock(root, "project-execution"),
        pytest.raises(ValueError, match="immutable"),
    ):
        api.bind_reservation_owned(root, selected.name, seed=changed)
    assert api.read_reservation(root, selected.name)["request"] == request
    path.write_bytes(b"tampered plan")
    assert api.preview_experiment(root).action == "refuse"


@pytest.mark.parametrize(
    "which", ["experiment link", "ledger parent link", "project link", "source link"]
)
def test_links_are_refused_and_targets_preserved(api, prepared, tmp_path, which):
    root, request, sources = prepared
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"keep")
    if which == "source link":
        link = tmp_path / "source-link.yml"
        link.symlink_to(sources[0].original_path)
        sources[0] = api.CapturedSource(sources[0].role, link, sources[0].data)
        with pytest.raises(ValueError, match="aliased"):
            admit(api, prepared)
    else:
        link = {
            "experiment link": root / "experiments",
            "ledger parent link": root / "experiments/_engine",
            "project link": tmp_path / "project-link",
        }[which]
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(outside, target_is_directory=True)
        selected = api.preview_experiment(link if which == "project link" else root)
        assert selected.action == "refuse"
    assert (outside / "sentinel").read_bytes() == b"keep"
    assert set(p.name for p in outside.iterdir()) == {"sentinel"}


@pytest.mark.parametrize(
    "field,invalid",
    [
        ("requested_seed", True),
        ("n_realizations", True),
        ("run_group_id_capacity", 0),
        ("water_year_start", "jan"),
        ("future_field", 1),
    ],
)
def test_request_schema_is_closed_and_never_coerces_types(
    api, prepared, field, invalid
):
    root, request, _ = prepared
    request["generation"][field] = invalid
    with pytest.raises(ValueError):
        api.validate_request(request, root)
    assert not root.exists()


@pytest.mark.parametrize("version", ["experiment-request/3", None, [], {}])
def test_request_rejects_invalid_schema_versions(api, prepared, version):
    root, request, _ = prepared
    request["schema_version"] = version
    with pytest.raises(ValueError, match="unsupported reservation request"):
        api.validate_request(request, root)


def test_full_candidate_name_length_is_checked_without_truncation(api, tmp_path):
    assert api.preview_experiment(tmp_path, "x" * 62).action == "refuse"
    assert api.preview_experiment(tmp_path, "x" * 61).name == "x" * 61 + "_01"


def test_unready_legacy_indicators_only_refuses(api, helper, tmp_path):
    fixture = helper.build_v2_experiment(tmp_path, publish=False)
    assert api.preview_experiment(fixture.project, helper.EXPERIMENT).action == "legacy"
    assert (
        api.preview_experiment(
            fixture.project, helper.EXPERIMENT, indicators_only=True
        ).action
        == "refuse"
    )


def test_recovery_admission_compares_initial_request_before_publication(api, prepared):
    root, request, _ = prepared

    def stop(boundary):
        if boundary == "ledger_prepared":
            raise RuntimeError("fault")

    with pytest.raises(RuntimeError):
        admit(api, prepared, failpoint=stop)
    request["generation"]["n_realizations"] = 2
    before = inventory(root)
    with pytest.raises(ValueError, match="n_realizations"):
        admit(api, prepared, recover=True)
    assert inventory(root) == before


def test_ready_resolution_and_simulation_bind_once_and_validate_upstream_evidence(
    api, helper, tmp_path, monkeypatch
):
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.experiment.simulation_record import (
        capture_simulation_sources_v2,
        freeze_simulation_v2,
        read_simulation_intent_v2,
        simulation_intent_v2,
    )
    from blueearth_cst.experiment.wf4_ancillary_descriptor import (
        preparation_identity_v2,
    )
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    fixture = helper.build_v2_experiment(tmp_path)
    root = fixture.project
    marker = read_collection_v2(fixture.collection_marker)
    manifest = file_reference(fixture.collection_marker, "project_root", root)
    request = {
        "schema_version": "experiment-request/1",
        "generation": None,
        "selection": {"mode": "explicit-manifest", "manifest": manifest},
        "input_catalogs": [],
        "simulation": {
            "climate_source": "era5",
            "model_root": "models/hydrology/wflow",
            "outvars": ["gwr"],
            "model_request": None,
        },
    }
    paths = [
        ("project", fixture.config),
        ("simulate_system", tmp_path / "simulate_system.yml"),
    ]
    sources = [
        api.CapturedSource(role, path, path.read_bytes()) for role, path in paths
    ]
    with archive_lock(root, "project-execution"):
        selected = api.admit_experiment_owned(
            root,
            request=request,
            sources=sources,
            creator_invocation_id=uuid.uuid4().hex,
        )
    resolution = {
        "resolution_mode": "explicit-manifest",
        "manifest": manifest,
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
        "run_ids": [item["run_id"] for item in marker["series"]],
    }
    with archive_lock(root, "project-execution"):
        resolved = api.bind_reservation_owned(
            root, selected.name, resolution=resolution
        )
        before = inventory(root)
        assert (
            api.bind_reservation_owned(root, selected.name, resolution=resolution)[
                "revision"
            ]
            == 1
        )
    assert inventory(root) == before
    assert resolved["state"] == "resolved"
    old_intent = read_simulation_intent_v2(fixture.root)
    target = root / "experiments" / selected.name
    for relative in ("_engine/model_reference.yml",):
        source = fixture.root / relative
        if source.exists():
            shutil.copyfile(source, target / relative)
    documents = copy.deepcopy(old_intent["documents"])
    documents["preparation"] = helper._preparation(
        root, target, tmp_path / "new-preparation"
    )
    identity = preparation_identity_v2(
        documents["preparation"], project_root=root, experiment_root=target
    )
    intent = simulation_intent_v2(
        selected.name,
        resolution,
        old_intent["simulator"],
        documents,
        identity,
    )
    invocation = uuid.uuid4().hex
    capture_simulation_sources_v2(fixture.config, root, invocation)
    freeze_simulation_v2(target, intent, invocation_id=invocation, command=["fixture"])
    binding = {
        "simulation_id": intent["simulation_id"],
        "intent": file_reference(
            target / "_engine/simulation_intent.json", "project_root", root
        ),
        "creator_archive": file_reference(
            target / "config/run_record.yml", "project_root", root
        ),
    }
    with archive_lock(root, "project-execution"):
        bound = api.bind_reservation_owned(root, selected.name, simulation=binding)
        before = inventory(root)
        assert (
            api.bind_reservation_owned(root, selected.name, simulation=binding)[
                "revision"
            ]
            == 2
        )
    assert bound["state"] == "simulation_bound"
    assert inventory(root) == before
    changed = copy.deepcopy(resolution)
    changed["collection_revision"] = "0" * 64
    with (
        archive_lock(root, "project-execution"),
        pytest.raises(ValueError, match="immutable"),
    ):
        api.bind_reservation_owned(root, selected.name, resolution=changed)
    (target / "config/run_record.yml").write_bytes(b"corrupt creator archive")
    assert api.preview_experiment(root).action == "refuse"


def test_published_continuation_calls_explicit_comparator_with_original_requests(
    api, prepared
):
    root, request, _ = prepared
    selected = admit(api, prepared)
    retained = copy.deepcopy(request)
    request["generation"]["n_realizations"] = 2
    observed = []

    def comparator(old, new):
        observed.append((old, new))

    continued = admit(api, prepared, request_comparator=comparator)
    assert continued.reservation_id == selected.reservation_id
    assert len(observed) == 1
    assert observed[0][0] == retained
    assert observed[0][1] is request
    assert api.read_reservation(root, selected.name)["request"] == retained
    with pytest.raises(ValueError, match="n_realizations"):
        admit(api, prepared)


@pytest.mark.skipif(
    os.name != "nt", reason="Windows transient-handle publication behavior"
)
def test_transient_windows_directory_handle_retries_same_creation(
    api, prepared, monkeypatch
):
    original = Path.rename
    observed = []

    def rename(path, target):
        observed.append((path, target))
        if len(observed) == 1:
            fault = PermissionError(13, "synthetic transient handle")
            fault.winerror = 5
            raise fault
        return original(path, target)

    monkeypatch.setattr(Path, "rename", rename)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    selected = admit(api, prepared)
    assert observed[0] == observed[1]
    assert (
        api.read_reservation(prepared[0], selected.name)["reservation_id"]
        == selected.reservation_id
    )


def test_new_integer_seed_reservation_can_bind_existing_auto_seed_collection(
    api, prepared, helper, tmp_path
):
    from blueearth_cst.experiment.content_identity import content_sha256
    from blueearth_cst.experiment.generation_plan import build_reuse_plan
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    fixture_base = tmp_path / "ready-fixture"
    fixture_base.mkdir()
    fixture = helper.build_v2_experiment(fixture_base)
    root = fixture.project
    _, request, sources = prepared
    creator = read_canonical_json(
        fixture.collection_marker.parent / "collection_intent.json"
    )
    seed_resolution = creator["documents"]["generation_config"]["seed_resolution"]
    explicit = seed_resolution["resolved_seed"]
    request["generation"]["requested_seed"] = explicit
    candidate = copy.deepcopy(creator)
    config = candidate["documents"]["generation_config"]
    config["seed"]["requested"] = explicit
    config["seed_resolution"]["seed_request"] = explicit
    config["seed_resolution"]["seed_resolution_id"] = content_sha256(
        {
            k: v
            for k, v in config["seed_resolution"].items()
            if k != "seed_resolution_id"
        }
    )
    candidate["document_digests"]["generation_config"] = content_sha256(config)
    plan_request = {
        "schema_version": "generation-request/2",
        "settings": {"seed": explicit},
    }
    plan = build_reuse_plan(
        {
            "request": plan_request,
            "n_realizations": 1,
            "n_design_points": 1,
            "capacity": 2,
        },
        candidate,
        fixture.collection_marker,
    )
    digest = content_sha256(plan)
    path = (
        root
        / "scenarios/_engine/requests"
        / content_sha256(plan_request)[:12]
        / f"{digest}.json"
    )
    path.parent.mkdir(parents=True)
    atomic_record(path, plan)
    with archive_lock(root, "project-execution"):
        selected = api.admit_experiment_owned(
            root,
            request=request,
            sources=sources,
            creator_invocation_id=uuid.uuid4().hex,
        )
        bound = api.bind_reservation_owned(
            root,
            selected.name,
            seed={
                "state": "bound",
                "requested": explicit,
                "resolved": explicit,
                "seed_resolution": config["seed_resolution"],
                "plan": file_reference(path, "project_root", root),
                "source_identity_sha256": candidate["identity_digests"][
                    "source_inventory"
                ],
            },
        )
    assert bound["seed"]["requested"] == explicit
    assert (
        plan["intent"]["documents"]["generation_config"]["seed_resolution"][
            "seed_request"
        ]
        == "auto"
    )
    assert bound["seed"]["seed_resolution"]["seed_request"] == explicit
