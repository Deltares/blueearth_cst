"""Fixture proof for the additive scenario-collection/3 reader and identity."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from shutil import copy2

import pytest

from blueearth_cst.experiment.collection_resolution import (
    discover_collections_v2,
    resolve_explicit_collection_v2,
)
from blueearth_cst.experiment.content_identity import (
    atomic_record,
    automatic_seed_v2,
    collection_id_v2,
    collection_revision_v2,
    content_sha256,
    identity_segment,
    repository_code_inventory,
)
from blueearth_cst.experiment.generation_plan import (
    GENERATION_CODE_ENTRIES,
    GENERATION_INVENTORY_EXCLUDED,
)
from blueearth_cst.experiment.scenario_collection_v2 import (
    _profile,
    read_collection_v2,
)
from blueearth_cst.shared.workflow_config_snapshot import file_reference
from tests._v2_experiment import collection_intent_v2


def test_wf4_only_evidence_does_not_enter_seed_or_collection_identity():
    intent = collection_intent_v2()
    _profile(intent)
    original_seed = intent["documents"]["generation_config"]["seed"]["resolved"]
    original_id = intent["collection_id"]
    assert intent["documents"]["generation_config"]["seed_resolution"][
        "projection_sha256"
    ] == ("4098d5a35ab1178648d6c20c1e662d84d38ef1c3bff957b6d34e04b81a1bf78a")
    assert original_seed == 1459665732
    assert (
        original_id
        != "e129e7f83e925771dc8d9cfbf1553b615a01c39f5561876c81f29f6e6f870a92"
    )
    wf4_only = {"elevation": "changed", "settings": "changed", "code": "changed"}
    wf4_only["elevation"] = "changed again"
    assert intent["collection_id"] == original_id
    assert (
        automatic_seed_v2(
            intent["documents"]["generation_config"]["seed_resolution"]["projection"]
        )
        == original_seed
    )
    changed = deepcopy(intent)
    changed["documents"]["source_inventory"]["sources"][0]["file"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="document digest"):
        _profile(changed)


def test_wf3_code_closure_ignores_wf4_only_source_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]

    def generation_inventory(repo):
        # Match the scientific identity; diagnostic imports are excluded in production.
        return repository_code_inventory(
            repo, GENERATION_CODE_ENTRIES, GENERATION_INVENTORY_EXCLUDED
        )

    inventory = generation_inventory(root)
    paths = {item["path"] for item in inventory}
    forbidden = {
        "blueearth_cst/climate_analysis/prepare_climate_data_catalog.py",
        "blueearth_cst/experiment/downscale_climate_forcing.py",
        "blueearth_cst/experiment/simulation_record.py",
        "blueearth_cst/experiment/indicator_groups.py",
        "blueearth_cst/experiment/wf4_ancillary_descriptor.py",
        "blueearth_cst/shared/snake_utils.py",
    }
    assert not paths & forbidden
    for relative in paths | forbidden:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(root / relative, target)
    original = generation_inventory(tmp_path)
    for relative in forbidden:
        path = tmp_path / relative
        path.write_bytes(path.read_bytes() + b"\n# WF4-only mutation\n")
    assert generation_inventory(tmp_path) == original

    def identities(code):
        intent = collection_intent_v2()
        generation = intent["documents"]["generation_config"]
        material = generation["seed_resolution"]["projection"]
        revision = content_sha256(code)
        material["provider_revision"] = revision
        seed = automatic_seed_v2(material)
        generation["seed"]["resolved"] = seed
        generation["weathergen"]["generate_weather"]["seed"] = seed
        resolution = generation["seed_resolution"]
        resolution["resolved_seed"] = seed
        resolution["projection_sha256"] = content_sha256(material)
        resolution["seed_resolution_id"] = content_sha256(
            {
                key: value
                for key, value in resolution.items()
                if key != "seed_resolution_id"
            }
        )
        intent["provider"]["revision"] = revision
        intent["documents"]["provider_code"] = code
        intent["identity_projections"]["generation_config"]["resolved_seed"] = seed
        intent["document_digests"] = {
            key: content_sha256(value) for key, value in intent["documents"].items()
        }
        intent["identity_digests"] = {
            key: content_sha256(intent["identity_projections"].get(key, value))
            for key, value in intent["documents"].items()
        }
        intent["collection_id"] = collection_id_v2(intent)
        _profile(intent)
        return revision, seed, intent["collection_id"]

    assert identities(generation_inventory(tmp_path)) == identities(original)
    changed = tmp_path / "blueearth_cst/experiment/scenario_provider.py"
    changed.write_bytes(changed.read_bytes() + b"\n# WF3 mutation\n")
    assert generation_inventory(tmp_path) != original


def test_collection_v2_refuses_unknown_nested_document_key():
    intent = collection_intent_v2()
    intent["documents"]["generation_config"]["unclassified"] = True
    intent["document_digests"]["generation_config"] = content_sha256(
        intent["documents"]["generation_config"]
    )
    with pytest.raises(ValueError, match="generation_config"):
        _profile(intent)


def test_explicit_and_auto_requests_with_same_seed_share_collection_identity():
    intent = collection_intent_v2()
    changed = deepcopy(intent)
    generation = changed["documents"]["generation_config"]
    resolved = generation["seed"]["resolved"]
    generation["seed"]["requested"] = resolved
    resolution = generation["seed_resolution"]
    resolution["seed_request"] = resolved
    resolution["seed_resolution_id"] = content_sha256(
        {key: value for key, value in resolution.items() if key != "seed_resolution_id"}
    )
    changed["document_digests"]["generation_config"] = content_sha256(generation)
    _profile(changed)
    assert changed["collection_id"] == intent["collection_id"]


def test_collection_v2_requires_marker_and_matching_data(tmp_path, monkeypatch):
    intent = collection_intent_v2()
    segment = identity_segment(intent["collection_id"], "collection_id")
    record = tmp_path / "scenarios" / "_engine" / "collections" / segment
    data = tmp_path / "scenarios" / segment
    record.mkdir(parents=True)
    (data / "config").mkdir(parents=True)
    (data / "series").mkdir()
    (data / "weathergenr" / "output").mkdir(parents=True)
    for role in ("basin_cells", "historical_climate"):
        path = tmp_path / "data" / f"{role}.dat"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"x")
    atomic_record(record / "collection_intent.json", intent)
    archive_path = data / "config" / "run_record.yml"
    archive_path.write_bytes(b"fixture archive")
    monkeypatch.setattr(
        "blueearth_cst.experiment.scenario_collection_v2.validate_archive",
        lambda path: {
            "owner": {"kind": "scenario_collection", "id": intent["collection_id"]},
            "workflow": "generate_scenarios",
            "generated_inputs": [products[2]],
        },
    )
    lookup = data / "scenario_run_lookup.csv"
    lookup.write_text(
        "run_id,evaluate,type,rlz,st_id\nrun_1,true,stochastic,1,\nrun_2,true,stochastic,1,1\n",
        encoding="utf-8",
    )
    perturbation = data / "perturbation_lookup.csv"
    perturbation.write_text(
        "st_id,month,temp_change,precip_change,precip_variance_change\n"
        + "".join(f"1,{month},0,0,0\n" for month in range(1, 13)),
        encoding="utf-8",
    )
    series = []
    for run_id in ("run_1", "run_2"):
        path = data / "series" / f"{run_id}.nc"
        path.write_bytes(run_id.encode())
        series.append(
            {
                "run_id": run_id,
                "file": file_reference(path, "project_root", tmp_path),
                "descriptor": {
                    "source_calendar": "standard",
                    "timestep": "daily",
                    "time_label": "interval_end",
                    "start": "2046-01-01",
                    "end": "2054-12-31",
                    "spatial_representation_sha256": sha256(b"spatial").hexdigest(),
                    "variables": [
                        {"name": "temp", "units": "degC", "missing_value": None}
                    ],
                },
            }
        )
    products = []
    for role, name in (
        ("sim_dates", "output/sim_dates.csv"),
        ("resampled_dates", "output/resampled_dates.csv"),
        ("weather_generation_input", "weather_generation_input.yml"),
    ):
        path = data / "weathergenr" / name
        path.write_bytes(role.encode())
        products.append(
            {"role": role, "file": file_reference(path, "project_root", tmp_path)}
        )
    marker = {
        "schema_version": "scenario-collection/3",
        "canonicalization_id": "collection-canon/1",
        "status": "ready",
        "collection_id": intent["collection_id"],
        "collection_revision": "0" * 64,
        "data_root": f"scenarios/{segment}",
        "intent": file_reference(
            record / "collection_intent.json", "record_directory", record
        ),
        "archive": file_reference(archive_path, "project_root", tmp_path),
        "scenario_run_lookup": file_reference(lookup, "project_root", tmp_path),
        "perturbation_lookup": file_reference(perturbation, "project_root", tmp_path),
        "series": series,
        "provider_products": products,
        "diagnostics": [],
    }
    marker["collection_revision"] = collection_revision_v2(marker, intent)
    # Readiness is never inferred from the data directory alone.
    with pytest.raises(FileNotFoundError):
        read_collection_v2(record / "collection.json")
    atomic_record(record / "collection.json", marker)
    before = (record / "collection.json").read_bytes()
    assert read_collection_v2(record / "collection.json") == marker
    assert read_collection_v2(record / "collection.json") == marker
    assert discover_collections_v2(tmp_path) == [marker]
    selection, selected = resolve_explicit_collection_v2(
        {"manifest_path": record / "collection.json"}
    )
    assert selection["collection_id"] == intent["collection_id"]
    assert selected == marker
    assert (record / "collection.json").read_bytes() == before
    (data / "series" / "run_2.nc").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact reference bytes differ"):
        read_collection_v2(record / "collection.json")
    assert (record / "collection.json").read_bytes() == before
