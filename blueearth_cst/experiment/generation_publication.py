"""Initialize and publish one pinned WF3 collection without re-planning."""

import csv
import hashlib
import math
import os
import re
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from blueearth_cst.experiment.content_identity import (
    atomic_record,
    collection_revision_v2,
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.forcing_descriptor import (
    collection_forcing_descriptor,
)
from blueearth_cst.experiment.generation_plan import (
    generator_seed_projection,
    read_pinned_plan,
)
from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
from blueearth_cst.shared.config_composition import (
    capture_configuration_sources,
    compose_config,
)
from blueearth_cst.shared.provenance import environment_file_hashes, toolbox_identity
from blueearth_cst.shared.wf3_science import ADVANCED_SETTINGS
from blueearth_cst.shared.workflow_config_snapshot import (
    file_reference,
    publish_archive,
    resolve_file_reference,
    run_record_document,
    validate_archive,
)

PROJECTION = ("project", "basin", "climate", "workflows.generate_scenarios")
SCENARIO_HEADER = ("run_id", "evaluate", "type", "rlz", "st_id")


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"duplicate installed generator key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _decode_generator(data: bytes) -> dict:
    """Decode only unambiguous plain YAML values accepted by the R binding."""
    text = data.decode("utf-8")
    for token in yaml.scan(text):
        if isinstance(token, (yaml.AliasToken, yaml.AnchorToken, yaml.TagToken)):
            raise ValueError("installed generator YAML uses alias, anchor, or tag")
        if isinstance(token, yaml.ScalarToken) and token.style is None:
            if token.value.lower() in {"yes", "no", "on", "off", "y", "n", "~"}:
                raise ValueError("installed generator YAML has ambiguous scalar")
    value = yaml.load(text, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("installed generator YAML must be a mapping")

    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("installed generator YAML has nonfinite value")
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def _paths(project_root: Path, plan: dict) -> dict[str, Path]:
    output = plan["outputs"]
    return {
        key: project_root / output[key]
        for key in (
            "data_root",
            "record_root",
            "archive_root",
            "generator_input",
            "scenario_run_lookup",
            "perturbation_lookup",
            "diagnostic_root",
        )
    }


def _source_paths(plan: dict, project_root: Path) -> dict[str, Path]:
    result = {}
    for item in plan["candidate_intent"]["documents"]["source_inventory"]["sources"]:
        result[item["role"]] = resolve_file_reference(
            item["file"], {"project_root": project_root}
        )
    return result


def _perturbation_rows(plan: dict) -> list[dict]:
    perturb = plan["intent"]["identity_projections"]["generation_config"][
        "climate_perturbations"
    ]
    temp = perturb["temp"]
    precip = perturb["precip"]
    temp_values = np.linspace(
        temp["mean"]["min"], temp["mean"]["max"], temp["n_levels"], axis=1
    )
    precip_values = np.linspace(
        precip["mean"]["min"],
        precip["mean"]["max"],
        precip["n_levels"],
        axis=1,
    )
    variance_values = np.linspace(
        precip["variance"]["min"],
        precip["variance"]["max"],
        precip["n_levels"],
        axis=1,
    )

    def level(value):
        return float(str(np.float32(value)))

    rows = []
    width = len(str(temp["n_levels"] * precip["n_levels"]))
    for j in range(temp["n_levels"]):
        for k in range(precip["n_levels"]):
            st_id = f"{len(rows) // 12 + 1:0{width}d}"
            for month in range(12):
                rows.append(
                    {
                        "st_id": st_id,
                        "month": month + 1,
                        "temp_change": level(temp_values[month, j]),
                        "precip_change": level(precip_values[month, k]) * 100 - 100,
                        "precip_variance_change": level(variance_values[month, k]) * 100
                        - 100,
                    }
                )
    return rows


def _observed_perturbation_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
            "st_id",
            "month",
            "temp_change",
            "precip_change",
            "precip_variance_change",
        ]:
            raise ValueError("provider perturbation columns differ")
        return [
            {
                "st_id": row["st_id"],
                "month": int(row["month"]),
                "temp_change": float(row["temp_change"]),
                "precip_change": float(row["precip_change"]),
                "precip_variance_change": float(row["precip_variance_change"]),
            }
            for row in reader
        ]


def validate_provider_inputs(
    plan: dict,
    project_root: Path,
    *,
    historical: Path,
    cells: Path,
    generator_yaml: Path,
    lookup: Path,
    root_run_ids: list[str] | None = None,
    output_dir: Path | None = None,
    ancestor: Path | None = None,
    row_id: str | None = None,
    output: Path | None = None,
) -> dict:
    """Bind actual provider positional arguments to the frozen WF3 plan."""
    project_root = Path(project_root).resolve()
    expected_sources = _source_paths(plan, project_root)
    observed_paths = {
        "historical_climate": Path(historical),
        "basin_cells": Path(cells),
    }
    for role, path in observed_paths.items():
        if path.resolve(strict=True) != expected_sources[role]:
            raise ValueError(f"provider {role} argument differs from pinned snapshot")
    yaml_path = _paths(project_root, plan)["generator_input"]
    if Path(generator_yaml).resolve(strict=True) != yaml_path:
        raise ValueError("provider YAML argument differs from installed generator")
    data = validate_installed_generator(plan, project_root)
    expected_generation = plan["intent"]["identity_projections"]["generation_config"]
    if (
        type(data["generate_weather"].get("seed")) is not int
        or data["generate_weather"]["seed"] != expected_generation["resolved_seed"]
    ):
        raise ValueError("provider resolved seed differs from frozen identity")
    observed_generation = {
        **expected_generation,
        "generator_settings": generator_seed_projection(data),
    }
    expected_rows = _perturbation_rows(plan)
    if (
        Path(lookup).resolve(strict=True)
        != _paths(project_root, plan)["perturbation_lookup"]
    ):
        raise ValueError("provider lookup argument differs from pinned lookup")
    observed_rows = _observed_perturbation_rows(Path(lookup))
    if observed_rows != expected_rows:
        raise ValueError("provider monthly perturbation rows differ from plan")
    # Bytes against the pinned snapshot's file reference, not the identity
    # projection: under generation-sources-identity/3 the projection carries a
    # content digest, which every provider job would otherwise re-derive.
    pinned_files = {
        item["role"]: item["file"]
        for item in plan["intent"]["documents"]["source_inventory"]["sources"]
    }
    expected_sources_identity = [
        {
            "role": role,
            "sha256": pinned_files[role]["sha256"],
            "size_bytes": pinned_files[role]["size_bytes"],
        }
        for role in sorted(observed_paths)
    ]
    observed_sources_identity = [
        {
            "role": role,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        for role, path in sorted(observed_paths.items())
    ]
    observed_identity = {
        "schema_version": "provider-input-identity/1",
        "generation_config": observed_generation,
        "sources": observed_sources_identity,
        "scenario_semantics_sha256": plan["intent"]["scenario_semantics_sha256"],
        "perturbation_rows_sha256": content_sha256(observed_rows),
    }
    expected_identity = {
        **observed_identity,
        "generation_config": expected_generation,
        "sources": expected_sources_identity,
        "perturbation_rows_sha256": content_sha256(expected_rows),
    }
    if observed_identity != expected_identity:
        raise ValueError("observed provider input identity differs from plan")
    rows = plan["rows"]
    roots = [item["run_id"] for item in rows if not item["st_id"]]
    if root_run_ids is not None:
        expected_output_dir = (
            project_root / plan["outputs"]["data_root"] / "weathergenr/output"
        )
        if (
            root_run_ids != roots
            or output_dir is None
            or Path(output_dir).resolve() != expected_output_dir
            or ancestor is not None
            or row_id is not None
            or output is not None
        ):
            raise ValueError("provider root argument mapping differs")
    else:
        selected = next((item for item in rows if item["run_id"] == row_id), None)
        if selected is None:
            raise ValueError("provider perturbed run ID differs")
        root = next(
            item
            for item in rows
            if item["rlz"] == selected["rlz"] and not item["st_id"]
        )
        series = {item["run_id"]: item["path"] for item in plan["outputs"]["series"]}
        # Every run, the root included, is transformed from its realization's
        # RAW generated series (t2608151154).
        raw = {
            item["run_id"]: item["path"]
            for item in plan["outputs"]["temporary_members"]
        }
        expected_ancestor = (project_root / raw[root["run_id"]]).resolve()
        expected_output = (project_root / series[row_id]).resolve()
        if (
            Path(ancestor).resolve(strict=True) != expected_ancestor
            or Path(output).resolve() != expected_output
        ):
            raise ValueError("provider perturbed member mapping differs")
    return observed_identity


def _installed_generator(plan: dict, project_root: Path) -> dict:
    """Render optional arguments and bind only the selected absolute execution root."""
    data = deepcopy(plan["documents"]["generation_config"]["weathergen"])
    for section, keys in (
        ("generate_weather", ("plot_dpi", "plot_device")),
        ("write_netcdf", ("file_suffix", "out_dir")),
    ):
        for key in keys:
            maybe = data[section][key]
            if maybe["present"]:
                data[section][key] = maybe["value"]
            else:
                del data[section][key]
    paths = _paths(project_root, plan)
    data["generate_weather"]["out_dir"] = str(
        (paths["data_root"] / "weathergenr").resolve()
    )
    _validate_supported_execution_choices(data)
    return data


def _validate_supported_execution_choices(data: dict) -> None:
    compression = data["write_netcdf"]["compression"]
    if type(compression) is not int or not 1 <= compression <= 9:
        raise ValueError("unsupported provider compression level")
    evaluation_cap = data["run_weather_generator"]["eval_max_grids"]
    if type(evaluation_cap) is not int or evaluation_cap < 1:
        raise ValueError("unsupported provider evaluation grid cap")
    if data["generate_weather"]["save_plots"] is True and evaluation_cap < 2:
        raise ValueError("provider plotting requires at least two evaluation grids")


def _relocated_generator(data: dict, plan: dict, project_root: Path) -> dict:
    """Rebind an installed ``out_dir`` written under another project root.

    The installed YAML is archived bytes, so a copied tree keeps the absolute
    ``out_dir`` of the root it was generated in. Only the trailing
    ``<data_root>/weathergenr`` components identify the collection; anything
    else still differs from the plan.
    """
    installed = data["generate_weather"].get("out_dir")
    tail = (*Path(plan["outputs"]["data_root"]).parts, "weathergenr")
    if not isinstance(installed, str):
        return data
    parts = tuple(part for part in re.split(r"[\\/]", installed) if part)
    if parts[-len(tail) :] != tail:
        return data
    rebound = deepcopy(data)
    rebound["generate_weather"]["out_dir"] = str(
        (_paths(project_root, plan)["data_root"] / "weathergenr").resolve()
    )
    return rebound


def validate_installed_generator(
    plan: dict, project_root: Path, *, relocatable: bool = False
) -> dict:
    """Require both exact archived bytes and semantic equality to the plan.

    ``relocatable`` admits an ``out_dir`` installed under another project root.
    Pass it only where the generator will not run again, so it cannot write
    into the root the tree was copied from.
    """
    path = _paths(project_root, plan)["generator_input"]
    # The receipt exists only after archive publication completed. Concurrent
    # Snakemake jobs can validate its immutable bytes without taking the same
    # Windows byte-range lock from several threads in one process.
    archive = validate_archive(project_root / plan["outputs"]["archive_root"])
    refs = [
        item["file"]
        for item in archive["generated_inputs"]
        if item["role"] == "weather_generation_input"
    ]
    if (
        len(refs) != 1
        or resolve_file_reference(refs[0], {"project_root": project_root}) != path
    ):
        raise ValueError("creator archive does not bind installed generator input")
    data = _decode_generator(path.read_bytes())
    compared = _relocated_generator(data, plan, project_root) if relocatable else data
    if compared != _installed_generator(plan, project_root):
        raise ValueError("installed generator arguments differ from pinned plan")
    _validate_supported_execution_choices(data)
    return data


def initialize_generation(
    project_root: Path,
    plan_path: Path,
    plan_sha256: str,
    *,
    config_path: Path,
    loaded_config: dict,
    invocation_id: str,
    lookup_path: Path,
    command: list[str],
) -> Path:
    """Install creator evidence, then publish a receipt as the last init write."""
    project_root = Path(project_root).resolve()
    plan = read_pinned_plan(plan_path, plan_sha256, project_root)
    paths = _paths(project_root, plan)
    receipt = Path(plan_path).parent / "initializations" / f"{invocation_id}.json"
    if receipt.exists():
        raise ValueError("generation initialization receipt already exists")
    if plan["decision"] == "reuse_ready":
        marker = read_collection_v2(paths["record_root"] / "collection.json")
        if marker["collection_revision"] != plan["collection_revision"]:
            raise ValueError("ready collection revision changed after planning")
        # A ready collection is never generated again, so a tree copied from
        # another root may keep its installed out_dir (t2610042152).
        validate_installed_generator(plan, project_root, relocatable=True)
        archive_ref = plan["creator_archive"]
    elif plan["decision"] == "create":
        if paths["record_root"].exists() or paths["data_root"].exists():
            raise ValueError("collection namespace appeared after create planning")
        generator = _installed_generator(plan, project_root)
        paths["record_root"].mkdir(parents=True, exist_ok=False)
        paths["data_root"].mkdir(parents=True, exist_ok=False)
        sources = _source_paths(plan, project_root)
        generator_path = paths["generator_input"]
        generator_path.parent.mkdir(parents=True)
        with generator_path.open("xb") as handle:
            handle.write(yaml.safe_dump(generator, sort_keys=False).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        # The installed YAML is an execution artifact. Its contents must be
        # checked against the plan before source capture and again before R.
        if _decode_generator(generator_path.read_bytes()) != generator:
            raise ValueError("installed generator input differs from plan")
        source_config = capture_configuration_sources(
            Path(config_path), "generate_scenarios", PROJECTION
        )
        for source in source_config:
            if source.original_path.read_bytes() != source.data:
                raise ValueError("WF3 config source changed during archive capture")
        current_config, _ = compose_config(
            yaml.safe_load(Path(config_path).read_bytes()),
            Path(config_path),
            entry="generate_scenarios",
            declared_sections=PROJECTION,
        )
        if current_config != loaded_config:
            raise ValueError("WF3 config changed after plan selection")
        references = [
            {
                "id": role,
                "role": role,
                "original_path": next(
                    item["original_path"]
                    for item in plan["candidate_intent"]["documents"][
                        "source_inventory"
                    ]["sources"]
                    if item["role"] == role
                ),
                "source_id": None,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for role, path in sorted(sources.items())
        ]
        record, payloads = run_record_document(
            workflow="generate_scenarios",
            invocation_id=invocation_id,
            owner_kind="scenario_collection",
            owner_id=plan["collection_id"],
            loaded_config=loaded_config,
            advanced_settings=ADVANCED_SETTINGS,
            projection=PROJECTION,
            toolbox=toolbox_identity(),
            environment=environment_file_hashes(),
            invocation={
                "entry_point": "scripts/generate_scenarios.py",
                "command": command,
                "targets": ["all"],
                "working_directory": str(Path.cwd()),
                "overrides": {},
            },
            sources=source_config,
            referenced_inputs=references,
            generated_inputs=[
                {
                    "role": "weather_generation_input",
                    "file": file_reference(
                        generator_path, "project_root", project_root
                    ),
                }
            ],
        )
        publish_archive(
            project_root,
            plan["collection_id"][:12],
            plan["outputs"]["archive_root"],
            record,
            payloads,
        )
        validate_installed_generator(plan, project_root)
        atomic_record(paths["record_root"] / "collection_intent.json", plan["intent"])
        shutil.copyfile(lookup_path, paths["perturbation_lookup"])
        with paths["scenario_run_lookup"].open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=SCENARIO_HEADER)
            writer.writeheader()
            writer.writerows({**row, "evaluate": "true"} for row in plan["rows"])
        archive_ref = file_reference(
            paths["archive_root"] / "run_record.yml", "project_root", project_root
        )
    else:
        raise ValueError("unknown generation plan decision")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_record(
        receipt,
        {
            "schema_version": "generation-initialization/2",
            "invocation_id": invocation_id,
            "generation_request_id": plan["generation_request_id"],
            "plan_sha256": plan_sha256,
            "collection_id": plan["collection_id"],
            "intent_sha256": plan["intent_sha256"],
            "candidate_intent_sha256": plan["candidate_intent_sha256"],
            "creator_archive": archive_ref,
            "decision": plan["decision"],
            "collection_revision": plan["collection_revision"],
            "workspace_path": (paths["data_root"] / "weathergenr")
            .relative_to(project_root)
            .as_posix(),
            "claim_token": uuid.uuid4().hex,
        },
    )
    return receipt


def publish_generation(
    project_root: Path, plan_path: Path, plan_sha256: str, invocation_id: str
) -> Path:
    """Validate every retained product and publish the ready marker last."""
    project_root = Path(project_root).resolve()
    plan = read_pinned_plan(plan_path, plan_sha256, project_root)
    paths = _paths(project_root, plan)
    marker_path = paths["record_root"] / "collection.json"
    receipt_path = Path(plan_path).parent / "initializations" / f"{invocation_id}.json"
    receipt = read_canonical_json(receipt_path)
    expected_archive = file_reference(
        project_root / plan["outputs"]["archive_root"] / "run_record.yml",
        "project_root",
        project_root,
    )
    expected_workspace = (
        (paths["data_root"] / "weathergenr").relative_to(project_root).as_posix()
    )
    if (
        receipt["schema_version"] != "generation-initialization/2"
        or receipt["plan_sha256"] != plan_sha256
        or receipt["collection_id"] != plan["collection_id"]
        or receipt["invocation_id"] != invocation_id
        or receipt["intent_sha256"] != plan["intent_sha256"]
        or receipt["candidate_intent_sha256"] != plan["candidate_intent_sha256"]
        or receipt["generation_request_id"] != plan["generation_request_id"]
        or receipt["decision"] != plan["decision"]
        or receipt["creator_archive"] != expected_archive
        or receipt["collection_revision"] != plan["collection_revision"]
        or receipt["workspace_path"] != expected_workspace
        or not isinstance(receipt["claim_token"], str)
        or re.fullmatch("[0-9a-f]{32}", receipt["claim_token"]) is None
    ):
        raise ValueError("generation receipt differs from pinned plan")
    if plan["decision"] == "reuse_ready":
        marker = read_collection_v2(marker_path)
        if marker["collection_revision"] != plan["collection_revision"]:
            raise ValueError("ready collection revision changed after planning")
        return marker_path
    if marker_path.exists():
        raise ValueError("ready marker appeared before publication")
    source_paths = _source_paths(plan, project_root)
    validate_provider_inputs(
        plan,
        project_root,
        historical=source_paths["historical_climate"],
        cells=source_paths["basin_cells"],
        generator_yaml=paths["generator_input"],
        lookup=paths["perturbation_lookup"],
        root_run_ids=[row["run_id"] for row in plan["rows"] if not row["st_id"]],
        output_dir=paths["data_root"] / "weathergenr/output",
    )
    for path in _source_paths(plan, project_root).values():
        # Resolve already checks the reference bytes; re-reading before marker
        # publication catches tampering after the provider consumed a snapshot.
        if not path.is_file():
            raise ValueError(f"pinned source disappeared: {path}")
    source_inventory = plan["intent"]["documents"]["source_inventory"]
    interpretation = source_inventory["interpretation"]
    reader = {
        "metadata": {
            "cst_unit_interpretation": {
                "revision": interpretation["revision"],
                "evidence": "generation source inventory",
                "variables": [
                    [item["name"], item["units"]]
                    for item in interpretation["variables"]
                ],
            }
        }
    }
    series = []
    for item in plan["outputs"]["series"]:
        path = project_root / item["path"]
        series.append(
            {
                "run_id": item["run_id"],
                "file": file_reference(path, "project_root", project_root),
                "descriptor": collection_forcing_descriptor(path, reader),
            }
        )
    date_products = [
        {
            "role": item["role"],
            "file": file_reference(
                project_root / item["path"], "project_root", project_root
            ),
        }
        for item in plan["outputs"]["date_products"]
    ]
    date_products.append(
        {
            "role": "weather_generation_input",
            "file": file_reference(
                paths["generator_input"], "project_root", project_root
            ),
        }
    )
    diagnostic_root = paths["diagnostic_root"]
    diagnostics = (
        [
            file_reference(path, "project_root", project_root)
            for path in sorted(diagnostic_root.rglob("*"))
            if path.is_file()
        ]
        if diagnostic_root.exists()
        else []
    )
    marker = {
        "schema_version": "scenario-collection/3",
        "canonicalization_id": "collection-canon/1",
        "status": "ready",
        "collection_id": plan["collection_id"],
        "collection_revision": "0" * 64,
        "data_root": plan["outputs"]["data_root"],
        "intent": file_reference(
            paths["record_root"] / "collection_intent.json",
            "project_root",
            project_root,
        ),
        "archive": receipt["creator_archive"],
        "scenario_run_lookup": file_reference(
            paths["scenario_run_lookup"], "project_root", project_root
        ),
        "perturbation_lookup": file_reference(
            paths["perturbation_lookup"], "project_root", project_root
        ),
        "series": series,
        "provider_products": date_products,
        "diagnostics": diagnostics,
    }
    marker["collection_revision"] = collection_revision_v2(marker, plan["intent"])
    atomic_record(marker_path, marker)
    read_collection_v2(marker_path)
    return marker_path
