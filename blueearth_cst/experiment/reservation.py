"""Numbered experiment reservation under caller-held project ownership.

Writing entry points require ``archive_lock(root, 'project-execution')`` for
their entire invocation. They deliberately do not acquire or release that lock:
the runner retains ownership through all producers. Preview is advisory and
never grants writing authority. This module does not construct scientific
requests or execute producers.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from blueearth_cst.experiment.catalog_identity import legacy_catalog_input
from blueearth_cst.experiment.content_identity import (
    atomic_record,
    canonical_json_bytes,
    confined_path,
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.shared.provenance import toolbox_identity
from blueearth_cst.shared.snake_utils import validate_experiment_name
from blueearth_cst.shared.workflow_config_snapshot import (
    _REPLACE_RETRY_ATTEMPTS,
    _REPLACE_RETRY_BASE_SECONDS,
    _REPLACE_RETRY_CAP_SECONDS,
    file_reference,
    resolve_file_reference,
    validate_archive,
)
from blueearth_cst.shared.yaml_documents import parse_yaml_document

_ALLOCATION_PATH = "experiments/_engine/allocation.json"
_TRANSACTIONS_PATH = "experiments/_engine/transactions"


@dataclass(frozen=True)
class CapturedSource:
    """Original file bytes already used by the caller to form its request."""

    role: str
    original_path: Path
    data: bytes


@dataclass(frozen=True)
class Selection:
    action: str
    project_root: Path
    name: str | None
    family: str | None
    number: int | None
    reason: str
    reservation_id: str | None = None
    advisory: bool = False


def _fields(value: Any, expected: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{label}: expected exactly {sorted(expected)}")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label}: expected integer >= {minimum}")
    return value


def _text(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label}: expected nonempty text")
    return value


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label}: expected lowercase SHA256")
    return value


def _uuid(value: Any, label: str) -> str:
    if type(value) is not str or re.fullmatch("[0-9a-f]{32}", value) is None:
        raise ValueError(f"{label}: expected lowercase UUID4 hex")
    if uuid.UUID(hex=value).version != 4:
        raise ValueError(f"{label}: expected UUID4")
    return value


def _timestamp(value: Any) -> str:
    if (
        type(value) is not str
        or re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z",
            value,
        )
        is None
    ):
        raise ValueError("created_at: expected UTC RFC3339 timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp is not UTC")
    return value


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _no_links(path: Path) -> None:
    """Check every existing ancestor before resolving, including junctions."""
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            if (
                part.is_symlink()
                or getattr(info, "st_file_attributes", 0)
                & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ):
                raise ValueError(f"aliased/reparse path is forbidden: {part}")


def _root(value: Path | str) -> Path:
    path = Path(value).absolute()
    _no_links(path)
    if path.exists() and not path.is_dir():
        raise ValueError(f"project root is not a directory: {path}")
    return path.resolve()


def _path(root: Path, relative: str) -> Path:
    lexical = root / relative
    _no_links(lexical)
    return confined_path(root, relative)


def _name(value: Any) -> str:
    return validate_experiment_name(value, Path("."))


def parse_numbered_name(name: str) -> tuple[str, int] | None:
    """Parse canonical numeric pins; dated legacy precedence is caller-owned."""
    _name(name)
    match = re.fullmatch(r"(.+)_([0-9]+)", name)
    if match is None:
        return None
    base, digits = match.groups()
    number = int(digits)
    if number < 1 or digits != f"{number:02d}":
        raise ValueError(f"noncanonical experiment number: {name}")
    _name(base)
    return base, number


def _dated(name: str) -> bool:
    return re.fullmatch(r".+_[0-9]{8}(?:_v[0-9]+)?", name) is not None


def _ref(value: Any, root: Path, *, verify: bool = False) -> dict:
    _fields(
        value,
        {"schema_version", "path_base", "path", "sha256", "size_bytes"},
        "file reference",
    )
    if (
        value["schema_version"] != "artifact-reference/1"
        or value["path_base"] != "project_root"
    ):
        raise ValueError("reference must use artifact-reference/1 and project_root")
    _digest(value["sha256"], "reference digest")
    _integer(value["size_bytes"], "reference size")
    _path(root, value["path"])
    if verify:
        resolve_file_reference(value, {"project_root": root})
    return value


def _window(value: Any, label: str) -> None:
    _fields(value, {"start", "end"}, label)
    if _integer(value["end"], label, 1) < _integer(value["start"], label, 1):
        raise ValueError(f"{label}: end precedes start")


def _local_inputs(value: Any) -> None:
    if type(value) is not list:
        raise ValueError("local_inputs must be a list")
    roles = []
    for item in value:
        _fields(item, {"role", "path", "sha256"}, "semantic local input")
        if item["role"] not in {"region", "output_locations"}:
            raise ValueError("unknown local input role")
        path = _text(item["path"], "local input path")
        if not Path(path).is_absolute() or Path(path).as_posix() != path:
            raise ValueError("local input path must be absolute POSIX")
        _digest(item["sha256"], "local input digest")
        roles.append(item["role"])
    if roles != sorted(set(roles)):
        raise ValueError("local input roles must be unique and sorted")


def validate_request(request: dict, project_root: Path | str) -> dict:
    """Validate the closed reservation envelope; upstream values are caller-normalized.

    The caller owns effective region, HydroMT catalogs and build-document
    validation using their execution parsers. No substitute physics schema is
    imposed here. WF3 perturbation and generator classifications are reused.
    """
    root = _root(project_root)
    _fields(
        request,
        {"schema_version", "generation", "selection", "input_catalogs", "simulation"},
        "request",
    )
    canonical_json_bytes(request)
    if request["schema_version"] not in (
        "experiment-request/1",
        "experiment-request/2",
    ):
        raise ValueError("unsupported reservation request")
    selection = _fields(request["selection"], {"mode", "manifest"}, "selection")
    if selection["mode"] not in {"project-generation", "explicit-manifest"}:
        raise ValueError("unsupported selection mode")
    if (selection["mode"] == "explicit-manifest") != (
        selection["manifest"] is not None
    ):
        raise ValueError("manifest presence disagrees with selection mode")
    if selection["manifest"] is not None:
        _ref(selection["manifest"], root)
    catalogs = request["input_catalogs"]
    if type(catalogs) is not list or any(type(item) is not dict for item in catalogs):
        raise ValueError("input_catalogs must contain semantic catalog documents")
    if request["schema_version"] == "experiment-request/2":
        for catalog in catalogs:
            _fields(catalog, {"document", "root"}, "catalog input")
            if type(catalog["document"]) is not dict:
                raise ValueError("catalog document must be a mapping")
            _text(catalog["root"], "catalog root")
    generation = request["generation"]
    if generation is None and selection["mode"] == "project-generation":
        raise ValueError("unresolved project-generation requires generation request")
    if generation is not None:
        from blueearth_cst.experiment.generation_plan import GENERATOR_SEED_FIELDS
        from blueearth_cst.experiment.scenario_collection_v2 import _perturbations

        _fields(
            generation,
            {
                "schema_version",
                "n_realizations",
                "simulation_window",
                "climate_perturbations",
                "water_year_start",
                "requested_seed",
                "generator_settings",
                "run_group_id_capacity",
                "source_request",
                "provider_code",
            },
            "generation request",
        )
        if generation["schema_version"] != "reservation-generation-request/1":
            raise ValueError("unsupported generation request")
        _integer(generation["n_realizations"], "n_realizations", 1)
        _integer(generation["run_group_id_capacity"], "run_group_id_capacity", 1)
        _window(generation["simulation_window"], "generation simulation_window")
        _perturbations(generation["climate_perturbations"])
        if (
            generation["water_year_start"]
            not in "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
        ):
            raise ValueError("water_year_start must be uppercase month")
        if generation["requested_seed"] != "auto":
            _integer(generation["requested_seed"], "requested_seed")
        settings = generation["generator_settings"]
        _fields(settings, set(GENERATOR_SEED_FIELDS), "generator settings")
        for section, (included, _) in GENERATOR_SEED_FIELDS.items():
            _fields(settings[section], included, f"generator settings.{section}")
        source = _fields(
            generation["source_request"],
            {
                "region",
                "climate_source",
                "historical_window",
                "hydrography",
                "basin_index",
                "local_inputs",
            },
            "source request",
        )
        for key in ("climate_source", "hydrography", "basin_index"):
            _text(source[key], key)
        _window(source["historical_window"], "historical_window")
        _local_inputs(source["local_inputs"])
        code = generation["provider_code"]
        if type(code) is not list or not code:
            raise ValueError("provider_code must be a nonempty inventory")
        paths = []
        for item in code:
            _fields(item, {"path", "sha256"}, "provider code")
            _path(root, item["path"])
            _digest(item["sha256"], "provider code digest")
            paths.append(item["path"])
        if paths != sorted(set(paths)):
            raise ValueError("provider code paths must be unique and sorted")
    simulation = request["simulation"]
    from blueearth_cst.experiment.simulation_runs import include_unperturbed

    include_unperturbed(simulation)
    _fields(
        {
            key: value
            for key, value in simulation.items()
            if key != "include_unperturbed"
        },
        {"climate_source", "model_root", "outvars", "model_request"},
        "declared simulation",
    )
    _text(simulation["climate_source"], "climate_source")
    _path(root, simulation["model_root"])
    outvars = simulation["outvars"]
    if (
        type(outvars) is not list
        or not outvars
        or any(type(v) is not str or not v for v in outvars)
        or len(set(outvars)) != len(outvars)
    ):
        raise ValueError("outvars must be unique nonempty canonical variable names")
    model = simulation["model_request"]
    if model is not None:
        _fields(
            model,
            {"basin", "climate", "engine", "simulation_window", "local_inputs"},
            "model request",
        )
        _fields(model["climate"], {"selected", "window"}, "model climate")
        _text(model["climate"]["selected"], "selected climate")
        _window(model["climate"]["window"], "model climate window")
        _fields(model["engine"], {"build_config", "waterbodies_config"}, "model engine")
        _window(model["simulation_window"], "model simulation window")
        _local_inputs(model["local_inputs"])
    return request


def compare_request(retained: Any, requested: Any) -> None:
    """Refuse the first changed scientific field, preserving JSON types."""

    # V2 keeps source YAML and root context, leaving adapter validation to the
    # workflow that consumes the catalog. Compare legacy records through that
    # explicit projection; their original request and digest remain immutable.
    if (
        isinstance(retained, dict)
        and isinstance(requested, dict)
        and (retained.get("schema_version"), requested.get("schema_version"))
        in (
            ("experiment-request/1", "experiment-request/2"),
            ("experiment-request/2", "experiment-request/1"),
        )
    ):

        def current_view(request: dict) -> dict:
            if request["schema_version"] == "experiment-request/2":
                return request
            return {
                **request,
                "schema_version": "experiment-request/2",
                "input_catalogs": [
                    legacy_catalog_input(item) for item in request["input_catalogs"]
                ],
            }

        retained, requested = current_view(retained), current_view(requested)

    def compare(old: Any, new: Any, path: str) -> None:
        if canonical_json_bytes(old) == canonical_json_bytes(new):
            return
        if type(old) is dict and type(new) is dict and set(old) == set(new):
            for key in sorted(old):
                compare(old[key], new[key], f"{path}.{key}")
        elif type(old) is list and type(new) is list and len(old) == len(new):
            for index, (left, right) in enumerate(zip(old, new, strict=True)):
                compare(left, right, f"{path}[{index}]")
        raise ValueError(
            f"scientific request differs at {path}: {old!r} -> {new!r}; use retained creator inputs or --new-experiment"
        )

    compare(retained, requested, "request")


def _creation(value: Any) -> dict:
    _fields(
        value,
        {
            "schema_version",
            "name",
            "family",
            "number",
            "reservation_id",
            "created_at",
            "creator_invocation_id",
            "note",
            "toolbox",
        },
        "creation",
    )
    if value["schema_version"] != "experiment-creation/1":
        raise ValueError("unsupported experiment metadata")
    _name(value["name"])
    _name(value["family"])
    if (
        value["name"]
        != f"{value['family']}_{_integer(value['number'], 'number', 1):02d}"
    ):
        raise ValueError("creation name/family/number differ")
    _uuid(value["reservation_id"], "reservation_id")
    _uuid(value["creator_invocation_id"], "creator_invocation_id")
    _timestamp(value["created_at"])
    if value["note"] is not None and type(value["note"]) is not str:
        raise ValueError("creation note must be text or null")
    toolbox = _fields(value["toolbox"], {"commit", "commit_source", "dirty"}, "toolbox")
    if toolbox["commit"] is not None:
        _text(toolbox["commit"], "toolbox commit")
    if toolbox["commit_source"] not in {None, "git", "baked"} or type(
        toolbox["dirty"]
    ) not in {bool, type(None)}:
        raise ValueError("invalid toolbox identity")
    return value


def _seed(value: Any, request: dict, root: Path, *, verify: bool = False) -> None:
    _fields(
        value,
        {
            "state",
            "requested",
            "resolved",
            "seed_resolution",
            "plan",
            "source_identity_sha256",
        },
        "seed",
    )
    generation = request["generation"]
    if generation is None:
        if value != {
            "state": "not_applicable",
            "requested": None,
            "resolved": None,
            "seed_resolution": None,
            "plan": None,
            "source_identity_sha256": None,
        }:
            raise ValueError("seed must be not_applicable without generation")
        return
    if canonical_json_bytes(value["requested"]) != canonical_json_bytes(
        generation["requested_seed"]
    ):
        raise ValueError("seed request differs from initial request")
    if value["state"] == "awaiting_sources":
        if any(
            value[field] is not None
            for field in (
                "resolved",
                "seed_resolution",
                "plan",
                "source_identity_sha256",
            )
        ):
            raise ValueError("awaiting_sources seed has binding evidence")
    elif value["state"] == "bound":
        _integer(value["resolved"], "resolved seed")
        _digest(value["source_identity_sha256"], "source identity")
        _ref(value["plan"], root, verify=verify)
        resolution = _fields(
            value["seed_resolution"],
            {
                "seed_request",
                "resolved_seed",
                "projection",
                "projection_sha256",
                "seed_resolution_id",
            },
            "seed resolution",
        )
        _digest(resolution["projection_sha256"], "seed projection")
        _digest(resolution["seed_resolution_id"], "seed resolution id")
        if (
            content_sha256(resolution["projection"]) != resolution["projection_sha256"]
            or content_sha256(
                {k: v for k, v in resolution.items() if k != "seed_resolution_id"}
            )
            != resolution["seed_resolution_id"]
        ):
            raise ValueError("seed resolution digests differ")
        if (
            canonical_json_bytes(resolution["seed_request"])
            != canonical_json_bytes(value["requested"])
            or type(resolution["resolved_seed"]) is not int
            or resolution["resolved_seed"] != value["resolved"]
        ):
            raise ValueError("seed resolution values differ")
        if verify:
            from blueearth_cst.experiment.generation_plan import read_pinned_plan
            from blueearth_cst.experiment.scenario_collection_v2 import _profile

            plan = read_pinned_plan(
                root / value["plan"]["path"], value["plan"]["sha256"], root
            )
            candidate = plan["candidate_intent"]
            _profile(candidate)
            config = candidate["documents"]["generation_config"]
            if (
                canonical_json_bytes(config["seed_resolution"])
                != canonical_json_bytes(resolution)
                or candidate["identity_digests"]["source_inventory"]
                != value["source_identity_sha256"]
            ):
                raise ValueError("seed binding differs from pinned plan")
            projection = resolution["projection"]
            for intent in (plan["intent"], candidate):
                for source in intent["documents"]["source_inventory"]["sources"]:
                    _ref(source["file"], root, verify=True)
            for key in (
                "n_realizations",
                "simulation_window",
                "climate_perturbations",
                "water_year_start",
                "generator_settings",
            ):
                compare_request(generation[key], projection[key])
            if config["run_group_id_capacity"] != generation[
                "run_group_id_capacity"
            ] or canonical_json_bytes(
                candidate["documents"]["provider_code"]
            ) != canonical_json_bytes(generation["provider_code"]):
                raise ValueError("bound plan capacity/provider differ from request")
    else:
        raise ValueError("unsupported seed state")


def _resolution(
    value: Any, root: Path, *, verify: bool = False, include: bool = True
) -> None:
    _fields(
        value,
        {
            "resolution_mode",
            "manifest",
            "collection_id",
            "collection_revision",
            "run_ids",
        },
        "resolution",
    )
    if value["resolution_mode"] not in {"project-generation", "explicit-manifest"}:
        raise ValueError("unsupported resolution mode")
    _ref(value["manifest"], root, verify=verify)
    _digest(value["collection_id"], "collection_id")
    _digest(value["collection_revision"], "collection_revision")
    runs = value["run_ids"]
    if (
        type(runs) is not list
        or not runs
        or any(
            type(v) is not str or re.fullmatch(r"run_[0-9]+", v) is None for v in runs
        )
        or len({len(v) for v in runs}) != 1
        or any(int(v[4:]) < 1 for v in runs)
        or [int(v[4:]) for v in runs] != sorted({int(v[4:]) for v in runs})
    ):
        raise ValueError(
            "run_ids must be unique, consistently padded run_ IDs in numeric order"
        )
    if verify:
        from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2

        marker = read_collection_v2(root / value["manifest"]["path"])
        from blueearth_cst.experiment.simulation_runs import selected_run_ids

        if (
            marker["collection_id"] != value["collection_id"]
            or marker["collection_revision"] != value["collection_revision"]
            or selected_run_ids(
                marker, root / value["manifest"]["path"], include=include
            )
            != runs
        ):
            raise ValueError("resolution differs from ready collection")


def _simulation(
    value: Any, root: Path, name: str, resolution: dict, *, verify: bool = False
) -> None:
    _fields(value, {"simulation_id", "intent", "creator_archive"}, "simulation binding")
    _digest(value["simulation_id"], "simulation_id")
    for key in ("intent", "creator_archive"):
        _ref(value[key], root, verify=verify)
    if (
        value["intent"]["path"] != f"experiments/{name}/_engine/simulation_intent.json"
        or value["creator_archive"]["path"]
        != f"experiments/{name}/config/run_record.yml"
    ):
        raise ValueError("simulation binding points to another experiment")
    if verify:
        from blueearth_cst.experiment.simulation_record import read_simulation_intent_v2

        intent = read_simulation_intent_v2(root / "experiments" / name)
        archive = validate_archive((root / value["creator_archive"]["path"]).parent)
        if intent["simulation_id"] != value["simulation_id"] or archive["owner"] != {
            "kind": "experiment",
            "id": value["simulation_id"],
        }:
            raise ValueError("simulation intent/creator archive disagree")
        compare_request(intent["collection"], resolution)


def validate_reservation(
    value: Any, project_root: Path | str, *, verify: bool = False
) -> dict:
    root = _root(project_root)
    _fields(
        value,
        {
            "schema_version",
            "canonicalization_id",
            "reservation_id",
            "name",
            "revision",
            "state",
            "request",
            "request_sha256",
            "seed",
            "resolution",
            "simulation",
        },
        "reservation",
    )
    if (
        value["schema_version"] != "experiment-reservation/1"
        or value["canonicalization_id"] != "collection-canon/1"
    ):
        raise ValueError("unsupported reservation")
    _uuid(value["reservation_id"], "reservation_id")
    _name(value["name"])
    _integer(value["revision"], "reservation revision")
    validate_request(value["request"], root)
    _digest(value["request_sha256"], "request_sha256")
    if content_sha256(value["request"]) != value["request_sha256"]:
        raise ValueError("reservation request digest differs")
    _seed(value["seed"], value["request"], root, verify=verify)
    resolution, simulation = value["resolution"], value["simulation"]
    expected = (
        "pending"
        if resolution is None
        else "resolved"
        if simulation is None
        else "simulation_bound"
    )
    if value["state"] != expected or (simulation is not None and resolution is None):
        raise ValueError("reservation state disagrees with bindings")
    if resolution is not None:
        from blueearth_cst.experiment.simulation_runs import include_unperturbed

        _resolution(
            resolution,
            root,
            verify=verify,
            include=include_unperturbed(value["request"]["simulation"]),
        )
        if value["seed"]["state"] == "awaiting_sources":
            raise ValueError("resolution cannot precede generation seed binding")
        if verify and value["seed"]["state"] == "bound":
            plan = read_canonical_json(root / value["seed"]["plan"]["path"])
            if (
                value["request"]["selection"]["mode"] == "project-generation"
                and plan["collection_id"] != resolution["collection_id"]
            ):
                raise ValueError("resolution differs from bound generation plan")
    if simulation is not None:
        _simulation(simulation, root, value["name"], resolution, verify=verify)
    return value


def _ledger(root: Path) -> dict:
    path = _path(root, _ALLOCATION_PATH)
    if not path.exists():
        return {
            "schema_version": "experiment-allocation/1",
            "revision": 0,
            "families": [],
        }
    value = read_canonical_json(path)
    _fields(value, {"schema_version", "revision", "families"}, "allocation ledger")
    if value["schema_version"] != "experiment-allocation/1":
        raise ValueError("unsupported allocation ledger")
    _integer(value["revision"], "ledger revision")
    if type(value["families"]) is not list:
        raise ValueError("families must be a list")
    bases, names, ids = [], [], []
    for family in value["families"]:
        _fields(family, {"base", "high_water", "allocations"}, "family")
        bases.append(_name(family["base"]))
        _integer(family["high_water"], "high_water", 1)
        allocations = family["allocations"]
        if type(allocations) is not list or not allocations:
            raise ValueError("family allocations must be nonempty")
        numbers = []
        for allocation in allocations:
            _fields(
                allocation,
                {
                    "number",
                    "name",
                    "reservation_id",
                    "origin",
                    "created_at",
                    "publication",
                    "presence",
                    "transaction",
                    "legacy_owner",
                    "tombstoned_at",
                },
                "allocation",
            )
            number = _integer(allocation["number"], "allocation number", 1)
            numbers.append(number)
            name = _name(allocation["name"])
            names.append(name)
            if name != f"{family['base']}_{number:02d}":
                raise ValueError("ledger name/family/number disagree")
            _timestamp(allocation["created_at"])
            if allocation["presence"] not in {"present", "tombstone"} or allocation[
                "publication"
            ] not in {"prepared", "published"}:
                raise ValueError("unsupported allocation status")
            if (allocation["presence"] == "tombstone") != (
                allocation["tombstoned_at"] is not None
            ):
                raise ValueError("tombstone timestamp disagrees")
            if allocation["tombstoned_at"] is not None:
                _timestamp(allocation["tombstoned_at"])
                if allocation["publication"] != "published":
                    raise ValueError("prepared allocation cannot be tombstoned")
            if allocation["origin"] == "reservation":
                ids.append(_uuid(allocation["reservation_id"], "reservation_id"))
                _ref(allocation["transaction"], root)
                if (
                    allocation["transaction"]["path"]
                    != f"{_TRANSACTIONS_PATH}/{allocation['reservation_id']}/transaction.json"
                    or allocation["legacy_owner"] is not None
                ):
                    raise ValueError("reservation allocation owner disagrees")
            elif allocation["origin"] == "legacy":
                _ref(allocation["legacy_owner"], root)
                if (
                    allocation["transaction"] is not None
                    or allocation["reservation_id"] is not None
                    or allocation["publication"] != "published"
                ):
                    raise ValueError("legacy allocation carries reservation evidence")
                if allocation["legacy_owner"]["path"] not in {
                    f"experiments/{name}/_engine/simulation.json",
                    f"experiments/{name}/_engine/simulation_intent.json",
                }:
                    raise ValueError("legacy owner points to another experiment")
            else:
                raise ValueError("unsupported allocation origin")
        if numbers != sorted(set(numbers)) or family["high_water"] != max(numbers):
            raise ValueError("allocation ordering/high_water disagree")
    if (
        bases != sorted(set(bases))
        or len(names) != len(set(names))
        or len(ids) != len(set(ids))
    ):
        raise ValueError(
            "ledger families/names/reservation IDs are duplicate or unsorted"
        )
    return value


def _legacy_owner(root: Path, name: str) -> dict:
    from blueearth_cst.experiment.simulation_record import (
        read_simulation_intent_v2,
        read_simulation_v2,
    )

    directory = _path(root, f"experiments/{name}")
    if not directory.is_dir():
        raise ValueError(f"occupied experiment is not a directory: {directory}")
    marker = _path(root, f"experiments/{name}/_engine/simulation.json")
    intent_path = _path(root, f"experiments/{name}/_engine/simulation_intent.json")
    archive_path = _path(root, f"experiments/{name}/config/run_record.yml")
    if marker.exists():
        read_simulation_v2(directory)
        return file_reference(marker, "project_root", root)
    if intent_path.exists() and archive_path.exists():
        intent = read_simulation_intent_v2(directory)
        archive = validate_archive(archive_path.parent)
        if archive["owner"] != {"kind": "experiment", "id": intent["simulation_id"]}:
            raise ValueError("legacy intent/creator archive disagree")
        return file_reference(intent_path, "project_root", root)
    raise ValueError(f"unknown/unpaired experiment occupant: {directory}")


def _add_allocation(ledger: dict, base: str, allocation: dict) -> None:
    family = next((f for f in ledger["families"] if f["base"] == base), None)
    if family is None:
        family = {"base": base, "high_water": allocation["number"], "allocations": []}
        ledger["families"].append(family)
        ledger["families"].sort(key=lambda f: f["base"])
    family["allocations"].append(allocation)
    family["allocations"].sort(key=lambda a: a["number"])
    family["high_water"] = max(family["high_water"], allocation["number"])


def _reconcile(root: Path, ledger: dict) -> None:
    directory = _path(root, "experiments")
    first = not _path(root, _ALLOCATION_PATH).exists()
    known = {a["name"]: a for f in ledger["families"] for a in f["allocations"]}
    if directory.exists() and not directory.is_dir():
        raise ValueError("experiments root is not a directory")
    for child in directory.iterdir() if directory.exists() else ():
        _no_links(child)
        if child.name == "_engine":
            if not child.is_dir():
                raise ValueError("experiment engine root is not a directory")
            continue
        _name(child.name)
        if _dated(child.name):
            continue
        pin = parse_numbered_name(child.name)
        if pin is None:
            continue
        if child.name in known:
            continue
        if not first:
            raise ValueError(f"unledgered numbered occupant: {child}")
        owner = _legacy_owner(root, child.name)
        _add_allocation(
            ledger,
            pin[0],
            {
                "number": pin[1],
                "name": child.name,
                "reservation_id": None,
                "origin": "legacy",
                "created_at": _now(),
                "publication": "published",
                "presence": "present",
                "transaction": None,
                "legacy_owner": owner,
                "tombstoned_at": None,
            },
        )
    for allocation in known.values():
        target = _path(root, f"experiments/{allocation['name']}")
        if target.exists() and not target.is_dir():
            raise ValueError(f"experiment target is not a directory: {target}")
        if (
            allocation["publication"] == "published"
            and allocation["presence"] == "present"
            and not target.exists()
        ):
            allocation["presence"] = "tombstone"
            allocation["tombstoned_at"] = _now()


def _save_ledger(root: Path, ledger: dict) -> None:
    path = _path(root, _ALLOCATION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger["revision"] += 1
    atomic_record(path, ledger, replace=path.exists())


def _allocation(ledger: dict, name: str) -> dict | None:
    return next(
        (a for f in ledger["families"] for a in f["allocations"] if a["name"] == name),
        None,
    )


def _flags(
    new: bool, note: str | None, indicators: bool, enabled: bool, recover: bool
) -> None:
    if any(type(flag) is not bool for flag in (new, indicators, enabled, recover)):
        raise ValueError("operation flags must be booleans")
    if note is not None and type(note) is not str:
        raise ValueError("note must be text or null")
    if note is not None and not new:
        raise ValueError("--note requires --new-experiment")
    if not enabled and (new or note is not None or recover):
        raise ValueError("WF4 disabled with experiment creation/note/recovery flags")
    if indicators and (new or note is not None or recover):
        raise ValueError("indicators-only cannot create or recover reservations")
    if recover and (new or note is not None):
        raise ValueError("recovery cannot combine with creation/note")


def _select(
    root: Path,
    ledger: dict,
    requested: str | None,
    *,
    new: bool,
    indicators: bool,
    recover: bool,
) -> Selection:
    name = "stress_test" if requested is None else _name(requested)
    exact = _path(root, f"experiments/{name}")
    allocated = _allocation(ledger, name)
    if allocated is None and exact.exists():
        _legacy_owner(root, name)
        if indicators:
            from blueearth_cst.experiment.simulation_record import read_simulation_v2

            read_simulation_v2(exact)
        if _dated(name):
            if new:
                raise ValueError(
                    "dated legacy name cannot be a new family; choose a different base"
                )
            return Selection(
                "legacy", root, name, None, None, "exact owned dated legacy name"
            )
        if parse_numbered_name(name) is None and not new:
            return Selection(
                "legacy", root, name, None, None, "exact owned unnumbered legacy name"
            )
    pin = parse_numbered_name(name)
    if pin is not None and new:
        raise ValueError("numbered pin cannot combine with --new-experiment")
    base = pin[0] if pin else name
    family = next((f for f in ledger["families"] if f["base"] == base), None)
    number = (
        pin[1]
        if pin
        else (family["high_water"] + 1 if new else family["high_water"])
        if family
        else 1
    )
    selected = _name(f"{base}_{number:02d}")
    allocation = _allocation(ledger, selected)
    if allocation is None:
        if recover:
            raise ValueError("no committed allocation exists for recovery")
        if indicators:
            return Selection(
                "missing",
                root,
                selected,
                base,
                number,
                "indicators-only never allocates",
            )
        return Selection(
            "create",
            root,
            selected,
            base,
            number,
            "unused pin" if pin else "new family number",
            advisory=True,
        )
    if allocation["presence"] == "tombstone":
        return Selection(
            "missing",
            root,
            selected,
            base,
            number,
            "allocation is permanently tombstoned",
            allocation["reservation_id"],
        )
    if allocation["publication"] == "prepared":
        return Selection(
            "continue" if recover else "refuse",
            root,
            selected,
            base,
            number,
            "interrupted publication; use --recover-reservation",
            allocation["reservation_id"],
        )
    if allocation["origin"] == "legacy":
        _ref(allocation["legacy_owner"], root, verify=True)
        _legacy_owner(root, selected)
        action = "legacy"
    else:
        _read_pair(root, allocation)
        action = "continue"
    if indicators:
        from blueearth_cst.experiment.simulation_record import read_simulation_v2

        read_simulation_v2(root / "experiments" / selected)
    return Selection(
        action,
        root,
        selected,
        base,
        number,
        "highest allocation" if not pin else "exact numbered pin",
        allocation["reservation_id"],
    )


def preview_experiment(
    project_root: Path | str,
    requested_name: str | None = None,
    *,
    new_experiment: bool = False,
    note: str | None = None,
    indicators_only: bool = False,
    wf4_enabled: bool = True,
    recover: bool = False,
) -> Selection:
    """Resolve read-only; would-be creation is advisory, never reserved."""
    root = Path(project_root).absolute()
    try:
        root = _root(project_root)
        _flags(new_experiment, note, indicators_only, wf4_enabled, recover)
        ledger = _ledger(root)
        _reconcile(root, ledger)
        selected = _select(
            root,
            ledger,
            requested_name,
            new=new_experiment,
            indicators=indicators_only,
            recover=recover,
        )
        transactions = _path(root, _TRANSACTIONS_PATH)
        referenced = {
            a["reservation_id"] for f in ledger["families"] for a in f["allocations"]
        }
        orphans = (
            [p.name for p in transactions.iterdir() if p.name not in referenced]
            if transactions.exists()
            else []
        )
        if orphans:
            selected = Selection(
                **{
                    **selected.__dict__,
                    "reason": f"{selected.reason}; orphan transaction evidence: {', '.join(sorted(orphans))}",
                }
            )
        return selected
    except (ValueError, TypeError, OSError) as error:
        return Selection("refuse", root, requested_name, None, None, str(error))


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _hit(failpoint: Callable[[str], None] | None, boundary: str) -> None:
    if failpoint is not None:
        failpoint(boundary)


def _source_roles(request: dict) -> set[str]:
    required = {"project", "simulate_system"}
    required.update(f"catalog_{i}" for i in range(len(request["input_catalogs"])))
    generation = request["generation"]
    if generation is not None:
        required.update({"generate_scenarios", "generator_template"})
        required.update(
            item["role"] for item in generation["source_request"]["local_inputs"]
        )
    model = request["simulation"]["model_request"]
    if model is not None:
        required.update({"build_model", "build_template", "waterbodies_template"})
        required.update(item["role"] for item in model["local_inputs"])
    return required


def verify_captured_sources(sources: Sequence[CapturedSource], request: dict) -> None:
    """Fail before mutation if captured dependencies no longer match original bytes."""
    roles = [source.role for source in sources]
    if len(roles) != len(set(roles)) or set(roles) != _source_roles(request):
        raise ValueError(
            f"source roles must match applicability: {sorted(_source_roles(request))}"
        )
    for source in sources:
        if type(source.data) is not bytes:
            raise ValueError("captured source data must be bytes")
        path = Path(source.original_path)
        if not path.is_absolute():
            raise ValueError("captured source path must be absolute")
        _no_links(path)
        if path.read_bytes() != source.data:
            raise ValueError(f"captured source changed before admission: {source.role}")


def _transaction(root: Path, allocation: dict) -> dict:
    reference = allocation["transaction"]
    _ref(reference, root, verify=True)
    transaction = read_canonical_json(root / reference["path"])
    _fields(
        transaction,
        {
            "schema_version",
            "reservation_id",
            "name",
            "base",
            "number",
            "creator_invocation_id",
            "created_at",
            "metadata",
            "initial_reservation",
            "source_capture",
            "metadata_sha256",
            "initial_reservation_sha256",
        },
        "creation transaction",
    )
    if transaction["schema_version"] != "experiment-creation-transaction/1":
        raise ValueError("unsupported creation transaction")
    metadata = _creation(transaction["metadata"])
    reservation = validate_reservation(transaction["initial_reservation"], root)
    for key in ("name", "reservation_id", "number", "created_at"):
        if canonical_json_bytes(transaction[key]) != canonical_json_bytes(
            allocation[key]
        ) or canonical_json_bytes(metadata[key]) != canonical_json_bytes(
            transaction[key]
        ):
            raise ValueError(f"transaction/ledger/metadata {key} disagree")
    if (
        transaction["base"] != metadata["family"]
        or transaction["creator_invocation_id"] != metadata["creator_invocation_id"]
        or reservation["name"] != metadata["name"]
        or reservation["reservation_id"] != metadata["reservation_id"]
    ):
        raise ValueError("transaction initial identities disagree")
    if (
        reservation["revision"] != 0
        or reservation["state"] != "pending"
        or reservation["seed"]["state"] == "bound"
    ):
        raise ValueError("transaction must retain initial pending reservation")
    _digest(transaction["metadata_sha256"], "metadata digest")
    if transaction["initial_reservation_sha256"] != content_sha256(reservation):
        raise ValueError("initial reservation digest differs")
    captures = transaction["source_capture"]
    if type(captures) is not list:
        raise ValueError("source_capture must be a list")
    roles = []
    prefix = f"{_TRANSACTIONS_PATH}/{metadata['reservation_id']}/sources/"
    for source in captures:
        _fields(source, {"role", "original_path", "file"}, "captured source")
        roles.append(source["role"])
        if (
            type(source["original_path"]) is not str
            or not Path(source["original_path"]).is_absolute()
            or Path(source["original_path"]).as_posix() != source["original_path"]
        ):
            raise ValueError("captured original_path must be absolute POSIX")
        _ref(source["file"], root, verify=True)
        if source["file"]["path"] != prefix + source["role"] + ".bin":
            raise ValueError("captured source path disagrees with role/transaction")
    if roles != sorted(_source_roles(reservation["request"])):
        raise ValueError("transaction capture roles differ from request applicability")
    return transaction


def _read_pair(
    root: Path, allocation: dict, *, stage: bool = False, initial: bool = False
) -> dict:
    transaction = _transaction(root, allocation)
    relative = (
        f"{_TRANSACTIONS_PATH}/{allocation['reservation_id']}/experiment"
        if stage
        else f"experiments/{allocation['name']}"
    )
    directory = _path(root, relative)
    metadata_path = _path(root, f"{relative}/experiment.yml")
    reservation_path = _path(root, f"{relative}/_engine/reservation.json")
    data = metadata_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != transaction["metadata_sha256"]:
        raise ValueError("experiment metadata bytes differ from immutable transaction")
    metadata = parse_yaml_document(data)
    if canonical_json_bytes(_creation(metadata)) != canonical_json_bytes(
        transaction["metadata"]
    ):
        raise ValueError("experiment metadata differs from transaction")
    reservation = validate_reservation(
        read_canonical_json(reservation_path), root, verify=not initial
    )
    original = transaction["initial_reservation"]
    for key in ("request", "request_sha256", "name", "reservation_id"):
        if canonical_json_bytes(reservation[key]) != canonical_json_bytes(
            original[key]
        ):
            raise ValueError(f"reservation changed immutable {key}")
    if initial:
        if content_sha256(reservation) != transaction["initial_reservation_sha256"]:
            raise ValueError("unpublished reservation differs from initial recipe")
        actual = set()
        for path in directory.rglob("*"):
            _no_links(path)
            actual.add(path.relative_to(directory).as_posix())
        if actual != {"experiment.yml", "_engine", "_engine/reservation.json"}:
            raise ValueError("unpublished experiment contains unknown payloads")
    return reservation


def admit_experiment_owned(
    project_root: Path | str,
    requested_name: str | None = None,
    *,
    request: dict,
    sources: Sequence[CapturedSource],
    creator_invocation_id: str,
    new_experiment: bool = False,
    note: str | None = None,
    indicators_only: bool = False,
    wf4_enabled: bool = True,
    recover: bool = False,
    request_comparator: Callable[[dict, dict], None] = compare_request,
    failpoint: Callable[[str], None] | None = None,
) -> Selection:
    """Admit/create under caller-held project ownership, never execute producers."""
    root = _root(project_root)
    _flags(new_experiment, note, indicators_only, wf4_enabled, recover)
    if not indicators_only:
        validate_request(request, root)
        verify_captured_sources(sources, request)
        _uuid(creator_invocation_id, "creator_invocation_id")
        if request["selection"]["manifest"] is not None:
            from blueearth_cst.experiment.scenario_collection_v2 import (
                read_collection_v2,
            )

            _ref(request["selection"]["manifest"], root, verify=True)
            read_collection_v2(root / request["selection"]["manifest"]["path"])
    ledger = _ledger(root)
    before = canonical_json_bytes(ledger)
    _reconcile(root, ledger)
    selected = _select(
        root,
        ledger,
        requested_name,
        new=new_experiment,
        indicators=indicators_only,
        recover=recover,
    )
    if selected.action in {"refuse", "missing"}:
        if before != canonical_json_bytes(ledger) and not indicators_only:
            _save_ledger(root, ledger)
        raise ValueError(selected.reason)
    if selected.action == "continue" and not indicators_only:
        allocation = _allocation(ledger, selected.name)
        if allocation["publication"] == "prepared":
            compare_request(
                _transaction(root, allocation)["initial_reservation"]["request"],
                request,
            )
            return recover_experiment_owned(root, selected.name, failpoint=failpoint)
        request_comparator(read_reservation(root, selected.name)["request"], request)
    if selected.action != "create":
        if before != canonical_json_bytes(ledger) and not indicators_only:
            _save_ledger(root, ledger)
        return selected
    reservation_id = uuid.uuid4().hex
    created_at = _now()
    metadata = {
        "schema_version": "experiment-creation/1",
        "name": selected.name,
        "family": selected.family,
        "number": selected.number,
        "reservation_id": reservation_id,
        "created_at": created_at,
        "creator_invocation_id": creator_invocation_id,
        "note": note,
        "toolbox": toolbox_identity(),
    }
    seed = {
        "state": "not_applicable"
        if request["generation"] is None
        else "awaiting_sources",
        "requested": None
        if request["generation"] is None
        else request["generation"]["requested_seed"],
        "resolved": None,
        "seed_resolution": None,
        "plan": None,
        "source_identity_sha256": None,
    }
    reservation = {
        "schema_version": "experiment-reservation/1",
        "canonicalization_id": "collection-canon/1",
        "reservation_id": reservation_id,
        "name": selected.name,
        "revision": 0,
        "state": "pending",
        "request": copy.deepcopy(request),
        "request_sha256": content_sha256(request),
        "seed": seed,
        "resolution": None,
        "simulation": None,
    }
    validate_reservation(reservation, root)
    relative = f"{_TRANSACTIONS_PATH}/{reservation_id}"
    transaction_dir = _path(root, relative)
    transaction_dir.mkdir(parents=True, exist_ok=False)
    captures = []
    for source in sorted(sources, key=lambda s: s.role):
        path = _path(root, f"{relative}/sources/{source.role}.bin")
        _write_bytes(path, source.data)
        captures.append(
            {
                "role": source.role,
                "original_path": Path(source.original_path).as_posix(),
                "file": file_reference(path, "project_root", root),
            }
        )
        _hit(failpoint, f"source_captured:{source.role}")
    _hit(failpoint, "sources_captured")
    metadata_bytes = yaml.safe_dump(
        metadata, sort_keys=True, allow_unicode=True
    ).encode("utf-8")
    _write_bytes(_path(root, f"{relative}/experiment/experiment.yml"), metadata_bytes)
    _hit(failpoint, "metadata_written")
    staged_reservation = _path(root, f"{relative}/experiment/_engine/reservation.json")
    staged_reservation.parent.mkdir(parents=True, exist_ok=True)
    atomic_record(staged_reservation, reservation)
    _hit(failpoint, "reservation_written")
    transaction = {
        "schema_version": "experiment-creation-transaction/1",
        "reservation_id": reservation_id,
        "name": selected.name,
        "base": selected.family,
        "number": selected.number,
        "creator_invocation_id": creator_invocation_id,
        "created_at": created_at,
        "metadata": metadata,
        "initial_reservation": reservation,
        "source_capture": captures,
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "initial_reservation_sha256": content_sha256(reservation),
    }
    transaction_path = _path(root, f"{relative}/transaction.json")
    atomic_record(transaction_path, transaction)
    _hit(failpoint, "transaction_written")
    allocation = {
        "number": selected.number,
        "name": selected.name,
        "reservation_id": reservation_id,
        "origin": "reservation",
        "created_at": created_at,
        "publication": "prepared",
        "presence": "present",
        "transaction": file_reference(transaction_path, "project_root", root),
        "legacy_owner": None,
        "tombstoned_at": None,
    }
    _add_allocation(ledger, selected.family, allocation)
    _save_ledger(root, ledger)
    _hit(failpoint, "ledger_prepared")
    _publish_directory(root, allocation)
    _hit(failpoint, "directory_published")
    _read_pair(root, allocation, initial=True)
    allocation["publication"] = "published"
    _save_ledger(root, ledger)
    _hit(failpoint, "ledger_published")
    return Selection(
        "create",
        root,
        selected.name,
        selected.family,
        selected.number,
        "committed creation",
        reservation_id,
    )


def _publish_directory(root: Path, allocation: dict) -> None:
    _read_pair(root, allocation, stage=True, initial=True)
    staged = _path(
        root,
        f"{_TRANSACTIONS_PATH}/{allocation['reservation_id']}/experiment",
    )
    target = _path(root, f"experiments/{allocation['name']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"unknown final occupant: {target}")
    # Windows rename refuses all existing destinations; ownership excludes other
    # sanctioned writers. Empty foreign directories are still explicitly refused.
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        _no_links(target)
        if target.exists():
            raise ValueError(f"unknown final occupant: {target}")
        try:
            staged.rename(target)
            return
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) != 5:
                raise
            if attempt == _REPLACE_RETRY_ATTEMPTS - 1:
                error.add_note(
                    f"directory publication failed after {_REPLACE_RETRY_ATTEMPTS} attempts; prepared allocation retained for --recover-reservation"
                )
                raise
            time.sleep(
                min(
                    _REPLACE_RETRY_BASE_SECONDS * 2**attempt, _REPLACE_RETRY_CAP_SECONDS
                )
            )


def recover_experiment_owned(
    project_root: Path | str,
    name: str,
    *,
    failpoint: Callable[[str], None] | None = None,
) -> Selection:
    """Complete only a verified committed creation recipe; never current config."""
    root = _root(project_root)
    _name(name)
    ledger = _ledger(root)
    allocation = _allocation(ledger, name)
    if allocation is None or allocation["origin"] != "reservation":
        raise ValueError("no committed reservation recipe for recovery")
    if allocation["presence"] == "tombstone":
        raise ValueError("tombstoned allocation cannot be recreated")
    if allocation["publication"] == "published":
        _read_pair(root, allocation)
        return Selection(
            "continue",
            root,
            name,
            parse_numbered_name(name)[0],
            allocation["number"],
            "published reservation already valid",
            allocation["reservation_id"],
        )
    _transaction(root, allocation)
    staged = _path(
        root,
        f"{_TRANSACTIONS_PATH}/{allocation['reservation_id']}/experiment",
    )
    target = _path(root, f"experiments/{name}")
    if staged.exists() == target.exists():
        raise ValueError(
            "recovery requires exactly one complete stage or matching final; evidence missing/ambiguous"
        )
    if staged.exists():
        _publish_directory(root, allocation)
        _hit(failpoint, "directory_published")
    _read_pair(root, allocation, initial=True)
    allocation["publication"] = "published"
    _save_ledger(root, ledger)
    _hit(failpoint, "ledger_published")
    return Selection(
        "continue",
        root,
        name,
        parse_numbered_name(name)[0],
        allocation["number"],
        "recovered committed creation",
        allocation["reservation_id"],
    )


def read_reservation(project_root: Path | str, name: str) -> dict:
    """Read validated published reservation, including all bound evidence."""
    root = _root(project_root)
    _name(name)
    allocation = _allocation(_ledger(root), name)
    if (
        allocation is None
        or allocation["origin"] != "reservation"
        or allocation["publication"] != "published"
        or allocation["presence"] != "present"
    ):
        raise ValueError("experiment is not a published live reservation")
    return _read_pair(root, allocation)


def bind_reservation_owned(
    project_root: Path | str,
    name: str,
    *,
    seed: dict | None = None,
    resolution: dict | None = None,
    simulation: dict | None = None,
) -> dict:
    """Bind verified immutable facts atomically under caller-held ownership."""
    root = _root(project_root)
    retained = read_reservation(root, name)
    candidate = copy.deepcopy(retained)
    for key, value in (
        ("seed", seed),
        ("resolution", resolution),
        ("simulation", simulation),
    ):
        if value is None:
            continue
        previous = retained[key]
        frozen = (
            previous is not None
            if key != "seed"
            else previous["state"] in {"bound", "not_applicable"}
        )
        if frozen and canonical_json_bytes(previous) != canonical_json_bytes(value):
            raise ValueError(f"cannot change immutable reservation {key}")
        candidate[key] = copy.deepcopy(value)
    candidate["state"] = (
        "pending"
        if candidate["resolution"] is None
        else "resolved"
        if candidate["simulation"] is None
        else "simulation_bound"
    )
    if canonical_json_bytes(candidate) == canonical_json_bytes(retained):
        return retained
    if seed is not None and seed.get("state") != "bound":
        raise ValueError("seed transition must bind complete source/plan evidence")
    candidate["revision"] += 1
    validate_reservation(candidate, root, verify=True)
    atomic_record(
        _path(root, f"experiments/{name}/_engine/reservation.json"),
        candidate,
        replace=True,
    )
    return candidate
