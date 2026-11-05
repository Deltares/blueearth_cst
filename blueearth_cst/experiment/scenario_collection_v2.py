"""Strict, additive reader for model-independent scenario-collection/3 records."""

import csv
import math
import re
from pathlib import Path
from typing import Any

from blueearth_cst.experiment.content_identity import (
    collection_id_v2,
    collection_revision_v2,
    content_sha256,
    identity_segment,
    read_canonical_json,
    scientific_source_entry,
)
from blueearth_cst.experiment.generation_plan import (
    GENERATOR_SEED_FIELDS,
    generator_seed_projection,
)
from blueearth_cst.shared.workflow_config_snapshot import (
    resolve_file_reference,
    validate_archive,
)

_DOCUMENTS = {"generation_config", "source_inventory", "provider_code", "environment"}
_SCENARIO_HEADER = ["run_id", "evaluate", "type", "rlz", "st_id"]
_PERTURBATION_HEADER = [
    "st_id",
    "month",
    "temp_change",
    "precip_change",
    "precip_variance_change",
]


def _fields(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    """Return a closed JSON object or refuse missing and surplus keys."""
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{name}: expected exactly {sorted(expected)}")
    return value


def _digest(value: Any, name: str) -> str:
    """Require one full lowercase SHA-256 value."""
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name}: expected full lowercase SHA-256")
    return value


def _positive(value: Any, name: str) -> int:
    """Require a positive integer without accepting booleans."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name}: expected positive integer")
    return value


def _window(value: Any) -> None:
    """Validate the frozen inclusive year window."""
    window = _fields(value, {"start", "end"}, "simulation_window")
    if _positive(window["end"], "end") < _positive(window["start"], "start"):
        raise ValueError("simulation_window end precedes start")


def _months(value: Any, name: str) -> None:
    """Require twelve finite monthly numbers without coercing booleans."""
    if (
        type(value) is not list
        or len(value) != 12
        or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in value
        )
    ):
        raise ValueError(f"{name}: expected twelve finite monthly numbers")


def _perturbations(value: Any) -> None:
    """Validate every configured stochastic stress dimension."""
    payload = _fields(value, {"temp", "precip", "spell_factors"}, "perturbations")
    for name in ("temp", "precip"):
        expected = {"n_levels", "trajectory", "mean"}
        if name == "precip":
            expected.add("variance")
        axis = _fields(payload[name], expected, f"perturbations.{name}")
        _positive(axis["n_levels"], f"{name}.n_levels")
        if axis["trajectory"] not in ("transient", "constant"):
            raise ValueError(f"{name}.trajectory is unsupported")
        for range_name in ("mean", "variance"):
            if range_name not in axis:
                continue
            range_value = _fields(axis[range_name], {"min", "max"}, range_name)
            _months(range_value["min"], f"{name}.{range_name}.min")
            _months(range_value["max"], f"{name}.{range_name}.max")
    spells = _fields(payload["spell_factors"], {"dry", "wet"}, "spell_factors")
    for name in ("dry", "wet"):
        _months(spells[name], f"spell_factors.{name}")


def _interpretation(value: Any) -> None:
    """Validate the resolved WF3 physical variable meaning."""
    item = _fields(
        value,
        {"revision", "variables", "calendar", "time_label"},
        "source interpretation",
    )
    if item["time_label"] != "interval_end":
        raise ValueError("source interpretation time label must be interval_end")
    for field in ("revision", "calendar"):
        if type(item[field]) is not str or not item[field]:
            raise ValueError(f"source interpretation {field} must be text")
    variables = item["variables"]
    if (
        type(variables) is not list
        or not variables
        or [variable["name"] for variable in variables]
        != sorted({variable["name"] for variable in variables})
    ):
        raise ValueError("source interpretation variables must be unique and sorted")
    for variable in variables:
        _fields(variable, {"name", "units"}, "interpreted variable")
        if not all(
            type(variable[field]) is str and variable[field]
            for field in ("name", "units")
        ):
            raise ValueError("interpreted variable name and units must be text")


def _generator(value: Any, resolved_seed: int, expected_settings: Any) -> None:
    """Refuse absent/unclassified upstream arguments and changed seed semantics."""
    if type(value) is not dict or set(value) != set(GENERATOR_SEED_FIELDS):
        raise ValueError("weathergen must contain exactly six classified sections")
    optional = {
        "generate_weather": {"plot_dpi", "plot_device"},
        "write_netcdf": {"file_suffix", "out_dir"},
    }
    for section, (included, excluded) in GENERATOR_SEED_FIELDS.items():
        fields = value[section]
        if type(fields) is not dict or set(fields) != included | excluded:
            raise ValueError(f"weathergen.{section} has missing or unknown arguments")
        for key in optional.get(section, set()):
            maybe = _fields(fields[key], {"present", "value"}, f"{section}.{key}")
            if type(maybe["present"]) is not bool or (
                not maybe["present"] and maybe["value"] is not None
            ):
                raise ValueError(f"{section}.{key} has invalid presence encoding")
    if value["generate_weather"]["seed"] != resolved_seed:
        raise ValueError("installed generator seed differs from resolution")
    if value["generate_weather"]["out_dir"] is not None:
        raise ValueError("embedded generator out_dir must be null")
    if generator_seed_projection(value) != expected_settings:
        raise ValueError(
            "seed material generator settings differ from resolved arguments"
        )


def _profile(intent: dict[str, Any]) -> None:
    """Validate closed embedded WF3 profiles and all declared H operations."""
    _fields(
        intent,
        {
            "schema_version",
            "canonicalization_id",
            "collection_id",
            "provider",
            "scenario_type",
            "scenario_spec",
            "scenario_semantics_sha256",
            "run_count",
            "run_group_id_capacity",
            "run_group_id_width",
            "documents",
            "document_digests",
            "identity_projections",
            "identity_digests",
        },
        "collection intent",
    )
    if (
        intent["schema_version"] != "scenario-collection-intent/3"
        or intent["canonicalization_id"] != "collection-canon/1"
        or intent["scenario_type"] != "stochastic"
    ):
        raise ValueError("unsupported collection intent version or provider binding")
    _digest(intent["collection_id"], "collection_id")
    _digest(intent["scenario_semantics_sha256"], "scenario_semantics_sha256")
    capacity = _positive(intent["run_group_id_capacity"], "run_group_id_capacity")
    if intent["run_group_id_width"] != len(str(capacity)):
        raise ValueError("run_group_id_width differs from capacity")
    provider = _fields(intent["provider"], {"name", "revision"}, "provider")
    if provider["name"] != "weathergenr":
        raise ValueError("unsupported stochastic provider")
    _digest(provider["revision"], "provider revision")
    spec = _fields(
        intent["scenario_spec"],
        {
            "scenario_type",
            "n_realizations",
            "n_design_points",
            "unperturbed_per_realization",
            "expected_run_count",
            "simulation_window",
            "pairing",
            "row_order",
        },
        "scenario_spec",
    )
    n_realizations = _positive(spec["n_realizations"], "n_realizations")
    n_points = _positive(spec["n_design_points"], "n_design_points")
    if (
        spec["scenario_type"] != "stochastic"
        or spec["unperturbed_per_realization"] != 1
        or spec["pairing"] != "paired_across_design_points"
        or spec["row_order"] != "rlz-major/unperturbed-first/st-id-ascending"
        or spec["expected_run_count"] != n_realizations * (n_points + 1)
        or intent["run_count"] != spec["expected_run_count"]
        or intent["run_count"] > capacity
    ):
        raise ValueError("scenario_spec does not describe the complete cross-product")
    _window(spec["simulation_window"])
    docs = _fields(intent["documents"], _DOCUMENTS, "documents")
    document_digests = _fields(
        intent["document_digests"], _DOCUMENTS, "document_digests"
    )
    identity_digests = _fields(
        intent["identity_digests"], _DOCUMENTS, "identity_digests"
    )
    projections = _fields(
        intent["identity_projections"],
        {"generation_config", "source_inventory"},
        "identity_projections",
    )
    for name in _DOCUMENTS:
        if _digest(document_digests[name], name) != content_sha256(docs[name]):
            raise ValueError(f"{name}: embedded document digest differs")
    for name in ("provider_code", "environment"):
        if identity_digests[name] != document_digests[name]:
            raise ValueError(f"{name}: identity must hash complete document")
    for name in projections:
        if _digest(identity_digests[name], name) != content_sha256(projections[name]):
            raise ValueError(f"{name}: scientific projection digest differs")
    if content_sha256(docs["provider_code"]) != provider["revision"]:
        raise ValueError("provider revision differs from code inventory")
    code = docs["provider_code"]
    if type(code) is not list or [item["path"] for item in code] != sorted(
        {item["path"] for item in code}
    ):
        raise ValueError("provider code inventory must be unique and sorted")
    for item in code:
        _fields(item, {"path", "sha256"}, "provider code entry")
        _digest(item["sha256"], "provider code bytes")
    env = _fields(docs["environment"], {"packages", "locks"}, "environment")
    if type(env["packages"]) is not dict or type(env["locks"]) is not dict:
        raise ValueError("environment maps are required")
    for value in env["locks"].values():
        _digest(value, "environment lock")
    if set(env["locks"]) - {
        "installed-python-distribution-metadata",
        "installed-conda-dependency-records",
        "Manifest.toml",
    }:
        raise ValueError("environment contains an unknown lock identity")
    if not env["packages"] or any(
        type(key) is not str or not key or type(value) is not str or not value
        for key, value in env["packages"].items()
    ):
        raise ValueError("environment package revisions must be nonempty text")
    generation = _fields(
        docs["generation_config"],
        {
            "seed",
            "seed_resolution",
            "run_group_id_capacity",
            "weathergen",
            "climate_perturbations",
        },
        "generation_config",
    )
    if generation["run_group_id_capacity"] != capacity:
        raise ValueError("generation capacity differs from intent")
    _perturbations(generation["climate_perturbations"])
    seed = _fields(generation["seed"], {"requested", "resolved"}, "seed")
    if type(seed["resolved"]) is not int or not 0 <= seed["resolved"] < 2**31 - 1:
        raise ValueError("resolved seed must be in the canonical integer domain")
    resolution = _fields(
        generation["seed_resolution"],
        {
            "seed_request",
            "resolved_seed",
            "projection",
            "projection_sha256",
            "seed_resolution_id",
        },
        "seed_resolution",
    )
    material = resolution["projection"]
    _fields(
        material,
        {
            "schema_version",
            "scenario_type",
            "n_realizations",
            "simulation_window",
            "climate_perturbations",
            "water_year_start",
            "provider_revision",
            "source_inventory_sha256",
            "generator_settings",
        },
        "generation seed material",
    )
    if material["schema_version"] != "generation-seed-material/2":
        raise ValueError("expected WF3-only generation-seed-material/2")
    if type(material["water_year_start"]) is not str or not re.fullmatch(
        r"[A-Z]{3}", material["water_year_start"]
    ):
        raise ValueError("seed material has invalid water-year month")
    if (
        material["scenario_type"] != "stochastic"
        or material["n_realizations"] != n_realizations
        or content_sha256(material["simulation_window"])
        != content_sha256(spec["simulation_window"])
        or content_sha256(material["climate_perturbations"])
        != content_sha256(generation["climate_perturbations"])
    ):
        raise ValueError("seed material differs from resolved generation settings")
    if (
        seed["requested"] != resolution["seed_request"]
        or seed["resolved"] != resolution["resolved_seed"]
        or _digest(resolution["projection_sha256"], "seed projection")
        != content_sha256(material)
        or _digest(resolution["seed_resolution_id"], "seed resolution")
        != content_sha256(
            {
                key: value
                for key, value in resolution.items()
                if key != "seed_resolution_id"
            }
        )
    ):
        raise ValueError("seed resolution does not bind its frozen material")
    if seed["requested"] == "auto":
        from blueearth_cst.experiment.content_identity import automatic_seed_v2

        if seed["resolved"] != automatic_seed_v2(material):
            raise ValueError("automatic seed differs from canonical material")
    elif type(seed["requested"]) is not int or seed["requested"] != seed["resolved"]:
        raise ValueError("explicit seed differs from resolved seed")
    _generator(
        generation["weathergen"], seed["resolved"], material["generator_settings"]
    )
    projected_generation = {
        "schema_version": "generation-config-identity/2",
        "resolved_seed": seed["resolved"],
        "generator_settings": material["generator_settings"],
        "climate_perturbations": generation["climate_perturbations"],
        "simulation_window": spec["simulation_window"],
        "n_realizations": n_realizations,
        "water_year_start": material["water_year_start"],
    }
    if content_sha256(projections["generation_config"]) != content_sha256(
        projected_generation
    ):
        raise ValueError("generation_config scientific projection differs")
    inventory = _fields(
        docs["source_inventory"], {"sources", "interpretation"}, "source_inventory"
    )
    sources = inventory["sources"]
    if type(sources) is not list or [item["role"] for item in sources] != sorted(
        {item["role"] for item in sources}
    ):
        raise ValueError("source inventory roles must be unique and sorted")
    for item in sources:
        _fields(item, {"role", "original_path", "file", "metadata"}, "source")
        if type(item["metadata"]) is not dict:
            raise ValueError("source metadata must be an evidence map")
    _interpretation(inventory["interpretation"])
    version = projections["source_inventory"].get("schema_version")
    selected = [
        scientific_source_entry(item["role"], item["file"], item["metadata"], version)
        for item in sources
        if item["role"] in ("basin_cells", "historical_climate")
    ]
    if [item["role"] for item in selected] != ["basin_cells", "historical_climate"]:
        raise ValueError("missing pinned WF3 scientific source")
    if content_sha256(selected) != material["source_inventory_sha256"]:
        raise ValueError("seed material differs from pinned source inventory")
    if material["provider_revision"] != provider["revision"]:
        raise ValueError("seed material differs from provider code")
    projected_sources = {
        "schema_version": version,
        "sources": selected,
        "interpretation": inventory["interpretation"],
    }
    if content_sha256(projections["source_inventory"]) != content_sha256(
        projected_sources
    ):
        raise ValueError("source inventory scientific projection differs")
    if intent["collection_id"] != collection_id_v2(intent):
        raise ValueError("collection identity differs")


def _csv(path: Path, header: list[str]) -> list[dict[str, str]]:
    """Read an exact lookup header without silently accepting extra cells."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != header:
            raise ValueError(f"{path}: unexpected lookup header")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path}: extra lookup cells")
    return rows


def _rows(rows: list[dict[str, str]], intent: dict[str, Any]) -> None:
    """Validate complete stochastic lineage reconstructed from lookup columns."""
    spec = intent["scenario_spec"]
    width = intent["run_group_id_width"]
    n_realizations = spec["n_realizations"]
    n_points = spec["n_design_points"]
    if len(rows) != spec["expected_run_count"]:
        raise ValueError("scenario lookup has incomplete cross-product")
    expected = []
    for realization in range(1, n_realizations + 1):
        for point in range(n_points + 1):
            expected.append(
                {
                    "run_id": f"run_{len(expected) + 1:0{width}d}",
                    "evaluate": "true",
                    "type": "stochastic",
                    "rlz": f"{realization:0{len(str(n_realizations))}d}",
                    "st_id": f"{point:0{len(str(n_points))}d}" if point else "",
                }
            )
    if rows != expected:
        raise ValueError("scenario lookup changes row order or stochastic ancestry")
    semantics = [
        {key: value for key, value in row.items() if key != "run_id"} for row in rows
    ]
    if content_sha256(semantics) != intent["scenario_semantics_sha256"]:
        raise ValueError("scenario semantic digest differs")


def _series_descriptor(value: Any) -> None:
    """Check the closed physical descriptor envelope for a retained series."""
    descriptor = _fields(
        value,
        {
            "source_calendar",
            "timestep",
            "time_label",
            "start",
            "end",
            "spatial_representation_sha256",
            "variables",
        },
        "series descriptor",
    )
    for key in ("source_calendar", "timestep", "start", "end"):
        if type(descriptor[key]) is not str or not descriptor[key]:
            raise ValueError(f"series descriptor {key} must be nonempty text")
    if descriptor["time_label"] != "interval_end":
        raise ValueError("series descriptor must retain interval-end labels")
    _digest(descriptor["spatial_representation_sha256"], "spatial representation")
    variables = descriptor["variables"]
    if type(variables) is not list or not variables:
        raise ValueError("series descriptor must list its physical variables")
    if [item["name"] for item in variables] != sorted(
        {item["name"] for item in variables}
    ):
        raise ValueError("series descriptor variables must be unique and sorted")
    for variable in variables:
        _fields(variable, {"name", "units", "missing_value"}, "series variable")
        if type(variable["name"]) is not str or not variable["name"]:
            raise ValueError("series variable name is empty")
        if type(variable["units"]) is not str or not variable["units"]:
            raise ValueError("series variable units are empty")
        missing = variable["missing_value"]
        if missing is not None and (
            type(missing) is not dict
            or not missing
            or not set(missing) <= {"_FillValue", "missing_value"}
        ):
            raise ValueError("series missing-value encoding has unknown keys")


def read_collection_v2(marker_path: Path) -> dict[str, Any]:
    """Read a ready v2 collection through its engine marker and checked data root."""
    marker_path = Path(marker_path)
    if marker_path.is_symlink():
        raise ValueError("collection marker cannot be a symbolic link")
    marker_path = marker_path.resolve(strict=True)
    record_dir = marker_path.parent
    if marker_path.name != "collection.json" or record_dir.parent.name != "collections":
        raise ValueError("collection discovery must start at its engine marker")
    if (
        record_dir.parent.parent.name != "_engine"
        or record_dir.parent.parent.parent.name != "scenarios"
    ):
        raise ValueError("collection marker is outside the scenario engine")
    project_root = record_dir.parents[3]
    marker = read_canonical_json(marker_path)
    _fields(
        marker,
        {
            "schema_version",
            "canonicalization_id",
            "status",
            "collection_id",
            "collection_revision",
            "data_root",
            "intent",
            "archive",
            "scenario_run_lookup",
            "perturbation_lookup",
            "series",
            "provider_products",
            "diagnostics",
        },
        "collection marker",
    )
    if (
        marker["schema_version"] != "scenario-collection/3"
        or marker["canonicalization_id"] != "collection-canon/1"
        or marker["status"] != "ready"
    ):
        raise ValueError(
            "unsupported or unready collection marker; legacy numeric run-ID collections must be regenerated"
        )
    full_id = _digest(marker["collection_id"], "collection_id")
    segment = identity_segment(full_id, "collection_id")
    if record_dir.name != segment or marker["data_root"] not in (
        f"scenarios/_engine/data/{segment}",
        f"scenarios/{segment}",  # Completed collections retain their original paths.
    ):
        raise ValueError("collection full ID, engine path, and data root differ")
    data_root = project_root / marker["data_root"]
    if not data_root.is_dir() or data_root.is_symlink():
        raise ValueError("collection data root is missing or aliased")
    anchors = {"project_root": project_root, "record_directory": record_dir}
    intent_path = resolve_file_reference(marker["intent"], anchors)
    if intent_path != record_dir / "collection_intent.json":
        raise ValueError("collection intent must be the sibling frozen record")
    intent = read_canonical_json(intent_path)
    _profile(intent)
    if intent["collection_id"] != full_id:
        raise ValueError("marker and intent collection identities differ")
    archive_path = resolve_file_reference(marker["archive"], anchors)
    if archive_path != data_root / "config" / "run_record.yml":
        raise ValueError("collection creator archive path differs")
    archive = validate_archive(archive_path.parent)
    if (
        archive["owner"] != {"kind": "scenario_collection", "id": full_id}
        or archive["workflow"] != "generate_scenarios"
    ):
        raise ValueError("collection creator archive owner differs")
    for source in intent["documents"]["source_inventory"]["sources"]:
        resolve_file_reference(source["file"], {"project_root": project_root})
    scenario_path = resolve_file_reference(marker["scenario_run_lookup"], anchors)
    perturbation_path = resolve_file_reference(marker["perturbation_lookup"], anchors)
    if scenario_path != data_root / "scenario_run_lookup.csv":
        raise ValueError("scenario lookup path differs")
    if perturbation_path != data_root / "perturbation_lookup.csv":
        raise ValueError("perturbation lookup path differs")
    rows = _csv(scenario_path, _SCENARIO_HEADER)
    _rows(rows, intent)
    perturbations = _csv(perturbation_path, _PERTURBATION_HEADER)
    expected_points = intent["scenario_spec"]["n_design_points"]
    if len(perturbations) != expected_points * 12:
        raise ValueError("perturbation lookup has incomplete monthly grid")
    for index, row in enumerate(perturbations):
        if row["st_id"] != f"{index // 12 + 1:0{len(str(expected_points))}d}" or row[
            "month"
        ] != str(index % 12 + 1):
            raise ValueError("perturbation lookup order differs")
        for field in _PERTURBATION_HEADER[2:]:
            if not math.isfinite(float(row[field])):
                raise ValueError("perturbation lookup contains nonfinite change")
    series = marker["series"]
    if type(series) is not list or [entry["run_id"] for entry in series] != [
        row["run_id"] for row in rows
    ]:
        raise ValueError("series entries differ from scenario lookup order")
    for entry in series:
        _fields(entry, {"run_id", "file", "descriptor"}, "series entry")
        _series_descriptor(entry["descriptor"])
        path = resolve_file_reference(entry["file"], anchors)
        if path != data_root / "series" / f"{entry['run_id']}.nc":
            raise ValueError("series path does not match its run ID")
    products = marker["provider_products"]
    if [entry["role"] for entry in products] != [
        "sim_dates",
        "resampled_dates",
        "weather_generation_input",
    ]:
        raise ValueError("provider product roles/order differ")
    for entry in products:
        _fields(entry, {"role", "file"}, "provider product")
        resolve_file_reference(entry["file"], anchors)
    generated = [
        item["file"]
        for item in archive["generated_inputs"]
        if item["role"] == "weather_generation_input"
    ]
    if len(generated) != 1 or generated[0] != products[2]["file"]:
        raise ValueError("creator archive does not bind installed generator YAML")
    resolve_file_reference(generated[0], {"project_root": project_root})
    diagnostics = marker["diagnostics"]
    if type(diagnostics) is not list or [ref["path"] for ref in diagnostics] != sorted(
        {ref["path"] for ref in diagnostics}
    ):
        raise ValueError("diagnostics must be unique and path-sorted")
    for ref in diagnostics:
        resolve_file_reference(ref, anchors)
    if _digest(
        marker["collection_revision"], "collection_revision"
    ) != collection_revision_v2(marker, intent):
        raise ValueError("collection revision differs from retained series and dates")
    return marker
