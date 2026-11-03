"""Readable per-set exports of immutable indicator sets, with committed pointers.

Each selected set publishes ``results/indicator_sets/<set-id>/``: its scoped run
and group tables, group membership and metadata, and a derived
``indicators.csv`` dictionary. Immutable objects stay authoritative; the
export is a replaceable view whose committed inventory lives under
``_engine/indicator_sets/names/<set-id>/``. Machine records never appear in the
readable folder.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from blueearth_cst.experiment import indicator_registry as registry
from blueearth_cst.experiment.content_identity import (
    atomic_record,
    content_sha256,
    read_canonical_json,
)
from blueearth_cst.experiment.indicator_plan import (
    COVERAGE_EVIDENCE_SCHEMA,
    read_indicator_set,
)
from blueearth_cst.experiment.legacy_indicator_names import indicator_names
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.run_log_core import plural

PUBLICATION_SCHEMA = "named-indicator-export/1"
CURRENT_SCHEMA = "named-indicator-current/1"
DICTIONARY_NAME = "indicators.csv"
DICTIONARY_COLUMNS = (
    "indicator",
    "set_id",
    "set_title",
    "set_description",
    "title",
    "description",
    "variable",
    "input_units",
    "output_units",
    "quantity",
    "method",
    "parameters",
    "equivalent_definition",
    "grain",
    "group_policy",
    "water_year_start",
    "calendar",
    "time_label",
    "block_assignment",
    "partial_blocks",
    "block_coverage",
    "missingness",
    "validity",
    "references",
    "authored_references",
    "availability",
    "unavailable_reason",
)
_FILE_NAME = re.compile(
    r"(?:group_members|group_metadata|indicators|[a-z][a-z0-9_]*_(?:run|group)_indicators)\.csv"
)


class NamedExportError(ValueError):
    """A named export cannot be published, refreshed or recovered safely."""


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise NamedExportError(f"path escapes the experiment: {relative}")
    return path


def _names_root(root: Path, set_id: str) -> Path:
    return root / "_engine/indicator_sets/names" / set_id


def _csv(rows, fields) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


# --------------------------------------------------------------- dictionary
def _block_text(block) -> str:
    return (
        f"{block['block_label']} {block['observed']}/{block['expected']} d"
        f" missing {block['missing']}"
    )


def _pattern_key(item) -> tuple:
    return (
        item["calendar"],
        item["timestep_seconds"],
        item["time_label"],
        item["anchor"],
        item["first_label"],
        item["last_label"],
        tuple(
            (
                b["block_label"],
                b["observed"],
                b["expected"],
                b["missing"],
                b["all_missing"],
            )
            for b in item["blocks"]
        ),
    )


def _pattern_text(item, pairs) -> str:
    blocks = item["blocks"]
    first, last = blocks[0], blocks[-1]
    runs = len({run for run, _, _ in pairs})
    members = ", ".join(
        f'"{run} at location {location}" ({role})' for run, location, role in pairs
    )
    return (
        f"{members}: anchor {item['anchor']}; {item['block_count']} blocks "
        f"({item['complete_label_blocks']} complete label blocks; first "
        f"{first['observed']}/{first['expected']} d; last {last['observed']}/{last['expected']} d); "
        f"labels {item['first_label']}..{item['last_label']}; missing {item['missing_total']}; "
        f"all-missing blocks {item['all_missing_blocks']}; {runs} run(s); "
        f"calendar {item['calendar']}; timestep {item['timestep_seconds']} s; "
        f"{item['time_label']}; blocks ["
        + "; ".join(_block_text(b) for b in blocks)
        + "]"
    )


def block_coverage(evidence, pairs) -> str:
    """Every distinct coverage pattern with its explicit run/location pairs.

    Patterns are sorted by every pattern field, pairs by (run, location, role);
    nothing is truncated or sampled. Equal count patterns do not claim equal
    missing positions; exact masks remain in the retained responses.
    """
    by_key = {
        (item["run_id"], item["variable"], item["location_id"]): item
        for item in evidence
    }
    patterns = {}
    for run, variable, location, role in pairs:
        item = by_key.get((run, variable, location))
        if item is None:
            raise NamedExportError(
                f"no committed coverage evidence for {run}/{variable}/{location}"
            )
        entry = patterns.setdefault(_pattern_key(item), (item, set()))
        entry[1].add((run, location, role))
    return " | ".join(
        _pattern_text(item, sorted(members))
        for _, (item, members) in sorted(patterns.items(), key=lambda kv: kv[0])
    )


def _parameters_text(parameters) -> str:
    return "; ".join(f"{key}={value}" for key, value in sorted(parameters.items()))


def dictionary_rows(resolved_set, manifest, evidence, water_year_start) -> list[dict]:
    """One readable dictionary row per indicator, resolved from immutable sources."""
    anchor = manifest["identity_projection"]["water_year_anchor"]
    reference = manifest["identity_projection"].get("reference") or {}
    series = {
        (item["run_id"], item["variable"], item["location_id"]) for item in evidence
    }
    declared = {item["name"]: item for item in manifest["declarations"]}
    dropped = {
        item["calculation"]["indicator"]: item["reasons"]
        for item in manifest.get("unavailable_calculations", [])
    }
    gaps = {}
    for item in manifest.get("unavailable_results", []):
        gaps.setdefault(item["indicator"], {}).setdefault(item["reason"], set()).add(
            item["location"]
        )
    rows = []
    for indicator in sorted(resolved_set.indicators, key=lambda item: item.indicator):
        declaration = indicator.declaration
        if indicator.indicator in dropped:
            availability = "unavailable"
            reason = "; ".join(dropped[indicator.indicator])
        elif indicator.indicator in gaps:
            availability = "partial"
            reason = "; ".join(
                f"{key} at {plural(len(locations), 'location')}"
                for key, locations in sorted(gaps[indicator.indicator].items())
            )
        else:
            availability, reason = "available", ""
        if availability != "unavailable" and (
            declared.get(indicator.indicator, {}).get("statistic")
            != declaration.statistic
        ):
            raise NamedExportError(
                f"{resolved_set.set_id}: {indicator.indicator} differs from its immutable declaration"
            )
        required = {item.variable for item in declaration.requirements}
        contract = registry.METHODS.get(indicator.method)
        if indicator.calculation_set:
            from types import SimpleNamespace

            from blueearth_cst.indicators import loader

            assumptions = loader.load_set(indicator.calculation_set).ASSUMPTIONS
            contract = SimpleNamespace(
                block_assignment=str(assumptions),
                partial_blocks=declaration.required_responses.coverage,
                missingness=declaration.required_responses.missingness,
                validity=declaration.value_validity,
            )
        pairs = {
            (run, var, location, "evaluated")
            for run, var, location in series
            if var in required
        }
        if declaration.reference:
            which = "wet" if declaration.statistic == "wetmonth_mean" else "dry"
            resolved = reference[which]
            pairs |= {
                (run, "q", resolved["reference_location_id"], "reference")
                for run in resolved["run_ids"]
            }
        cover = [item for item in evidence if item["variable"] in required]
        crop_water = declaration.calculation_set == "crop-water"
        assignment = (
            f"{assumptions['season_labels']}; {assumptions['recharge_year']}"
            if crop_water
            else f"{anchor} ({registry.block_bins(anchor)} water-year blocks); "
            f"{contract.block_assignment}"
        )
        rows.append(
            {
                "indicator": indicator.indicator,
                "set_id": resolved_set.set_id,
                "set_title": resolved_set.title,
                "set_description": resolved_set.description,
                "title": indicator.title,
                "description": indicator.description,
                "variable": ", ".join(sorted(required)),
                "input_units": ", ".join(
                    f"{item.variable}: {item.units}"
                    for item in declaration.requirements
                )
                if len(required) > 1
                else declaration.required_responses.units,
                "output_units": registry.output_units(declaration),
                "quantity": registry.quantity(declaration),
                "method": indicator.method,
                "parameters": _parameters_text(indicator.parameters),
                "equivalent_definition": indicator.equivalent_definition,
                "grain": declaration.grain,
                "group_policy": declaration.group_reduction,
                "water_year_start": (
                    "Oct (recharge); fixed seasons" if crop_water else water_year_start
                ),
                "calendar": "; ".join(sorted({item["calendar"] for item in cover})),
                "time_label": "; ".join(sorted({item["time_label"] for item in cover})),
                "block_assignment": assignment,
                "partial_blocks": contract.partial_blocks,
                "block_coverage": (
                    "Complete-period counts and coverage: indicator_diagnostics.json/period_values"
                    if crop_water
                    else block_coverage(
                        evidence, sorted((r, v, loc, role) for r, v, loc, role in pairs)
                    )
                ),
                "missingness": contract.missingness,
                "validity": contract.validity,
                "references": "; ".join(indicator.references),
                "authored_references": "; ".join(indicator.authored_references),
                "availability": availability,
                "unavailable_reason": reason,
            }
        )
    return rows


# -------------------------------------------------------------- publication
def legacy_claims(root: Path) -> dict[str, str]:
    """Readable result roots claimed by read-only indicator-set/3 manifests."""
    claims = {}
    for marker in sorted((root / "_engine/indicator_sets").glob("*/indicators.json")):
        manifest = read_canonical_json(marker)
        result = str(manifest.get("result_root", ""))
        if result.startswith("results/indicator_sets/"):
            claims[result.split("/")[2]] = marker.parent.name
    return claims


def read_current_publication(
    root: Path, set_id: str, *, verify_export: bool = True
) -> dict:
    """The committed publication of one named set, verified end to end."""
    root = Path(root).resolve()
    names = _names_root(root, set_id)
    legacy = not (names / "current.json").exists()
    if legacy:
        names = root / "_engine/metric_sets/names" / set_id
    current = read_canonical_json(names / "current.json")
    if set(current) != {"schema_version", "publication_id", "publication"} or (
        current["schema_version"]
        != ("named-metric-current/1" if legacy else CURRENT_SCHEMA)
    ):
        raise NamedExportError(f"{set_id}: unsupported current pointer")
    path = _confined(root, current["publication"]["path"])
    if file_sha256(path) != current["publication"]["sha256"]:
        raise NamedExportError(f"{set_id}: publication record digest differs")
    record = read_canonical_json(path)
    expected = {
        "schema_version",
        "set_id",
        "indicator_set_id",
        "indicator_manifest",
        "definition_source",
        "resolved_metadata",
        "files",
        "publication_id",
        "created_at",
    }
    normalized = indicator_names(record) if legacy else record
    if (
        set(normalized) != expected
        or normalized["schema_version"] != PUBLICATION_SCHEMA
    ):
        raise NamedExportError(f"{set_id}: unsupported publication record")
    semantic = {
        k: v for k, v in record.items() if k not in {"publication_id", "created_at"}
    }
    if (
        record["set_id"] != set_id
        or record["publication_id"] != content_sha256(semantic)
        or record["publication_id"] != current["publication_id"]
    ):
        raise NamedExportError(f"{set_id}: publication identity differs")
    record = normalized
    for reference in (record["indicator_manifest"], record["definition_source"]):
        if file_sha256(_confined(root, reference["path"])) != reference["sha256"]:
            raise NamedExportError(f"{set_id}: {reference['path']} differs")
    if verify_export:
        destination = (
            root
            / ("results/metric_sets" if legacy else "results/indicator_sets")
            / set_id
        )
        present = (
            {p.name for p in destination.iterdir()} if destination.is_dir() else set()
        )
        if present != set(record["files"]) or any(
            file_sha256(destination / name) != digest
            for name, digest in record["files"].items()
        ):
            raise NamedExportError(
                f"{set_id}: results/indicator_sets/{set_id} differs from its committed "
                "inventory; rerun to restore it"
            )
    return record


def _owned_names(root: Path, set_id: str) -> set[str] | None:
    """Filenames this set owns, or None when it has no committed publication."""
    names = _names_root(root, set_id)
    if not (names / "current.json").exists():
        return None
    record = read_current_publication(root, set_id, verify_export=False)
    owned = set(record["files"])
    pending = names / "pending.json"
    if pending.exists():
        owned |= set(read_canonical_json(pending)["files"])
    return owned


def publish_named_set(
    root, resolved_set, manifest_path, *, water_year_start: str
) -> dict:
    """Publish or refresh one named export; return its committed record.

    The caller holds project execution ownership. Files are staged with
    rollback copies under ``_engine``; the per-name pointer is written last.
    """
    root = Path(root).resolve()
    set_id = resolved_set.set_id
    manifest = read_indicator_set(root, Path(manifest_path))
    if manifest["schema_version"] not in (
        "indicator-set/4",
        "indicator-set/5",
        "indicator-set/6",
    ):
        raise NamedExportError(f"{set_id}: named exports require indicator-set/4 to /6")
    engine = root / manifest["engine_root"]
    evidence = read_canonical_json(engine / manifest["coverage_evidence"]["path"])
    if evidence.get("schema_version") != COVERAGE_EVIDENCE_SCHEMA:
        raise NamedExportError(f"{set_id}: unsupported coverage evidence")
    rows = dictionary_rows(resolved_set, manifest, evidence["series"], water_year_start)
    payloads = {}
    for reference in [
        manifest["group_members"],
        manifest["group_metadata"],
        *(item["file"] for item in manifest["tables"]),
    ]:
        payload = (root / reference["path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise NamedExportError(f"{set_id}: immutable {reference['path']} differs")
        payloads[Path(reference["path"]).name] = payload
    payloads[DICTIONARY_NAME] = _csv(rows, DICTIONARY_COLUMNS)
    if any(not _FILE_NAME.fullmatch(name) for name in payloads):
        raise NamedExportError(f"{set_id}: invalid export filename")

    names = _names_root(root, set_id)
    destination = root / "results/indicator_sets" / set_id
    for path in (root / "results/indicator_sets", destination, names):
        if path.is_symlink() or (path.exists() and path.resolve() != path):
            raise NamedExportError(f"{set_id}: {path} is aliased")
    claims = legacy_claims(root)
    if set_id in claims:
        raise NamedExportError(
            f"{set_id}: results/indicator_sets/{set_id} is the result root of legacy "
            f"indicator set {claims[set_id]}; choose another set id (contents preserved)"
        )
    owned = _owned_names(root, set_id)
    if destination.exists():
        if owned is None:
            raise NamedExportError(
                f"{set_id}: results/indicator_sets/{set_id} exists but no committed "
                "publication of this set owns it (contents preserved)"
            )
        unknown = sorted(p.name for p in destination.iterdir() if p.name not in owned)
        if unknown or any(not p.is_file() for p in destination.iterdir()):
            raise NamedExportError(
                f"{set_id}: results/indicator_sets/{set_id} holds files this set does not "
                f"own: {unknown} (contents preserved)"
            )

    source = resolved_set.source
    source_bytes = source.text.encode("utf-8")
    definition = names / "definitions" / f"{source.sha256}.yml"
    record = {
        "schema_version": PUBLICATION_SCHEMA,
        "set_id": set_id,
        "indicator_set_id": manifest["indicator_set_id"],
        "indicator_manifest": {
            "path": _relative(root, engine / "indicators.json"),
            "sha256": file_sha256(engine / "indicators.json"),
        },
        "definition_source": {
            "path": _relative(root, definition),
            "sha256": source.sha256,
        },
        "resolved_metadata": rows,
        "files": {
            name: hashlib.sha256(p).hexdigest() for name, p in sorted(payloads.items())
        },
    }
    record["publication_id"] = content_sha256(record)
    record["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    publication = names / "publications" / f"{record['publication_id']}.json"

    if (names / "current.json").exists():
        try:
            current = read_current_publication(root, set_id)
            if current["publication_id"] == record["publication_id"]:
                return current
        except NamedExportError:
            pass  # interrupted or edited export: restore below

    definition.parent.mkdir(parents=True, exist_ok=True)
    if definition.exists():
        if file_sha256(definition) != source.sha256:
            raise NamedExportError(
                f"{set_id}: captured definition {definition} differs"
            )
    else:
        atomic_bytes(definition, source_bytes)
    publication.parent.mkdir(parents=True, exist_ok=True)
    if publication.exists():
        record = read_canonical_json(publication)
    else:
        atomic_record(publication, record)

    staging = names / f".stage-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    atomic_record(names / "pending.json", {"files": record["files"]}, replace=True)
    destination.mkdir(parents=True, exist_ok=True)
    targets = set(payloads) | {p.name for p in destination.iterdir()}
    changed, rollback_failed = [], False
    try:
        for name, payload in payloads.items():
            (staging / name).write_bytes(payload)
        for name in targets:
            if (destination / name).exists():
                shutil.copyfile(destination / name, staging / f"{name}.previous")
        try:
            for name in sorted(targets):
                if name in payloads:
                    os.replace(staging / name, destination / name)
                else:
                    (destination / name).unlink()
                changed.append(name)
            atomic_record(
                names / "current.json",
                {
                    "schema_version": CURRENT_SCHEMA,
                    "publication_id": record["publication_id"],
                    "publication": {
                        "path": _relative(root, publication),
                        "sha256": file_sha256(publication),
                    },
                },
                replace=True,
            )
        except OSError:
            try:
                for name in reversed(changed):
                    backup = staging / f"{name}.previous"
                    if backup.exists():
                        os.replace(backup, destination / name)
                    else:
                        (destination / name).unlink(missing_ok=True)
            except OSError as exc:
                rollback_failed = True
                raise NamedExportError(
                    f"{set_id}: export rollback failed; backups kept in {staging}"
                ) from exc
            raise
    finally:
        if not rollback_failed:
            shutil.rmtree(staging, ignore_errors=True)
    (names / "pending.json").unlink(missing_ok=True)
    return read_current_publication(root, set_id)


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a sibling temporary file and an atomic replace."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def legacy_flat_export(root: Path) -> list[str]:
    """Paths of a pre-named flat export, reported for migration and preserved."""
    root = Path(root)
    tracking = root / "_engine/indicator_sets/current.json"
    if not tracking.is_file():
        return []
    record = read_canonical_json(tracking)
    files = record.get("files") if isinstance(record, dict) else None
    return [tracking.as_posix()] + [
        (root / "results/indicator_sets" / name).as_posix()
        for name in sorted(files or {})
    ]
