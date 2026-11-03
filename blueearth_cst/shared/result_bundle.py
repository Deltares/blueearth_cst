"""Portable WF2 result bundles: explicit export and validated import.

A bundle is one ZIP holding a WF2 product, the exact retained configuration
snapshots that produced it, the basin polygon it was computed over, and a
manifest with every member's size and SHA-256. Source-machine paths survive
only as historical text; nothing in a bundle needs the source project, its
`_shared/` execution inputs, links, locks or `.snakemake/` state.

The working project keeps its one configuration home under `config/`. Bundles
carry snapshots; import never writes the destination's configuration archives,
spatial foundation, invocation history or logs. It only creates the project's
runtime execution lock under `config/_engine/archive-transactions/`. Import supports reading and
interpreting results, not runner adoption: no future no-recompute behaviour
is implied.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from blueearth_cst.shared import product_record

SCHEMA_VERSION = "cst-result-bundle/1"
WORKFLOW = "analyze_projections"
MANIFEST = "manifest.json"
REGION_MEMBER = "dependencies/region.geojson"
# Files without which a WF2 product cannot be read or attributed.
REQUIRED = (
    "report.md",
    "summary/provenance.json",
    "summary/composition.csv",
    "_engine/analysis_record.json",
)
# Scientific settings whose difference changes what the results mean.
CONFIGURATION_FACTS = (
    "ensemble",
    "models",
    "scenarios",
    "members",
    "variables",
    "future_windows",
    "reference_window",
)
_TEXT_SUFFIXES = {".csv", ".json", ".log", ".md", ".tsv", ".txt", ".yml", ".yaml"}


class BundleError(ValueError):
    """A bundle cannot be written or published without a false claim."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _payload(path: Path) -> list[Path]:
    """Every file below `path`, refusing links rather than following them."""
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or item.is_junction():
            raise BundleError(f"linked path cannot be exported: {item}")
        if item.is_file():
            files.append(item)
    return files


def _region_candidates(project: Path, product: Path) -> list[Path]:
    from blueearth_cst.shared.wf3_science import region_geojson_path

    imported = sorted((product / "_engine/imported").glob("*/" + REGION_MEMBER))
    return [Path(region_geojson_path(project)), *imported]


def _region(project: Path, product: Path, expected: str) -> Path:
    """The polygon the product was computed over, by WKB fingerprint."""
    from blueearth_cst.projections.series_identity import region_fingerprint

    for path in _region_candidates(project, product):
        if path.is_file() and region_fingerprint(path) == expected:
            return path
    raise BundleError(
        "no region.geojson matches summary/provenance.json's region fingerprint; "
        "the product no longer describes the project's current basin polygon"
    )


def _snapshots(project: Path, product: Path) -> list[Path]:
    """Retained project snapshots, then snapshots an earlier import carried."""
    root = project / "config/_engine/configuration-snapshots" / WORKFLOW
    local = sorted(root.iterdir()) if root.is_dir() else []
    carried = sorted((product / "_engine/imported").glob("*/configuration/*"))
    return [
        item
        for item in (*local, *carried)
        if item.is_dir()
        and not item.name.startswith(".")
        and (item / "run_record.yml").is_file()
    ]


def _snapshot_by_digest(
    project: Path, product: Path, provenance: Mapping[str, Any]
) -> Path:
    """The retained snapshot whose configuration digests produced provenance."""
    from blueearth_cst.shared.workflow_config_snapshot import validate_archive

    wanted = (
        provenance.get("effective_config_sha256"),
        provenance.get("configuration_inputs_sha256"),
    )
    for directory in _snapshots(project, product):
        projection = validate_archive(directory)["configuration_projection"]
        if (
            projection["effective_config_sha256"],
            projection["configuration_inputs_sha256"],
        ) == wanted:
            return directory
    raise BundleError(
        "no retained configuration snapshot matches summary/provenance.json's "
        "configuration digests; the producing configuration cannot be bundled"
    )


def _snapshot_of(project: Path, product: Path, generation: Mapping[str, Any]) -> Path:
    """A generation's snapshot by run-record digest, wherever it is retained."""
    from blueearth_cst.shared.provenance import file_sha256

    reference = generation["configuration"]["run_record"]
    if not reference:
        raise BundleError("a producing generation has no configuration reference")
    for directory in _snapshots(project, product):
        if file_sha256(directory / "run_record.yml") == reference["sha256"]:
            return directory
    raise BundleError(f"configuration snapshot unavailable: {reference['path']}")


def _snapshot_files(directory: Path) -> list[str]:
    from blueearth_cst.shared.configuration_archive import _archive_files
    from blueearth_cst.shared.workflow_config_snapshot import validate_archive

    return _archive_files(directory, validate_archive(directory))


def _verified_attribution(
    product: Path, record: Mapping[str, Any] | None, names: list[str]
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Trust a record's credit only where current bytes still match it."""
    current = product_record.inventory(product)
    if record is None:
        producers = {name: None for name in current}
        summary = {"mode": "legacy-unattributed", "state": "unrecorded"}
    else:
        if record["state"] != "complete":
            raise BundleError(
                "the product is incomplete: its latest changing run failed; rerun WF2 "
                "until it succeeds before exporting"
            )
        producers = {}
        for name, state in current.items():
            entry = record["files"].get(name, {})
            matches = entry.get("sha256") == state["sha256"]
            producers[name] = entry.get("producer") if matches else None
        summary = {
            "mode": "production-record",
            "state": record["state"],
            "missing_recorded_files": sorted(set(record["files"]) - set(current)),
        }
    summary["unattributed_files"] = sum(value is None for value in producers.values())
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        raise BundleError(f"product lacks required files: {', '.join(missing)}")
    return producers, summary


def _compatibility(
    product: Path, provenance: Mapping[str, Any], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    analysis = json.loads(
        (product / "_engine/analysis_record.json").read_text(encoding="utf-8")
    )
    section = configuration["loaded_config"]["workflows"][WORKFLOW]
    return {
        "region_fingerprint": provenance["region_fingerprint"],
        "variable_spec": provenance.get("variable_spec"),
        "water_year_start": analysis.get("water_year_start"),
        "reference_window": provenance.get("reference_window"),
        "horizon_windows": provenance.get("horizon_windows"),
        "windows": analysis.get("windows"),
        "composition": provenance.get("composition"),
        "series": [
            {
                key: source.get(key)
                for key in ("series_key", "catalog_entry", "members", "series_digest")
            }
            for source in provenance.get("sources", [])
        ],
        "methods": {
            "provenance_schema_version": provenance.get("schema_version"),
            "analysis_record_schema_version": analysis.get("schema_version"),
            "reducer_module_hash": provenance.get("reducer_module_hash"),
            "weighting_scheme": provenance.get("weighting_scheme"),
            "catalog_crawled_on": provenance.get("catalog_crawled_on"),
        },
        "configuration": {key: section.get(key) for key in CONFIGURATION_FACTS},
    }


def _historical_references(members: Mapping[str, Path], project: Path) -> list[str]:
    """Text payloads naming the source project: history, never dependencies."""
    # JSON escapes Windows separators, so look for that spelling as well.
    needles = {str(project), project.as_posix(), json.dumps(str(project))[1:-1]}
    found = []
    for name, path in members.items():
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(needle in text for needle in needles):
            found.append(name)
    return found


def export_bundle(
    project_root: Path, output: Path, *, ensemble: str | None = None
) -> dict[str, Any]:
    """Write one verified WF2 bundle; refuse rather than omit or overclaim."""
    from blueearth_cst.shared.provenance import file_sha256, toolbox_identity
    from blueearth_cst.shared.workflow_config_snapshot import (
        archive_lock,
        validate_archive,
    )

    project = Path(project_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise BundleError(f"bundle already exists: {output}")
    with archive_lock(project, "project-execution"):
        if ensemble:
            product = project / "data/climate/projections" / _safe_member(ensemble)
        elif (project / "config" / WORKFLOW / "run_record.yml").is_file():
            product = product_record.product_root(project, WORKFLOW)
        else:
            raise BundleError(
                "no captured WF2 configuration names the product; pass --ensemble"
            )
        if product is None or not product.is_dir():
            raise BundleError(f"no WF2 product at {product}")
        files = _payload(product)
        names = [path.relative_to(product).as_posix() for path in files]
        record = product_record.read_record(product)
        producers, attribution = _verified_attribution(product, record, names)
        provenance = json.loads(
            (product / "summary/provenance.json").read_text(encoding="utf-8")
        )
        region = _region(project, product, provenance["region_fingerprint"])
        primary = _snapshot_by_digest(project, product, provenance)
        directories = {primary}
        if record is not None:
            owners = {value for value in producers.values() if value is not None}
            directories |= {
                _snapshot_of(project, product, record["generations"][owner])
                for owner in owners
            }
        members: dict[str, Path] = {
            f"product/{product.name}/{name}": path for name, path in zip(names, files)
        }
        snapshots = []
        keys = {
            directory: file_sha256(directory / "run_record.yml")
            for directory in directories
        }
        for directory in sorted(directories, key=keys.get):
            archive = validate_archive(directory)
            projection = archive["configuration_projection"]
            for name in _snapshot_files(directory):
                members[f"configuration/{keys[directory]}/{name}"] = directory / name
            snapshots.append(
                {
                    "path": f"configuration/{keys[directory]}",
                    "effective_config_sha256": projection["effective_config_sha256"],
                    "configuration_inputs_sha256": projection[
                        "configuration_inputs_sha256"
                    ],
                }
            )
        members[REGION_MEMBER] = region
        inventory = []
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            entry = {
                "path": name,
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
            }
            relative = name.removeprefix(f"product/{product.name}/")
            if relative in producers:
                entry["producer"] = producers[relative]
            inventory.append(entry)
        bundle_id = _sha256_bytes(
            "\n".join(f"{item['path']} {item['sha256']}" for item in inventory).encode()
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "workflow": WORKFLOW,
            "product": {
                "name": product.name,
                "path": f"product/{product.name}",
                "destination": product.relative_to(project).as_posix(),
            },
            "created_at_utc": _now(),
            "exported_by": toolbox_identity(),
            "source": {"project_root": str(project), "historical_only": True},
            "attribution": attribution,
            "configuration": {
                "provenance_snapshot": f"configuration/{keys[primary]}",
                "snapshots": snapshots,
            },
            "compatibility": _compatibility(
                product, provenance, validate_archive(primary)
            ),
            "dependencies": {
                "region": {
                    "path": REGION_MEMBER,
                    "fingerprint": provenance["region_fingerprint"],
                }
            },
            "historical_path_references": _historical_references(members, project),
            "files": inventory,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(handle)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                for name, path in sorted(members.items()):
                    archive.write(path, name)
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {
        "bundle": str(output),
        "bundle_id": bundle_id,
        "product": manifest["product"]["destination"],
        "files": len(inventory),
        "size_bytes": output.stat().st_size,
        "attribution": attribution,
        "configuration_snapshots": [item["path"] for item in snapshots],
        "historical_path_references": manifest["historical_path_references"],
    }


def _safe_member(name: str) -> str:
    """Accept only relative POSIX member names that stay inside a root."""
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise BundleError(f"unsafe bundle member name: {name!r}")
    return name


IMPORT_RECORD_VERSION = "cst-result-import/1"
IMPORTED = "_engine/imported"
# Short directory names keep extracted paths inside Windows' default MAX_PATH.
_SHORT = 12


def _read_bundle(bundle: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Validate a bundle's structure and every checksum; nothing is written."""
    try:
        archive = zipfile.ZipFile(bundle)
    except (OSError, zipfile.BadZipFile) as error:
        raise BundleError(f"not a readable bundle: {error}") from error
    with archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise BundleError("bundle repeats a member name")
        for info in infos:
            _safe_member(info.filename)
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise BundleError(f"bundle member is not a file: {info.filename}")
        if MANIFEST not in names:
            raise BundleError("bundle has no manifest.json")
        try:
            manifest = json.loads(archive.read(MANIFEST))
        except (ValueError, zipfile.BadZipFile) as error:
            raise BundleError(f"unreadable manifest: {error}") from error
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise BundleError(
                f"unsupported bundle schema: {manifest.get('schema_version')!r}"
            )
        if manifest.get("workflow") != WORKFLOW:
            raise BundleError(f"unsupported workflow: {manifest.get('workflow')!r}")
        product = _safe_member(manifest["product"]["name"])
        if "/" in product or manifest["product"]["path"] != f"product/{product}":
            raise BundleError("manifest names an invalid product directory")
        listed: dict[str, Mapping[str, Any]] = {}
        for item in manifest["files"]:
            name = _safe_member(item["path"])
            if name in listed or not (
                name.startswith((f"product/{product}/", "configuration/"))
                or name == REGION_MEMBER
            ):
                raise BundleError(f"manifest lists an invalid member: {name}")
            listed[name] = item
        unlisted = sorted(set(names) - set(listed) - {MANIFEST})
        missing = sorted(set(listed) - set(names))
        if unlisted or missing:
            raise BundleError(
                f"bundle members differ from manifest: unlisted {unlisted}, "
                f"missing {missing}"
            )
        payload = {}
        for name, item in listed.items():
            try:
                data = archive.read(name)
            except zipfile.BadZipFile as error:
                raise BundleError(f"damaged member {name}: {error}") from error
            if len(data) != item["size_bytes"] or _sha256_bytes(data) != item["sha256"]:
                raise BundleError(f"member bytes differ from manifest: {name}")
            payload[name] = data
    identity = _sha256_bytes(
        "\n".join(
            f"{path} {listed[path]['sha256']}" for path in sorted(listed)
        ).encode()
    )
    if identity != manifest.get("bundle_id"):
        raise BundleError("manifest inventory differs from its bundle id")
    absent = [name for name in REQUIRED if f"product/{product}/{name}" not in payload]
    if absent or REGION_MEMBER not in payload:
        raise BundleError(f"bundle lacks required files: {absent or [REGION_MEMBER]}")
    return manifest, payload


def _check_region(
    project: Path, manifest: Mapping[str, Any], region: bytes
) -> dict[str, Any]:
    """Compare polygons by WKB fingerprint; equal bounding boxes prove nothing."""
    import io

    from blueearth_cst.projections.series_identity import region_fingerprint
    from blueearth_cst.shared.wf3_science import region_geojson_path

    expected = manifest["compatibility"]["region_fingerprint"]
    if (
        manifest["dependencies"]["region"]["fingerprint"] != expected
        or region_fingerprint(io.BytesIO(region)) != expected
    ):
        raise BundleError("bundled region differs from the results' region")
    destination = Path(region_geojson_path(project))
    if not destination.is_file():
        return {"status": "unverified", "reason": "destination has no region.geojson"}
    if region_fingerprint(destination) != expected:
        raise BundleError(
            "destination basin polygon differs from the bundle's; "
            "these results describe another region"
        )
    return {
        "status": "verified",
        "destination": destination.relative_to(project).as_posix(),
    }


def _check_configuration(
    project: Path, manifest: Mapping[str, Any], config: Path | None
) -> dict[str, Any]:
    """Name every scientific setting the destination configures differently."""
    if config is not None:
        from blueearth_cst.shared.config_composition import load_composed_config

        section = load_composed_config(config)["workflows"].get(WORKFLOW)
        compared = str(Path(config).resolve())
    elif (project / "config" / WORKFLOW / "run_record.yml").is_file():
        from blueearth_cst.shared.workflow_config_snapshot import read_archive

        record = read_archive(project, WORKFLOW, f"config/{WORKFLOW}")
        section = record["loaded_config"]["workflows"][WORKFLOW]
        compared = f"config/{WORKFLOW}/run_record.yml"
    else:
        return {"compared_with": None, "differences": {}}
    if section is None:
        return {"compared_with": compared, "differences": {}, "note": "no WF2 section"}
    bundle = manifest["compatibility"]["configuration"]
    differences = {
        key: {"bundle": bundle.get(key), "destination": section.get(key)}
        for key in CONFIGURATION_FACTS
        if bundle.get(key) != section.get(key)
    }
    return {"compared_with": compared, "differences": differences}


def _same_import(target: Path, manifest: Mapping[str, Any]) -> bool:
    """An earlier import of exactly these bytes, untouched since."""
    imported = target / IMPORTED / manifest["bundle_id"][:_SHORT]
    if not (imported / MANIFEST).is_file():
        return False
    prefix = f"{manifest['product']['path']}/"
    expected = {
        item["path"].removeprefix(prefix): item["sha256"]
        for item in manifest["files"]
        if item["path"].startswith(prefix)
    }
    current = {
        path.relative_to(target).as_posix(): path
        for path in _payload(target)
        if not path.is_relative_to(imported)
    }
    return set(current) == set(expected) and all(
        _sha256_bytes(path.read_bytes()) == expected[name]
        for name, path in current.items()
    )


def _staged_name(name: str, manifest: Mapping[str, Any], imported: str) -> str:
    product = f"{manifest['product']['path']}/"
    if name.startswith(product):
        return name.removeprefix(product)
    if name.startswith("configuration/"):
        snapshot, rest = name.removeprefix("configuration/").split("/", 1)
        return f"{imported}/configuration/{snapshot[:_SHORT]}/{rest}"
    return f"{imported}/{name}"


def import_bundle(
    bundle: Path,
    project_root: Path,
    *,
    config: Path | None = None,
    accept_differences: bool = False,
) -> dict[str, Any]:
    """Publish a verified bundle's product atomically, or change nothing."""
    from blueearth_cst.shared.provenance import toolbox_identity
    from blueearth_cst.shared.workflow_config_snapshot import (
        _replace,
        _write_durable,
        archive_lock,
        validate_archive,
    )

    manifest, payload = _read_bundle(Path(bundle))
    project = Path(project_root).resolve(strict=True)
    name = manifest["product"]["name"]
    target = project / "data/climate/projections" / name
    imported = f"{IMPORTED}/{manifest['bundle_id'][:_SHORT]}"
    with archive_lock(project, "project-execution"):
        if target.exists():
            if _same_import(target, manifest):
                return {
                    "status": "already_imported",
                    "product": target.relative_to(project).as_posix(),
                    "bundle_id": manifest["bundle_id"],
                }
            raise BundleError(
                f"destination already has a WF2 product at {target}; "
                "import refuses to replace or merge it"
            )
        region = _check_region(project, manifest, payload[REGION_MEMBER])
        configuration = _check_configuration(project, manifest, config)
        if configuration["differences"] and not accept_differences:
            raise BundleError(
                "destination configuration differs in "
                f"{', '.join(configuration['differences'])}; "
                "rerun with --accept-differences to import anyway"
            )
        configuration["accepted"] = bool(configuration["differences"])
        target.parent.mkdir(parents=True, exist_ok=True)
        # The project lock is held, so any staged sibling is an interrupted import.
        stale = sorted(target.parent.glob(f".{name}.import-*.staged"))
        for directory in stale:
            _remove_tree(target.parent, directory)
        staging = target.parent / f".{name}.import-{os.urandom(8).hex()}.staged"
        try:
            for member, data in payload.items():
                path = staging / _staged_name(member, manifest, imported)
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_durable(path, data)
            for snapshot in manifest["configuration"]["snapshots"]:
                short = snapshot["path"].removeprefix("configuration/")[:_SHORT]
                validate_archive(staging / imported / "configuration" / short)
            _write_durable(
                staging / imported / MANIFEST,
                json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n",
            )
            record = {
                "schema_version": IMPORT_RECORD_VERSION,
                "bundle_id": manifest["bundle_id"],
                "imported_at_utc": _now(),
                "imported_by": toolbox_identity(),
                "source_project_root": manifest["source"]["project_root"],
                "attribution": manifest["attribution"],
                "region": region,
                "configuration": configuration,
                "runner_adoption": (
                    "none: no runner invocation produced these files here; a later "
                    "WF2 run may recompute them"
                ),
            }
            _write_durable(
                staging / imported / "import_record.json",
                json.dumps(record, indent=2, sort_keys=True).encode() + b"\n",
            )
            _replace(staging, target)
        finally:
            if staging.exists():
                _remove_tree(target.parent, staging)
    return {
        "status": "imported",
        "product": target.relative_to(project).as_posix(),
        "bundle_id": manifest["bundle_id"],
        "files": sum(1 for item in payload if item.startswith("product/")),
        "import_record": f"{target.relative_to(project).as_posix()}/{imported}/"
        "import_record.json",
        "region": region,
        "configuration": configuration,
        "removed_interrupted_imports": [path.name for path in stale],
    }


def _remove_tree(parent: Path, directory: Path) -> None:
    import shutil

    if directory.parent != parent or not directory.name.endswith(".staged"):
        raise BundleError(f"refusing to remove {directory}")
    if any(path.is_symlink() or path.is_junction() for path in directory.rglob("*")):
        raise BundleError(f"staged import contains links: {directory}")
    shutil.rmtree(directory)
