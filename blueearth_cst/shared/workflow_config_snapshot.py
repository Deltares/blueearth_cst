"""Record the config an artifact-scoped workflow ran under, beside the artifact.

WF0-WF2 are singletons per project, so each writes one overwritable record
under ``config/<workflow>/``. WF3 and WF4 are not: a project holds one
directory per scenario collection and one per experiment, so a single
``config/<workflow>/`` slot would be claimed by whichever ran last and
would describe the others wrongly. Their snapshot therefore lives INSIDE the
identified directory, which is the only placement that stays true when a
second collection is generated.

**The snapshot is UNREFERENCED on purpose.** ``simulation_id`` and
``collection_id`` hash digests and never path strings, and neither
``simulation.json`` nor ``collection.json`` carries a directory listing -- so a
file the identity documents do not name changes no identity and invalidates no
retained indicator set. Recording these fields INSIDE ``collection_intent.json``
or ``simulation.json`` would have been the tidier shape, and would have
rewritten every existing experiment's identity to buy provenance. It would also
fail closed rather than silently: ``read_simulation_intent_v2`` compares the
record's key set against an exact expected set and raises on any extra field.

What it closes, none of which the digest documents carried:

* **which SOURCE file the settings came from**, and whether it has changed
  since. The collection records the resolved VALUES -- which is the stronger
  fact for reproducing numbers -- but nothing anywhere in the scenario or
  experiment trees named the file they were resolved from, so the audit
  question "is this still the config that produced it?" had no answer;
* **``operation:``**, a WF4 config key that reached no record at all;
* a view in the same vocabulary and format as the source config, rather than
  settings split across two JSON provenance documents in a different one.

It is a RECORD, not a re-run button. The collection also depends on the staged
forcing and the generator template, which ``source_inventory.json`` pins by
sha256 and this file does not carry -- the same limit the WF1 snapshot has.
"""

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import yaml

from blueearth_cst.shared.provenance import (
    collection_sha256,
    configuration_inputs_digest,
    effective_config_digest,
    environment_file_hashes,
    file_sha256,
    short_digest,
)

SCHEMA_VERSION = 1

#: The snapshot's filename, in every bin that holds one. Deliberately the same
#: name WF0-WF2 use under ``config/<workflow>/``: a reader who has learned
#: one has learned all five, and the directory already says which workflow and
#: which artifact it belongs to. Spelling the workflow into the name as well
#: would restate the directory and stutter in the per-workflow bins.
SNAPSHOT_NAME = "composed_config.yml"

#: Header planted at the top of every snapshot. ``yaml.safe_dump`` drops
#: comments from the document, so a reader who opens this file without having
#: read the bin's README would have no way to tell a record from a source they
#: may edit -- which is exactly the mistake the two-tier config split exists to
#: prevent.
_HEADER = (
    "# Written by the run. Editing this file changes nothing: the next\n"
    "# execution of this workflow overwrites it, and a retained collection or\n"
    "# experiment is immutable. To change what a run does, edit the config you\n"
    "# pass to --configfile and run the workflow again.\n"
)


def snapshot_document(
    workflow: str,
    source_config_path: Union[str, Path],
    workflow_config_path: Union[str, Path, None],
    composed_config: Mapping[str, Any],
) -> dict:
    """Assemble the snapshot for one artifact-scoped workflow run.

    Parameters
    ----------
    workflow
        The workflow key, as it appears under ``workflows:``.
    source_config_path
        The project file the user passed to ``--configfile``. Hashed as
        invoked, so the record names the file a reader would go back to.
    workflow_config_path
        That workflow's own settings file, as ``compose_config`` resolved it.
        ``None`` when the project declared no ``config_path`` for it, which is
        legal and must not be mistaken for a missing record.
    composed_config
        The composed config the workflow is running under -- the WHOLE
        composed mapping, not a projection of it. That is what the WF0-WF2
        snapshots hold, and the point of this file is that a reader who has
        learned one snapshot has learned all five. A projection belongs to a
        DIGEST, whose job is to not move when another workflow's section
        changes; this is a record, whose job is to say what ran.
    """
    # Both paths RESOLVED, and both the same way. The project file arrives
    # absolute (Snakemake's `workflow.configfiles[0]`) while the workflow file
    # arrives relative to the run directory, so recording them as given would
    # put two different kinds of path in one document and leave a reader unable
    # to tell which anchor the relative one used. Absolute matches what
    # `run_record.yml` already records for the same field.
    source_path = Path(source_config_path).resolve()
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "workflow": workflow,
        "source_config": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
        },
    }
    if workflow_config_path is None:
        # Recorded as an explicit null rather than an absent key: "this project
        # declared no settings file for this workflow" and "this record predates
        # the field" must not read the same way.
        document["workflow_config"] = None
    else:
        workflow_path = Path(workflow_config_path).resolve()
        document["workflow_config"] = {
            "path": str(workflow_path),
            "sha256": file_sha256(workflow_path),
        }
    document["effective_config"] = dict(composed_config)
    return document


def render_snapshot(document: Mapping[str, Any]) -> str:
    """Render the snapshot as the bytes written into the artifact directory.

    ``sort_keys=True`` for the same reason the WF0-WF2 snapshots use it: two
    runs of the same configuration must produce the same bytes, or a byte
    comparison between two projects reports a difference that is only key order.
    """
    return _HEADER + yaml.safe_dump(dict(document), sort_keys=True)


def snapshot_bytes(
    workflow: str,
    source_config_path: Union[str, Path],
    workflow_config_path: Union[str, Path, None],
    composed_config: Mapping[str, Any],
) -> bytes:
    """Build and render in one step, for a caller that writes bytes.

    WF3 writes its snapshot through ``write_collection_payload``, which takes
    bytes and refuses to write at all once the collection is sealed -- so the
    snapshot has to be a value the caller can hand over, not a path this module
    writes to itself.
    """
    document = snapshot_document(
        workflow,
        source_config_path,
        workflow_config_path,
        composed_config,
    )
    return render_snapshot(document).encode("utf-8")


def composed_workflow_section(
    project: Mapping[str, Any], workflow: str, settings: Mapping[str, Any]
) -> dict:
    """Rebuild ``compose_config``'s shape for an entry point that does not use it.

    WF4 reads the project file and its own settings file directly
    (``simulation_settings``), so there is no composed mapping to snapshot. This
    reproduces the one ``compose_config`` would have produced: ``config_path``
    dropped from every stanza -- it is a pointer to a file, not a setting, and
    the composed document has already followed it -- and this workflow's
    settings merged into its own stanza beside ``enabled``.

    Without this the WF4 snapshot would be a different shape from the other
    four, and the promise that one snapshot teaches you all five would be false
    exactly where a reader is least able to check it.
    """
    stanzas = {
        name: {
            key: value for key, value in dict(stanza).items() if key != "config_path"
        }
        for name, stanza in dict(project.get("workflows") or {}).items()
    }
    stanzas.setdefault(workflow, {}).update(dict(settings))
    composed = {key: value for key, value in project.items() if key != "workflows"}
    composed["workflows"] = stanzas
    return composed


def write_snapshot(
    directory: Union[str, Path],
    workflow: str,
    source_config_path: Union[str, Path],
    workflow_config_path: Union[str, Path, None],
    composed_config: Mapping[str, Any],
) -> Path:
    """Write the snapshot into ``directory`` and return the path written."""
    target = Path(directory) / SNAPSHOT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        snapshot_bytes(
            workflow,
            source_config_path,
            workflow_config_path,
            composed_config,
        )
    )
    return target


RUN_RECORD_VERSION = "run-record/2"
ARCHIVE_VERSION = "source-archive/2"
_LEGACY_ARCHIVE_VERSION = "source-archive/1"
_ARCHIVE_RESERVED_NAMES = frozenset(
    {"run_record.yml", "sources.md", "catalogs", "sources", "_engine", "predecessor"}
)
TRANSACTION_VERSION = "archive-transaction/1"


@dataclass(frozen=True)
class CapturedSource:
    """One resolved source identity and the exact bytes captured before parse."""

    id: str
    role: str
    original_path: Path
    data: bytes
    git_blob: str | None = None


@dataclass(frozen=True)
class RerunResolution:
    """Validated relocated archive and its explicit source/path resolver."""

    composed_config: dict[str, Any]
    workflow_config_paths: dict[str, str]
    sources: tuple[CapturedSource, ...]
    archive_directory: Path
    project_root: Path
    original_project_root: Path
    source_paths: Mapping[str, Path]
    external_paths: Mapping[str, Path]

    def resolve_path(self, locator: str | Path) -> Path:
        """Map a recorded locator without basename search or old-path fallback."""
        original = Path(locator)
        if not original.is_absolute():
            raise ValueError(
                "rerun locator must be absolute; resolve run-directory paths first"
            )
        identity = str(original)
        if identity in self.source_paths:
            return self.source_paths[identity]
        try:
            relative = original.relative_to(self.original_project_root)
        except ValueError:
            if identity in self.external_paths:
                return self.external_paths[identity]
            raise ValueError(f"unmapped external rerun path: {identity}") from None
        if relative == Path("."):
            return self.project_root
        return _confined(self.project_root, relative.as_posix())


def capture_sources(
    sources: Sequence[tuple[str, str, str | Path]],
) -> tuple[CapturedSource, ...]:
    """Read each distinct source once; never reread it during archive writing."""
    captured: dict[Path, bytes] = {}
    result = []
    ids = set()
    for source_id, role, raw_path in sources:
        if not source_id or source_id in ids or not role:
            raise ValueError("source ids must be unique and roles nonempty")
        ids.add(source_id)
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"source is not a file: {path}")
        if path not in captured:
            captured[path] = path.read_bytes()
        result.append(CapturedSource(source_id, role, path, captured[path]))
    return tuple(result)


_CATALOG_ROLES = frozenset({"projection_store_index"})


def _is_catalog_role(role: str) -> bool:
    """A catalog itself, or ``projection_store_index`` (cmip6_store_index.json),
    which ships beside its ``projection_catalog`` and is kept next to it for
    convenience even though its own name doesn't say "catalog"."""
    return "catalog" in role or role in _CATALOG_ROLES


def source_archive_paths(sources: Sequence[CapturedSource]) -> dict[str, str]:
    """Place captured files beside the record, with catalogs in ``catalogs/``.

    Original locations remain in the record and ``SOURCES.md``. Names are
    compared case-insensitively on every platform so archives stay portable.
    Reserved names and colliding basenames gain a deterministic suffix from
    their original path and bytes; equal bytes at different paths stay distinct.
    """
    if not sources:
        raise ValueError("an archive needs at least the project source")
    paths: dict[str, str] = {}
    occupied: dict[str, str | None] = dict.fromkeys(_ARCHIVE_RESERVED_NAMES)
    for source in sorted(sources, key=lambda item: item.id):
        bucket = "catalogs/" if _is_catalog_role(source.role) else ""
        name = source.original_path.name
        if not name or name in {".", ".."} or any(c in name for c in "\\/:"):
            raise ValueError(f"unsafe source basename: {name!r}")
        candidate = f"{bucket}{name}"
        key = candidate.casefold()
        original = str(source.original_path)
        if key in occupied and occupied[key] != original:
            digest = short_digest(
                hashlib.sha256(
                    original.encode("utf-8") + b"\0" + source.data
                ).hexdigest()
            )
            stem = source.original_path.stem
            suffix = source.original_path.suffix
            candidate = f"{bucket}{stem}-{digest}{suffix}"
            key = candidate.casefold()
            counter = 2
            while key in occupied and occupied[key] != original:
                candidate = f"{bucket}{stem}-{digest}-{counter}{suffix}"
                key = candidate.casefold()
                counter += 1
        occupied[key] = original
        paths[source.id] = candidate
    return paths


def write_source_index(sources: Sequence[CapturedSource]) -> str:
    """Render ``SOURCES.md``: the archive-name -> original-directory map.

    Generated at publish time from the same data as ``run_record.yml``'s
    ``original_path`` field -- it is a convenience mirror, not a second source
    of truth, and carries no bytes of its own for :func:`validate_archive` to
    check. Read this when the flat name alone does not say where a file came
    from; read ``run_record.yml`` when you need the fact to be load-bearing.
    """
    paths = source_archive_paths(sources)
    rows = sorted(
        (paths[source.id], str(source.original_path.parent)) for source in sources
    )
    lines = [
        "# Source directories",
        "",
        "Captured files beside this index do not carry their original directory --",
        "this is a generated index, not a source file itself. The same fact is",
        "recorded per source in `run_record.yml`'s `original_path` field.",
        "",
        "| archived as | original directory |",
        "|---|---|",
    ]
    lines.extend(f"| `{archived}` | `{directory}` |" for archived, directory in rows)
    return "\n".join(lines) + "\n"


def _source_entries(
    sources: Sequence[CapturedSource],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    paths = source_archive_paths(sources)
    entries = []
    payloads = {}
    for source in sorted(sources, key=lambda item: item.id):
        archive_path = paths[source.id]
        digest = hashlib.sha256(source.data).hexdigest()
        entries.append(
            {
                "id": source.id,
                "role": source.role,
                "original_path": str(source.original_path),
                "archived_path": archive_path,
                "path_base": "record_directory",
                "sha256": digest,
                "size_bytes": len(source.data),
                "recoverable": True,
                "git_blob": source.git_blob,
            }
        )
        previous = payloads.setdefault(archive_path, source.data)
        if previous != source.data:
            raise ValueError(f"conflicting source bytes: {archive_path}")
    payloads["SOURCES.md"] = write_source_index(sources).encode("utf-8")
    return entries, payloads


def run_record_document(
    *,
    workflow: str,
    invocation_id: str,
    owner_kind: str,
    owner_id: str,
    loaded_config: Mapping[str, Any],
    advanced_settings: Mapping[str, Any],
    projection: Sequence[str],
    toolbox: Mapping[str, Any],
    invocation: Mapping[str, Any],
    sources: Sequence[CapturedSource],
    referenced_inputs: Sequence[Mapping[str, Any]] = (),
    generated_inputs: Sequence[Mapping[str, Any]] = (),
    rerun_adjustments: Sequence[Mapping[str, Any]] = (),
    environment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Build run-record/2 and its immutable captured source payloads."""
    entries, payloads = _source_entries(sources)
    by_id = {entry["id"]: entry for entry in entries}
    if "project" not in by_id:
        raise ValueError("captured sources must include id 'project'")
    workflow_entries = [
        entry for entry in entries if entry["role"] == f"workflow_config_{workflow}"
    ]
    if len(workflow_entries) > 1:
        raise ValueError("multiple workflow config sources")
    effective_sha = effective_config_digest(
        loaded_config, advanced_settings, projection
    )
    env = dict(environment if environment is not None else environment_file_hashes())
    references = sorted(
        (dict(item) for item in referenced_inputs), key=lambda item: item["id"]
    )
    if len({item["id"] for item in references}) != len(references):
        raise ValueError("duplicate referenced input id")
    for reference in references:
        if set(reference) != {
            "id",
            "role",
            "original_path",
            "source_id",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("referenced input must use archive-referenced-input/1")
        if reference["source_id"] is not None and reference["source_id"] not in by_id:
            raise ValueError("referenced input names uncaptured source")
        if reference["source_id"] is not None:
            source_entry = by_id[reference["source_id"]]
            if (reference["sha256"], reference["size_bytes"]) != (
                source_entry["sha256"],
                source_entry["size_bytes"],
            ):
                raise ValueError("referenced input differs from captured source")
    generated = sorted(
        (dict(item) for item in generated_inputs),
        key=lambda item: (item["role"], item["file"]["path"]),
    )
    for item in generated:
        if set(item) != {"role", "file"} or set(item["file"]) != {
            "schema_version",
            "path_base",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("generated input must use archive-generated-input/1")
        if item["file"]["schema_version"] != "artifact-reference/1":
            raise ValueError("unknown generated input reference")
    rerun = {
        "project_source": "project",
        "workflow": workflow,
        "path_policy": "captured-source-map/1",
        "adjustments": list(rerun_adjustments),
    }
    config_projection = {
        "paths": list(projection),
        "digest_schema_version": 2,
        "effective_config_sha256": effective_sha,
        "configuration_inputs_sha256": configuration_inputs_digest(
            effective_sha, toolbox, env, references
        ),
    }
    record = {
        "schema_version": RUN_RECORD_VERSION,
        "archive_schema_version": ARCHIVE_VERSION,
        "workflow": workflow,
        "invocation_id": invocation_id,
        "owner": {"kind": owner_kind, "id": owner_id},
        "loaded_config": dict(loaded_config),
        "advanced_settings": dict(advanced_settings),
        "configuration_projection": config_projection,
        "toolbox": dict(toolbox),
        "environment": env,
        "invocation": dict(invocation),
        "source_config": "project",
        "workflow_config": workflow_entries[0]["id"] if workflow_entries else None,
        "source_files": entries,
        "referenced_inputs": references,
        "generated_inputs": generated,
        "rerun": rerun,
    }
    record["archive_id"] = collection_sha256(
        {
            "schema_version": ARCHIVE_VERSION,
            "workflow": workflow,
            "invocation_id": invocation_id,
            "source_files": [
                {
                    key: entry[key]
                    for key in ("id", "role", "archived_path", "sha256", "size_bytes")
                }
                for entry in entries
            ],
            "loaded_config": record["loaded_config"],
            "advanced_settings": record["advanced_settings"],
            "configuration_projection": config_projection,
            "referenced_inputs": references,
            "generated_inputs": generated,
            "rerun": rerun,
        }
    )
    _validate_run_record_shape(record)
    return record, payloads


def _record_bytes(record: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(dict(record), sort_keys=True, allow_unicode=True).encode(
        "utf-8"
    )


def _write_durable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


_WINDOWS_REPLACE_RETRY = os.name == "nt"
#: Bounded retry budget for a Windows-only transient rename failure (WinError 5)
#: seen when an antivirus/indexer briefly holds a handle open on a just-written
#: file or directory -- the identical pattern already handled in
#: dev/scripts/stage_data.py's `_write_zarr` for zarr's own metadata rename.
#: 10 attempts at a flat 20ms (0.2s total) turned out too short for a live run:
#: t2609231529 hit WinError 5 renaming a whole directory (`config/<workflow>`,
#: whose scan window plausibly outlasts a single file's). Exponential backoff,
#: capped per attempt so it cannot itself become the multi-minute stall it is
#: meant to absorb, sums to ~5s over 12 attempts -- the same order of magnitude
#: that already resolved this pattern for zarr writes.
_REPLACE_RETRY_ATTEMPTS = 12
_REPLACE_RETRY_BASE_SECONDS = 0.05
_REPLACE_RETRY_CAP_SECONDS = 0.5


def _replace(source: Path, target: Path) -> None:
    """Replace a file or directory despite momentary Windows reader handles."""
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            if (
                not _WINDOWS_REPLACE_RETRY
                or error.winerror != 5
                or attempt == _REPLACE_RETRY_ATTEMPTS - 1
            ):
                raise
            delay = min(
                _REPLACE_RETRY_BASE_SECONDS * (2**attempt), _REPLACE_RETRY_CAP_SECONDS
            )
            time.sleep(delay)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_durable(
            temp,
            (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        _replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def archive_lock(project_root: Path, owner_key: str) -> Iterator[None]:
    """Cross-process archive lock shared by the writer and sanctioned readers."""
    if not owner_key or any(char in owner_key for char in "/\\:. "):
        raise ValueError("invalid archive owner key")
    if os.name == "nt":
        import ctypes

        drive = project_root.drive
        if (
            drive.startswith("\\\\")
            or ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\") == 4
        ):
            raise ValueError(
                "network project roots lack the required archive lock guarantee"
            )
    path = (
        project_root
        / "config"
        / "_engine"
        / "archive-transactions"
        / f"{owner_key}.lock"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            # Windows permits locking beyond EOF. Initialize only while owned;
            # reading/writing the byte beforehand races another process's lock.
            if os.fstat(handle.fileno()).st_size == 0:
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _confined(root: Path, relative: str) -> Path:
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError(f"archive path is not confined POSIX: {relative}")
    path = Path(relative)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"unconfined archive path: {relative}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"archive path escapes root: {relative}")
    return result


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REF_BASES = frozenset(
    {"project_root", "experiment_root", "record_directory", "request_directory"}
)


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has unclassified or missing fields")


def _digest(value: Any, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _reference_shape(reference: Any, name: str) -> None:
    _keys(
        reference,
        {"schema_version", "path_base", "path", "sha256", "size_bytes"},
        name,
    )
    if (
        reference["schema_version"] != "artifact-reference/1"
        or reference["path_base"] not in _REF_BASES
    ):
        raise ValueError(f"{name} has an unsupported version or base")
    _digest(reference["sha256"], f"{name}.sha256")
    if type(reference["size_bytes"]) is not int or reference["size_bytes"] < 0:
        raise ValueError(f"{name}.size_bytes must be nonnegative")
    # Validate lexical path constraints before an anchor exists.
    relative = reference["path"]
    if (
        not isinstance(relative, str)
        or "\\" in relative
        or ":" in relative
        or relative.startswith("/")
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError(f"{name}.path is not confined relative POSIX")


def _validate_run_record_shape(record: Any) -> None:
    """Reject unknown/missing fields before checking any archive digest."""
    _keys(
        record,
        {
            "schema_version",
            "archive_schema_version",
            "archive_id",
            "workflow",
            "invocation_id",
            "owner",
            "loaded_config",
            "advanced_settings",
            "configuration_projection",
            "toolbox",
            "environment",
            "invocation",
            "source_config",
            "workflow_config",
            "source_files",
            "referenced_inputs",
            "generated_inputs",
            "rerun",
        },
        "run-record/2",
    )
    if record["schema_version"] != RUN_RECORD_VERSION or record[
        "archive_schema_version"
    ] not in (ARCHIVE_VERSION, _LEGACY_ARCHIVE_VERSION):
        raise ValueError("unsupported run record or source archive")
    _digest(record["archive_id"], "archive_id")
    for key in ("workflow", "invocation_id"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{key} must be nonempty")
    _keys(record["owner"], {"kind", "id"}, "owner")
    if record["owner"]["kind"] not in {
        "workflow",
        "scenario_collection",
        "simulation",
        "experiment",
        "invocation",
    }:
        raise ValueError("unknown archive owner kind")
    if not isinstance(record["owner"]["id"], str) or not record["owner"]["id"]:
        raise ValueError("owner id must be nonempty")
    for key in ("loaded_config", "advanced_settings", "toolbox", "environment"):
        if not isinstance(record[key], dict):
            raise ValueError(f"{key} must be a mapping")
    projection = record["configuration_projection"]
    _keys(
        projection,
        {
            "paths",
            "digest_schema_version",
            "effective_config_sha256",
            "configuration_inputs_sha256",
        },
        "configuration_projection",
    )
    if (
        projection["digest_schema_version"] != 2
        or type(projection["paths"]) is not list
        or not all(isinstance(path, str) and path for path in projection["paths"])
    ):
        raise ValueError("invalid configuration projection")
    _digest(projection["effective_config_sha256"], "effective_config_sha256")
    _digest(projection["configuration_inputs_sha256"], "configuration_inputs_sha256")
    _keys(
        record["invocation"],
        {"entry_point", "command", "targets", "working_directory", "overrides"},
        "invocation",
    )
    invocation = record["invocation"]
    if (
        not all(
            isinstance(invocation[key], str)
            for key in ("entry_point", "working_directory")
        )
        or not all(
            isinstance(invocation[key], list)
            and all(isinstance(item, str) for item in invocation[key])
            for key in ("command", "targets")
        )
        or not isinstance(invocation["overrides"], dict)
    ):
        raise ValueError("invalid invocation evidence")
    sources = record["source_files"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_files must be nonempty")
    source_ids = []
    archive_paths = {}
    for entry in sources:
        _keys(
            entry,
            {
                "id",
                "role",
                "original_path",
                "archived_path",
                "path_base",
                "sha256",
                "size_bytes",
                "recoverable",
                "git_blob",
            },
            "source-entry/1",
        )
        if not all(
            isinstance(entry[key], str) and entry[key]
            for key in ("id", "role", "original_path")
        ):
            raise ValueError("invalid source identity")
        _reference_shape(
            {
                "schema_version": "artifact-reference/1",
                "path_base": entry["path_base"],
                "path": entry["archived_path"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            },
            "source entry",
        )
        relative = entry["archived_path"]
        parts = relative.split("/")
        if record["archive_schema_version"] == _LEGACY_ARCHIVE_VERSION:
            valid_path = relative.startswith("sources/")
        else:
            valid_path = (
                len(parts) == 1 and relative.casefold() not in _ARCHIVE_RESERVED_NAMES
            ) or (len(parts) == 2 and parts[0] == "catalogs")
        if entry["path_base"] != "record_directory" or not valid_path:
            raise ValueError("source entry has wrong archive base")
        if type(entry["recoverable"]) is not bool or not entry["recoverable"]:
            raise ValueError("captured source must be recoverable")
        if entry["git_blob"] is not None and not isinstance(entry["git_blob"], str):
            raise ValueError("invalid source git blob")
        source_ids.append(entry["id"])
        key = (
            relative.casefold()
            if record["archive_schema_version"] == ARCHIVE_VERSION or os.name == "nt"
            else relative
        )
        identity = (
            relative,
            entry["original_path"],
            entry["sha256"],
            entry["size_bytes"],
        )
        if key in archive_paths and archive_paths[key] != identity:
            raise ValueError("source archive paths collide")
        archive_paths[key] = identity
    if source_ids != sorted(set(source_ids)):
        raise ValueError("source ids or archive paths are duplicate/unsorted")
    if record["source_config"] not in source_ids:
        raise ValueError("project source pointer is absent")
    by_id = {entry["id"]: entry for entry in sources}
    if by_id[record["source_config"]]["role"] != "project_config":
        raise ValueError("project source pointer has wrong role")
    workflow_source = record["workflow_config"]
    if workflow_source is not None and (
        workflow_source not in by_id
        or by_id[workflow_source]["role"] != f"workflow_config_{record['workflow']}"
    ):
        raise ValueError("workflow source pointer has wrong role")
    references = record["referenced_inputs"]
    if not isinstance(references, list):
        raise ValueError("referenced_inputs must be a list")
    reference_ids = []
    for item in references:
        _keys(
            item,
            {"id", "role", "original_path", "source_id", "sha256", "size_bytes"},
            "archive-referenced-input/1",
        )
        if not all(
            isinstance(item[key], str) and item[key]
            for key in ("id", "role", "original_path")
        ):
            raise ValueError("invalid referenced input identity")
        _digest(item["sha256"], "referenced input sha256")
        if type(item["size_bytes"]) is not int or item["size_bytes"] < 0:
            raise ValueError("invalid referenced input size")
        if item["source_id"] is not None:
            if item["source_id"] not in by_id or (
                item["sha256"],
                item["size_bytes"],
            ) != (
                by_id[item["source_id"]]["sha256"],
                by_id[item["source_id"]]["size_bytes"],
            ):
                raise ValueError("referenced input differs from captured source")
        reference_ids.append(item["id"])
    if reference_ids != sorted(set(reference_ids)):
        raise ValueError("referenced input ids are duplicate/unsorted")
    generated = record["generated_inputs"]
    if not isinstance(generated, list):
        raise ValueError("generated_inputs must be a list")
    generated_sort = []
    for item in generated:
        _keys(item, {"role", "file"}, "archive-generated-input/1")
        if not isinstance(item["role"], str) or not item["role"]:
            raise ValueError("generated input role must be nonempty")
        _reference_shape(item["file"], "generated input FileRef")
        generated_sort.append((item["role"], item["file"]["path"]))
    if generated_sort != sorted(set(generated_sort)):
        raise ValueError("generated input roles/paths are duplicate/unsorted")
    rerun = record["rerun"]
    _keys(
        rerun,
        {"project_source", "workflow", "path_policy", "adjustments"},
        "archive-rerun/1",
    )
    if (
        rerun["project_source"] != record["source_config"]
        or rerun["workflow"] != record["workflow"]
        or rerun["path_policy"] != "captured-source-map/1"
        or not isinstance(rerun["adjustments"], list)
        or len(rerun["adjustments"]) > 1
    ):
        raise ValueError("invalid rerun declaration")
    for overlay in rerun["adjustments"]:
        _keys(
            overlay,
            {
                "schema_version",
                "source_record",
                "source_map",
                "project_root_mapping",
                "external_mappings",
                "overrides",
                "output_project_dir",
            },
            "resolver-overlay/1",
        )
        if overlay["schema_version"] != "resolver-overlay/1":
            raise ValueError("unknown resolver overlay")
        _reference_shape(overlay["source_record"], "predecessor FileRef")
        if overlay["source_record"]["path_base"] != "record_directory":
            raise ValueError("predecessor FileRef needs record_directory base")
        _keys(overlay["project_root_mapping"], {"from", "to"}, "project_root_mapping")
        if not all(
            isinstance(overlay["project_root_mapping"][key], str)
            and overlay["project_root_mapping"][key]
            for key in ("from", "to")
        ):
            raise ValueError("invalid project root mapping")
        if not isinstance(overlay["overrides"], dict) or not isinstance(
            overlay["output_project_dir"], str
        ):
            raise ValueError("invalid rerun overrides/output")
        if not isinstance(overlay["source_map"], list) or not isinstance(
            overlay["external_mappings"], list
        ):
            raise ValueError("invalid rerun maps")
        for mapping in overlay["source_map"]:
            _keys(
                mapping, {"source_id", "original_path", "captured"}, "source_map entry"
            )
            _reference_shape(mapping["captured"], "captured FileRef")
            if mapping["captured"]["path_base"] != "record_directory":
                raise ValueError("captured FileRef needs record_directory base")
            if not all(
                isinstance(mapping[key], str) and mapping[key]
                for key in ("source_id", "original_path")
            ):
                raise ValueError("invalid source map identity")
        source_map_ids = [item["source_id"] for item in overlay["source_map"]]
        if source_map_ids != sorted(set(source_map_ids)):
            raise ValueError("source map ids are duplicate/unsorted")
        for mapping in overlay["external_mappings"]:
            _keys(
                mapping,
                {"id", "from", "to", "sha256", "size_bytes"},
                "external mapping",
            )
            _digest(mapping["sha256"], "external mapping sha256")
            if (
                not all(
                    isinstance(mapping[key], str) and mapping[key]
                    for key in ("id", "from", "to")
                )
                or type(mapping["size_bytes"]) is not int
                or mapping["size_bytes"] < 0
            ):
                raise ValueError("invalid external mapping identity")
        external_ids = [item["id"] for item in overlay["external_mappings"]]
        if external_ids != sorted(set(external_ids)):
            raise ValueError("external mapping ids are duplicate/unsorted")


def file_reference(path: Path, path_base: str, root: Path) -> dict[str, Any]:
    """Bind an existing file to a validated relative artifact-reference/1."""
    if path_base not in {
        "project_root",
        "experiment_root",
        "record_directory",
        "request_directory",
    }:
        raise ValueError(f"unknown reference base: {path_base}")
    root = Path(root).resolve()
    path = Path(path).resolve(strict=True)
    relative = path.relative_to(root).as_posix()
    _confined(root, relative)
    return {
        "schema_version": "artifact-reference/1",
        "path_base": path_base,
        "path": relative,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def resolve_file_reference(
    reference: Mapping[str, Any], anchors: Mapping[str, Path]
) -> Path:
    """Resolve and byte-check a FileRef against a caller supplied owning root."""
    if (
        set(reference)
        != {"schema_version", "path_base", "path", "sha256", "size_bytes"}
        or reference["schema_version"] != "artifact-reference/1"
    ):
        raise ValueError("unknown artifact reference")
    base = reference["path_base"]
    if (
        base
        not in {
            "project_root",
            "experiment_root",
            "record_directory",
            "request_directory",
        }
        or base not in anchors
    ):
        raise ValueError(f"unbound artifact reference base: {base}")
    path = _confined(Path(anchors[base]), reference["path"])
    if (
        path.stat().st_size != reference["size_bytes"]
        or file_sha256(path) != reference["sha256"]
    ):
        raise ValueError(f"artifact reference bytes differ: {reference['path']}")
    return path


def validate_archive(directory: Path) -> dict[str, Any]:
    """Read a complete run-record/2 archive and verify each captured byte hash."""
    record_path = directory / "run_record.yml"
    record = yaml.safe_load(record_path.read_bytes())
    _validate_run_record_shape(record)
    config_projection = record["configuration_projection"]
    if config_projection["digest_schema_version"] != 2:
        raise ValueError("unsupported effective configuration digest")
    effective_sha = effective_config_digest(
        record["loaded_config"], record["advanced_settings"], config_projection["paths"]
    )
    if config_projection["effective_config_sha256"] != effective_sha:
        raise ValueError("effective configuration digest differs")
    if config_projection["configuration_inputs_sha256"] != configuration_inputs_digest(
        effective_sha,
        record["toolbox"],
        record["environment"],
        record["referenced_inputs"],
    ):
        raise ValueError("configuration input digest differs")
    seen = set()
    for entry in record["source_files"]:
        source_id = entry["id"]
        if source_id in seen or entry["path_base"] != "record_directory":
            raise ValueError("invalid source entry")
        seen.add(source_id)
        path = _confined(directory, entry["archived_path"])
        data = path.read_bytes()
        if (
            len(data) != entry["size_bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"source bytes differ: {source_id}")
    expected = collection_sha256(
        {
            "schema_version": record["archive_schema_version"],
            "workflow": record["workflow"],
            "invocation_id": record["invocation_id"],
            "source_files": [
                {
                    key: entry[key]
                    for key in ("id", "role", "archived_path", "sha256", "size_bytes")
                }
                for entry in record["source_files"]
            ],
            "loaded_config": record["loaded_config"],
            "advanced_settings": record["advanced_settings"],
            "configuration_projection": record["configuration_projection"],
            "referenced_inputs": record["referenced_inputs"],
            "generated_inputs": record["generated_inputs"],
            "rerun": record["rerun"],
        }
    )
    if expected != record["archive_id"]:
        raise ValueError("archive id differs")
    for overlay in record["rerun"]["adjustments"]:
        if overlay.get("schema_version") != "resolver-overlay/1":
            raise ValueError("unknown rerun overlay")
        source_record = overlay["source_record"]
        if source_record["path_base"] != "record_directory":
            raise ValueError("rerun predecessor reference has wrong base")
        predecessor_path = _confined(directory, source_record["path"])
        if (
            predecessor_path.stat().st_size != source_record["size_bytes"]
            or file_sha256(predecessor_path) != source_record["sha256"]
        ):
            raise ValueError("rerun predecessor record differs")
        validate_archive(predecessor_path.parent)
        mapped_ids = {item["source_id"] for item in overlay["source_map"]}
        if mapped_ids != seen or len(mapped_ids) != len(overlay["source_map"]):
            raise ValueError("rerun source map does not cover captured sources")
        for source_map in overlay["source_map"]:
            entry = next(
                item
                for item in record["source_files"]
                if item["id"] == source_map["source_id"]
            )
            if source_map["original_path"] != entry["original_path"]:
                raise ValueError("rerun original source identity differs")
            ref = source_map["captured"]
            if ref["path_base"] != "record_directory":
                raise ValueError("rerun source reference has wrong base")
            path = _confined(directory, ref["path"])
            if (
                path.stat().st_size != ref["size_bytes"]
                or file_sha256(path) != ref["sha256"]
            ):
                raise ValueError("rerun mapped source differs")
    return record


def _matches_archive(directory: Path, expected_hash: str | None) -> bool:
    if expected_hash is None or not directory.is_dir():
        return False
    try:
        if file_sha256(directory / "run_record.yml") != expected_hash:
            return False
        validate_archive(directory)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return False
    return True


def _transaction_paths(project_root: Path, owner_key: str) -> tuple[Path, Path]:
    journal_dir = project_root / "config" / "_engine" / "archive-transactions"
    journal_dir.mkdir(parents=True, exist_ok=True)
    return journal_dir / f"{owner_key}.json", journal_dir / "completed"


def _checked_transaction_paths(
    project_root: Path, tx: Mapping[str, Any]
) -> tuple[Path, Path, Path]:
    return tuple(
        _confined(project_root, tx[key]) for key in ("target", "staged", "previous")
    )


def _verify_candidates(project_root: Path, tx: Mapping[str, Any]) -> None:
    """Refuse any present candidate whose bytes contradict its assigned slot."""
    target, staged, previous = _checked_transaction_paths(project_root, tx)
    old_sha = tx["old_record_sha256"]
    new_sha = tx["new_record_sha256"]
    if target.exists() and not (
        _matches_archive(target, old_sha) or _matches_archive(target, new_sha)
    ):
        raise ValueError("target archive contradicts transaction")
    if staged.exists() and not _matches_archive(staged, new_sha):
        raise ValueError("staged archive contradicts transaction")
    if previous.exists() and not _matches_archive(previous, old_sha):
        raise ValueError("previous archive contradicts transaction")


def _verify_terminal(project_root: Path, tx: Mapping[str, Any]) -> None:
    _verify_candidates(project_root, tx)
    target, _, _ = _checked_transaction_paths(project_root, tx)
    state = tx["state"]
    if state == "committed" and _matches_archive(target, tx["new_record_sha256"]):
        return
    if state == "rolled_back" and _matches_archive(target, tx["old_record_sha256"]):
        return
    if (
        state == "unpublished"
        and tx["old_record_sha256"] is None
        and not target.exists()
    ):
        return
    raise ValueError(f"archive terminal state is corrupt: {state}")


def _recover_locked(project_root: Path, owner_key: str) -> str | None:
    journal, _ = _transaction_paths(project_root, owner_key)
    if not journal.exists():
        return None
    tx = json.loads(journal.read_text(encoding="utf-8"))
    if tx.get("schema_version") != TRANSACTION_VERSION or tx.get("owner") != owner_key:
        raise ValueError("unknown archive transaction")
    target, staged, previous = _checked_transaction_paths(project_root, tx)
    _verify_candidates(project_root, tx)
    if tx["state"] in {"committed", "rolled_back", "unpublished"}:
        _verify_terminal(project_root, tx)
        return tx["state"]
    if _matches_archive(target, tx["new_record_sha256"]):
        tx["state"] = "committed"
    elif _matches_archive(target, tx["old_record_sha256"]):
        if not _matches_archive(staged, tx["new_record_sha256"]):
            raise ValueError("staged archive contradicts transaction")
        tx["state"] = "rolled_back"
    elif not target.exists() and _matches_archive(previous, tx["old_record_sha256"]):
        _replace(previous, target)
        if not _matches_archive(target, tx["old_record_sha256"]):
            raise ValueError("restored archive differs")
        tx["state"] = "rolled_back"
    elif (
        tx["old_record_sha256"] is None
        and not target.exists()
        and _matches_archive(staged, tx["new_record_sha256"])
    ):
        tx["state"] = "unpublished"
    else:
        raise ValueError("archive transaction candidates contradict recorded digests")
    _atomic_json(journal, tx)
    _verify_terminal(project_root, tx)
    return tx["state"]


def recover_archive(project_root: Path, owner_key: str) -> str | None:
    """Recover a publication under the lock; return its terminal state."""
    project_root = Path(project_root).resolve()
    with archive_lock(project_root, owner_key):
        return _recover_locked(project_root, owner_key)


def read_archive(
    project_root: Path, owner_key: str, target_relative: str
) -> dict[str, Any]:
    """Recover/refuse an unfinished writer, then read one matching generation."""
    project_root = Path(project_root).resolve()
    with archive_lock(project_root, owner_key):
        _recover_locked(project_root, owner_key)
        target = _confined(project_root, target_relative)
        return validate_archive(target)


def publish_archive(
    project_root: Path,
    owner_key: str,
    target_relative: str,
    record: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    *,
    failure_after: str | None = None,
) -> str:
    """Publish record and source bytes as one recoverable directory generation.

    ``failure_after`` is a deterministic fault point used by transaction tests.
    The caller must already own the broader project execution lock.
    """
    project_root = Path(project_root).resolve()
    target = _confined(project_root, target_relative)
    if target.name != "config" and not (
        target.parent == project_root / "config"
        and target.name in {"analyze_climate", "build_model", "analyze_projections"}
    ):
        raise ValueError(
            "archive target must be a workflow or artifact config directory"
        )
    journal, completed = _transaction_paths(project_root, owner_key)
    with archive_lock(project_root, owner_key):
        _recover_locked(project_root, owner_key)
        if journal.exists():
            tx_old = json.loads(journal.read_text(encoding="utf-8"))
            _verify_terminal(project_root, tx_old)
            completed.mkdir(parents=True, exist_ok=True)
            archived_journal = completed / f"{tx_old['transaction_id']}.json"
            if archived_journal.exists():
                raise ValueError("completed transaction id collision")
            _replace(journal, archived_journal)
        if target.exists():
            if not target.is_dir():
                raise ValueError("archive target is not a directory")
            if any(target.iterdir()):
                validate_archive(target)
                old_sha = file_sha256(target / "run_record.yml")
            else:
                target.rmdir()
                old_sha = None
        else:
            old_sha = None
        txid = uuid.uuid4().hex
        staged = target.with_name(f".{target.name}.{txid}.staged")
        previous = target.with_name(f".{target.name}.{txid}.previous")
        if staged.exists() or previous.exists():
            raise ValueError("archive transaction path already exists")
        staged.mkdir(parents=True)
        try:
            for relative, data in payloads.items():
                _write_durable(_confined(staged, relative), data)
            record_data = _record_bytes(record)
            _write_durable(staged / "run_record.yml", record_data)
            validate_archive(staged)
            new_sha = hashlib.sha256(record_data).hexdigest()
            tx = {
                "schema_version": TRANSACTION_VERSION,
                "transaction_id": txid,
                "owner": owner_key,
                "invocation_id": record["invocation_id"],
                "target": target.relative_to(project_root).as_posix(),
                "staged": staged.relative_to(project_root).as_posix(),
                "previous": previous.relative_to(project_root).as_posix(),
                "old_record_sha256": old_sha,
                "new_record_sha256": new_sha,
                "archive_id": record["archive_id"],
                "state": "prepared",
            }
            _atomic_json(journal, tx)
            if failure_after == "prepared":
                raise RuntimeError("injected failure after prepared")
            if old_sha is not None:
                _replace(target, previous)
                tx["state"] = "old_detached"
                _atomic_json(journal, tx)
            if failure_after == "old_detached":
                raise RuntimeError("injected failure after old_detached")
            _replace(staged, target)
            tx["state"] = "new_installed"
            _atomic_json(journal, tx)
            if failure_after == "new_installed":
                raise RuntimeError("injected failure after new_installed")
            if not _matches_archive(target, new_sha):
                raise ValueError("installed archive differs from staged record")
            tx["state"] = "committed"
            _atomic_json(journal, tx)
            return new_sha
        except Exception:
            # Leave journal and candidates for locked, evidence-based recovery.
            if not journal.exists():
                shutil.rmtree(staged)
            raise


def capture_rerun_predecessor(
    predecessor_directory: Path,
    *,
    predecessor_project_root: Path,
    new_project_root: Path,
    output_project_dir: str,
    external_mappings: Sequence[Mapping[str, Any]] = (),
    overrides: Mapping[str, Any] | None = None,
) -> tuple[tuple[CapturedSource, ...], dict[str, bytes], dict[str, Any]]:
    """Copy verified predecessor evidence into a new archive without old inputs.

    Return new source captures, immutable predecessor payloads and the closed
    resolver overlay. The caller supplies explicit external data remappings;
    this function never searches by basename or rereads original source paths.
    """
    predecessor_directory = Path(predecessor_directory).resolve()
    old = validate_archive(predecessor_directory)
    archive_id = old["archive_id"]
    prefix = f"predecessor/{archive_id}"
    preserved: dict[str, bytes] = {}
    for path in predecessor_directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"predecessor evidence contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(predecessor_directory).as_posix()
            preserved[f"{prefix}/{relative}"] = path.read_bytes()
    record_bytes = preserved[f"{prefix}/run_record.yml"]
    source_record = {
        "schema_version": "artifact-reference/1",
        "path_base": "record_directory",
        "path": f"{prefix}/run_record.yml",
        "sha256": hashlib.sha256(record_bytes).hexdigest(),
        "size_bytes": len(record_bytes),
    }
    captured = []
    for entry in old["source_files"]:
        data = preserved[f"{prefix}/{entry['archived_path']}"]
        captured.append(
            CapturedSource(
                entry["id"],
                entry["role"],
                Path(entry["original_path"]),
                data,
                entry["git_blob"],
            )
        )
    new_paths = source_archive_paths(captured)
    source_map = []
    for entry in sorted(old["source_files"], key=lambda item: item["id"]):
        data = next(source.data for source in captured if source.id == entry["id"])
        source_map.append(
            {
                "source_id": entry["id"],
                "original_path": entry["original_path"],
                "captured": {
                    "schema_version": "artifact-reference/1",
                    "path_base": "record_directory",
                    "path": new_paths[entry["id"]],
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                },
            }
        )
    mapped = sorted(
        (dict(item) for item in external_mappings), key=lambda item: item["id"]
    )
    if len({item["id"] for item in mapped}) != len(mapped):
        raise ValueError("duplicate external mapping id")
    required_external = {
        item["id"]: item
        for item in old["referenced_inputs"]
        if item["source_id"] is None
    }
    if {item["id"] for item in mapped} != set(required_external):
        raise ValueError("rerun requires an explicit mapping for every external input")
    for item in mapped:
        if set(item) != {"id", "from", "to", "sha256", "size_bytes"}:
            raise ValueError("external mapping has unclassified fields")
        prior = required_external[item["id"]]
        if (item["from"], item["sha256"], item["size_bytes"]) != (
            prior["original_path"],
            prior["sha256"],
            prior["size_bytes"],
        ):
            raise ValueError("external mapping differs from recorded input")
        target = Path(item["to"])
        if (
            not target.is_file()
            or target.stat().st_size != item["size_bytes"]
            or file_sha256(target) != item["sha256"]
        ):
            raise ValueError(f"external rerun input bytes differ: {item['id']}")
    overlay = {
        "schema_version": "resolver-overlay/1",
        "source_record": source_record,
        "source_map": source_map,
        "project_root_mapping": {
            "from": str(Path(predecessor_project_root).resolve()),
            "to": str(Path(new_project_root).resolve()),
        },
        "external_mappings": mapped,
        "overrides": dict(
            old["invocation"]["overrides"] if overrides is None else overrides
        ),
        "output_project_dir": output_project_dir,
    }
    return tuple(captured), preserved, overlay


def resolve_rerun_execution(
    project_root: Path,
    owner_key: str,
    target_relative: str,
    *,
    declared_sections: Sequence[str] = (),
    external_relocations: Mapping[str, str | Path] | None = None,
) -> RerunResolution:
    """Resolve a relocated rerun archive into captured composition and paths.

    Original A paths need not exist. The relocated project root is the caller's
    validated C; external inputs are byte checked at the locator supplied by
    the overlay or by an explicit C-time relocation for that input ID.
    """
    from blueearth_cst.shared.config_composition import compose_captured_config

    root = Path(project_root).resolve()
    record = read_archive(root, owner_key, target_relative)
    adjustments = record["rerun"]["adjustments"]
    if len(adjustments) != 1:
        raise ValueError("archive has no rerun resolver overlay")
    overlay = adjustments[0]
    archive_directory = _confined(root, target_relative)
    source_paths = {
        item["original_path"]: resolve_file_reference(
            item["captured"], {"record_directory": archive_directory}
        )
        for item in overlay["source_map"]
    }
    captured = tuple(
        CapturedSource(
            entry["id"],
            entry["role"],
            Path(entry["original_path"]),
            source_paths[entry["original_path"]].read_bytes(),
            entry["git_blob"],
        )
        for entry in record["source_files"]
    )
    project_source = next(
        source for source in captured if source.id == record["source_config"]
    )
    composed, _ = compose_captured_config(
        project_source,
        captured,
        record["workflow"],
        declared_sections,
        overlay["overrides"],
    )
    project_section = composed.get("project")
    if isinstance(project_section, dict):
        project_section["project_dir"] = str(root)
    workflow_paths = {
        entry["role"].removeprefix("workflow_config_"): str(
            source_paths[entry["original_path"]]
        )
        for entry in record["source_files"]
        if entry["role"].startswith("workflow_config_")
    }
    requested_external = dict(external_relocations or {})
    expected_ids = {item["id"] for item in overlay["external_mappings"]}
    if set(requested_external) - expected_ids:
        raise ValueError("unknown external relocation id")
    external_paths = {}
    for item in overlay["external_mappings"]:
        path = Path(requested_external.get(item["id"], item["to"]))
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise ValueError(f"external rerun input bytes differ: {item['id']}")
        external_paths[item["from"]] = path.resolve()
    return RerunResolution(
        composed,
        workflow_paths,
        captured,
        archive_directory,
        root,
        Path(overlay["project_root_mapping"]["from"]),
        source_paths,
        external_paths,
    )
