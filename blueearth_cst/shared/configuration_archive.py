"""Keep WF0--WF2 configuration generations separate from launch attempts."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.workflow_config_snapshot import (
    _replace,
    _write_durable,
    archive_lock,
    read_archive,
    validate_archive,
)

WORKFLOWS = ("analyze_climate", "build_model", "analyze_projections")

CONFIG_README = """\
# Configuration records

Edit the source configuration supplied to the runner to change a run; generated
files here are records, not editable run inputs.

`analyze_climate/`, `build_model/` and `analyze_projections/` contain the latest
exact source captures and `run_record.yml`. `SOURCES.md` lists original source paths.
Unchanged settings reuse their snapshot.
Immutable historical snapshots, stable execution inputs, locks and attempt history
live under `_engine/`. Attempt commands and outcomes are presented in the timestamped
run summaries under `logs/`.

WF3 configuration belongs to its scenario collection. WF4 configuration belongs to
`experiments/<name>/config/`. Project-wide experiment numbering and recovery live in
`experiments/_engine/`.
"""


def same_configuration(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare captured provenance, excluding facts belonging to one attempt."""
    attempt_fields = {"archive_id", "invocation_id", "invocation"}
    return {key: value for key, value in left.items() if key not in attempt_fields} == {
        key: value for key, value in right.items() if key not in attempt_fields
    }


def snapshot_root(project: Path, workflow: str) -> Path:
    return project / "config/_engine/configuration-snapshots" / workflow


def _inside(project: Path, path: Path) -> Path:
    resolved = path.resolve()
    if resolved == project.resolve() or not resolved.is_relative_to(project.resolve()):
        raise ValueError("configuration maintenance path escapes project")
    if path.is_symlink() or path.is_junction():
        raise ValueError("configuration maintenance refuses linked paths")
    return resolved


def retain_archive(project: Path, directory: Path) -> Path:
    """Preserve the exact record and captured sources at an immutable address.

    The caller owns the project execution lock. Existing run-record/2 bytes and
    identities remain unchanged, including creator provenance in older projects.
    """
    record = validate_archive(directory)
    checksum = file_sha256(directory / "run_record.yml")
    destination = _inside(
        project, snapshot_root(project, record["workflow"]) / checksum
    )
    if destination.exists():
        if (destination / "run_record.yml").read_bytes() != (
            directory / "run_record.yml"
        ).read_bytes():
            raise ValueError("immutable configuration snapshot differs")
        validate_archive(destination)
        return destination / "run_record.yml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{checksum}.{uuid.uuid4().hex}.staged")
    try:
        # Copy only declared configuration bytes, not operational leftovers.
        staging.mkdir()
        for name in _archive_files(directory, record):
            source = _inside(project, directory / name)
            target = _inside(project, staging / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_durable(target, source.read_bytes())
        validate_archive(staging)
        _replace(staging, destination)
    finally:
        if staging.exists():
            _remove_directory(project, staging)
    return destination / "run_record.yml"


def matching_archive(project: Path, record: Mapping[str, Any]) -> Path | None:
    """Find a previously retained configuration when settings return to it."""
    root = snapshot_root(project, record["workflow"])
    if root.exists():
        for directory in sorted(root.iterdir()):
            if directory.name.startswith(".") or not directory.is_dir():
                continue
            previous = validate_archive(directory)
            if same_configuration(previous, record):
                return directory
    return None


def archive_payloads(directory: Path, record: Mapping[str, Any]) -> dict[str, bytes]:
    return {
        name: (directory / name).read_bytes()
        for name in _archive_files(directory, record)
        if name != "run_record.yml"
    }


def _archive_files(directory: Path, record: Mapping[str, Any]) -> list[str]:
    """Include the checked source closure of rerun predecessors as well."""
    result = [
        "run_record.yml",
        *(item["archived_path"] for item in record["source_files"]),
    ]
    if (directory / "SOURCES.md").exists():
        result.append("SOURCES.md")
    for adjustment in record["rerun"]["adjustments"]:
        predecessor = directory / adjustment["source_record"]["path"]
        previous = validate_archive(predecessor.parent)
        for name in _archive_files(predecessor.parent, previous):
            result.append((predecessor.parent / name).relative_to(directory).as_posix())
    return result


def _remove_directory(project: Path, directory: Path) -> None:
    directory = _inside(project, directory)
    if any(path.is_symlink() or path.is_junction() for path in directory.rglob("*")):
        raise ValueError("configuration maintenance refuses linked descendants")
    shutil.rmtree(directory)


def finish_archive_transaction(project: Path, workflow: str) -> None:
    """Retire committed replacement machinery after retaining both generations."""
    root = project / "config/_engine/archive-transactions"
    journal = root / f"{workflow}.json"
    if not journal.exists():
        return
    transaction = json.loads(journal.read_text(encoding="utf-8"))
    if transaction["state"] != "committed" or transaction["owner"] != workflow:
        raise ValueError("configuration transaction has not committed")
    target = _inside(project, project / transaction["target"])
    expected = (project / "config" / workflow).resolve()
    previous = _inside(project, project / transaction["previous"])
    if target != expected or previous != expected.with_name(
        f".{workflow}.{transaction['transaction_id']}.previous"
    ):
        raise ValueError("configuration transaction names unexpected directories")
    if file_sha256(target / "run_record.yml") != transaction["new_record_sha256"]:
        raise ValueError("committed configuration differs from transaction")
    retain_archive(project, target)
    if previous.exists():
        if file_sha256(previous / "run_record.yml") != transaction["old_record_sha256"]:
            raise ValueError("previous configuration differs from transaction")
        retain_archive(project, previous)
        _remove_directory(project, previous)
    journal.unlink()


def remove_launch_context(project: Path, path: Path) -> None:
    """Remove a completed child's private launch context, preserving inputs."""
    path = _inside(project, path)
    base = (project / "config/_engine/execution-configs").resolve()
    if not path.is_relative_to(base) or not path.name.startswith("launch-"):
        raise ValueError("not a configuration launch context")
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        # Cleanup must not replace the child's observed failure or exit code.
        print(f"(launch-context cleanup deferred: {error})", file=sys.stderr)


def remove_stale_launch_contexts(project: Path, workflow: str) -> None:
    """Collect prior launch contexts while holding the project execution lock."""
    root = project / "config/_engine/execution-configs" / workflow
    if root.exists():
        for path in root.glob("*/launch-*.json"):
            context = json.loads(path.read_text(encoding="utf-8"))
            if (
                context["workflow"] != workflow
                or Path(context["project_root"]).resolve() != project.resolve()
            ):
                raise ValueError("launch context belongs to another workflow/project")
            remove_launch_context(project, path)


def write_config_readme(project: Path) -> Path:
    """Explain the archive layout without duplicating captured settings."""
    readme = project / "config/README.md"
    if not readme.exists() or readme.read_text(encoding="utf-8") != CONFIG_README:
        readme.write_text(CONFIG_README, encoding="utf-8")
    return readme


def maintenance_report(project: Path) -> dict[str, Any]:
    """Inspect existing configuration state without changing it."""
    project = project.resolve(strict=True)
    config = project / "config"
    workflows = [
        name for name in WORKFLOWS if (config / name / "run_record.yml").is_file()
    ]
    contexts = list((config / "_engine/execution-configs").glob("*/*/launch-*.json"))
    missing = 0
    for path in (config / "_engine/invocations").glob("*.json"):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        reference = attempt.get("configuration", {}).get("run_record")
        if reference is None:
            continue
        target = _inside(project, project / reference["path"])
        if not target.is_file() or file_sha256(target) != reference["sha256"]:
            missing += 1
    return {
        "project": str(project),
        "workflows": workflows,
        "snapshots": len(
            list(
                (config / "_engine/configuration-snapshots").glob("*/*/run_record.yml")
            )
        ),
        "launch_contexts": len(contexts),
        "completed_transactions": len(
            list((config / "_engine/archive-transactions/completed").glob("*.json"))
        ),
        "unavailable_history_references": missing,
    }


def maintain_archives(project: Path) -> dict[str, Any]:
    """Migrate retained WF0--WF2 archives and retire completed launch machinery.

    Source files, execution YAML, attempt history and scientific outputs are
    preserved. Missing historical record bytes are reported, never invented.
    """
    project = project.resolve(strict=True)
    if not any(
        (project / "config" / workflow / "run_record.yml").is_file()
        for workflow in WORKFLOWS
    ):
        raise ValueError("project has no captured WF0--WF2 configuration archives")
    with archive_lock(project, "project-execution"):
        for workflow in WORKFLOWS:
            directory = project / "config" / workflow
            if not directory.exists():
                continue
            read_archive(project, workflow, f"config/{workflow}")
            retain_archive(project, directory)
            finish_archive_transaction(project, workflow)
            remove_stale_launch_contexts(project, workflow)
        # Completed journals are recovery bookkeeping, not configuration history.
        # Preserve any surviving displaced archive before retiring its journal.
        completed = project / "config/_engine/archive-transactions/completed"
        for journal in completed.glob("*.json"):
            transaction = json.loads(journal.read_text(encoding="utf-8"))
            workflow = transaction.get("owner")
            if workflow not in WORKFLOWS or transaction.get("state") != "committed":
                continue
            if journal.stem != transaction["transaction_id"]:
                raise ValueError("completed transaction filename differs")
            previous = _inside(project, project / transaction["previous"])
            expected = (
                project
                / "config"
                / f".{workflow}.{transaction['transaction_id']}.previous"
            )
            if previous != expected.resolve():
                raise ValueError("completed transaction names unexpected directory")
            if previous.exists():
                if (
                    file_sha256(previous / "run_record.yml")
                    != transaction["old_record_sha256"]
                ):
                    raise ValueError(
                        "previous configuration differs from completed transaction"
                    )
                retain_archive(project, previous)
                _remove_directory(project, previous)
            _inside(project, journal).unlink()
        write_config_readme(project)
        return maintenance_report(project)
