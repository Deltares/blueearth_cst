"""Pre-parse capture and archive admission for WF0--WF2 launches.

The source buffers are captured before Snakemake sees a configfile. External
dependency files (data catalogs, build/waterbodies configs, observations) are
staged content-addressed and shared across workflows, so an unchanged
dependency resolves to the same execution path from WF0, WF1 or WF2 alike --
this is what lets the shared-foundation rules (delineate_region and friends)
skip on a rerun instead of rebuilding every time a different workflow last
touched them. Project/workflow execution documents use canonical filenames
and are keyed by their parsed settings and selected dependency contents, not
source locations or another workflow's scheduling choices. Original paths,
bytes, and enable flags remain in run-record/2.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from blueearth_cst.shared import execution_outputs, invocation_history, product_record
from blueearth_cst.shared.config_composition import (
    capture_configuration_sources,
    compose_captured_config,
    compose_config,
)
from blueearth_cst.shared.config_paths import TOOLBOX_ROOT
from blueearth_cst.shared.configuration_archive import (
    archive_payloads,
    finish_archive_transaction,
    matching_archive,
    remove_launch_context,
    remove_stale_launch_contexts,
    retain_archive,
    same_configuration,
    write_config_readme,
)
from blueearth_cst.shared.provenance import (
    environment_file_hashes,
    short_digest,
    toolbox_identity,
)
from blueearth_cst.shared.wf3_science import ADVANCED_SETTINGS
from blueearth_cst.shared.workflow_config_snapshot import (
    CapturedSource,
    archive_lock,
    capture_sources,
    file_reference,
    publish_archive,
    read_archive,
    resolve_file_reference,
    run_record_document,
)

WORKFLOWS = ("analyze_climate", "build_model", "analyze_projections")
PROJECTION = ("project", "basin", "climate", "model")
CONTEXT_ENV = "BLUEEARTH_WF012_CAPTURE_CONTEXT"
# Bump whenever staging logic changes generated execution YAML while the
# captured sources can remain byte-identical. Including this in the bundle
# digest retains older immutable execution views instead of overwriting them.
EXECUTION_CONFIG_SCHEMA_VERSION = "4"


def split_config_overrides(extra: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    """Consume Snakemake top-level config assignments before source capture."""
    overrides: dict[str, Any] = {}
    forwarded: list[str] = []
    index = 0
    while index < len(extra):
        token = extra[index]
        if token in {"--configfile", "-s", "--snakefile"} or token.startswith(
            ("--configfile=", "--snakefile=")
        ):
            raise ValueError(f"execution option bypasses captured config: {token}")
        if token == "--config":
            index += 1
            if index == len(extra) or extra[index].startswith("-"):
                raise ValueError("--config requires a key=value assignment")
            while index < len(extra) and not extra[index].startswith("-"):
                assignment = extra[index]
                if "=" not in assignment:
                    raise ValueError(f"invalid --config assignment: {assignment}")
                key, value = assignment.split("=", 1)
                if not key or key in overrides:
                    raise ValueError(f"duplicate or empty --config key: {key}")
                overrides[key] = yaml.safe_load(value)
                index += 1
            continue
        if token.startswith("--config="):
            assignment = token.removeprefix("--config=")
            if "=" not in assignment:
                raise ValueError(f"invalid --config assignment: {assignment}")
            key, value = assignment.split("=", 1)
            if not key or key in overrides:
                raise ValueError(f"duplicate or empty --config key: {key}")
            overrides[key] = yaml.safe_load(value)
        else:
            forwarded.append(token)
        index += 1
    return overrides, forwarded


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable execution input changed: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def stage_shared_dependency(
    project_root: Path, dependency_id: str, source: Path
) -> Path:
    """Stage one immutable dependency at its cross-workflow canonical path."""
    source = Path(source).resolve(strict=True)
    data = source.read_bytes()
    digest = short_digest(hashlib.sha256(data).hexdigest())
    suffix = source.suffix.lower()
    if suffix == ".yaml":
        suffix = ".yml"
    destination = (
        Path(project_root).resolve()
        / "config"
        / "_engine"
        / "execution-configs"
        / "_shared"
        / dependency_id
        / digest
        / f"{dependency_id}{suffix}"
    )
    _write_once(destination, data)
    return destination


def _file_specifications(
    workflow: str, composed: Mapping[str, Any]
) -> list[tuple[str, str, Path]]:
    """Collect local config dependencies actually selected by this workflow."""
    project = composed["project"]
    settings = composed["workflows"][workflow]
    defaults = TOOLBOX_ROOT if composed.get("schema_version") == 3 else Path(".")
    values: list[tuple[str, str, Any]] = []
    catalogs = project["catalog"]
    for index, catalog in enumerate(
        catalogs if isinstance(catalogs, list) else [catalogs]
    ):
        values.append((f"project_catalog_{index}", "data_catalog", catalog))
    if workflow == "analyze_projections":
        values.append(("projection_catalog", "projection_catalog", settings["catalog"]))
        values.append(
            (
                "projection_index",
                "projection_store_index",
                str(Path(settings["catalog"]).parent / "cmip6_store_index.json"),
            )
        )
    # The vector-foundation rule is byte-identical across WF0-WF2 and consumes
    # this basin-owned gauge file in every entry point. Staging it only for WF1
    # makes the same rule alternate between the original absolute path and a
    # staged path, so Snakemake reports a changed input set on every workflow
    # switch and rebuilds the shared foundation.
    locations = composed.get("basin", {}).get("output_locations")
    if isinstance(locations, str):
        values.append(("output_locations", "output_locations", locations))
    if workflow == "build_model":
        engine = settings.get("engine") or {}
        values.extend(
            [
                (
                    "build_config",
                    "build_config",
                    engine.get(
                        "build_config",
                        str(defaults / "config/defaults/wflow_build_model.yml"),
                    ),
                ),
                (
                    "waterbodies_config",
                    "waterbodies_config",
                    engine.get(
                        "waterbodies_config",
                        str(defaults / "config/defaults/wflow_update_waterbodies.yml"),
                    ),
                ),
            ]
        )
        for variable, locator in sorted((settings.get("observations") or {}).items()):
            if isinstance(locator, str):
                values.append(
                    (
                        f"observation_{len(values)}",
                        f"observation_{variable}",
                        locator,
                    )
                )
    result = []
    for source_id, role, locator in values:
        if not isinstance(locator, str):
            raise ValueError(f"{source_id} must be a file locator")
        path = Path(locator)
        if path.is_file():
            result.append((source_id, role, path.resolve()))
        elif path.is_absolute() or path.suffix in {".yml", ".yaml"}:
            raise ValueError(f"configured source is missing: {path}")
        # A registered HydroMT catalog name is not a local source file.
    return result


def _check_relocatable_yaml(source: CapturedSource) -> None:
    if "catalog" not in source.role:
        return
    value = yaml.safe_load(source.data)
    roots = value.get("meta", {}).get("roots", []) if isinstance(value, Mapping) else []
    if roots and all(
        isinstance(root, str)
        and (
            Path(root).is_absolute()
            or root.startswith("/")
            or "://" in root
            or (len(root) > 2 and root[1:3] in {":/", ":\\"})
        )
        for root in roots
    ):
        return

    def inspect(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if key == "uri" and isinstance(child, str):
                    if "://" not in child and not Path(child).is_absolute():
                        raise ValueError(
                            f"catalog has relative URI requiring an anchored execution view: "
                            f"{source.original_path}: {child}"
                        )
                inspect(child)
        elif isinstance(item, list):
            for child in item:
                inspect(child)

    inspect(value)


def _replace_paths(item: Any, paths: Mapping[Path, Path]) -> Any:
    if isinstance(item, dict):
        return {key: _replace_paths(value, paths) for key, value in item.items()}
    if isinstance(item, list):
        return [_replace_paths(value, paths) for value in item]
    if isinstance(item, str):
        resolved = Path(item).resolve()
        if resolved in paths:
            return str(paths[resolved])
    return item


def prepare_workflow(
    workflow: str,
    project_config: Path,
    project_root: Path,
    *,
    command: Sequence[str],
    targets: Sequence[str],
    overrides: Mapping[str, Any] | None = None,
    entry_point: str = "scripts/run_workflow.py",
    invocation_id: str | None = None,
) -> tuple[Path, Path]:
    """Capture, stage, and publish one exact WF0--WF2 archive.

    Returns the generated execution project config and launch context path.
    P3 can call this with its child command/id without changing capture logic.
    """
    if workflow not in WORKFLOWS:
        raise ValueError(f"unsupported capture workflow: {workflow}")
    project_config = Path(project_config).resolve(strict=True)
    project_root = Path(project_root).resolve()
    existing = project_root / "config" / workflow
    previous = None
    if existing.exists():
        try:
            previous = read_archive(project_root, workflow, f"config/{workflow}")
            retain_archive(project_root, existing)
            finish_archive_transaction(project_root, workflow)
        except (OSError, ValueError) as error:
            raise ValueError(
                f"existing {workflow} output archive predates run-record/2 or is corrupt; "
                "use a fresh project root"
            ) from error
    projection = (*PROJECTION, f"workflows.{workflow}")
    initial = capture_configuration_sources(project_config, workflow, projection)
    composed, _ = compose_captured_config(
        initial[0], initial, workflow, projection, overrides
    )
    declared_root = Path(composed["project"]["project_dir"]).resolve()
    if declared_root != project_root:
        raise ValueError(f"project root differs from captured config: {declared_root}")
    custom = capture_sources(_file_specifications(workflow, composed))
    for source in custom:
        _check_relocatable_yaml(source)
    sources = (*initial, *custom)
    # Content-addressed by the FILE's own bytes, in a bucket shared across
    # workflows -- not nested under this workflow's own `stage` (bundle digest).
    # WF0/WF1/WF2 declare byte-identical shared-foundation rules (delineate_region,
    # delineate_subbasins_and_rivers, extract_climate_datasets) that consume a
    # dependency such as the project data catalog. Staging it under `stage` kept
    # a separate copy per workflow, at a path that changed whenever ANY other
    # part of that workflow's bundle changed -- so switching between workflows
    # made Snakemake see a changed input set on every shared rule and rebuild
    # the whole foundation on every run, even with nothing to redo. Keying on
    # the dependency's own hash instead means an unchanged file resolves to the
    # same execution path no matter which workflow staged it first.
    dependency_root = (
        project_root / "config" / "_engine" / "execution-configs" / "_shared"
    )
    replacements: dict[Path, Path] = {}
    generated_paths = []
    # `projection_catalog` and `projection_index` (cmip6_store_index.json) are a
    # crawl PAIR the Snakefile finds by guessing "beside the catalog" rather than
    # reading a resolved path for each: `STORE_INDEX = Path(DATA_SOURCES).parent /
    # "cmip6_store_index.json"` (analyze_projections.smk), because "the store
    # index sits beside the catalog that generated it" is a D12 invariant, not a
    # coincidence. Content-addressing each file by its OWN bytes broke that guess
    # the moment their bytes differ: two different hashes put them in two
    # different subfolders under the shared "projection_catalog" bucket, so the
    # sibling guess never resolves and `load_pins` silently returns `{}` even
    # though the index file exists and genuinely has pins for the entry -- it is
    # simply one folder over. Hash the PAIR together instead, so they always
    # land in the same subfolder; either file changing still moves the shared
    # hash, so this stays content-addressed.
    pair_ids = {"projection_catalog", "projection_index"}
    pair_sources = [source for source in custom if source.id in pair_ids]
    pair_hash = None
    if len(pair_sources) == len(pair_ids):
        combined = b"".join(
            hashlib.sha256(source.data).digest()
            for source in sorted(pair_sources, key=lambda item: item.id)
        )
        pair_hash = short_digest(hashlib.sha256(combined).hexdigest())
    for source in custom:
        dependency_dir = (
            "projection_catalog" if source.id == "projection_index" else source.id
        )
        content_hash = (
            pair_hash
            if source.id in pair_ids and pair_hash is not None
            else short_digest(hashlib.sha256(source.data).hexdigest())
        )
        if pair_hash is None or source.id not in pair_ids:
            destination = stage_shared_dependency(
                project_root, dependency_dir, source.original_path
            )
        else:
            destination = (
                dependency_root
                / dependency_dir
                / content_hash
                / (
                    "cmip6_store_index.json"
                    if source.id == "projection_index"
                    else "projection_catalog.yml"
                )
            )
            _write_once(destination, source.data)
        generated_paths.append((f"execution_{source.id}", destination))
        replacements[source.original_path] = destination
    workflow_source = next(
        (source for source in initial if source.role == f"workflow_config_{workflow}"),
        None,
    )
    workflow_document = (
        _replace_paths(yaml.safe_load(workflow_source.data), replacements)
        if workflow_source is not None
        else None
    )
    project_document = _replace_paths(yaml.safe_load(initial[0].data), replacements)
    if overrides:
        project_document.update(_replace_paths(dict(overrides), replacements))
    if composed.get("schema_version") == 3:
        # Resolve against the original declarations BEFORE relocating execution
        # YAML. CapturedSource still holds the exact, portable source bytes.
        project_document = _replace_paths(
            {key: value for key, value in composed.items() if key != "workflows"},
            replacements,
        )
        project_document["workflows"] = {
            workflow: {"enabled": composed["workflows"][workflow]["enabled"]}
        }
        if workflow_source is not None:
            workflow_document = _replace_paths(
                {
                    key: value
                    for key, value in composed["workflows"][workflow].items()
                    if key != "enabled"
                },
                replacements,
            )
    # Scheduling choices and source locators remain in the exact archive. Only
    # this workflow's executable settings belong in its immutable config view.
    project_document["workflows"] = {
        workflow: {
            key: value
            for key, value in project_document["workflows"][workflow].items()
            if key != "config_path"
        }
    }
    material = {
        "execution_config_schema": EXECUTION_CONFIG_SCHEMA_VERSION,
        "project": project_document,
        "workflow": workflow_document,
        "dependencies": [
            (source.id, hashlib.sha256(source.data).hexdigest()) for source in custom
        ],
    }
    digest = short_digest(
        hashlib.sha256(
            yaml.safe_dump(material, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )
    stage = (
        project_root / "config" / "_engine" / "execution-configs" / workflow / digest
    )
    if workflow_source is not None:
        execution_workflow = stage / "workflow.yml"
        _write_once(
            execution_workflow,
            yaml.safe_dump(workflow_document, sort_keys=True).encode("utf-8"),
        )
        generated_paths.append(("execution_workflow_config", execution_workflow))
    else:
        execution_workflow = None
    if execution_workflow is not None:
        project_document["workflows"][workflow]["config_path"] = str(execution_workflow)
    execution_project = stage / "project.yml"
    _write_once(
        execution_project,
        yaml.safe_dump(project_document, sort_keys=True).encode("utf-8"),
    )
    generated_paths.append(("execution_project_config", execution_project))
    loaded, _ = compose_config(
        project_document, execution_project, workflow, projection
    )
    references = [
        {
            "id": source.id,
            "role": source.role,
            "original_path": str(source.original_path),
            "source_id": source.id,
            "sha256": hashlib.sha256(source.data).hexdigest(),
            "size_bytes": len(source.data),
        }
        for source in custom
    ]
    invocation_id = invocation_id or uuid.uuid4().hex
    record, payloads = run_record_document(
        workflow=workflow,
        invocation_id=invocation_id,
        owner_kind="workflow",
        owner_id=workflow,
        loaded_config=loaded,
        advanced_settings=ADVANCED_SETTINGS,
        projection=projection,
        toolbox=toolbox_identity(),
        environment=environment_file_hashes(),
        invocation={
            "entry_point": entry_point,
            "command": list(command),
            "targets": list(targets),
            "working_directory": str(Path.cwd()),
            "overrides": dict(overrides or {}),
        },
        sources=sources,
        referenced_inputs=references,
        generated_inputs=[
            {"role": role, "file": file_reference(path, "project_root", project_root)}
            for role, path in generated_paths
        ],
    )
    relative = f"config/{workflow}"
    if previous is not None and same_configuration(previous, record):
        record = previous
    else:
        prior = matching_archive(project_root, record)
        if prior is not None:
            from blueearth_cst.shared.workflow_config_snapshot import validate_archive

            record = validate_archive(prior)
            payloads = archive_payloads(prior, record)
        publish_archive(project_root, workflow, relative, record, payloads)
        finish_archive_transaction(project_root, workflow)
    write_config_readme(project_root)
    remove_stale_launch_contexts(project_root, workflow)
    context = stage / f"launch-{invocation_id}.json"
    _write_once(
        context,
        (
            json.dumps(
                {
                    "workflow": workflow,
                    "execution_config": str(execution_project),
                    "archive_id": record["archive_id"],
                    "project_root": str(project_root),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return execution_project, context


#: Snakemake options under which no rule body runs and nothing scientific is
#: written: planning (`--dry-run`), bookkeeping (`--unlock`, `--touch`, output deletion) and
#: reporting (listings, summaries, graphs, lint).
_NON_PRODUCING_OPTIONS = frozenset(
    {
        "--dry-run",
        "--dryrun",
        "--unlock",
        "-n",
        "--touch",
        "-t",
        "--delete-all-output",
        "--delete-temp-output",
        "--list",
        "--list-rules",
        "-l",
        "--list-target-rules",
        "--lt",
        "--summary",
        "-S",
        "--detailed-summary",
        "-D",
        "--dag",
        "--rulegraph",
        "--filegraph",
        "--lint",
    }
)


def non_producing_invocation(argv: Sequence[str]) -> bool:
    """Whether a Snakemake command line runs no rule body and writes no science."""
    return any(item in _NON_PRODUCING_OPTIONS for item in argv)


def require_capture(
    workflow: str, config_path: str, *, dry_run: bool | None = None
) -> dict[str, Any] | None:
    """Refuse raw production before any rule or scientific write is admitted.

    ``dry_run`` defaults to reading the process's own command line: any
    non-producing Snakemake invocation (``non_producing_invocation``) is
    admitted without the launcher's capture context.
    """
    if dry_run is None:
        dry_run = non_producing_invocation(sys.argv)
    context_path = os.environ.get(CONTEXT_ENV)
    if context_path is None:
        if dry_run:
            return None
        raise ValueError(
            f"{workflow} requires scripts/run_workflow.py for pre-parse source capture"
        )
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    if (
        context["workflow"] != workflow
        or Path(context["execution_config"]).resolve() != Path(config_path).resolve()
    ):
        raise ValueError("workflow capture context does not match configfile")
    record = read_archive(Path(context["project_root"]), workflow, f"config/{workflow}")
    if record["archive_id"] != context["archive_id"]:
        raise ValueError("workflow capture archive differs from launch context")
    for item in record["generated_inputs"]:
        resolve_file_reference(
            item["file"], {"project_root": Path(context["project_root"])}
        )
    return record


def captured_projection(
    record: Mapping[str, Any] | None,
    composed: Mapping[str, Any],
    advanced_settings: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Bind Snakefile rule triggers to the exact published run record."""
    if record is None:
        return None
    if (
        record["loaded_config"] != composed
        or record["advanced_settings"] != advanced_settings
    ):
        raise ValueError("parsed workflow settings differ from captured run record")
    projection = record["configuration_projection"]
    return (
        projection["effective_config_sha256"],
        projection["configuration_inputs_sha256"],
    )


@execution_outputs.capture_launch
def run_workflow(
    workflow: str,
    project_config: Path,
    project_root: Path,
    *,
    cores: int = 3,
    targets: Sequence[str] = ("all",),
    dry_run: bool = False,
    keep_going: bool = False,
    extra: Sequence[str] = (),
) -> int:
    """One captured WF0--WF2 execution with a common invocation record."""
    if workflow not in WORKFLOWS or cores < 1:
        raise ValueError("invalid workflow or core count")
    path, record = invocation_history.start(
        Path(project_root),
        workflow=workflow,
        entry_point="scripts/run_workflow.py",
        command=["snakemake", *targets, "-s", f"{workflow}.smk"],
        targets=list(targets),
        mode="dry_run" if dry_run else "execute",
        contract_mode="new_schema",
        invocation_id=os.environ.get(invocation_history.INVOCATION_ENV),
        parent_invocation_id=os.environ.get(invocation_history.PARENT_ENV),
    )
    try:
        return _run_workflow_started(
            workflow,
            project_config,
            project_root,
            cores=cores,
            targets=targets,
            dry_run=dry_run,
            keep_going=keep_going,
            extra=extra,
            path=path,
            record=record,
        )
    except BaseException as error:
        invocation_history.finish(path, record, exit_code=None, error=error)
        raise


def _run_workflow_started(
    workflow: str,
    project_config: Path,
    project_root: Path,
    *,
    cores: int,
    targets: Sequence[str],
    dry_run: bool,
    keep_going: bool,
    extra: Sequence[str],
    path: Path,
    record: dict[str, Any],
) -> int:
    config_path = Path(project_config).resolve()
    project = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime_dir = TOOLBOX_ROOT if project.get("schema_version") == 3 else Path.cwd()
    if project.get("workflows", {}).get(workflow, {}).get("enabled") is False:
        raise ValueError(
            f"Workflow {workflow} is disabled in {config_path}. "
            f"Set workflows.{workflow}.enabled to true in that "
            "project configuration file, then rerun the command."
        )
    overrides, forwarded = (
        split_config_overrides(extra) if not dry_run else ({}, list(extra))
    )
    command = [
        "snakemake",
        *targets,
        "-c",
        str(cores),
        "-s",
        f"{workflow}.smk",
        "--configfile",
        str(Path(project_config).resolve()),
    ]
    if keep_going:
        command.append("--keep-going")
    if dry_run:
        command.append("--dry-run")
    command.extend(forwarded)
    with archive_lock(Path(project_root).resolve(), "project-execution"):
        if dry_run:
            # Dry-runs do not publish creator archives. They may parse the raw
            # config for diagnostics, and do not claim exact source capture.
            from blueearth_cst.shared.windows_job import run_project_child

            result = run_project_child(
                command,
                cwd=runtime_dir,
                env={**os.environ, **execution_outputs.environment(path, record)},
                writing=False,
            )
            invocation_history.finish(path, record, exit_code=result)
            return result
        execution, context = prepare_workflow(
            workflow,
            project_config,
            project_root,
            command=command,
            targets=targets,
            overrides=overrides,
            invocation_id=record["invocation_id"],
        )
        try:
            archive = read_archive(Path(project_root), workflow, f"config/{workflow}")
            projection = archive["configuration_projection"]
            record["configuration"].update(
                source_config_sha256=archive["source_files"][0]["sha256"],
                effective_config_sha256=projection["effective_config_sha256"],
                configuration_inputs_sha256=projection["configuration_inputs_sha256"],
                run_record=file_reference(
                    retain_archive(
                        Path(project_root), Path(project_root) / "config" / workflow
                    ),
                    "project_root",
                    Path(project_root),
                ),
                archive_state="retained",
            )
            invocation_history.update(path, record)
            command[command.index("--configfile") + 1] = str(execution)
            environment = os.environ.copy()
            environment.update(execution_outputs.environment(path, record))
            environment[CONTEXT_ENV] = str(context)
            from blueearth_cst.shared.windows_job import run_project_child

            observation = product_record.observe(workflow, Path(project_root))
            try:
                result = run_project_child(command, cwd=runtime_dir, env=environment)
            except BaseException as error:
                invocation_history.finish(path, record, exit_code=None, error=error)
                product_record.publish(observation, path, record)
                raise
        finally:
            remove_launch_context(Path(project_root), context)
        invocation_history.finish(path, record, exit_code=result)
        product_record.publish(observation, path, record)
        return result
