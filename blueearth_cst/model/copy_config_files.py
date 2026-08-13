"""Snapshot source and effective workflow configuration into ``project_dir``."""

import os
import shutil
import subprocess
import uuid
from os.path import join
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import yaml

from blueearth_cst.shared.gauges import is_unset, warn_if_low_gauge_ids
from blueearth_cst.shared.provenance import (
    configuration_inputs_digest,
    effective_config_digest,
    effective_config_document,
    environment_file_hashes,
    file_sha256,
    toolbox_identity,
)
from blueearth_cst.shared.snake_utils import log_row

#: Repository root, three levels up from ``blueearth_cst/model/copy_config_files.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def copy_config_files(
    config: Union[str, Path],
    config_out_path: Union[str, Path],
    other_config_files: Optional[Mapping[Union[str, Path], Union[str, Path]]] = None,
    reference_roles: Optional[Mapping[Union[str, Path], str]] = None,
    run_record_path: Union[str, Path, None] = None,
    effective_config: Optional[Mapping] = None,
    advanced_settings: Optional[Mapping] = None,
    workflow_name: Optional[str] = None,
    projection: Optional[Sequence[str]] = None,
):
    """
    Snapshot the snake config and its referenced config files into project_dir.

    R07 B9 changed this from "one derived output directory" to explicit
    per-file routing, because the project config snapshot is now split by
    KIND -- runs/, catalogs/, templates/, generated/. That is a signature
    change, not a rename: one output_dir cannot serve four destinations.

    A referenced file is copied only when the toolbox repository cannot give it
    back (see :func:`_tracked_blob`). Whether a file is copied is therefore a
    property of the FILE, not of the bin it lands in: ``data_sources``,
    ``model_build_config`` and ``waterbodies_config`` hold arbitrary paths, so
    a project may name a site-specific catalog that lives nowhere in the
    toolbox, and a bin-level rule would discard exactly the file the policy
    exists to protect.

    Parameters
    ----------
    config : Union[str, Path]
        path to the snake config file
    config_out_path : Union[str, Path]
        FULL destination path for the snake config snapshot (the rule declares
        it, so the bin choice lives in the Snakefile rather than here)
    other_config_files : Mapping[src, dest_dir], optional
        each referenced config file mapped to the directory its kind belongs
        in. Missing files are recorded as logical identifiers rather than
        copied -- hydromt's predefined catalogs have no path on disk.
    reference_roles : Mapping[src, role], optional
        the ROLE a referenced file plays, which becomes its destination name
        when it is copied. A file with no declared role keeps its own
        basename. Roles exist because two configured paths can share a
        basename, and the old ``dest_dir / source_path.name`` silently
        overwrote one with the other.
    run_record_path : path-like, optional
        FULL destination path for ``run_record.yml``. When supplied,
        ``effective_config``, ``advanced_settings`` and ``workflow_name`` are
        required.
    effective_config : Mapping, optional
        Snakemake's merged config dictionary, after command-line overrides.
    advanced_settings : Mapping, optional
        Resolved toolbox-wide settings applied outside the project config.
    workflow_name : str, optional
        Workflow the record describes.
    projection : Sequence[str], optional
        the workflow's declared consumed-key paths. ``None`` records the whole
        config, which is over-inclusive rather than wrong.

    Raises
    ------
    ValueError
        if two referenced files resolve to the same destination path. Raising
        is the point: the previous behaviour overwrote one with the other and
        left a project claiming to hold an input it had actually lost.
    """
    source_config_path = Path(config)
    current_config_path = Path(config_out_path)
    current_config_path.parent.mkdir(parents=True, exist_ok=True)
    log_row(
        f"Copying {current_config_path.name} to {current_config_path.parent}",
        module="config",
    )
    shutil.copyfile(source_config_path, current_config_path)

    references = dict(other_config_files or {})
    roles = {str(key): value for key, value in (reference_roles or {}).items()}
    toolbox = toolbox_identity()
    referenced_inputs = _snapshot_references(references, roles, toolbox)

    record_values = (
        run_record_path,
        effective_config,
        advanced_settings,
        workflow_name,
    )
    if any(value is not None for value in record_values):
        if any(value is None for value in record_values):
            raise ValueError(
                "run_record_path, effective_config, advanced_settings, and "
                "workflow_name must be provided together"
            )
        _write_run_record(
            run_record_path=Path(run_record_path),
            source_config_path=source_config_path,
            effective_config=effective_config,
            advanced_settings=advanced_settings,
            workflow_name=workflow_name,
            projection=projection,
            toolbox=toolbox,
            referenced_inputs=referenced_inputs,
        )


def _snapshot_references(
    references: Mapping[Union[str, Path], Union[str, Path]],
    roles: Mapping[str, str],
    toolbox: Mapping[str, object],
) -> list[dict]:
    """Apply the copy predicate to every referenced file, and copy what it says.

    Returns one ``referenced_inputs`` entry per reference, in a stable order so
    the record does not churn on dictionary iteration order.
    """
    entries: list[dict] = []
    claimed: dict[Path, str] = {}
    for config_file, dest_dir in sorted(
        references.items(), key=lambda item: (str(item[1]), str(item[0]))
    ):
        origin = str(config_file)
        source_path = Path(origin)
        role = roles.get(origin) or source_path.stem

        if not source_path.is_file():
            # A logical identifier: hydromt's predefined catalogs are named,
            # not pathed. Nothing to hash and nothing to copy, so the record
            # carries the name and says so with nulls.
            entries.append(
                {
                    "role": role,
                    "origin": origin,
                    "recoverable": False,
                    "archived_path": None,
                    "git_blob": None,
                    "sha256": None,
                    "size_bytes": None,
                }
            )
            continue

        blob = _tracked_blob(source_path, toolbox)
        if blob is not None:
            entries.append(
                {
                    "role": role,
                    "origin": _repo_relative(source_path),
                    "recoverable": True,
                    "archived_path": None,
                    "git_blob": blob,
                    "sha256": file_sha256(source_path),
                    "size_bytes": source_path.stat().st_size,
                }
            )
            continue

        destination_dir = Path(dest_dir)
        destination = destination_dir / f"{role}{source_path.suffix}"
        previous = claimed.get(destination.resolve())
        if previous is not None:
            raise ValueError(
                f"two referenced config files both map to {destination}: "
                f"{previous!r} and {origin!r}. Copying both would lose one of "
                "them silently; give them distinct roles instead."
            )
        claimed[destination.resolve()] = origin

        destination_dir.mkdir(parents=True, exist_ok=True)
        log_row(f"Copying {source_path.name} to {destination_dir}", module="config")
        shutil.copyfile(source_path, destination)
        entries.append(
            {
                "role": role,
                "origin": origin,
                "recoverable": False,
                "archived_path": destination.as_posix(),
                "git_blob": None,
                "sha256": file_sha256(source_path),
                "size_bytes": source_path.stat().st_size,
            }
        )
    return entries


def _tracked_blob(source_path: Path, toolbox: Mapping[str, object]) -> Optional[str]:
    """Return the git blob id when the toolbox repository can give this file back.

    The whole copy policy hangs off this question: copy a referenced file into
    the project only when the repository cannot reproduce it. A file inside the
    checkout, tracked at the recorded commit, and locally unmodified is
    reproducible from git, so copying it would duplicate what version control
    already holds.

    Returns ``None`` -- meaning "copy it" -- whenever the answer is anything
    other than a confident yes. That covers the deployed-image case by
    construction: a container has no ``.git``, so every query fails, the commit
    is null or baked, and every referenced file is copied. That is the correct
    outcome there rather than a degradation, because in an image that cannot be
    interrogated the copies are the only way the project can say what it ran
    with.
    """
    if not toolbox.get("commit"):
        return None
    try:
        resolved = source_path.resolve()
        resolved.relative_to(_REPO_ROOT)
    except (OSError, ValueError):
        return None
    if _git_query(["git", "ls-files", "--error-unmatch", str(resolved)]) is None:
        return None
    status = _git_query(["git", "status", "--porcelain", "--", str(resolved)])
    if status is None or status.strip():
        return None
    blob = _git_query(["git", "rev-parse", f"HEAD:./{_repo_relative(resolved)}"])
    if blob is None or not blob.strip():
        return None
    return blob.strip()


def _repo_relative(path: Path) -> str:
    """Return a path relative to the toolbox root, or the path as given."""
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except (OSError, ValueError):
        return str(path)


def _git_query(command: list) -> Optional[str]:
    """Run a git tracking query, swallowing every failure.

    Separate from ``provenance._run_metadata_command`` on purpose: that one
    answers "which revision is this", while these answer "can the repository
    give this file back". Both swallow failures, for the same reason -- a
    provenance helper must never crash a run -- but a failure here means
    "copy it", not "record a null".
    """
    try:
        result = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return getattr(result, "stdout", "")


def _write_run_record(
    *,
    run_record_path: Path,
    source_config_path: Path,
    effective_config: Mapping,
    advanced_settings: Mapping,
    workflow_name: str,
    projection: Optional[Sequence[str]],
    toolbox: Mapping[str, object],
    referenced_inputs: Sequence[Mapping],
) -> None:
    """Write the current-run record atomically.

    Atomically because this file is the project's answer to "what did the last
    run use". A half-written record read by a person or a tool is worse than no
    record: it looks authoritative.
    """
    environment = environment_file_hashes()
    digested = effective_config_document(
        effective_config, advanced_settings, projection
    )
    effective_sha = effective_config_digest(
        effective_config, advanced_settings, projection
    )
    document = {
        # This record's schema, which is the digested document's as well --
        # they move together, so one version pins both.
        "schema_version": digested["schema_version"],
        "workflow": workflow_name,
        "toolbox": dict(toolbox),
        "environment": environment,
        "source_config": {
            "path": str(source_config_path),
            "sha256": file_sha256(source_config_path),
        },
        "projection": digested["projection"],
        "effective_config": digested["project_config"],
        "advanced_settings": digested["advanced_settings"],
        "effective_config_sha256": effective_sha,
        "configuration_inputs_sha256": configuration_inputs_digest(
            effective_sha, toolbox, environment, referenced_inputs
        ),
        "referenced_inputs": list(referenced_inputs),
    }

    run_record_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = run_record_path.with_name(
        f".{run_record_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump(document, stream, sort_keys=True, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, run_record_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    log_row(
        f"Run record for {workflow_name}: {run_record_path}",
        module="config",
    )


def _warn_on_low_gauge_ids(locations_path):
    """Advisory read of ``output_locations`` for the wflow_id convention.

    Rule 1.01 is the earliest point that sees this file, so a warning here
    reaches the user BEFORE rule 1.05 writes the ids into the model and a
    renumbering would cost a rebuild.

    CSV only, and every failure is swallowed: hydromt accepts several formats
    (GeoJSON, a catalog entry name) and owns the actual reading. Re-implementing
    that here would be exactly the "re-engineer how hydromt handles data" this
    repo forbids. A format we cannot cheaply parse simply goes unchecked --
    an advisory that skips is fine; one that breaks a valid run is not.
    """
    if os.path.splitext(str(locations_path))[1].lower() != ".csv":
        return
    try:
        import pandas as pd

        frame = pd.read_csv(locations_path)
        if "wflow_id" in frame.columns:
            warn_if_low_gauge_ids(frame["wflow_id"].tolist(), locations_path)
    except Exception:  # noqa: BLE001 - advisory only; never fail the rule
        return


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        # Get the in and out path of the snake (main) config file
        config_snake = sm.input.config_snake
        config_snake_out = sm.output.config_snake_out

        # R07 B9: the project config snapshot is split by KIND, so this is a
        # signature change rather than a rename -- one derived output_dir can
        # no longer serve. The snake config lands where the rule declared it
        # (config/runs/, or the experiment dir for wf3); catalogs go to
        # config/catalogs/; verbatim snapshots of shipped templates go to
        # config/templates/. Generated run-time configs live in
        # config/generated/, written by their own rules, not copied here.
        config_dir = sm.params.config_dir
        catalogs_dir = join(config_dir, "catalogs")
        templates_dir = join(config_dir, "templates")
        # Fifth bin (2026-08-01). The two OPTIONAL observation inputs live
        # outside the repository AND outside project_dir, referenced by
        # absolute path (R07 O-01), so without this the finished project cannot
        # say what it was evaluated against -- the metrics table would cite
        # gauges and observations that exist only on the machine that ran it.
        # Same provenance role as config/catalogs/, hence the same home.
        observations_dir = join(config_dir, "observations")

        # Get other config files to copy based on workflow name, each routed
        # to the bin its KIND belongs in.
        workflow_name = sm.params.workflow_name
        other_config_files = {}
        # The ROLE a file plays becomes its destination name when it is copied.
        # Only the roles whose basenames can collide need declaring; anything
        # else keeps its own stem.
        reference_roles = {}
        data_sources = sm.params.data_catalogs
        if workflow_name == "model_creation":
            other_config_files[sm.input.config_build] = templates_dir
            other_config_files[sm.input.config_waterbodies] = templates_dir
            reference_roles[str(sm.input.config_build)] = "model_build_config"
            reference_roles[str(sm.input.config_waterbodies)] = "waterbodies_config"
        if isinstance(data_sources, (list, tuple)):
            for src in data_sources:
                other_config_files[src] = catalogs_dir
        else:
            other_config_files[data_sources] = catalogs_dir

        # The observation inputs, when configured. Checked EXPLICITLY rather
        # than routed through the skip-missing loop above: that skip exists for
        # hydromt's predefined catalogs, which legitimately have no path on
        # disk, whereas a configured observations path that is not a file is a
        # typo -- and a silently skipped typo is precisely the failure mode
        # that cost this workflow its whole evaluation output once already
        # (dev/tasks/, the gauge-name entry).
        if workflow_name == "model_creation":
            # sm.INPUT, not sm.params: both keys became declared inputs on
            # 2026-08-02 so a file EDIT retriggers the rules that read them.
            # An unset key contributes no entry at all, so getattr's default is
            # the no-observations case; a configured-but-missing file never
            # reaches here, because Snakemake refuses to run the rule.
            for key in ("output_locations", "observations_timeseries"):
                path = getattr(sm.input, key, None)
                if is_unset(path):
                    continue
                other_config_files[str(path)] = observations_dir
                # The two keys that made collision safety necessary: both are
                # arbitrary absolute paths, so nothing stops a project pointing
                # them at two files called `data.csv` in different directories.
                reference_roles[str(path)] = key
                if key == "output_locations":
                    _warn_on_low_gauge_ids(path)

        # Call the main function
        copy_config_files(
            config=config_snake,
            config_out_path=config_snake_out,
            other_config_files=other_config_files,
            reference_roles=reference_roles,
            run_record_path=sm.output.run_record,
            effective_config=sm.params.effective_config,
            advanced_settings=sm.params.advanced_settings,
            workflow_name=workflow_name,
            projection=getattr(sm.params, "config_projection", None),
        )
