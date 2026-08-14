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

#: Roles archived into the project unconditionally, recoverable or not.
#:
#: R4's predicate asks whether the TOOLBOX can give a file back. For the two
#: observation inputs that is the wrong question: they are the project's record
#: of what the model was evaluated against, and a finished project must be able
#: to state that on its own. The predicate also makes the bin invisible exactly
#: when the inputs happen to live inside the checkout -- the test fixture case,
#: where `gauge_points` and `observations_timeseries` point at tracked
#: `test_case/test_data/` CSVs and nothing is ever written.
#:
#: The duplication is ACCEPTED, not overlooked: a gauge record can be large
#: (435 KB in the test fixture, more on a real basin). Do not "optimize" this
#: back into a predicate -- that is exactly the behaviour it replaced.
#:
#: Role-keyed rather than bin-keyed on purpose: §5.5's argument that copying is
#: a property of the FILE still holds, and `__main__` already assigns exactly
#: these two role names.
ALWAYS_ARCHIVED_ROLES = frozenset({"output_locations", "observations_timeseries"})

#: Dropped beside the run record, because this bin has two genuine traps.
#:
#: Everything here LOOKS like configuration and is not: it is written by the
#: run, so an edit is silently overwritten the next time the workflow executes.
#: Someone will eventually open the copy nearest the outputs and change it.
#:
#: And the per-workflow name promises a scope the content does not deliver --
#: `snake_config_model_creation.yml` is the WHOLE config, so the two copies are
#: byte-identical whenever both workflows ran under the same one. Measured
#: 2026-08-13: the owner opened them, read the identical bytes as duplication,
#: and asked. The `projection` field of `run_record.yml` is the scoped view;
#: the copies keep whole-config content because their paths are
#: baseline-fingerprinted AND the WF1 copy is a mandatory declared input of
#: WF3's guard rule, so the name cannot be fixed by renaming the file.
_RUNS_README = """\
# What is in this directory

Everything here is **written by the run**. Editing any of it changes nothing:
the next execution overwrites it.

To change what a run does, edit the **source config** you pass to
`--configfile`, then run the workflow again.

| File | What it is |
|---|---|
| `snake_config_<workflow>.yml` | the **whole** source config, copied verbatim -- see below |
| `<workflow>/run_record.yml` | what the run resolved to: toolbox commit, environment hashes, the settings actually consumed, and the external inputs referenced |
| `journal.jsonl` | append-only ledger, two lines per run (start and outcome) |
| `invocations/` | one manifest per `scripts/run_workflows.py` invocation |

## Why the `snake_config_<workflow>.yml` files look identical

Each one is the **whole** source config, not that workflow's section of it. The
per-workflow name records **which workflow last wrote the file**, not what is
inside -- so all of them carry every section, including the two workflows' that
did not write them.

Byte-identical copies are therefore the healthy case: it means both workflows
last ran under the same config. They diverge when one workflow has run since the
config changed and the other has not, and that divergence is exactly what WF3's
consistency guard reads to refuse an experiment whose model was built under
different settings.

For the settings a given workflow actually **consumed**, read the `projection`
and `effective_config` fields of its `run_record.yml` -- that is the
workflow-scoped view this file's name suggests but does not provide.

## Reading `run_record.yml`

It is **current-only** -- it describes the most recent run of that workflow and
is replaced, not accumulated. `journal.jsonl` is the history.

Two digests, answering different questions:

- `effective_config_sha256` -- the settings the workflow was asked to run under.
- `configuration_inputs_sha256` -- those settings **plus** the toolbox commit,
  the lock files, and the bytes of every referenced catalog and template.

Compare runs with the **wide** one: the narrow one does not move when the code
or a referenced file changes. Neither covers scientific data identity, because
a remote or mutable dataset can change under an unchanged catalog entry.

Two records assert the same CODE only when both have a non-null `commit` and
`dirty: false`. With `dirty: true`, the commit does not fully identify what ran.

## Reading `journal.jsonl`

It records **executed** runs. An invocation that finds everything up to date
does no work and appends nothing, so the line count is an exact count of
executions and a lower bound on invocations -- a gap in the dates means no work
was done in that period, not that nobody ran the command.

## Why some referenced files are here and others are not

A catalog or template is copied into the project only when the toolbox
repository cannot give it back; one that is tracked and unmodified in the
checkout is recorded by its git blob id instead. `run_record.yml` lists every
reference either way, so nothing goes unrecorded.

The **basin data inputs are an exception and are always copied**, into
`config/basin_data/`. They are this project's record of what the model was
built and evaluated against, and a finished project should be able to state
that without the toolbox checkout beside it -- so for them "can the toolbox
give it back?" is the wrong question. Their entries carry `archived_path` AND, when the file
was also tracked, `recoverable: true` and a `git_blob`.
"""


def _write_runs_readme(directory: Path) -> None:
    """Write the bin's own README, unconditionally.

    Unconditionally rather than only-when-absent: this file is documentation
    the toolbox ships, so a stale copy in an old project should be refreshed,
    and nothing a user writes here is meant to survive anyway -- which is
    precisely what it says.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "README.md").write_text(_RUNS_README, encoding="utf-8")
    except OSError as error:  # never fail a run over a README
        log_row(f"could not write {directory}/README.md: {error}", module="config")


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
    back (see :func:`_tracked_blob`) — except for the roles in
    :data:`ALWAYS_ARCHIVED_ROLES`, which are archived either way. Whether a file
    is copied is therefore a property of the FILE and its ROLE, never of the bin
    it lands in: ``data_sources``, ``model_build_config`` and
    ``waterbodies_config`` hold arbitrary paths, so a project may name a
    site-specific catalog that lives nowhere in the toolbox, and a bin-level
    rule would discard exactly the file the policy exists to protect.

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
    # Beside the flat config copy, which is the bin a user actually opens --
    # not beside the record, which sits one level down under a per-workflow
    # subdirectory (and, for WF3, in the experiment's own config dir).
    _write_runs_readme(current_config_path.parent)

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
    claimed: dict[str, str] = {}
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
        if blob is not None and role not in ALWAYS_ARCHIVED_ROLES:
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
        # normcase, not resolve(): on Windows `resolve()` reports the real
        # on-disk casing for a path that EXISTS and preserves the given casing
        # for one that does not, so two destinations differing only in case
        # would collide on a re-run and silently overwrite on a fresh one --
        # the same file lost, but only sometimes. normcase folds case on
        # win32 and is a no-op on POSIX, so the answer stops depending on
        # whether a previous run left the file behind.
        key = os.path.normcase(os.path.abspath(destination))
        previous = claimed.get(key)
        if previous is not None:
            raise ValueError(
                f"two referenced config files both map to {destination}: "
                f"{previous!r} and {origin!r}. Copying both would lose one of "
                "them silently; give them distinct roles instead."
            )
        claimed[key] = origin

        destination_dir.mkdir(parents=True, exist_ok=True)
        log_row(f"Copying {source_path.name} to {destination_dir}", module="config")
        shutil.copyfile(source_path, destination)
        entries.append(
            {
                "role": role,
                # `recoverable` and `archived_path` used to be mutually
                # exclusive -- a file was recorded OR copied. They are now
                # independent, because an always-archived role can be both:
                # `recoverable` keeps its own meaning (the toolbox can
                # reproduce this file) and `archived_path` becomes the only
                # field that says where the copy is. `blob` is in scope and is
                # strictly more information than the `None` this hardcoded.
                "origin": origin,
                "recoverable": blob is not None,
                "archived_path": destination.as_posix(),
                "git_blob": blob,
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
        # Fifth bin (2026-08-01), named `observations/` until 2026-08-14. The
        # OPTIONAL basin data inputs live outside the repository AND outside
        # project_dir, referenced by absolute path (R07 O-01), so without this
        # the finished project cannot say what it was evaluated against -- the
        # metrics table would cite gauges and observations that exist only on
        # the machine that ran it. Same provenance role as config/catalogs/,
        # hence the same home. The name is BASIN DATA, not `observations`,
        # because only one of the two files is an observation: the other
        # declares where the model reports. Local basin-scoped tabular inputs
        # is the property they share, and the one future files will share too.
        basin_data_dir = join(config_dir, "basin_data")

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
                other_config_files[str(path)] = basin_data_dir
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
