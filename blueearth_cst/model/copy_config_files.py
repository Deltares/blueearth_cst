"""Snapshot source and effective workflow configuration into ``project_dir``."""
import json
import os
from os.path import join
from pathlib import Path
import shutil
from typing import Mapping, Optional, Union

import yaml

from blueearth_cst.shared.gauges import is_unset, warn_if_low_gauge_ids
from blueearth_cst.shared.provenance import (
    effective_config_digest,
    effective_config_document,
    file_sha256,
    snapshot_bundle_digest,
)
from blueearth_cst.shared.snake_utils import log_row


def copy_config_files(
    config: Union[str, Path],
    config_out_path: Union[str, Path],
    other_config_files: Optional[Mapping[Union[str, Path], Union[str, Path]]] = None,
    snapshot_dir: Union[str, Path, None] = None,
    effective_config: Optional[Mapping] = None,
    advanced_settings: Optional[Mapping] = None,
    workflow_name: Optional[str] = None,
):
    """
    Snapshot the snake config and its referenced config files into project_dir.

    R07 B9 changed this from "one derived output directory" to explicit
    per-file routing, because the project config snapshot is now split by
    KIND -- runs/, catalogs/, templates/, generated/. That is a signature
    change, not a rename: one output_dir cannot serve four destinations.

    Parameters
    ----------
    config : Union[str, Path]
        path to the snake config file
    config_out_path : Union[str, Path]
        FULL destination path for the snake config snapshot (the rule declares
        it, so the bin choice lives in the Snakefile rather than here)
    other_config_files : Mapping[src, dest_dir], optional
        each referenced config file mapped to the directory its kind belongs
        in. Missing files are skipped -- hydromt's predefined catalogs have no
        path on disk.
    snapshot_dir : path-like, optional
        content-addressed bundle directory. When supplied, ``effective_config``,
        ``advanced_settings``, and ``workflow_name`` are required.
    effective_config : Mapping, optional
        Snakemake's merged config dictionary, after command-line overrides.
    advanced_settings : Mapping, optional
        Resolved toolbox-wide settings applied outside the project config.
    workflow_name : str, optional
        Workflow recorded in the immutable bundle manifest.

    """
    source_config_path = Path(config)
    current_config_path = Path(config_out_path)
    current_config_path.parent.mkdir(parents=True, exist_ok=True)
    log_row(
        f"Copying {current_config_path.name} to {current_config_path.parent}",
        module="config",
    )
    shutil.copyfile(source_config_path, current_config_path)

    # Copy every other config file into the bin its KIND belongs in
    references = dict(other_config_files or {})
    for config_file, dest_dir in references.items():
        # Check if the file does exist
        # (eg predefined catalogs of hydromt do not have a path)
        source_path = Path(config_file)
        if source_path.is_file():
            destination_dir = Path(dest_dir)
            destination_dir.mkdir(parents=True, exist_ok=True)
            log_row(
                f"Copying {source_path.name} to {destination_dir}", module="config"
            )
            shutil.copyfile(source_path, destination_dir / source_path.name)

    snapshot_values = (
        snapshot_dir,
        effective_config,
        advanced_settings,
        workflow_name,
    )
    if any(value is not None for value in snapshot_values):
        if any(value is None for value in snapshot_values):
            raise ValueError(
                "snapshot_dir, effective_config, advanced_settings, and "
                "workflow_name must be provided together"
            )
        _write_snapshot_bundle(
            source_config_path=source_config_path,
            snapshot_dir=Path(snapshot_dir),
            effective_config=effective_config,
            advanced_settings=advanced_settings,
            workflow_name=workflow_name,
            referenced_files=references,
        )


def _write_snapshot_bundle(
    *,
    source_config_path: Path,
    snapshot_dir: Path,
    effective_config: Mapping,
    advanced_settings: Mapping,
    workflow_name: str,
    referenced_files: Mapping[Union[str, Path], Union[str, Path]],
) -> None:
    """Write a deterministic bundle of resolved settings and referenced files."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_config_path, snapshot_dir / "source.yml")

    effective_document = effective_config_document(
        effective_config, advanced_settings
    )
    effective_document["effective_config_sha256"] = effective_config_digest(
        effective_config, advanced_settings
    )
    with (snapshot_dir / "effective.yml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            effective_document,
            stream,
            sort_keys=True,
            allow_unicode=True,
        )

    entries = []
    reference_descriptors = []
    for source, current_destination in sorted(
        referenced_files.items(), key=lambda item: (str(item[1]), str(item[0]))
    ):
        source_text = str(source)
        source_path = Path(source_text)
        kind = Path(current_destination).name
        if source_path.is_file():
            reference_descriptors.append(
                {"kind": kind, "identifier": source_text, "path": source_text}
            )
            digest = file_sha256(source_path)
            archive_path = (
                Path("files") / kind / f"{digest[:12]}-{source_path.name}"
            )
            archive_target = snapshot_dir / archive_path
            archive_target.parent.mkdir(parents=True, exist_ok=True)
            if not archive_target.exists():
                shutil.copyfile(source_path, archive_target)
            entries.append(
                {
                    "archived_path": archive_path.as_posix(),
                    "kind": kind,
                    "sha256": digest,
                    "size_bytes": source_path.stat().st_size,
                    "source": source_text,
                    "status": "archived",
                }
            )
        else:
            reference_descriptors.append(
                {"kind": kind, "identifier": source_text}
            )
            entries.append(
                {
                    "archived_path": None,
                    "kind": kind,
                    "sha256": None,
                    "size_bytes": None,
                    "source": source_text,
                    "status": "logical_identifier",
                }
            )

    bundle_digest = snapshot_bundle_digest(
        effective_config,
        advanced_settings,
        source_config_path,
        reference_descriptors,
    )
    manifest = {
        "effective_config_sha256": effective_document["effective_config_sha256"],
        "referenced_files": entries,
        "schema_version": "1",
        "snapshot_bundle_sha256": bundle_digest,
        "source_config": {
            "archived_path": "source.yml",
            "sha256": file_sha256(source_config_path),
            "source": str(source_config_path),
        },
        "workflow": workflow_name,
    }
    manifest_path = snapshot_dir / "referenced-files.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


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
        data_sources = sm.params.data_catalogs
        if workflow_name == "model_creation":
            other_config_files[sm.input.config_build] = templates_dir
            other_config_files[sm.input.config_waterbodies] = templates_dir
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
        # (dev/followups.md, the gauge-name entry).
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
                if key == "output_locations":
                    _warn_on_low_gauge_ids(path)

        # Call the main function
        copy_config_files(
            config=config_snake,
            config_out_path=config_snake_out,
            other_config_files=other_config_files,
            snapshot_dir=sm.output.snapshot_bundle,
            effective_config=sm.params.effective_config,
            advanced_settings=sm.params.advanced_settings,
            workflow_name=workflow_name,
        )
