"""Snapshot the snake config and its referenced config files into project_dir."""
import os
from os.path import join, dirname
from pathlib import Path
from typing import Union, Mapping, Optional

from blueearth_cst.shared.gauges import is_unset, warn_if_low_gauge_ids
from blueearth_cst.shared.snake_utils import log_row


def copy_config_files(
    config: Union[str, Path],
    config_out_path: Union[str, Path],
    other_config_files: Optional[Mapping[Union[str, Path], Union[str, Path]]] = None,
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

    """
    # Copy the snake config file to its declared destination
    os.makedirs(dirname(config_out_path), exist_ok=True)
    log_row(f"Copying {os.path.basename(config_out_path)} to "
            f"{dirname(config_out_path)}", module="config")
    with open(config, "r") as f:
        snake_config = f.read()
    with open(config_out_path, "w") as f:
        f.write(snake_config)

    # Copy every other config file into the bin its KIND belongs in
    for config_file, dest_dir in (other_config_files or {}).items():
        # Check if the file does exist
        # (eg predefined catalogs of hydromt do not have a path)
        if os.path.isfile(config_file):
            with open(config_file, "r") as f:
                content = f.read()
            config_name = os.path.basename(config_file)
            os.makedirs(dest_dir, exist_ok=True)
            log_row(f"Copying {config_name} to {dest_dir}", module="config")
            with open(join(dest_dir, config_name), "w") as f:
                f.write(content)


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
            for key in ("output_locations", "observations_timeseries"):
                path = getattr(sm.params, key, None)
                if is_unset(path):
                    continue
                if not os.path.isfile(str(path)):
                    raise ValueError(
                        f"{key} is set to {path!r} but no such file exists. "
                        f"Leave the key at `null` to run without observations, "
                        f"or fix the path -- it is read by rule 1.05 (gauges) "
                        f"and rule 1.11 (evaluation), both of which would "
                        f"otherwise skip it without saying so."
                    )
                other_config_files[str(path)] = observations_dir
                if key == "output_locations":
                    _warn_on_low_gauge_ids(path)

        # Call the main function
        copy_config_files(
            config=config_snake,
            config_out_path=config_snake_out,
            other_config_files=other_config_files,
        )

    else:
        copy_config_files(
            config="config/snake_config_model_test.yml",
            output_dir="test_case/test/config",
            config_out_name=None,
            other_config_files=[],
        )
