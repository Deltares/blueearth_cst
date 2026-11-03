"""Run the configured simulation operation through one validated Snakemake invocation."""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blueearth_cst.experiment.simulation_record import capture_simulation_sources_v2
from blueearth_cst.experiment.simulation_runner import (
    current_targets,
    simulation_command,
    simulation_selection,
    simulation_settings,
)
from blueearth_cst.shared import (
    execution_outputs,
    invocation_history,
    resource_tracking,
)
from blueearth_cst.shared.console_style import run_summary
from blueearth_cst.shared.provenance import file_sha256
from blueearth_cst.shared.snake_utils import warning_count
from blueearth_cst.shared.windows_job import run_project_child
from blueearth_cst.shared.workflow_config_snapshot import archive_lock


@execution_outputs.capture_launch
def main(argv=None):
    console_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-dir", type=Path)
    parser.add_argument(
        "--target",
        nargs="+",
        default=["all"],
        help="legacy operation configs only; operations/indicator_sets infer the target",
    )
    parser.add_argument("--cores", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forceall", action="store_true")
    parser.add_argument("--new-experiment", action="store_true")
    parser.add_argument("--note")
    parser.add_argument("--recover-reservation", action="store_true")
    parser.add_argument(
        "--traceback", action="store_true", help="show exception causes for diagnosis"
    )
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.dry_run:
        extra.append("--dry-run")
    if args.forceall:
        extra.append("--forceall")
    preview = args.dry_run or any(
        flag in extra for flag in ("-n", "--dry-run", "--unlock")
    )
    if args.cores < 1:
        parser.error("--cores must be positive")
    if args.project_dir is None:
        # Config-only syntax has reduced startup-failure coverage: its root is
        # unavailable until the configuration can be parsed.
        try:
            project, _ = simulation_settings(args.config)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            if args.traceback:
                raise
            parser.error(str(exc))
        root = Path(project["project"]["project_dir"]).resolve()
    else:
        root = args.project_dir.resolve()
    record_path, record = invocation_history.start(
        root,
        workflow="simulate_system",
        entry_point="scripts/simulate_system.py",
        command=[
            "simulate_system",
            "--config",
            args.config,
            "--target",
            *args.target,
            *extra,
        ],
        targets=list(args.target),
        mode="dry_run" if preview else "execute",
        contract_mode="new_schema",
        invocation_id=os.environ.get(invocation_history.INVOCATION_ENV),
        parent_invocation_id=os.environ.get(invocation_history.PARENT_ENV),
    )
    tally_key = "CST_WARNING_TALLY"
    previous_tally = os.environ.get(tally_key)
    directory = root / record["execution_directory"]
    os.environ[tally_key] = str(directory / "_parts" / f".{record_path.stem}.warnings")
    try:
        project, settings, selection = simulation_selection(args.config)
        if Path(project["project"]["project_dir"]).resolve() != root:
            raise ValueError("--project-dir differs from composed project root")
        record["operation"] = settings["operation"]
        record["indicator_sets"] = [item.set_id for item in selection.sets]
        command, environment = simulation_command(
            args.config,
            args.target,
            args.cores,
            extra,
            snapshot_path=record_path.with_suffix(".indicator-selection"),
        )
        environment.update(execution_outputs.environment(record_path, record))
        record["configuration"]["source_config_sha256"] = file_sha256(Path(args.config))
        with archive_lock(root, "project-execution"):
            from blueearth_cst.experiment.reservation_context import admit_context_owned

            context = admit_context_owned(
                Path(args.config),
                Path(__file__).resolve().parents[1],
                root,
                record["invocation_id"],
                new_experiment=args.new_experiment,
                note=args.note,
                recover=args.recover_reservation,
                preview=preview,
            )
            if not preview:
                from blueearth_cst.experiment.simulation_runner import (
                    validate_retained_indicator_sets,
                )

                validate_retained_indicator_sets(
                    root / "experiments" / context.name, command
                )
            if not preview and settings["operation"] == "simulate-and-indicators":
                context.bind_selection()
                capture = capture_simulation_sources_v2(
                    args.config, root, record["invocation_id"]
                )
                environment["CST_SIMULATION_CAPTURE"] = str(capture)
                record["configuration"]["archive_state"] = "pending"
            else:
                record["configuration"]["archive_state"] = "not_applicable"
            environment.update(context.environment())
            environment["CST_SIMULATION_INVOCATION_ID"] = record["invocation_id"]
            started = record_path.with_suffix(".snakemake-started")
            environment["CST_SIMULATION_STARTED"] = str(started)
            environment["CST_SIMULATION_FINALIZE"] = "1"
            if record.get("resources", {}).get("directory"):
                environment[resource_tracking.DIRECTORY_ENV] = record["resources"][
                    "directory"
                ]
            invocation_history.update(record_path, record)
            print("WF4 targets: " + " ".join(args.target), flush=True)
            resource_tracking.phase(record_path, "snakemake")
            result = run_project_child(
                command,
                env=environment,
                cwd=Path(__file__).resolve().parents[1],
                writing=not preview,
            )
            resource_tracking.phase(record_path, "finalization")
            if (
                not preview
                and (result == 0 or started.is_file())
                and selection.indicators
                and current_targets(args.target) != ["simulations_only"]
            ):
                from blueearth_cst.experiment.indicator_export import (
                    publish_selected_sets,
                    report_publication,
                )

                # Per-set completion, still under project execution ownership:
                # every ready set publishes even when another set's job failed,
                # and the invocation is nonzero unless every set is current.
                # A parse/DAG refusal started no jobs and must not attempt exports.
                experiment_root = root / "experiments" / context.name
                resource_tracking.phase(record_path, "indicator_exports")
                outcomes = publish_selected_sets(
                    experiment_root, project, settings, selection
                )
                record["indicator_set_outcomes"] = outcomes
                if not report_publication(experiment_root, outcomes) and result == 0:
                    result = 1
        invocation_history.finish(record_path, record, exit_code=result)
        if not preview:
            sys.stdout.flush()
            summary = run_summary(
                "wf4 simulate_system",
                root,
                None,
                None,
                elapsed_seconds=time.monotonic() - console_started,
                failed=result != 0,
                log_parts_dir=str(
                    directory / "_parts" / f"wf4-{record_path.stem}" / "logs"
                ),
                warnings=warning_count(),
            )
            sys.stderr.write("\n" + summary + "\n")
            if result == 0:
                sys.stderr.write(f"{datetime.now():%H:%M:%S} - DONE all\n")
            sys.stderr.flush()
        return result
    except ValueError as exc:
        invocation_history.finish(record_path, record, exit_code=None, error=exc)
        if args.traceback:
            raise
        parser.error(str(exc))
    except BaseException as exc:
        invocation_history.finish(record_path, record, exit_code=None, error=exc)
        raise
    finally:
        if previous_tally is None:
            os.environ.pop(tally_key, None)
        else:
            os.environ[tally_key] = previous_tally


if __name__ == "__main__":
    raise SystemExit(main())
