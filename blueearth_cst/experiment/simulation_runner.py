"""Shared operation/target contract for the two public simulation runners."""

import os
import sys
import uuid
from pathlib import Path

import yaml


def simulation_settings(config_path):
    """Read the simulation file without opening generation or model files.

    The named interface (``operations``/``indicator_sets``) is validated and
    normalized to the internal ``operation`` the Snakemake branches select on:
    ``indicators-only`` for indicators alone, otherwise ``simulate-and-indicators``.
    """
    project, settings, _ = simulation_selection(config_path)
    return project, settings


def simulation_selection(config_path):
    """Return ``(project, settings, selection)`` for either WF4 interface."""
    from blueearth_cst.experiment import indicator_sets

    path = Path(config_path).resolve()
    project = yaml.safe_load(path.read_text(encoding="utf-8"))
    from blueearth_cst.shared.config_paths import (
        resolve_project_paths,
        resolve_workflow_paths,
    )

    project = resolve_project_paths(project, path)
    workflows = project.get("workflows", {})
    if "run_stress_test" in workflows:
        raise ValueError(
            "run_stress_test is retired; migrate to generate_scenarios and "
            "simulate_system with scripts/migrate_project_config.py from "
            "toolbox v0.3.0"
        )
    stanza = workflows.get("simulate_system", {})
    if set(stanza) != {"enabled", "config_path"} or not isinstance(
        stanza["enabled"], bool
    ):
        raise ValueError("simulate_system requires a closed enabled/config_path stanza")
    if stanza["enabled"] is False:
        raise ValueError(
            f"Workflow simulate_system is disabled in {path}. "
            "Set workflows.simulate_system.enabled to true in that "
            "project configuration file, then rerun the command."
        )
    workflow_path = path.parent / stanza["config_path"]
    settings = (
        indicator_sets.load_unique_yaml(workflow_path.read_text(encoding="utf-8")) or {}
    )
    from blueearth_cst.shared.config_composition import normalize_indicator_settings

    settings = resolve_workflow_paths(
        settings, workflow_path, "simulate_system", project.get("schema_version", 2)
    )
    settings = normalize_indicator_settings(settings)
    from blueearth_cst.experiment.simulation_runs import include_unperturbed

    include_unperturbed(settings)
    if indicator_sets.uses_named_selection(settings):
        selection = indicator_sets.workflow_selection(settings, workflow_path)
        operation = (
            "indicators-only"
            if selection.operations == ("indicators",)
            else "simulate-and-indicators"
        )
        return project, {**settings, "operation": operation}, selection
    if settings.get("operation") not in {"simulate-and-indicators", "indicators-only"}:
        raise ValueError(
            "simulate_system.operation must be simulate-and-indicators or indicators-only; "
            "or use operations: [simulations, indicators]"
        )
    return project, settings, indicator_sets.workflow_selection(settings, workflow_path)


#: Target names retired 2026-09-24, accepted for one release and rewritten to
#: the current rule name before Snakemake sees them.
LEGACY_TARGETS = {
    "responses": "simulations_only",
    "indicators": "simulations_and_indicators",
    "metrics": "simulations_and_indicators",
}


def current_targets(targets):
    """``targets`` with any retired target name replaced by its current one."""
    return [LEGACY_TARGETS.get(target, target) for target in targets]


def check_required_variables(
    sets, available, where, *, evaluation="strict", locations=None
):
    """Refuse (strict) or return (partial) selected indicators lacking inputs.

    ``available`` names the response variables. Set-declared absent parameters
    are checked at ``locations`` (variable -> retained ids) when known, else only
    where absence is location-independent. Each gap names the set, indicator,
    missing inputs and a full description; never schedules a simulation.
    """
    from blueearth_cst.indicators import loader
    from blueearth_cst.shared.run_log_core import plural

    missing = []
    for item in sets:
        for indicator in item.indicators:
            label = f"{item.set_id}:{indicator.indicator} ({indicator.method})"
            declaration = indicator.declaration
            absent = [
                requirement.variable
                for requirement in declaration.requirements
                if requirement.variable not in available
            ]
            gaps = (
                ()
                if absent or indicator.pending_baseline
                else loader.unavailable(
                    declaration, (locations or {}).get(declaration.table_variable)
                )
            )
            if absent:
                text = f"{label} needs response {', '.join(absent)}"
                inputs = absent
            elif gaps:
                where_absent = (
                    "every location"
                    if any(location is None for location, _ in gaps)
                    else plural(len(gaps), "location")
                )
                reasons = sorted({reason for _, reason in gaps})
                # Reason codes are ``missing_<kind>:<input>``; rows name the input.
                inputs = [reason.split(":", 1)[-1] for reason in reasons]
                text = f"{label} lacks {', '.join(reasons)} at {where_absent}"
            else:
                continue
            missing.append(
                {
                    "set_id": item.set_id,
                    "indicator": indicator.indicator,
                    "inputs": inputs,
                    "text": text,
                    "kind": "response" if absent else "parameter",
                }
            )
    if missing and evaluation == "strict":
        remedies = []
        if all(gap["kind"] == "parameter" for gap in missing):
            where = "the selected indicator parameters"
        if any(gap["kind"] == "response" for gap in missing):
            remedies.append(
                "Supply the missing response through model.outvars in a separate simulation (--new-experiment)."
            )
        if any(gap["kind"] == "parameter" for gap in missing):
            remedies.append(
                "Supply the missing indicator parameter (for Qcrit, thresholds)."
            )
        raise ValueError(
            f"selected indicators lack inputs in {where}: "
            + "; ".join(gap["text"] for gap in missing)
            + ". "
            + " ".join(remedies)
            + " Alternatively exclude the indicator with indicator_selection, or set "
            "indicator_evaluation: partial."
        )
    return missing


def report_unavailable(gaps):
    """One warning row per set naming what a partial invocation cannot calculate."""
    from blueearth_cst.shared.run_log_core import log_row, plural
    from blueearth_cst.shared.snake_utils import listed

    by_set = {}
    for gap in gaps:
        by_set.setdefault(gap["set_id"], []).append(gap)
    for set_id, items in by_set.items():
        inputs = sorted({value for gap in items for value in gap["inputs"]})
        log_row(
            f"{set_id}: inputs missing ({', '.join(inputs)}); "
            f"{plural(len(items), 'indicator')} will be published as unavailable: "
            f"{listed([gap['indicator'] for gap in items], limit=4)}",
            level="WARNING",
        )


def simulation_indicator_sets(
    project, settings, selection, *, report=False, prepare_baseline=True
):
    """Resolve sets for the simulation path and preflight model outputs.

    ``report`` logs partial-mode gaps once, from the runner rather than every
    Snakemake parse.
    """
    from blueearth_cst.experiment import indicator_sets
    from blueearth_cst.shared.indicator_tables import indicator_tables
    from blueearth_cst.shared.snake_utils import DEFAULT_WFLOW_OUTVARS

    sets = indicator_sets.selected_sets(
        project,
        settings,
        selection,
        simulation_path=True,
        retained_variables=lambda: [],
        legacy_context="simulation_rule",
        prepare_baseline=prepare_baseline,
    )
    if not selection.legacy:
        outvars = (project.get("model") or {}).get("outvars", DEFAULT_WFLOW_OUTVARS)
        gaps = check_required_variables(
            sets,
            set(indicator_tables(outvars)),
            "the configured model.outvars",
            evaluation=selection.evaluation,
        )
        if report:
            report_unavailable(gaps)
    from blueearth_cst.experiment.simulation_runs import include_unperturbed

    if not include_unperturbed(settings):
        dependent = [
            f"{item.set_id}:{indicator.indicator}"
            for item in sets
            for indicator in item.indicators
            if indicator.declaration.reference == "unperturbed_runs"
        ]
        if dependent:
            raise ValueError(
                "include_unperturbed: false cannot calculate indicators requiring "
                "unperturbed reference runs: "
                + ", ".join(dependent)
                + ". Select an indicator set without these indicators, use "
                "operations: [simulations], or set include_unperturbed: true."
            )
    return sets


def resolve_targets(config_path, targets, *, prepare_baseline=True):
    """Return ``(operation, targets)``, refusing operation bypass before Snakemake.

    The named interface infers its single target; legacy mode keeps its
    explicit-target validation.
    """
    project, settings, selection = simulation_selection(config_path)
    operation = settings["operation"]
    if not selection.legacy:
        if current_targets(targets) != ["all"]:
            raise ValueError(
                "UnsupportedOperationTarget: operations/indicator_sets infer the "
                "target; omit --target"
            )
        if selection.indicators:
            if selection.simulations:
                simulation_indicator_sets(
                    project,
                    settings,
                    selection,
                    report=True,
                    prepare_baseline=prepare_baseline,
                )
            else:
                from blueearth_cst.experiment.indicator_plan import (
                    indicators_only_configuration,
                )
                from blueearth_cst.experiment.simulation_record import (
                    read_simulation_intent_v2,
                )

                root, sets, _ = indicators_only_configuration(
                    config_path, prepare_baseline=prepare_baseline
                )
                request = read_simulation_intent_v2(root)["documents"][
                    "response_request"
                ]
                report_unavailable(
                    check_required_variables(
                        sets,
                        {item["variable"] for item in request["variables"]},
                        "the retained simulation responses",
                        evaluation=selection.evaluation,
                        locations={
                            item["variable"]: [str(v) for v in item["locations"]]
                            for item in request["variables"]
                        },
                    )
                )
        return operation, ["all" if selection.indicators else "simulations_only"]
    supported = (
        "simulate-and-indicators + all/simulations_only; "
        "indicators-only + simulations_and_indicators or one selected indicator output"
    )
    if len(targets) != 1:
        raise ValueError(f"UnsupportedOperationTarget: {supported}")
    target = current_targets(targets)[0]
    if operation == "simulate-and-indicators":
        allowed = target in {"all", "simulations_only"}
    else:
        allowed = target == "simulations_and_indicators" or (
            Path(target).resolve().as_posix() in selected_output_paths(config_path)
        )
    if not allowed:
        raise ValueError(f"UnsupportedOperationTarget: {supported}")
    return operation, [target]


def selected_output_paths(config_path):
    """Every file an indicators-only invocation's selected plans name."""
    from blueearth_cst.experiment.indicator_plan import (
        build_indicator_plan,
        current_indicator_request,
        indicators_only_configuration,
    )
    from blueearth_cst.experiment.indicator_sets import calculation_records

    root, sets, anchor = indicators_only_configuration(config_path)
    paths = set()
    for item in sets:
        plan = build_indicator_plan(
            root,
            current_indicator_request(
                root, calculation_records(item.indicators), anchor
            ),
        )
        paths |= {*plan["targets"].values(), *plan["tables"].values()}
    return paths


def validate_targets(config_path, targets):
    """Refuse operation bypass before Snakemake can build any DAG."""
    return resolve_targets(config_path, targets)[0]


def simulation_command(
    config_path, targets, cores, extra=(), *, snapshot_path=None, prepare_baseline=True
):
    """Build one checked invocation; callers control lifecycle reporting.

    ``snapshot_path`` receives the admitted indicator-set sources so every
    Snakemake process resolves the same bytes.
    """
    operation, targets = resolve_targets(
        config_path, targets, prepare_baseline=prepare_baseline
    )
    _, _, selection = simulation_selection(config_path)
    extra = list(extra)
    # Independent sets: one set's refused reduction must not stop the others.
    if len(selection.sets) > 1 and "--keep-going" not in extra:
        extra.append("--keep-going")
    # Target selection and config belong to this runner. Passthrough is limited
    # to execution controls so a second target cannot bypass the check above.
    switches = {
        "--dry-run",
        "-n",
        "--forceall",
        "-F",
        "--printshellcmds",
        "-p",
        "--keep-going",
        "--rerun-incomplete",
        "--notemp",
        "--unlock",
    }
    value_switches = {
        "--resources",
        "--set-resources",
        "--set-threads",
        "--forcerun",
        "--rerun-triggers",
    }
    index = 0
    while index < len(extra):
        option = extra[index]
        index += 1
        if option in switches:
            continue
        if option not in value_switches:
            raise ValueError(
                "unsupported simulation execution option; target/config overrides are refused"
            )
        start = index
        while index < len(extra) and not extra[index].startswith("-"):
            index += 1
        if index == start:
            raise ValueError(f"{option} requires a value")
    repo = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "snakemake",
        *current_targets(targets),
        "-s",
        str(repo / "simulate_system.smk"),
        "--configfile",
        str(Path(config_path).resolve()),
        "-c",
        str(cores),
        *extra,
    ]
    environment = {
        **os.environ,
        "CST_SIMULATION_OPERATION": operation,
        "CST_SIMULATION_TARGET": current_targets(targets)[0],
        "CST_SIMULATION_INVOCATION_ID": uuid.uuid4().hex,
    }
    if snapshot_path is not None:
        from blueearth_cst.experiment import indicator_sets

        environment[indicator_sets.SNAPSHOT_ENV] = str(
            indicator_sets.write_selection_snapshot(selection, Path(snapshot_path))
        )
    return command, environment


def resolve_selected_collection(config_path, repository):
    """Consume an exact ready collection; never invoke its producers."""
    selection, marker, _ = resolve_collection_with_plan(config_path, repository)
    return selection, marker


def resolve_collection_with_plan(config_path, repository):
    """Return ready selection, marker and its validated generation plan reference.

    Explicit manifests have no generation plan binding. Project selections retain
    the exact plan checked here so experiment admission can bind its seed without
    rebuilding a candidate or rereading the mutable request pointer.
    """
    from blueearth_cst.experiment.collection_resolution import (
        GeneratedCollectionUnavailable,
        resolve_explicit_collection_v2,
    )
    from blueearth_cst.experiment.content_identity import (
        content_sha256,
        read_canonical_json,
    )
    from blueearth_cst.experiment.generation_plan import (
        generation_configuration,
        read_pinned_plan,
    )
    from blueearth_cst.experiment.scenario_collection_v2 import read_collection_v2
    from blueearth_cst.shared.config_composition import compose_config

    project, settings = simulation_settings(config_path)
    if "scenario_collection" in settings:
        selection, marker = resolve_explicit_collection_v2(
            settings["scenario_collection"]
        )
        verify_selected_collection_source(project, selection, marker)
        return selection, marker, None
    composed, _ = compose_config(
        project,
        config_path,
        entry="generate_scenarios",
        declared_sections=(
            "project",
            "basin",
            "climate",
            "workflows.generate_scenarios",
        ),
    )
    generation = generation_configuration(composed, repository)
    command = f'python scripts/run_workflows.py --config "{config_path}" --project-dir "{generation["project_dir"]}"'
    pointer_path = Path(generation["request_path"])
    if not pointer_path.is_file():
        raise GeneratedCollectionUnavailable(
            f"missing {generation['request_path']}; run {command}"
        )
    try:
        pointer = read_canonical_json(pointer_path)
        if (
            set(pointer)
            != {
                "schema_version",
                "generation_request_id",
                "plan_path",
                "path_base",
                "plan_sha256",
            }
            or pointer["schema_version"] != "scenario-request/2"
            or pointer["path_base"] != "request_directory"
        ):
            raise ValueError("unsupported scenario request pointer")
        if Path(pointer["plan_path"]).name != pointer["plan_path"]:
            raise ValueError("scenario plan escapes request directory")
        plan_path = pointer_path.parent / pointer["plan_path"]
        plan = read_pinned_plan(
            plan_path,
            pointer["plan_sha256"],
            Path(generation["project_dir"]),
        )
        if pointer["generation_request_id"] != plan["generation_request_id"] or pointer[
            "generation_request_id"
        ] != content_sha256(generation["request"]):
            raise ValueError("project request pointer differs from pinned plan")
        marker_path = (
            Path(generation["project_dir"])
            / plan["outputs"]["record_root"]
            / "collection.json"
        )
        marker = read_collection_v2(marker_path)
        if marker["collection_id"] != plan["collection_id"]:
            raise ValueError("pinned collection identity differs")
        if (
            plan["decision"] == "reuse_ready"
            and marker["collection_revision"] != plan["collection_revision"]
        ):
            raise ValueError("pinned collection revision differs")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        from blueearth_cst.experiment.collection_resolution import (
            GeneratedCollectionStale,
        )

        raise GeneratedCollectionStale(f"{pointer_path}: {exc}; run {command}") from exc
    selection = {
        "resolution_mode": "project-generation",
        "manifest_path": marker_path.resolve().as_posix(),
        "collection_id": marker["collection_id"],
        "collection_revision": marker["collection_revision"],
    }
    verify_selected_collection_source(project, selection, marker)
    from blueearth_cst.shared.workflow_config_snapshot import file_reference

    plan_reference = file_reference(
        plan_path, "project_root", generation["project_dir"]
    )
    if plan_reference["sha256"] != pointer["plan_sha256"]:
        raise ValueError("generation plan changed after selection")
    return selection, marker, plan_reference


def verify_selected_collection_source(project, selection, marker):
    """Bind WF4 elevation/PET choice to the collection creator's climate source."""
    from blueearth_cst.shared.workflow_config_snapshot import (
        resolve_file_reference,
        validate_archive,
    )

    path = Path(selection["manifest_path"]).resolve(strict=True)
    record_dir = path.parent
    project_root = record_dir.parents[3]
    configured_root = Path(project["project"]["project_dir"]).resolve()
    if project_root != configured_root:
        raise ValueError("selected collection is outside the WF4 project root")
    archive_path = resolve_file_reference(
        marker["archive"],
        {"project_root": project_root, "record_directory": record_dir},
    )
    archive = validate_archive(archive_path.parent)
    if archive["owner"] != {
        "kind": "scenario_collection",
        "id": marker["collection_id"],
    }:
        raise ValueError("collection creator archive owner differs")
    selected = archive["loaded_config"]["climate"]["selected"]
    if selected != project["climate"]["selected"]:
        raise ValueError(
            f"WF4 climate source {project['climate']['selected']!r} differs from "
            f"collection creator source {selected!r}"
        )
    return selected


FORCE_OPTIONS = ("--forceall", "-F", "--forcerun", "-R", "--forcetargets", "-f")


def validate_retained_indicator_sets(experiment_root, extra):
    """Before a forced run, refuse if any retained indicator set is not intact.

    A valid set's producer returns it unchanged, so it cannot fail and trigger
    Snakemake's failed-job cleanup of its declared outputs; an invalid one is
    reported here, before any forced producer can touch the retained files.
    """
    if not any(option in FORCE_OPTIONS for option in extra):
        return
    from blueearth_cst.experiment.indicator_plan import read_indicator_set
    from blueearth_cst.experiment.legacy_indicator_names import saved_set_markers

    root = Path(experiment_root)
    for marker in saved_set_markers(root):
        read_indicator_set(root, marker)
