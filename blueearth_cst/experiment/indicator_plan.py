"""Response-dependent indicator identities, complete keys and immutable result sets."""

import csv
import io
import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from blueearth_cst.experiment import gev_lmoments, return_level_validation
from blueearth_cst.experiment.content_identity import (
    canonical_json_bytes,
    content_sha256,
    identity_segment,
    read_canonical_json,
    repository_code_inventory,
)
from blueearth_cst.experiment.indicator_artifacts import (
    ImmutableIndicatorSetError,
    _validate_tables,
    read_indicator_set,
)
from blueearth_cst.experiment.indicator_groups import indicator_groups
from blueearth_cst.experiment.indicator_registry import (
    DeclaredBundle,
    indicator_unit_index,
    resolve_month_reference,
    series_coverage,
)
from blueearth_cst.experiment.indicator_sets import declarations_for_calculations
from blueearth_cst.experiment.response_inventory import read_response_inventory_v2
from blueearth_cst.experiment.scenario_rows import ScenarioRow
from blueearth_cst.experiment.simulation_record import (
    IndicatorsOnlySimulationUnavailable,
    atomic_record,
    read_simulation_intent_v2,
    read_simulation_v2,
)
from blueearth_cst.experiment.wflow_response_reader import (
    NativeRunArtifacts,
    ResponseRequest,
    open_responses,
)
from blueearth_cst.indicators import loader
from blueearth_cst.shared.snake_utils import log_row, plural

#: The published return-level provenance beside each v2 indicator set's marker.
RETURN_LEVEL_EVIDENCE_FILENAME = "return_level_evidence.json"
RETURN_LEVEL_EVIDENCE_SCHEMA = "return-level-evidence/1"
#: Per-series block coverage beside each indicator-set/4 marker; diagnostic
#: evidence bound by the response digest, not an identity input.
COVERAGE_EVIDENCE_FILENAME = "coverage_evidence.json"
COVERAGE_EVIDENCE_SCHEMA = "coverage-evidence/1"


class IndicatorPlanStale(ValueError):
    """A scheduling plan cannot certify the requested live inputs or responses."""


class IndicatorsOnlyIdentityMismatch(ValueError):
    """An explicit collection assertion differs from the retained simulation."""


def _require_v2(root):
    """Require a completed v2 simulation before reading retained responses.

    v1 simulation records existed only between 2026-09-11 and 2026-09-22, and
    no release wrote one, so this toolbox reads v2 alone.
    """
    if (
        not (root / "_engine/simulation.json").is_file()
        and (root / "config/simulation.json").is_file()
    ):
        raise IndicatorsOnlySimulationUnavailable(
            f"{root.name}: written in the pre-release v1 simulation record "
            "format, which this toolbox no longer reads; simulate it again "
            "under a new experiment name"
        )
    if not (root / "_engine/simulation.json").is_file():
        raise IndicatorsOnlySimulationUnavailable(
            f"No completed simulation is available for experiment {root.name!r}: "
            f"missing {root / '_engine/simulation.json'}. "
            "Indicators-only operations require retained simulation responses, "
            "including for a dry run. Set operations: [simulations, indicators] "
            "to run a simulation first, or select the experiment_name of an "
            "existing completed simulation."
        )


def _plain(value):
    """Convert known dataclass tuple fields to the persisted JSON array dialect."""
    return json.loads(json.dumps(value, allow_nan=False))


def indicator_anchor(project: dict) -> str:
    """The water-year resample anchor every indicator request uses.

    One owner, so the planner, both Snakemake branches and post-run
    publication cannot disagree. Bound through
    ``snake_utils.water_year_end_anchor`` (``YE-DEC`` for January, ``YE-SEP``
    for October): for midnight daily labels its right-closed bins hold the
    same days as the former ``YS-*`` anchor, so values and result keys are
    unchanged (design v3 O2 parity evidence); only the anchor recorded in the
    request, and so the indicator-set identity, differs.
    """
    from blueearth_cst.shared.snake_utils import (
        resolve_water_year_start,
        water_year_end_anchor,
    )

    climate = project.get("climate") or {}
    return water_year_end_anchor(
        resolve_water_year_start(climate.get("water_year_start"))
    )


def indicators_only_configuration(config_path, *, prepare_baseline=True):
    """Read only the selected experiment and indicator settings.

    This projection never opens generation or model workflow files.
    """
    import yaml

    from blueearth_cst.experiment.indicator_sets import load_unique_yaml
    from blueearth_cst.shared.snake_utils import validate_experiment_name

    path = Path(config_path).resolve()
    project = yaml.safe_load(path.read_text(encoding="utf-8"))
    from blueearth_cst.shared.config_paths import (
        resolve_project_paths,
        resolve_workflow_paths,
    )

    project = resolve_project_paths(project, path)
    descriptor = project["workflows"]["simulate_system"]
    if set(descriptor) != {"enabled", "config_path"} or not isinstance(
        descriptor["enabled"], bool
    ):
        raise ValueError(
            "indicators-only requires an enabled simulate_system config_path"
        )
    workflow = load_unique_yaml(
        (path.parent / descriptor["config_path"]).read_text(encoding="utf-8")
    )
    from blueearth_cst.shared.config_composition import normalize_indicator_settings

    workflow = resolve_workflow_paths(
        workflow,
        path.parent / descriptor["config_path"],
        "simulate_system",
        project.get("schema_version", 2),
    )
    workflow = normalize_indicator_settings(workflow)
    project_root = Path(project["project"]["project_dir"]).resolve()
    from blueearth_cst.experiment.reservation_context import resolve_experiment_name

    name = validate_experiment_name(
        resolve_experiment_name(project_root, workflow.get("experiment_name")),
        project_root,
    )
    root = project_root / "experiments" / name
    _require_v2(root)
    read_simulation_v2(root)
    intent_v2 = read_simulation_intent_v2(root)
    retained_collection = intent_v2["collection"]
    if "scenario_collection" in workflow:
        asserted = workflow["scenario_collection"]
        retained = retained_collection
        for field in ("collection_id", "collection_revision"):
            if asserted.get(field) not in (None, retained[field]):
                raise IndicatorsOnlyIdentityMismatch(
                    f"{field}: retained={retained[field]!r}; asserted={asserted[field]!r}"
                )
    from blueearth_cst.experiment import indicator_sets

    workflow_path = path.parent / descriptor["config_path"]
    selection = indicator_sets.workflow_selection(workflow, workflow_path)

    def retained_variables():
        request = intent_v2["documents"]["response_request"]
        return [item["variable"] for item in request["variables"]]

    sets = indicator_sets.selected_sets(
        project,
        workflow,
        selection,
        simulation_path=False,
        retained_variables=retained_variables,
        legacy_context="indicators_only",
        prepare_baseline=prepare_baseline,
    )
    climate = project.get("climate") or {}
    anchor = indicator_anchor(project)
    used = {"experiment_name", "indicators", "scenario_collection"}
    if not selection.legacy:
        used |= set(indicator_sets.SELECTION_KEYS)
    ignored = [f"simulate_system.{key}" for key in sorted(workflow) if key not in used]
    ignored += [
        f"climate.{key}" for key in sorted(climate) if key != "water_year_start"
    ]
    outvars_used = (
        workflow.get("indicators") is None
        if selection.legacy
        else any(item.source.set_id == indicator_sets.DEFAULT_SET for item in sets)
    )
    ignored += [
        f"model.{key}"
        for key in sorted(project.get("model") or {})
        if key != "outvars" or not outvars_used
    ]
    if ignored:
        print(
            f"Indicators-only uses retained simulation and generation inputs -> {(root / '_engine/simulation.json').as_posix()}"
            + f", {retained_collection.get('manifest_path', retained_collection.get('manifest', {}).get('path', 'retained collection'))}"
            + "; current settings not used: "
            + ", ".join(ignored),
            flush=True,
        )
    return root, sets, anchor


#: Indicator-stage dependency roots. lmoments3 joins them because the return-level
#: estimator is now part of what this stage's identity must describe (D5).
INDICATOR_ENVIRONMENT_ROOTS = (
    "numpy",
    "pandas",
    "scipy",
    "xarray",
    "xclim",
    "netCDF4",
    "pyproj",
    "PyYAML",
    "lmoments3",
)


def resolve_indicator_environment():
    """Observe the LIVE indicator-stage environment, including D7 source hashes.

    Always a fresh observation. It is never derived from a retained request, a
    cached planning descriptor or lock-file intent, because the whole point is
    to notice that the executing environment has drifted from the planned one.
    """
    from blueearth_cst.experiment.content_identity import stage_environment

    environment = stage_environment(list(INDICATOR_ENVIRONMENT_ROOTS))
    return {
        **environment,
        "return_level_estimator": {
            "estimator_id": gev_lmoments.ESTIMATOR_ID,
            "package": "lmoments3",
            "version": gev_lmoments.DEPENDENCY_VERSION,
            "source_sha256": dict(sorted(gev_lmoments.observed_source().items())),
        },
    }


def current_indicator_request(experiment_root, calculations, anchor):
    """Resolve stage-three identity without any live source/model or Julia access."""
    return current_indicator_requests(experiment_root, [calculations], anchor)[0]


def current_indicator_requests(experiment_root, calculation_sets, anchor):
    """Resolve a batch against one fresh simulation/code/environment observation.

    Callers may retain these identities for DAG scheduling within one invocation.
    Execution still checks live code, responses and environment independently.
    No observation is cached across calls to this function.
    """
    if not calculation_sets:
        return []
    root = Path(experiment_root).resolve()
    # Combined runs resolve request identities after freezing the intent, before
    # publishing responses. Planning and indicators-only entry still require a
    # completed simulation; request naming only needs the checked frozen intent.
    if not (root / "_engine/simulation_intent.json").is_file():
        _require_v2(root)
    if (root / "_engine/simulation.json").is_file():
        record = read_simulation_v2(root)
        schema = record["schema_version"]
    elif (root / "_engine/simulation_intent.json").is_file():
        # The v2 freeze has committed but responses have not published
        # _engine/simulation.json yet -- the DAG-build-time evaluation this
        # feeds (checkpoint prepare_indicator_plan, rule all's target lambda)
        # runs long before that, so the request's identity comes from the
        # frozen INTENT rather than waiting on a completed simulation.
        #
        # simulation_schema_version is pinned to the "simulation/3" literal
        # (matching build_indicator_plan), never the intent's
        # own "simulation-intent/2" tag -- the checkpoint wildcard is fixed
        # from this request before responses complete, and params.request is
        # re-resolved (this time via the v2-complete branch above) once the
        # job actually runs, so the two resolutions must hash identically.
        record = read_simulation_intent_v2(root)
        schema = "simulation/3"
    else:
        raise IndicatorsOnlySimulationUnavailable(
            f"{root.name}: no frozen simulation intent; simulate the experiment first"
        )
    environment = resolve_indicator_environment()
    validation = return_level_validation.build_declaration()
    response_request = read_simulation_intent_v2(root)["documents"]["response_request"]
    code_by_sets = {}
    requests = []
    for calculations in calculation_sets:
        available, _, _ = split_availability(calculations, response_request)
        contributing = _calculation_sets(declarations_for_calculations(available))
        if contributing not in code_by_sets:
            # A request with nothing calculable observes no calculation code.
            code_by_sets[contributing] = (
                _indicator_code_inventory(contributing) if contributing else []
            )
        request = indicator_request(
            record["simulation_id"],
            calculations,
            anchor,
            environment,
            validation,
            response_request,
            code_inventory=code_by_sets[contributing],
        )
        request["simulation_schema_version"] = schema
        requests.append(request)
    return requests


def split_availability(calculations, response_request):
    """Split selected records into calculable ones and explicit unavailable entries.

    An indicator needing a response the simulation never requested is dropped
    whole; a set-declared absent input (``loader.unavailable``) marks locations
    unavailable while the indicator stays calculable elsewhere. Returns sorted
    ``(calculations, unavailable_calculations, unavailable_results)``.
    """
    locations = {
        item["variable"]: item["locations"] for item in response_request["variables"]
    }
    calculations = sorted(calculations, key=lambda record: record["indicator"])
    available, dropped, results = [], [], []
    for record, declaration in zip(
        calculations, declarations_for_calculations(calculations), strict=True
    ):
        missing = sorted(
            {item.variable for item in declaration.requirements} - set(locations)
        )
        if missing:
            dropped.append(
                {
                    "calculation": record,
                    "reasons": [f"missing_response:{item}" for item in missing],
                }
            )
            continue
        available.append(record)
        retained = [str(item) for item in locations[declaration.table_variable]]
        for location, reason in loader.unavailable(declaration, retained):
            for item in retained if location is None else [location]:
                results.append(
                    {"indicator": declaration.name, "location": item, "reason": reason}
                )
    results.sort(key=lambda row: (row["indicator"], row["location"], row["reason"]))
    return available, dropped, results


def selected_calculations(request):
    """Every selected record of a request or manifest, calculable or unavailable.

    Manifests before indicator-set/6 have no unavailable calculations.
    """
    return sorted(
        [
            *request.get("calculations", []),
            *(
                item["calculation"]
                for item in request.get("unavailable_calculations", [])
            ),
        ],
        key=lambda record: record["indicator"],
    )


def indicator_request(
    simulation_id,
    calculations,
    anchor,
    environment,
    validation,
    response_request,
    *,
    code_inventory=None,
):
    """Compute a scheduling request before response values or reference months exist.

    ``calculations`` are the selected set records; they are sorted by indicator key
    so selection order, set ids and presentation never enter numerical identity.
    Availability against the frozen ``response_request`` is part of the request.
    """
    calculations, dropped, unavailable = split_availability(
        calculations, response_request
    )
    registry = declarations_for_calculations(calculations)
    # A NEW request must carry the D4 declaration. Legacy acceptance is a read
    # compatibility guarantee (D6), never a write escape hatch, so the retained
    # pre-C record is refused here rather than silently re-emitted.
    if return_level_validation.is_legacy(validation):
        raise ValueError(
            "the legacy unassessed return-level record cannot be written into a "
            "new indicator request; it is readable, not writable"
        )
    return_level_validation.verify_declaration(
        validation, return_level_validation.load_report_bytes()
    )
    # A request with nothing calculable observes no calculation code.
    code = (
        (_indicator_code_inventory(_calculation_sets(registry)) if registry else [])
        if code_inventory is None
        else code_inventory
    )
    return {
        "schema_version": "indicator-request/5",
        "simulation_id": simulation_id,
        "tokens": sorted({item.table_variable for item in registry}),
        "response_variables": sorted(
            {
                requirement.variable
                for item in registry
                for requirement in item.requirements
            }
        ),
        "calculations": _plain(calculations),
        "unavailable_calculations": _plain(dropped),
        "unavailable_results": unavailable,
        "declarations": [_plain(asdict(item)) for item in registry],
        "grouping": "stochastic/st_id including empty unperturbed key",
        "reference": "unperturbed evaluated runs; first native q location; monthly sum; first month on ties",
        "water_year_anchor": anchor,
        "code_inventory": code,
        "indicator_environment": environment,
        "return_level_validation": validation,
    }


def _calculation_sets(declarations) -> tuple[str, ...]:
    """Resolve actual calculation owners, including legacy definitions."""
    return tuple(
        sorted(
            {loader.local_declaration(item).calculation_set for item in declarations}
        )
    )


def _indicator_code_inventory(set_ids):
    """Observe calculation dependencies; resolved upstream inputs are sealed.

    Snakemake owns scheduling. This inventory guards immutable calculation reuse,
    not invocation logging, configuration capture or upstream implementation.
    """
    repo = Path(__file__).resolve().parents[2]
    indicator_root = repo / "blueearth_cst/indicators"
    sources = ["blueearth_cst/experiment/indicator_plan.py"]
    sources.extend(
        path.relative_to(repo).as_posix() for path in loader.execution_sources(set_ids)
    )
    # Independent checkers have fresh attribution in verification receipts.
    excluded = [
        # Readers still validate retained simulation/collection identities and
        # bytes. Their producers cannot change a calculation over those inputs.
        "blueearth_cst/experiment/simulation_record.py",
        "blueearth_cst/shared/workflow_config_snapshot.py",
        # Selection, experiment and water-year configuration enter the request
        # as resolved values. Changes to those values still change its identity.
        "blueearth_cst/experiment/reservation_context.py",
        "blueearth_cst/shared/config_composition.py",
        # Used for logging, naming and the resolved water-year anchor above.
        # Reducers receive the anchor explicitly; no default is consulted.
        "blueearth_cst/shared/snake_utils.py",
    ]
    excluded.extend(
        path.relative_to(repo).as_posix()
        for path in indicator_root.rglob("*.py")
        if path.name in {"checks.py", "checking.py"}
    )
    return repository_code_inventory(repo, sources, excluded=excluded)


def _collection(simulation):
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    root = Path(simulation["_root"])
    project = root.parent.parent
    path = resolve_file_reference(
        simulation["collection"]["manifest"], {"project_root": project}
    )
    # read_simulation_v2 already validated this complete collection closure.
    # Its content-addressed marker still has to match before reusing metadata.
    collection = read_canonical_json(path)
    intent = read_canonical_json(
        resolve_file_reference(
            collection["intent"],
            {"project_root": project, "record_directory": path.parent},
        )
    )
    with resolve_file_reference(
        collection["scenario_run_lookup"],
        {"project_root": project, "record_directory": path.parent},
    ).open(encoding="utf-8", newline="") as handle:
        rows = _v2_scenario_rows(csv.DictReader(handle))
    return collection, intent, rows


def _v2_scenario_rows(records):
    """Adapt v2's compact scenario lookup to the legacy indicator grouping rows."""
    expected = {"run_id", "evaluate", "type", "rlz", "st_id"}
    values = list(records)
    if any(set(record) != expected for record in values):
        raise IndicatorPlanStale("v2 scenario lookup has unknown fields")
    roots = {}
    for record in values:
        if record["evaluate"] not in {"true", "false"}:
            raise IndicatorPlanStale("v2 scenario lookup evaluate flag is invalid")
        if not record["run_id"] or not record["type"] or not record["rlz"]:
            raise IndicatorPlanStale("v2 scenario lookup has incomplete identifiers")
        if not record["st_id"]:
            if record["rlz"] in roots:
                raise IndicatorPlanStale(
                    "v2 scenario lookup has duplicate realization roots"
                )
            roots[record["rlz"]] = record["run_id"]
    if not roots:
        raise IndicatorPlanStale("v2 scenario lookup has no realization roots")
    rows = []
    for record in values:
        if record["st_id"] and record["rlz"] not in roots:
            raise IndicatorPlanStale(
                "v2 scenario lookup member lacks a realization root"
            )
        rows.append(
            ScenarioRow(
                run_id=record["run_id"],
                derived_from=roots[record["rlz"]] if record["st_id"] else "",
                evaluated=record["evaluate"] == "true",
                scenario_type=record["type"],
                payload=(("rlz", record["rlz"]), ("st_id", record["st_id"])),
            )
        )
    return tuple(rows)


def _native_runs_v2(root, inventory):
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    result = {}
    for artifact in inventory["artifacts"]:
        run = artifact["run_id"]
        series = next(item for item in inventory["series"] if item["run_id"] == run)
        result[run] = NativeRunArtifacts(
            resolve_file_reference(artifact["file"], {"experiment_root": root}),
            resolve_file_reference(
                series["native_selector"]["toml"], {"experiment_root": root}
            ),
            None,
        )
    return result


def build_indicator_plan(experiment_root, request, *, _observed_code=None):
    """Resolve units, references and final identity from complete retained responses.

    ``_observed_code`` is a fresh observation shared only within the caller's
    current planning/publication phase. Execution callers omit it and recheck.
    """
    root = Path(experiment_root).resolve()
    _require_v2(root)
    simulation = read_simulation_v2(root)
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    # Readiness has just checked the full intent, collection and native response
    # closure. Reuse those verified documents, checking their exact bytes against
    # the readiness references instead of walking the same closures again.
    anchors = {"project_root": root.parent.parent}
    intent_v2 = read_canonical_json(
        resolve_file_reference(simulation["intent"], anchors)
    )
    simulation = {
        **simulation,
        "_root": root,
        "collection": intent_v2["collection"],
    }
    if request["simulation_id"] != simulation["simulation_id"]:
        raise IndicatorPlanStale("indicator request selects a different simulation")
    if request.get("schema_version") != "indicator-request/5":
        raise IndicatorPlanStale(
            f"unsupported indicator request {request.get('schema_version')!r}"
        )
    live = indicator_request(
        simulation["simulation_id"],
        selected_calculations(request),
        request["water_year_anchor"],
        request["indicator_environment"],
        request["return_level_validation"],
        intent_v2["documents"]["response_request"],
        code_inventory=_observed_code,
    )
    live["simulation_schema_version"] = "simulation/3"
    if live != request:
        raise IndicatorPlanStale("indicator declarations or invoked code changed")
    inventory = read_canonical_json(
        resolve_file_reference(simulation["response_inventory"], anchors)
    )
    collection, intent, rows = _collection(simulation)
    source_descriptors = [item["descriptor"] for item in collection["series"]]
    if any(
        item["source_calendar"] != inventory["temporal_preparation"]["source_calendar"]
        for item in source_descriptors
    ):
        raise IndicatorPlanStale(
            "response source calendar differs from retained collection"
        )
    specification = intent["scenario_spec"]
    groups, reference_runs, _ = indicator_groups(
        rows,
        n_realizations=specification["n_realizations"],
        st_num=specification["n_design_points"],
        unit_id_capacity=intent["run_group_id_capacity"],
    )
    selected = set(intent_v2["collection"]["run_ids"])
    groups = {
        key: tuple(run for run in members if run in selected)
        for key, members in groups.items()
        if any(run in selected for run in members)
    }
    reference_runs = tuple(run for run in reference_runs if run in selected)
    evaluated_runs = [
        row.run_id for row in rows if row.evaluated and row.run_id in selected
    ]
    registry = declarations_for_calculations(request["calculations"])
    if not reference_runs and any(item.reference for item in registry):
        raise IndicatorPlanStale(
            "Selected indicators require unperturbed reference runs, which were not simulated. "
            "Select an indicator set without those indicators, or simulate a new experiment "
            "with include_unperturbed: true."
        )
    bundles = tuple(
        DeclaredBundle("st_id", key, members) for key, members in groups.items()
    )
    unit_rows = indicator_unit_index(
        [row.run_id for row in rows],
        evaluated_runs,
        bundles,
        unit_id_capacity=intent["run_group_id_capacity"],
    )
    units = _v2_run_groups([asdict(item) for item in unit_rows])
    bundle_ids = {
        bundle.canonical_key: f"group_{number:0{intent['run_group_id_width']}d}"
        for number, bundle in enumerate(
            sorted(bundles, key=lambda item: (item.bundle_by, item.canonical_key)), 1
        )
    }
    frozen_request = intent_v2["documents"]["response_request"]
    locations = {
        item["variable"]: item["locations"] for item in frozen_request["variables"]
    }
    required = {
        requirement.variable for item in registry for requirement in item.requirements
    }
    if not required <= set(locations):
        raise IndicatorPlanStale(
            f"missing retained response requirements: {sorted(required - set(locations))}; a new simulation is required"
        )
    for indicator in registry:
        primary = locations[indicator.table_variable]
        for requirement in indicator.requirements:
            if locations[requirement.variable] != primary:
                raise IndicatorPlanStale(
                    f"{indicator.name}: paired response locations differ for "
                    f"{requirement.variable}"
                )
    references = {}
    if any(item.reference for item in registry):
        native = _native_runs_v2(root, inventory)
        series = tuple(
            item
            for run in reference_runs
            for item in open_responses(run, native[run], ResponseRequest(("q",)))
        )
        references = {
            which: _plain(
                asdict(
                    resolve_month_reference(
                        series, expected_run_ids=reference_runs, which=which
                    )
                )
            )
            for which in ("wet", "dry")
        }
    expected_keys = []
    for indicator in registry:
        ids = list(evaluated_runs) if indicator.grain == "run" else []
        ids += list(bundle_ids.values())
        expected_keys.extend(
            [indicator.name, location, unit]
            for unit in ids
            for location in locations[indicator.table_variable]
        )
    expected_keys.sort()
    definition = {
        "declarations": request["declarations"],
        "unavailable_calculations": request["unavailable_calculations"],
        "unavailable_results": request["unavailable_results"],
        "grouping": request["grouping"],
        "reference": request["reference"],
        "resolved_references": references,
        "water_year_anchor": request["water_year_anchor"],
        "code_inventory": request["code_inventory"],
        "return_level_validation": request["return_level_validation"],
        "group_summary": "equal-weight run reductions; NaN propagation; fixed Class-C reference; pooled return levels",
    }
    definition_digest, environment_digest = (
        content_sha256(definition),
        content_sha256(request["indicator_environment"]),
    )
    identity_projection = {
        "simulation_id": simulation["simulation_id"],
        "response_identity_sha256": inventory["response_identity_sha256"],
        "indicator_definition_sha256": definition_digest,
        "indicator_environment_sha256": environment_digest,
        "groups": {key: list(value) for key, value in groups.items()},
        "schema_version": "indicator-set-identity/5",
        "run_groups": units,
        "reference": references,
        "water_year_anchor": request["water_year_anchor"],
        "return_level_validation_sha256": content_sha256(
            request["return_level_validation"]
        ),
    }
    identity = content_sha256(identity_projection)
    engine = (
        root / "_engine/indicator_sets" / identity_segment(identity, "indicator_set_id")
    )
    # indicator-set/4 data always lives under its engine root; legacy /3
    # results/<digest>/ roots are never adopted by a new request.
    destination = engine / "data"
    plan = {
        "schema_version": "indicator-request/5",
        "indicator_request_id": content_sha256(request),
        "request": request,
        "response_inventory_sha256": inventory["response_inventory_sha256"],
        "resolved_references": references,
        "units": units,
        "expected_result_keys": expected_keys,
        "groups": {key: list(value) for key, value in groups.items()},
        "bundle_run_group_ids": bundle_ids,
        "indicator_definition": definition,
        "indicator_definition_sha256": definition_digest,
        "indicator_environment_sha256": environment_digest,
        "identity_projection": identity_projection,
        "indicator_set_id": identity,
        "targets": {
            "manifest": (
                root
                / "_engine/indicator_sets"
                / identity_segment(identity, "indicator_set_id")
                / "indicators.json"
            ).as_posix(),
            "unit_index": (destination / "group_members.csv").as_posix(),
            "group_metadata": (destination / "group_metadata.csv").as_posix(),
            "environment": (
                root
                / "_engine/indicator_sets"
                / identity_segment(identity, "indicator_set_id")
                / "indicator_environment.json"
            ).as_posix(),
            "benchmark": (engine / return_level_validation.REPORT_FILENAME).as_posix(),
        },
        # Written by the same job as `targets` but not declared to Snakemake:
        # their number varies with the selection, while rule outputs are fixed.
        "tables": {
            f"{token}_{scope}": (
                destination / f"{token}_{scope}_indicators.csv"
            ).as_posix()
            for token in request["tokens"]
            for scope in ("run", "group")
        },
    }
    plan["request_sha256"] = content_sha256(plan)
    return plan


def _v2_run_groups(units):
    """Rename legacy in-memory unit rows for v2 indicator-set persistence."""
    expected = {"unit_id", "grain", "member_run_id"}
    if any(set(item) != expected for item in units):
        raise IndicatorPlanStale("indicator unit rows have unknown fields")
    return [
        {
            "run_group_id": item["unit_id"],
            "grain": item["grain"],
            "run_id": item["member_run_id"],
        }
        for item in units
    ]


def _indicator_request_path(root: Path, identity: str) -> Path:
    """The engine's file for one indicator request.

    A lone file per identity, so it gets a FILENAME rather than a directory
    holding one entry (t2609152104 change 3). It lives in the scope's `_engine/`
    bin because nothing here is for a reader: it exists so a run can be refused
    when its inputs moved.
    """
    return (
        root
        / "_engine"
        / "indicator_requests"
        / f"{identity_segment(identity, 'indicator_request_id')}.json"
    )


def write_indicator_plan(experiment_root, request):
    """Publish rebuildable checkpoint state after complete native validation."""
    root = Path(experiment_root).resolve()
    plan = build_indicator_plan(root, request)
    path = _indicator_request_path(root, plan["indicator_request_id"])
    if path.resolve() != path:
        raise IndicatorPlanStale("indicator planning path is aliased")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_record(path, plan, replace=True)
    log_row(
        f"{identity_segment(plan['indicator_set_id'], 'indicator_set_id')}: "
        f"{plural(len(plan['request']['declarations']), 'declaration')} over "
        f"{plural(len(plan['expected_result_keys']), 'result key')}",
        module="indicators",
    )
    return plan


def _rebased_plan(stored, root, expected):
    """The stored plan as written at ``root``, when only its root differs.

    A plan's targets are absolute, so a tree copied from another project root
    keeps that root's (t2610042152). The copy is accepted only when every
    target and table path keeps its root-relative tail under one common old
    root and the stored self-digest still matches; anything else is returned
    unchanged and so still differs.
    """
    here = root.as_posix()
    old_roots = set()
    rebased = {}
    for field in ("targets", "tables"):
        paths = stored.get(field)
        if not isinstance(paths, dict) or set(paths) != set(expected[field]):
            return stored
        for key, target in expected[field].items():
            tail = target[len(here) :]
            old = paths[key]
            if not (target.startswith(here + "/") and isinstance(old, str)):
                return stored
            if not old.endswith(tail):
                return stored
            old_roots.add(old[: -len(tail)])
        rebased[field] = expected[field]
    body = {key: value for key, value in stored.items() if key != "request_sha256"}
    if len(old_roots) != 1 or stored.get("request_sha256") != content_sha256(body):
        return stored
    rebased = {**body, **rebased}
    rebased["request_sha256"] = content_sha256(rebased)
    return rebased


def verify_indicator_plan(experiment_root, request, *, _observed_code=None):
    """Validate existing plans even when timestamps would schedule no job."""
    root = Path(experiment_root).resolve()
    path = _indicator_request_path(root, content_sha256(request))
    stored = read_canonical_json(path)
    expected = build_indicator_plan(root, request, _observed_code=_observed_code)
    if _rebased_plan(stored, root, expected) != expected:
        raise IndicatorPlanStale(
            f"indicator request expected={expected['request_sha256']} "
            f"observed={stored.get('request_sha256')}"
        )
    return expected


def _csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def check_live_indicator_environment(plan):
    """Compare the live indicator-stage environment with the retained one (D5).

    Rebuilding a plan from its own retained descriptor cannot detect drift --
    it would compare the descriptor with itself -- so the environment is
    observed afresh here and both the canonical bytes and their digest must
    agree. A mismatch aborts before any fitting and leaves no ready marker.
    """
    retained = plan["request"]["indicator_environment"]
    observed = resolve_indicator_environment()
    if canonical_json_bytes(observed) != canonical_json_bytes(retained):
        raise IndicatorPlanStale(
            "the live indicator-stage environment differs from the retained "
            f"descriptor: observed {content_sha256(observed)}, "
            f"retained {content_sha256(retained)}"
        )


def reduce_indicator_plan(
    experiment_root, plan, *, numerical_checks=False, diagnostics=None
):
    """Reduce the selected complete contract, keeping Class-C values at run grain."""
    from blueearth_cst.experiment.export_wflow_results import _format_value
    from blueearth_cst.experiment.indicator_registry import MonthReference

    root = Path(experiment_root).resolve()
    if build_indicator_plan(root, plan["request"]) != plan:
        raise IndicatorPlanStale("indicator plan changed before reduction")
    check_live_indicator_environment(plan)
    inventory = read_response_inventory_v2(root)
    native = _native_runs_v2(root, inventory)
    registry = declarations_for_calculations(plan["request"]["calculations"])
    local_registry = [(item, loader.local_declaration(item)) for item in registry]
    references = {
        key: MonthReference(
            **{
                **value,
                "run_ids": tuple(value["run_ids"]),
                "counts": tuple(tuple(row) for row in value["counts"]),
            }
        )
        for key, value in plan["resolved_references"].items()
    }
    output = {token: [] for token in plan["request"]["tokens"]}
    unavailable = {}
    for row in plan["request"]["unavailable_results"]:
        unavailable.setdefault(row["indicator"], {})[row["location"]] = row["reason"]
    evidence = []
    coverage = {}
    anchor = plan["request"]["water_year_anchor"]
    variables = tuple(plan["request"]["response_variables"])
    for group, members in plan["groups"].items():
        # A request with nothing calculable reads no responses.
        series = tuple(
            item
            for run in (members if variables else ())
            for item in open_responses(run, native[run], ResponseRequest(variables))
        )
        for item in series:
            coverage.setdefault(item.key, series_coverage(item, anchor))
        prepared = {}
        for indicator, local in local_registry:
            token = indicator.table_variable
            required_variables = {item.variable for item in indicator.requirements}
            # Unavailable locations never reach the calculation; they are
            # published as explicit rows below.
            blocked = unavailable.get(indicator.name, {})
            variable = [
                item
                for item in series
                if item.variable in required_variables
                and str(item.location_id) not in blocked
            ]
            reference = (
                references["wet" if indicator.statistic == "wetmonth_mean" else "dry"]
                if indicator.reference
                else None
            )
            if not variable:
                batches = []
            elif indicator.grain == "bundle":
                result = loader.evaluate(
                    local,
                    variable,
                    expected_run_ids=members,
                    anchor=anchor,
                    validation=plan["request"]["return_level_validation"],
                    _prepared=prepared,
                )
                batches = [(plan["bundle_run_group_ids"][group], result)]
                evidence.append(
                    {
                        "indicator": indicator.name,
                        "run_group_id": batches[0][0],
                        "locations": [
                            _plain(asdict(item)) for item in result.metadata["evidence"]
                        ],
                    }
                )
            else:
                batches = [
                    (
                        run,
                        loader.evaluate(
                            local,
                            [item for item in variable if item.run_id == run],
                            anchor=anchor,
                            reference=reference,
                            _prepared=prepared,
                        ),
                    )
                    for run in members
                ]
                batches.append(
                    (
                        plan["bundle_run_group_ids"][group],
                        loader.mean_run_results(
                            local,
                            dict(batches),
                            expected_run_ids=members,
                        ),
                    )
                )
            for unit, checked in batches:
                values = checked.values
                if numerical_checks or (
                    diagnostics is not None and indicator.calculation_set
                ):
                    is_group = unit.startswith("group_")
                    inputs = (
                        variable
                        if is_group
                        else [item for item in variable if item.run_id == unit]
                    )
                    context = {
                        "anchor": anchor,
                        "reference": reference,
                        "expected_run_ids": members if is_group else (unit,),
                        "validation": plan["request"]["return_level_validation"],
                        "group": is_group,
                    }
                    if numerical_checks:
                        loader.check(indicator, inputs, checked, **context)
                    if diagnostics is not None and indicator.calculation_set:
                        diagnostics.append(
                            {
                                "indicator": indicator.name,
                                "unit": unit,
                                "statuses": dict(checked.statuses),
                                "metadata": {
                                    k: v
                                    for k, v in checked.metadata.items()
                                    if k != "evidence"
                                },
                                "tables": [
                                    {
                                        "name": table.name,
                                        "columns": list(table.data.columns),
                                        "rows": json.loads(
                                            table.data.to_json(orient="records")
                                        ),
                                    }
                                    for table in checked.diagnostics
                                ],
                            }
                        )
                for location, value in values.items():
                    output[token].append(
                        {
                            "indicator": indicator.name,
                            "location": str(location),
                            "run_group_id": unit,
                            "value": _format_value(np.float32(value)),
                            "status": "undefined" if np.isnan(value) else "available",
                            "reason": "",
                        }
                    )
            units = list(members) if indicator.grain == "run" else []
            units.append(plan["bundle_run_group_ids"][group])
            for location, reason in sorted(blocked.items()):
                output[token].extend(
                    {
                        "indicator": indicator.name,
                        "location": location,
                        "run_group_id": unit,
                        "value": "",
                        "status": "unavailable",
                        "reason": reason,
                    }
                    for unit in units
                )
    actual = [
        (row["indicator"], row["location"], row["run_group_id"])
        for rows in output.values()
        for row in rows
    ]
    if len(actual) != len(set(actual)) or sorted(actual) != [
        tuple(key) for key in plan["expected_result_keys"]
    ]:
        raise ImmutableIndicatorSetError(
            "reduced result keys differ from independent expected set"
        )
    return output, evidence, [coverage[key] for key in sorted(coverage)]


def publish_indicator_set(experiment_root, plan):
    """Publish a whole declared set, preserving all bytes on exact ready reuse."""
    if plan["schema_version"] != "indicator-request/5":
        raise IndicatorPlanStale(
            f"unsupported indicator plan {plan['schema_version']!r}"
        )
    return _publish_indicator_set_v2(experiment_root, plan)


def _publish_indicator_set_v2(experiment_root, plan):
    """Publish immutable data beside its engine marker; preserve ready layouts."""
    root = Path(experiment_root).resolve()
    short = identity_segment(plan["indicator_set_id"], "indicator_set_id")
    engine = root / "_engine/indicator_sets" / short
    result = engine / "data"
    legacy_result = root / "results/indicator_sets" / short
    marker = engine / "indicators.json"
    if any(
        path.is_symlink() or path.is_junction() or path.resolve() != path
        for path in (engine, result, legacy_result)
    ):
        raise ImmutableIndicatorSetError("indicator-set directory is aliased")
    if marker.exists():
        return read_indicator_set(root, marker)
    check_live_indicator_environment(plan)
    # Snakemake pre-creates declared output parents. Accept only the empty
    # expected data directory, never a partly written or foreign payload.
    if (
        engine.exists()
        and (not engine.is_dir() or any(path != result for path in engine.iterdir()))
    ) or any(
        path.exists() and (not path.is_dir() or any(path.iterdir()))
        for path in (result, legacy_result)
    ):
        raise ImmutableIndicatorSetError("partial indicator set cannot be overwritten")
    diagnostics = []
    if any("set" in item for item in plan["request"]["calculations"]):
        tables, evidence, coverage = reduce_indicator_plan(
            root, plan, diagnostics=diagnostics
        )
    else:
        tables, evidence, coverage = reduce_indicator_plan(root, plan)
    declarations_by_name = {
        item["name"]: item for item in plan["request"]["declarations"]
    }
    _validate_tables(
        tables, plan["expected_result_keys"], declarations_by_name, plan["units"]
    )
    payloads = {
        f"{token}_{scope}_indicators.csv": _csv_bytes(
            ["indicator", "location", f"{scope}_id", "value", "status", "reason"],
            sorted(
                [
                    {
                        **{k: v for k, v in row.items() if k != "run_group_id"},
                        f"{scope}_id": row["run_group_id"],
                    }
                    for row in rows
                    if row["run_group_id"].startswith(f"{scope}_")
                ],
                key=lambda row: (
                    row["indicator"],
                    row["location"],
                    row[f"{scope}_id"],
                ),
            ),
        )
        for token, rows in tables.items()
        for scope in ("run", "group")
    }
    memberships = [
        {"group_id": row["run_group_id"], "run_id": row["run_id"]}
        for row in plan["units"]
        if row["grain"] == "bundle"
    ]
    metadata = [
        {"group_id": identity, "st_id": key, "is_reference": str(not key).lower()}
        for key, identity in plan["bundle_run_group_ids"].items()
    ]
    lookup = _csv_bytes(["group_id", "run_id"], memberships)
    metadata_payload = _csv_bytes(["group_id", "st_id", "is_reference"], metadata)
    environment = canonical_json_bytes(plan["request"]["indicator_environment"])
    # Each return level's provenance: usable block counts per member against
    # the required minimum, the fitted parameters, fit and shape coverage, and
    # the extraction / missingness / partial-block policies applied. Without it
    # a published return level is a bare number that cannot be audited.
    evidence_payload = canonical_json_bytes(
        {"schema_version": RETURN_LEVEL_EVIDENCE_SCHEMA, "bundles": _plain(evidence)}
    )
    coverage_payload = canonical_json_bytes(
        {"schema_version": COVERAGE_EVIDENCE_SCHEMA, "series": coverage}
    )
    diagnostics_payload = canonical_json_bytes(
        {
            "schema_version": "indicator-diagnostics/1",
            "results": _plain(diagnostics),
        }
    )
    report = return_level_validation.load_report_bytes()
    return_level_validation.verify_declaration(
        plan["request"]["return_level_validation"], report
    )
    import hashlib

    def ref(path, payload):
        return {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}

    result.mkdir(parents=True, exist_ok=True)
    engine.mkdir(parents=True, exist_ok=True)
    # Stage every file beside the engine root, then move each into place; the
    # manifest is written last, so an interrupted publish is never ready.
    staging = engine.parent / f".stage-{short}-{uuid.uuid4().hex}"
    staging.mkdir()
    files = {
        **{result / name: payload for name, payload in payloads.items()},
        result / "group_members.csv": lookup,
        result / "group_metadata.csv": metadata_payload,
        engine / "indicator_environment.json": environment,
        engine / return_level_validation.REPORT_FILENAME: report,
        engine / RETURN_LEVEL_EVIDENCE_FILENAME: evidence_payload,
        engine / COVERAGE_EVIDENCE_FILENAME: coverage_payload,
        engine / "indicator_diagnostics.json": diagnostics_payload,
    }
    try:
        staged = []
        for number, (path, payload) in enumerate(files.items()):
            temporary = staging / f"{number}.part"
            temporary.write_bytes(payload)
            staged.append((temporary, path))
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        # Without a manifest the set stays unready and is refused as partial.
        shutil.rmtree(staging, ignore_errors=True)
    simulation = read_simulation_v2(root)
    result_relative = result.relative_to(root).as_posix()
    from blueearth_cst.indicators import loader

    calculation_sets = sorted(
        {
            loader.local_declaration(item).calculation_set
            for item in declarations_for_calculations(plan["request"]["calculations"])
        }
    )
    manifest = {
        "schema_version": "indicator-set/6",
        "canonicalization_id": "collection-canon/1",
        "status": "ready",
        "indicator_set_id": plan["indicator_set_id"],
        "simulation_id": simulation["simulation_id"],
        "identity_projection": plan["identity_projection"],
        "response_inventory": {
            "path": "../../response_inventory.json",
            "sha256": plan["response_inventory_sha256"],
        },
        "environment": ref("indicator_environment.json", environment),
        "environment_sha256": content_sha256(plan["request"]["indicator_environment"]),
        "groups": plan["groups"],
        "run_groups": plan["units"],
        "group_members": ref(f"{result_relative}/group_members.csv", lookup),
        "group_metadata": ref(
            f"{result_relative}/group_metadata.csv", metadata_payload
        ),
        "calculations": plan["request"]["calculations"],
        "unavailable_calculations": plan["request"]["unavailable_calculations"],
        "unavailable_results": plan["request"]["unavailable_results"],
        "declarations": plan["request"]["declarations"],
        "execution_context": {
            "request": plan["request"],
            "indicator_definition": plan["indicator_definition"],
            "set_assumptions": {
                set_id: dict(loader.load_set(set_id).ASSUMPTIONS)
                for set_id in calculation_sets
            },
        },
        "expected_result_keys": plan["expected_result_keys"],
        "tables": [
            {
                "token": token,
                "scope": scope,
                "file": ref(
                    f"{result_relative}/{token}_{scope}_indicators.csv",
                    payloads[f"{token}_{scope}_indicators.csv"],
                ),
            }
            for token in sorted(tables)
            for scope in ("run", "group")
        ],
        "return_level_benchmark": ref(return_level_validation.REPORT_FILENAME, report),
        "return_level_evidence": ref(RETURN_LEVEL_EVIDENCE_FILENAME, evidence_payload),
        "coverage_evidence": ref(COVERAGE_EVIDENCE_FILENAME, coverage_payload),
        "diagnostics": ref("indicator_diagnostics.json", diagnostics_payload),
        "result_root": result_relative,
        "engine_root": f"_engine/indicator_sets/{short}",
    }
    manifest["indicators_manifest_sha256"] = content_sha256(manifest)
    atomic_record(marker, manifest)
    return read_indicator_set(root, marker)
