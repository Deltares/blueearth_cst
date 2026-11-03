# WF3: model-independent generation and immutable collection publication.
import os
import sys
import time
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.config_composition import compose_config
from blueearth_cst.shared.workflow_archive_launch import non_producing_invocation
from blueearth_cst.shared.snake_utils import catalog_root, declare_path_tokens, declare_project_root, declare_warning_tally, patch_psutil_windows_benchmark, warning_count
from blueearth_cst.shared.wf3_science import index_width
from blueearth_cst.shared.console_style import declared_step
from blueearth_cst.shared.console_style import install_console_style, open_run_header, pre_dag_step, RuleRegistry, rule_banner, run_summary, target_banner
from blueearth_cst.experiment.generation_plan import generation_configuration
from blueearth_cst.experiment.scenario_rows import stochastic_rows
patch_psutil_windows_benchmark()
config_path = workflow.configfiles[0]
CONFIG_PROJECTION = ("project", "basin", "climate", "workflows.generate_scenarios")
config, WORKFLOW_CONFIG_PATHS = compose_config(config, config_path, entry="generate_scenarios",
    declared_sections=CONFIG_PROJECTION)
WF_CONFIG_PATHS = sorted(WORKFLOW_CONFIG_PATHS.values())
SOURCE_ONLY = os.environ.get("CST_GENERATION_PHASE") == "source"
V2_MODE = os.environ.get("CST_GENERATION_PHASE") == "generation"
if not SOURCE_ONLY and not V2_MODE:
    raise ValueError("WF3 requires the owned source or generation phase")
GENERATION = generation_configuration(
    config, workflow.basedir, stage=bool(os.environ.get("CST_GENERATION_OWNED"))
)
project_dir = GENERATION["project_dir"]
REGION = GENERATION["region"]
CLIMATE_STORE = GENERATION["store"]
store_dir = CLIMATE_STORE.store_dir
_scenario_request_path = GENERATION["request_path"]
wg_dir = (Path(_scenario_request_path).parent / "generation").as_posix()
lookup_path = f"{wg_dir}/config/stress_test_lookup.csv"
V2_PLAN = None
if V2_MODE:
    from blueearth_cst.experiment.generation_plan import read_pinned_plan
    V2_PLAN = read_pinned_plan(
        Path(os.environ["CST_GENERATION_PLAN_PATH"]),
        os.environ["CST_GENERATION_PLAN_SHA256"],
        Path(project_dir),
    )
    if V2_PLAN["request"] != GENERATION["request"]:
        raise ValueError("pinned generation request differs from parsed WF3 settings")
# WF3's run records are keyed by the SCENARIO-REQUEST fingerprint, because a
# project can hold several requests at once and WF3 has no user-facing name to
# key on the way WF4 keys on its experiment.
#
# Since t2609152107 the `scenarios/requests/<fingerprint>/` DIRECTORY is itself
# named by the first SHORT_DIGEST_CHARS of that fingerprint, so this is a read
# of the directory name rather than a slice of it. The log, the benchmark table
# and both `_parts/` trees therefore cannot drift from the directory: one
# constant sets all four.
#
# Still not passed through `short_digest`, which RAISES on a value that is not a
# digest. This runs at parse time, so a request directory with an unexpected
# name would fail every WF3 invocation over a cosmetic key.
_plan_key = Path(_scenario_request_path).parent.name
LOG_PARTS_DIR = f"{project_dir}/logs/_parts/generate_scenarios/{_plan_key}"
BENCH_PARTS_DIR = f"{project_dir}/benchmarks/_parts/generate_scenarios/{_plan_key}"
WORKFLOW_LOG_NAME = f"wf3_generate_scenarios_{_plan_key}.log"
from blueearth_cst.shared.execution_outputs import workflow_paths, declare_warning_tally

# One tally per run, opened at parse time and published to every job process
# through the environment: a rule prints its warnings from a process of its
# own, and the verdict below is written by this one. See
# `snake_utils.declare_warning_tally`.
declare_warning_tally(project_dir, WORKFLOW_LOG_NAME)
BENCHMARKS_NAME = f"wf3_benchmarks_{_plan_key}.md"
_paths = workflow_paths(project_dir, "wf3", WORKFLOW_LOG_NAME, BENCHMARKS_NAME, LOG_PARTS_DIR, BENCH_PARTS_DIR)
WORKFLOW_LOG_PATH, BENCHMARK_OUTPUT_PATH = _paths.log, _paths.benchmark
LOG_PARTS_DIR, BENCH_PARTS_DIR = _paths.log_parts, _paths.benchmark_parts
RULES = RuleRegistry(LOG_PARTS_DIR, BENCH_PARTS_DIR)
LOG_RULES = RULES.log_rules
DELINEATE_REGION = RULES.logged("3.01", "delineate_region", summary="delineate the basin boundary")
EXTRACT_CLIMATE_DATASETS = RULES.logged("3.02", "extract_climate_datasets", summary="extract a basin climate grid")
PREPARE_PERTURBATION_GRID = RULES.logged("3.03", "prepare_perturbation_grid", summary="prepare monthly climate perturbation settings")
if V2_MODE:
    GENERATE_WEATHER_REALIZATIONS = RULES.logged(
        "3.07", "generate_weather_realizations",
        summary="generate unperturbed stochastic weather realizations",
    )
    PERTURB_CLIMATE_REALIZATIONS = RULES.logged(
        "3.08", "perturb_climate_realizations",
        summary="produce one scenario per stochastic realization",
    )
PUBLISH_SCENARIO_COLLECTION = RULES.banner_only("3.09", "publish_scenario_collection", summary="validate the scenario artifacts and mark the collection ready")
GATHER_BENCHMARKS = RULES.banner_only("3.10", "gather_benchmarks", summary="merge generation timing and resource-use records")
GATHER_LOGS = RULES.banner_only("3.11", "gather_logs", summary="merge generation logs")
INVOCATION_ID = os.environ.setdefault("CST_GENERATION_INVOCATION_ID", uuid.uuid4().hex)
RLZ_NUM = GENERATION["n_realizations"]
ST_NUM = GENERATION["n_design_points"]
ST_WIDTH, RLZ_WIDTH = index_width(ST_NUM), index_width(RLZ_NUM)
_row_capacity = GENERATION["capacity"]
SCENARIO_ROWS = stochastic_rows(RLZ_NUM, ST_NUM, unit_id_capacity=_row_capacity)
_rows_by_id = {row.run_id: row for row in SCENARIO_ROWS}
_root_rows = tuple(row for row in SCENARIO_ROWS if not row.derived_from)
stress_test_cfg = GENERATION["config"]["climate_perturbations"]
_generation_catalogs = GENERATION["catalogs"]
_generation_request = GENERATION["request"]
clim_source = GENERATION["source"]

# The run's key folders, stated ONCE. `run_header` prints them at the top of the
# console and every rule log repeats them in its own header; `log_row` and the
# console tee rewrite every path below to these names. No `model` row -- WF3
# builds none, and that is the point of the R12 split: generation is
# model-independent. `scenarios` is the WF3 output ROOT rather than one request,
# because a project holds several requests at once and the digest-named
# directory under it is the part a reader is telling apart.
#
# Dropped by the R12 split (868c4b7c) along with the rest of the apparatus and
# restored 2026-09-16; `run_stress_test.smk` carried the same block.
declare_path_tokens(
    data=catalog_root(_generation_catalogs),
    climate=store_dir,
    scenarios=f"{project_dir}/scenarios",
)
declare_project_root(project_dir)

SOURCE_TARGETS = [f"{store_dir}/extract_historical.nc", f"{store_dir}/basin_cells.csv", lookup_path]
V2_MARKER = (
    f"{project_dir}/{V2_PLAN['outputs']['record_root']}/collection.json"
    if V2_MODE else None
)
V2_TARGET = (
    V2_MARKER if V2_MODE and V2_PLAN["decision"] == "create"
    else os.environ.get("CST_GENERATION_RECEIPT_PATH")
)
WF3_TARGETS = SOURCE_TARGETS if SOURCE_ONLY else [
    V2_TARGET,
    *_paths.gather_targets.values(),
]

# all
rule all:
    message: target_banner("all", WF3_TARGETS, project_dir, compact=True)
    input:
        WF3_TARGETS,

# 3.01  delineate_region
rule delineate_region:
    message: DELINEATE_REGION.banner()
    input:
        **REGION.inputs,
    params:
        **REGION.params,
    output:
        **REGION.outputs,
    log:
        DELINEATE_REGION.log(),
    benchmark:
        DELINEATE_REGION.benchmark(),
    script: REGION.script


# 3.02  extract_climate_datasets
rule extract_climate_datasets:
    message: EXTRACT_CLIMATE_DATASETS.banner()
    input:
        **CLIMATE_STORE.inputs,
    params:
        **CLIMATE_STORE.params,
    output:
        **CLIMATE_STORE.outputs,
    log:
        EXTRACT_CLIMATE_DATASETS.log(),
    benchmark:
        EXTRACT_CLIMATE_DATASETS.benchmark(),
    script:
        CLIMATE_STORE.script

# 3.03  prepare_perturbation_grid
rule prepare_perturbation_grid:
    message: PREPARE_PERTURBATION_GRID.banner()
    input:
        config = ancient(config_path),
        config_workflows = ancient(WF_CONFIG_PATHS),
    params:
        stress_test_cfg = stress_test_cfg,
    output:
        lookup_csv = lookup_path,
    log:
        PREPARE_PERTURBATION_GRID.log(),
    benchmark:
        PREPARE_PERTURBATION_GRID.benchmark(),
    script:
        "blueearth_cst/experiment/prepare_cst_parameters.py"

if V2_MODE and V2_PLAN["decision"] != "create":
    # Reusing a published collection: planning created nothing, but the steps
    # still list, so the table shows the whole pipeline.
    declared_step("3.04", "snapshot_generation_inputs")
    declared_step("3.05", "claim_scenario_collection")
    declared_step("3.06", "prepare_weather_generator_settings")

if V2_MODE and V2_PLAN["decision"] == "create":
    # Done by scripts/generate_scenarios.py before this DAG was built: the plan
    # is frozen, the collection claimed and the generator input written.
    pre_dag_step("3.04", "snapshot_generation_inputs")
    pre_dag_step("3.05", "claim_scenario_collection")
    pre_dag_step("3.06", "prepare_weather_generator_settings")
    from blueearth_cst.experiment.generation_publication import (
        publish_generation,
        validate_provider_inputs,
    )
    from blueearth_cst.experiment.scenario_provider import (
        SourceInputs,
        generate_roots,
        transform,
    )
    from blueearth_cst.experiment.forcing_descriptor import ClimateArtifact
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference

    _v2_root = Path(project_dir).resolve()
    _v2_data = _v2_root / V2_PLAN["outputs"]["data_root"]
    _v2_receipt = Path(os.environ["CST_GENERATION_RECEIPT_PATH"])
    _v2_yaml = _v2_root / V2_PLAN["outputs"]["generator_input"]
    _v2_lookup = _v2_root / V2_PLAN["outputs"]["perturbation_lookup"]
    _v2_rows = {row["run_id"]: row for row in V2_PLAN["rows"]}
    _v2_root_ids = [row["run_id"] for row in V2_PLAN["rows"] if not row["st_id"]]
    _v2_derived_ids = [row["run_id"] for row in V2_PLAN["rows"] if row["st_id"]]
    # Every member is written once, straight to its retained series path; only
    # the roots pass through the generator's own output directory on the way.
    _v2_series = {
        item["run_id"]: _v2_root / item["path"]
        for item in V2_PLAN["outputs"]["series"]
    }
    _v2_generator_output = _v2_data / "weathergenr" / "output"
    _v2_sources = {
        item["role"]: item["file"]
        for item in V2_PLAN["intent"]["documents"]["source_inventory"]["sources"]
    }

    def _v2_input(role):
        return resolve_file_reference(
            _v2_sources[role], {"project_root": _v2_root}
        )

    # The generator's RAW root per realization. Every published run -- the
    # unperturbed baseline included -- is transformed from it, so the baseline
    # takes the same quantile-mapping path as the grid (t2608151154).
    _v2_raw = {
        item["run_id"]: _v2_root / item["path"]
        for item in V2_PLAN["outputs"]["temporary_members"]
    }

    def _v2_ancestor(wc):
        row = _v2_rows[wc.run_id]
        root = next(
            item for item in V2_PLAN["rows"]
            if item["rlz"] == row["rlz"] and not item["st_id"]
        )
        return _v2_raw[root["run_id"]].as_posix()

    rule generate_weather_realizations:
        message: GENERATE_WEATHER_REALIZATIONS.banner()
        input:
            receipt=_v2_receipt.as_posix(),
            historical=_v2_input("historical_climate").as_posix(),
            cells=_v2_input("basin_cells").as_posix(),
            yaml=_v2_yaml.as_posix(),
        output:
            # temp(): consumed only by the perturbation step, which publishes
            # every run's series from them.
            roots=[temp(_v2_raw[run_id].as_posix()) for run_id in _v2_root_ids],
            dates=[(_v2_root / item["path"]).as_posix() for item in V2_PLAN["outputs"]["date_products"]],
        log:
            GENERATE_WEATHER_REALIZATIONS.log()
        benchmark:
            GENERATE_WEATHER_REALIZATIONS.benchmark()
        run:
            source_inputs = SourceInputs(
                historical_climate=Path(input.historical),
                weathergen_config=Path(input.yaml),
                basin_cells=Path(input.cells),
                output_dir=_v2_generator_output,
                rlz_width=len(str(RLZ_NUM)),
                st_width=len(str(ST_NUM)),
            )
            validate_provider_inputs(
                V2_PLAN, _v2_root,
                historical=source_inputs.historical_climate,
                cells=source_inputs.basin_cells,
                generator_yaml=source_inputs.weathergen_config,
                lookup=_v2_lookup, root_run_ids=_v2_root_ids,
                output_dir=source_inputs.output_dir,
            )
            root_rows = tuple(_rows_by_id[run_id] for run_id in _v2_root_ids)
            generate_roots(root_rows, source_inputs, log_path=Path(log[0]))

    rule perturb_climate_realizations:
        message: PERTURB_CLIMATE_REALIZATIONS.banner(context="run {wildcards.run_id}")
        input:
            receipt=_v2_receipt.as_posix(),
            ancestor=_v2_ancestor,
            lookup=_v2_lookup.as_posix(),
            yaml=_v2_yaml.as_posix(),
        output:
            (_v2_data / "series" / "{run_id}.nc").as_posix()
        wildcard_constraints:
            run_id="|".join(_v2_root_ids + _v2_derived_ids),
        log:
            PERTURB_CLIMATE_REALIZATIONS.log("{run_id}")
        benchmark:
            PERTURB_CLIMATE_REALIZATIONS.benchmark("{run_id}")
        run:
            validate_provider_inputs(
                V2_PLAN, _v2_root,
                historical=_v2_input("historical_climate"),
                cells=_v2_input("basin_cells"),
                generator_yaml=Path(input.yaml), lookup=Path(input.lookup),
                ancestor=Path(input.ancestor), row_id=wildcards.run_id,
                output=Path(output[0]),
            )
            row = _rows_by_id[wildcards.run_id]
            root_id = row.derived_from or row.run_id
            Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
            transform(
                row,
                ClimateArtifact(root_id, Path(input.ancestor)),
                weathergen_config=Path(input.yaml),
                lookup_csv=Path(input.lookup),
                output_path=Path(output[0]),
                log_path=Path(log[0]),
            )
            if not row.derived_from:
                from blueearth_cst.shared.run_log_core import log_row
                log_row(f"{row.run_id}: unperturbed baseline ready", module="generation")

    rule publish_scenario_collection:
        message: PUBLISH_SCENARIO_COLLECTION.banner()
        input:
            receipt=_v2_receipt.as_posix(),
            series=[(_v2_root / item["path"]).as_posix() for item in V2_PLAN["outputs"]["series"]],
            dates=[(_v2_root / item["path"]).as_posix() for item in V2_PLAN["outputs"]["date_products"]],
            generator=_v2_yaml.as_posix(),
            scenario_lookup=(_v2_root / V2_PLAN["outputs"]["scenario_run_lookup"]).as_posix(),
            perturbation_lookup=_v2_lookup.as_posix(),
        output:
            V2_MARKER
        run:
            publish_generation(
                _v2_root,
                Path(os.environ["CST_GENERATION_PLAN_PATH"]),
                os.environ["CST_GENERATION_PLAN_SHA256"],
                INVOCATION_ID,
            )
            from blueearth_cst.shared.run_log_core import log_row, plural
            log_row(f"Scenario collection ready: {plural(len(V2_PLAN['rows']), 'run')}", module="generation")


# 3.10 / 3.11 -- merge the run's benchmark and log parts, generation phase only:
# the source phase runs first, and merging there would consume its parts before
# the generation rules wrote theirs. The terminal is the ready marker (create)
# or this invocation's receipt (reuse); both are new per run, so the merge is too.
# The gathers re-run when the COLLECTION or its sources change -- never on
# V2_TARGET alone, which on a reuse is the per-invocation receipt: a new file
# every run, so the gathers re-ran on every otherwise up-to-date invocation.
GATHER_INPUTS = [V2_MARKER, *SOURCE_TARGETS] if V2_MODE else []

if V2_MODE:
    rule gather_benchmarks:
        message: GATHER_BENCHMARKS.banner()
        input:
            GATHER_INPUTS,
        output:
            BENCHMARK_OUTPUT_PATH,
            BENCHMARK_OUTPUT_PATH.removesuffix(".md") + ".tsv",
        params:
            parts_dir = BENCH_PARTS_DIR,
            workflow_num = 3,
        script: "blueearth_cst/shared/merge_benchmarks.py"

    rule gather_logs:
        message: GATHER_LOGS.banner()
        input:
            GATHER_INPUTS,
        output:
            WORKFLOW_LOG_PATH,
        params:
            rules = LOG_RULES,
            parts_dir = LOG_PARTS_DIR,
        script: "blueearth_cst/shared/merge_logs.py"


# --------------------------------------------------------------------------
# Console: the run's own opening and closing block, and the Snakemake restyle.
#
# Restored 2026-09-16. The R12 split (868c4b7c) carried the rule bodies out of
# `run_stress_test.smk` but not the file scaffolding, so WF3 ran with no
# `_ConsoleHandler` at all: the banners it did keep printed unstyled among
# Snakemake's un-suppressed scheduler chatter, which is what the owner saw.
# --------------------------------------------------------------------------

# Wall clock for the end-of-run summary. Taken at PARSE, not in `onstart`:
# Snakemake exposes no run duration to these handlers, and parse-to-finish is
# the interval a person actually waited. It therefore includes DAG construction
# -- and on WF3 that is not free, since the parse-time collection validation
# above reads the retained plan.
_RUN_STARTED = time.monotonic()


def _summary(failed):
    """Print the end-of-run block to STDERR, beside Snakemake's own output.

    stderr because that is where Snakemake writes its console, so a redirect
    that captures one captures both. Never raises: a summary that broke a
    successful run would be the worst possible trade for a convenience.
    """
    try:
        # One write, blank line before it -- see the note on `_header`.
        sys.stderr.write(
            "\n"
            + run_summary(
                "wf3 generate_scenarios",
                project_dir,
                WORKFLOW_LOG_NAME,
                BENCHMARKS_NAME,
                elapsed_seconds=time.monotonic() - _RUN_STARTED,
                failed=failed,
                log_parts_dir=LOG_PARTS_DIR,
                warnings=warning_count(),
            )
            + "\n"
        )
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        # Nested, because sys.stderr may be exactly what failed above. An
        # OSError escaping here surfaces as an error in this Snakefile and
        # masks the rule that actually failed (observed 2026-08-17, wf0).
        try:
            print(f"(run summary unavailable: {exc})", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass


def _header():
    """Print the start-of-run block to STDERR, mirroring `_summary`.

    Takes no arguments on purpose. Every name it reports is read INSIDE the
    guard, so a value that turns out not to resolve in the `onstart` namespace
    costs a line of console rather than the run.
    """
    try:
        # One write, carrying its own blank line on both sides: the block is the
        # first thing this toolbox puts on the console and it must not open
        # flush against Snakemake's preamble nor close flush against its
        # `Job stats:`. `open_run_header` owns that; the spacing is its business.
        open_run_header(
            "wf3 generate_scenarios", project_dir, config_path,
            **{
                "historical climate": clim_source,
                "historical period": f"{config['climate']['window']['start']}–{config['climate']['window']['end']}",
                "scenario period": f"{GENERATION['start']}–{GENERATION['end']}",
                "weather realizations": RLZ_NUM,
                "climate perturbations": ST_NUM,
                "scenario series": len(SCENARIO_ROWS),
                "water year": f"Starts in {_generation_request['water_year_start'].title()}",
            },
        )
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        # Nested, for the reason given on `_summary` -- and it matters more
        # here: this runs from `onstart`, so a raise aborts the run before any
        # rule executes at all.
        try:
            print(f"(run header unavailable: {exc})", file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass


onstart:
    if not os.environ.get("CST_GENERATION_OWNED") and not non_producing_invocation(sys.argv):
        raise ValueError("WF3 execution requires an owned generation launcher")
    # Restyle Snakemake's own console output into this toolbox's grammar (one
    # line per job start and end). Here and not at parse time: the logging
    # stack does not exist yet then. Fail-open; see install_console_style.
    install_console_style()
    _header()


onsuccess:
    _summary(failed=False)


onerror:
    _summary(failed=True)
