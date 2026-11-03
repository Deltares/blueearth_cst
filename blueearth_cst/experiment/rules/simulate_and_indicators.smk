# WF4 stage two. Every collection artifact is an external, validated leaf.
import contextlib
import shlex
import subprocess
import tempfile
import os
import sys
from blueearth_cst.shared.output_capture import captured_output
from blueearth_cst.experiment.simulation_runner import simulation_indicator_sets, simulation_selection, resolve_selected_collection
from blueearth_cst.experiment.simulation_runs import include_unperturbed, selected_run_ids
from blueearth_cst.experiment.simulation_record import live_simulation_inputs_v2, read_simulation_intent_v2, read_simulation_v2, SimulationFrozenError
from blueearth_cst.experiment.wf4_ancillary_descriptor import resolve_wf4_preparation
from blueearth_cst.climate_analysis.prepare_climate_data_catalog import resolved_unit_interpretation
from blueearth_cst.experiment.reservation_context import resolve_experiment_name
from blueearth_cst.experiment.batch_sizing import count_active_cells, disk_headroom_bytes, measure_member_footprint, resolve_batch_plan, resolve_batch_size, split_evenly
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, DEFAULT_JULIA_THREADS, declare_path_tokens, declare_project_root, julia_prefix, resolve_water_year_start, validate_experiment_name
from blueearth_cst.shared.console_style import RuleRegistry, defer_rows, defer_warning, target_banner
from blueearth_cst.shared.run_log_core import log_row, plural

project, my_cfg, _selection = simulation_selection(config_path)
project_dir = Path(project["project"]["project_dir"]).resolve().as_posix()
experiment = resolve_experiment_name(Path(project_dir), my_cfg.get("experiment_name"))
experiment = validate_experiment_name(experiment, project_dir)
exp_dir = f"{project_dir}/experiments/{experiment}"
runs_dir = f"{exp_dir}/hydrology/wflow"
results_dir = f"{exp_dir}/results"
# Engine bookkeeping, collected per scope so a reader learns to ignore one
# directory rather than meeting machinery at three depths (t2609152104).
engine_dir = f"{exp_dir}/_engine"
basin_dir = f"{project_dir}/models/hydrology/wflow"
SELECTION, COLLECTION = resolve_selected_collection(config_path, REPOSITORY)
_intent = read_canonical_json(Path(SELECTION["manifest_path"]).parent / "collection_intent.json")
SIM_WINDOW_START = _intent["scenario_spec"]["simulation_window"]["start"]
SIM_WINDOW_END = _intent["scenario_spec"]["simulation_window"]["end"]
RUN_IDS = selected_run_ids(COLLECTION, SELECTION["manifest_path"], include=include_unperturbed(my_cfg))
_retained_selection = None
stored = None
if Path(f"{engine_dir}/simulation.json").exists():
    from blueearth_cst.shared.workflow_config_snapshot import resolve_file_reference
    stored = read_simulation_v2(exp_dir)
    # Readiness has checked the intent and full collection. Reuse its exact
    # referenced metadata for selection and the subsequent live-input check.
    _retained_selection = read_canonical_json(resolve_file_reference(
        stored["intent"], {"project_root": Path(project_dir)}))["collection"]
elif Path(f"{engine_dir}/simulation_intent.json").exists():
    _retained_selection = read_simulation_intent_v2(exp_dir)["collection"]
elif Path(f"{engine_dir}/reservation.json").exists():
    from blueearth_cst.experiment.reservation import read_reservation
    _retained_selection = read_reservation(Path(project_dir), experiment)["resolution"]
if _retained_selection is not None:
    if (_retained_selection["collection_id"] != COLLECTION["collection_id"]
        or _retained_selection["collection_revision"] != COLLECTION["collection_revision"]
        or _retained_selection["run_ids"] != RUN_IDS):
        raise SimulationFrozenError("selected collection differs from retained experiment; use --new-experiment")
    SELECTION["resolution_mode"] = _retained_selection["resolution_mode"]
SERIES = {item["run_id"]: item["file"]["path"] for item in COLLECTION["series"]}
LOG_PARTS_DIR = f"{project_dir}/logs/_parts/simulate_system/{experiment}"
BENCH_PARTS_DIR = f"{project_dir}/benchmarks/_parts/simulate_system/{experiment}"
WORKFLOW_LOG_NAME = f"wf4_simulate_system_{experiment}.log"
BENCHMARKS_NAME = f"wf4_benchmarks_{experiment}.md"
from blueearth_cst.shared.execution_outputs import workflow_paths
_paths = workflow_paths(project_dir, "wf4", WORKFLOW_LOG_NAME, BENCHMARKS_NAME, LOG_PARTS_DIR, BENCH_PARTS_DIR)
WORKFLOW_LOG_PATH, BENCHMARK_OUTPUT_PATH = _paths.log, _paths.benchmark
LOG_PARTS_DIR, BENCH_PARTS_DIR = _paths.log_parts, _paths.benchmark_parts
RULES = RuleRegistry(LOG_PARTS_DIR, BENCH_PARTS_DIR)
LOG_RULES = RULES.log_rules
WRITE_MODEL_FINGERPRINT = RULES.logged("4.01", "write_model_fingerprint", summary="record the model inputs for this experiment")
CHECK_MODEL_UNCHANGED = RULES.logged("4.02", "check_model_unchanged", summary="compare the live model with the experiment reference")
SNAPSHOT_SIMULATION_INPUTS = RULES.banner_only("4.03", "snapshot_simulation_inputs", summary="freeze the selected runs, model, and simulation settings")
DOWNSCALE_SCENARIO_SERIES = RULES.logged("4.04", "downscale_scenario_series", summary="prepare model-grid forcing for the selected scenario run")
RUN_WFLOW_SIMULATIONS = RULES.logged("4.05", "run_wflow_simulations", summary="run Wflow for selected scenario runs")
PUBLISH_WFLOW_OUTPUTS = RULES.banner_only("4.06", "publish_wflow_outputs", summary="verify and retain the native simulation responses")
GATHER_LOGS = RULES.banner_only("4.09", "gather_logs", after_checkpoint=True, summary="merge simulation workflow logs")
GATHER_BENCHMARKS = RULES.banner_only("4.10", "gather_benchmarks", after_checkpoint=True, summary="merge simulation timing and resource-use records")
# Resolved once per invocation from the runner's selection snapshot. Legacy
# configs keep the old indicators/outvars precedence (indicator_sets.legacy_tokens).
INDICATOR_SETS = simulation_indicator_sets(project, my_cfg, _selection) if _selection.indicators else ()
from blueearth_cst.experiment.indicator_plan import indicator_anchor
INDICATOR_ANCHOR = indicator_anchor(project)

# The run's key folders, stated ONCE -- the same block WF0/WF1/WF2 carry, and
# the one `run_stress_test.smk` had before the R12 split (868c4b7c) dropped it.
# `run_header` prints them at the top of the console, every rule log repeats
# them in its own header, and `log_row` plus the console tee rewrite every path
# below to these names. Declared longest-root-last so `experiment` -- under
# which a WF4 run writes almost everything -- wins over `project`.
#
# No `data` row unless the collection's catalog declares a local root: WF4 reads
# no global data catalog of its own, it reads the collection WF3 retained.
declare_path_tokens(
    model=basin_dir,
    scenarios=f"{project_dir}/scenarios",
    experiment=exp_dir,
)
declare_project_root(project_dir)

def _selected_collection(wc):
    return SELECTION["manifest_path"]

def _current_preparation():
    catalogs = project["project"]["catalog"]
    catalogs = catalogs if isinstance(catalogs, list) else [catalogs]
    source = project["climate"]["selected"]
    # Warnings print under the run header, not above it (`defer_rows`).
    with captured_output(replay=defer_rows):
        return resolve_wf4_preparation(
            Path(project_dir), Path(exp_dir), climate_source=source,
            catalogs=catalogs,
            unit_interpretation=resolved_unit_interpretation(catalogs, source),
        )
def _live_simulation_inputs():
    preparation = _current_preparation()
    with captured_output(replay=defer_rows):
        with tempfile.TemporaryDirectory(dir=project_dir) as directory:
            return live_simulation_inputs_v2(
                exp_dir, project_root=project_dir, model_root=basin_dir,
                collection_path=SELECTION["manifest_path"], collection=COLLECTION,
                resolution_mode=SELECTION["resolution_mode"], preparation=preparation,
                simulation_window={"start": SIM_WINDOW_START, "end": SIM_WINDOW_END},
                julia_command=shlex.split(julia_prefix(1)),
                header_path=Path(directory) / "headers.txt",
                include_unperturbed=include_unperturbed(my_cfg))

_simulation_complete = False
if stored is not None:
    from blueearth_cst.experiment.simulation_reuse import check_completed_simulation_inputs
    check_completed_simulation_inputs(
        exp_dir, project_root=project_dir, model_root=basin_dir,
        collection_path=SELECTION["manifest_path"], collection=COLLECTION,
        resolution_mode=SELECTION["resolution_mode"], preparation=_current_preparation(),
        simulation_window={"start": SIM_WINDOW_START, "end": SIM_WINDOW_END},
        julia_command=shlex.split(julia_prefix(1)),
        include_unperturbed=include_unperturbed(my_cfg), _ready_simulation=stored)
    _simulation_complete = True

def _frozen_simulation(wc):
    if _simulation_complete:
        return f"{exp_dir}/_engine/simulation_intent.json"
    return checkpoints.snapshot_simulation_inputs.get().output.simulation

# The console counts resolved inputs after the indicator checkpoints.
# all
rule all:
    message: target_banner("all", None, compact=True)
    input:
        f"{engine_dir}/simulation.json",
        f"{engine_dir}/response_inventory.json",
        lambda wc: _indicator_plan_paths(wc),
        lambda wc: _selected_indicator_outputs(wc),
        # Managed invocations merge diagnostics in the launcher's finalizer.
        *_paths.gather_targets.values(),

if not _simulation_complete:
    # 4.01  write_model_fingerprint
    rule write_model_fingerprint:
        message: WRITE_MODEL_FINGERPRINT.banner()
        input:
            model_ready=ancient(f"{basin_dir}/.outputs_configured"),
            model_toml=ancient(f"{basin_dir}/wflow_sbm.toml"),
        params:
            model_dir=basin_dir,
            project_dir=project_dir,
        output:
            model_reference=update(f"{engine_dir}/model_reference.yml"),
        log:
            WRITE_MODEL_FINGERPRINT.log(),
        script: "../write_model_reference.py"

    # 4.03  snapshot_simulation_inputs
    checkpoint snapshot_simulation_inputs:
        message: SNAPSHOT_SIMULATION_INPUTS.banner()
        input:
            collection=_selected_collection,
            model_reference=f"{exp_dir}/.model_reference_ok",
        output:
            simulation=update(f"{exp_dir}/_engine/simulation_intent.json"),
            archive=update(f"{exp_dir}/config/run_record.yml"),
        run:
            from blueearth_cst.experiment.simulation_record import freeze_simulation_v2
            intent = _live_simulation_inputs()
            freeze_simulation_v2(
                exp_dir, intent,
                invocation_id=os.environ["CST_SIMULATION_INVOCATION_ID"],
                command=["simulate_system", "--config", config_path, "--target", "simulations_only"],
            )

    # 4.02  check_model_unchanged
    rule check_model_unchanged:
        message: CHECK_MODEL_UNCHANGED.banner()
        input:
            model_reference=f"{engine_dir}/model_reference.yml",
            model_toml=ancient(f"{basin_dir}/wflow_sbm.toml"),
        params:
            model_dir=basin_dir,
            experiment=experiment,
        output:
            ok=temp(touch(f"{exp_dir}/.model_reference_ok")),
        log:
            CHECK_MODEL_UNCHANGED.log(),
        script: "../check_model_reference.py"

    # 4.04  downscale_scenario_series
    rule downscale_scenario_series:
        message: DOWNSCALE_SCENARIO_SERIES.banner(context="run {wildcards.run_id}")
        wildcard_constraints:
            run_id="(?:" + "|".join(RUN_IDS) + ")",
        input:
            nc=lambda wc: (Path(project_dir) / SERIES[wc.run_id]).as_posix(),
            collection=_selected_collection,
            simulation=_frozen_simulation,
            model_reference_ok=f"{exp_dir}/.model_reference_ok",
        output:
            nc=temp(f"{runs_dir}/forcing/inmaps_{{run_id}}.nc"),
            toml=update(f"{runs_dir}/run_settings/{{run_id}}.toml"),
            catalog=temp(f"{runs_dir}/run_settings/{{run_id}}.yml"),
            temporal=temp(f"{runs_dir}/run_settings/{{run_id}}.temporal.json"),
        params:
            model_dir=basin_dir,
            run_id=lambda wc: wc.run_id,
            validated_collection=COLLECTION,
            native_output_path=f"{runs_dir}/output/{{run_id}}.csv",
            native_log_path=f"{runs_dir}/output/_log/{{run_id}}.log",
            sim_window_start=SIM_WINDOW_START,
            sim_window_end=SIM_WINDOW_END,
        threads: 1
        resources:
            mem_mb=2048,
        log:
            DOWNSCALE_SCENARIO_SERIES.log("{run_id}"),
        benchmark:
            DOWNSCALE_SCENARIO_SERIES.benchmark("{run_id}"),
        script: "../downscale_climate_forcing.py"

    compute = my_cfg.get("compute") or {}
    _cores = int(workflow.cores or 1)
    # Threads per batch and batches at once come from the batch plan
    # (t2609242342): project `compute` keys, then `advanced_settings.batching`,
    # then a regime chosen by the model's active cell count. A `--set-threads`
    # on the batch rules still wins, as before.
    import psutil
    from dataclasses import asdict
    import time
    _batch_decision_epoch = time.time()
    _plan = resolve_batch_plan(count_active_cells(f"{basin_dir}/staticmaps.nc"), _cores,
        ADVANCED_SETTINGS["batching"], compute, psutil.virtual_memory().available)
    _threads = min([_plan.threads, *[int(value) for name, value in
        getattr(workflow.resource_settings, "overwrite_threads", {}).items() if name.startswith("run_wflow_simulations_batch_")]])
    BATCH_PLAN_SUMMARY = f"Up to {plural(_plan.max_parallel, 'simulation')} at once; {plural(_threads, 'Julia thread')} per simulation"
    if _plan.active_cells is not None:
        BATCH_PLAN_SUMMARY += f"; {_plan.active_cells} active model cells"
    sizing = resolve_batch_size(member_count=len(RUN_IDS), cores=_plan.max_parallel,
        batch_size_max=compute.get("batch_size_max"), explicit=compute.get("batch_size"),
        footprint=measure_member_footprint(basin_dir, SIM_WINDOW_START, SIM_WINDOW_END, write_states=False),
        headroom_bytes=disk_headroom_bytes(project_dir,
            fraction=ADVANCED_SETTINGS["defaults"]["batch_disk_headroom_fraction"], headroom_gb=compute.get("disk_headroom_gb")))
    # Observations stay out of rule params: changing free RAM must not itself
    # become a reason for Snakemake to rerun a simulation.
    _batch_resource_decision = {"selected_at_epoch_s": _batch_decision_epoch,
        "plan": asdict(_plan), "sizing": asdict(sizing),
        "settings": ADVANCED_SETTINGS["batching"], "compute": compute,
        "batches": split_evenly(RUN_IDS, sizing.batch_size)}
    if sizing.warning:
        # A bare `print` carried no stamp and no module column at all,
        # which put it further outside the grammar than the `logger`
        # calls the other workflows used. Deferred rather than printed
        # because this is parse time -- `defer_warning` says why.
        defer_warning(sizing.warning, module='batching')
    for batch, members in enumerate(split_evenly(RUN_IDS, sizing.batch_size)):
        # 4.05  run_wflow_simulations_batch_<b> — one bounded batch of retained runs
        rule:
            name: RUN_WFLOW_SIMULATIONS.job_name(f"batch_{batch}")
            # Member COUNT rather than the span `run_stress_test.smk` printed:
            # the post-R12 batch is keyed by opaque run ids, so a span would read
            # as two digests rather than as a range. The count is the part that
            # says how long this line will sit there.
            message: RUN_WFLOW_SIMULATIONS.banner(part=f"batch_{batch}", context=plural(len(members), "scenario run"))
            input:
                simulation=_frozen_simulation,
                forcing=[f"{runs_dir}/forcing/inmaps_{run}.nc" for run in members],
                tomls=[f"{runs_dir}/run_settings/{run}.toml" for run in members],
            output:
                csvs=[update(f"{runs_dir}/output/{run}.csv") for run in members],
            threads: _threads
            resources:
                mem_mb=2048,
            params:
                records=[str(batch), *[value for run in members for value in
                    (run, f"{runs_dir}/run_settings/{run}.toml", f"{runs_dir}/output/{run}.csv")]],
                julia=lambda wc, threads: julia_prefix(threads),
            log:
                RUN_WFLOW_SIMULATIONS.log(f"batch_{batch}"),
            benchmark:
                RUN_WFLOW_SIMULATIONS.benchmark(f"batch_{batch}"),
            run:
                from blueearth_cst.experiment.batch_staging import run_staged_batch
                read_simulation_intent_v2(exp_dir)
                if Path(f"{engine_dir}/simulation.json").exists() or any(Path(p).exists() for p in output.csvs):
                    raise SimulationFrozenError("native responses already exist; use a new experiment name")
                # Finished members of a failed attempt wait in staging, keyed by
                # run id; only the rest run (t2608071217, batch_staging.py).
                run_staged_batch(params.records, input.simulation, f"{engine_dir}/batch_staging",
                    [sys.executable, "-u", str(REPOSITORY / "blueearth_cst/shared/run_logged.py"),
                     str(log[0]), "--", *shlex.split(params.julia),
                     str(REPOSITORY / "blueearth_cst/experiment/run_wflow_batch.jl")])

    # 4.06  publish_wflow_outputs
    rule publish_wflow_outputs:
        message: PUBLISH_WFLOW_OUTPUTS.banner()
        input:
            simulation=_frozen_simulation,
            csvs=[f"{runs_dir}/output/{run}.csv" for run in RUN_IDS],
            tomls=[f"{runs_dir}/run_settings/{run}.toml" for run in RUN_IDS],
            temporal=[f"{runs_dir}/run_settings/{run}.temporal.json" for run in RUN_IDS],
        output:
            inventory=update(f"{engine_dir}/response_inventory.json"),
            simulation=update(f"{engine_dir}/simulation.json"),
        run:
            from blueearth_cst.experiment.response_inventory import publish_response_inventory_v2
            from blueearth_cst.experiment.simulation_record import publish_simulation_v2
            from blueearth_cst.experiment.wflow_response_reader import NativeRunArtifacts
            native = {run: NativeRunArtifacts(Path(csv), Path(toml), Path(temporal))
                for run, csv, toml, temporal in zip(RUN_IDS, input.csvs, input.tomls, input.temporal)}
            evidence = read_canonical_json(Path(input.temporal[0]))
            if any(item["descriptor"]["source_calendar"] != evidence["source_calendar"] for item in COLLECTION["series"]):
                raise ValueError("response temporal evidence differs from retained source calendar")
            inventory = publish_response_inventory_v2(exp_dir, native, evidence)
            published = publish_simulation_v2(exp_dir, inventory)
            log_row(f"Simulation outputs ready: {plural(len(RUN_IDS), 'run')} retained", module="responses")
            from blueearth_cst.experiment.console_reporting import record_indicator_detail
            record_indicator_detail(exp_dir, f"Simulation {published['simulation_id']}; response inventory → {output.inventory}")

# simulations_only -- target (unnumbered): simulate, stop before indicators
rule simulations_only:
    # A TARGET AGGREGATOR, so it says what the run produced -- the grammar
    # WF0, WF1 and WF2 each close with. It carried a plain `rule_banner` and
    # printed two bare lines, which is the same job in a second grammar.
    # ONE LINE: Snakemake's parser takes a keyword's body as a single
    # expression and rejects a call split across lines here.
    message: target_banner("simulations_only", [f"{engine_dir}/response_inventory.json"], project_dir, compact=True)
    input:
        f"{engine_dir}/response_inventory.json",
        f"{engine_dir}/simulation.json",

# 4.09  gather_logs
rule gather_logs:
    message: GATHER_LOGS.banner()
    input:
        lambda wc: _indicator_plan_paths(wc),
        lambda wc: _selected_indicator_outputs(wc),
    output:
        WORKFLOW_LOG_PATH,
    params:
        rules=LOG_RULES,
        parts_dir=LOG_PARTS_DIR,
    script: "../../shared/merge_logs.py"

# 4.10  gather_benchmarks
rule gather_benchmarks:
    message: GATHER_BENCHMARKS.banner()
    input:
        lambda wc: _indicator_plan_paths(wc),
        lambda wc: _selected_indicator_outputs(wc),
    output:
        BENCHMARK_OUTPUT_PATH,
        BENCHMARK_OUTPUT_PATH.removesuffix(".md") + ".tsv",
    params:
        parts_dir=BENCH_PARTS_DIR,
        workflow_num=4,
    script: "../../shared/merge_benchmarks.py"
