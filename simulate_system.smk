# WF4: supported through scripts/simulate_system.py or scripts/run_workflows.py.
import os
import sys
import time
from pathlib import Path
REPOSITORY = Path(workflow.basedir)
sys.path.insert(0, str(REPOSITORY))
from blueearth_cst.experiment.content_identity import content_sha256, read_canonical_json
from blueearth_cst.experiment.simulation_runner import simulation_selection, simulation_settings
from blueearth_cst.shared.snake_utils import declare_warning_tally, patch_psutil_windows_benchmark, warning_count
from blueearth_cst.shared.execution_outputs import declare_warning_tally
from blueearth_cst.shared.console_style import install_console_style, open_run_header, run_summary, target_banner
from blueearth_cst.shared.provenance import SHORT_DIGEST_CHARS, short_digest
patch_psutil_windows_benchmark()
config_path = workflow.configfiles[0]
_console_project, _settings = simulation_settings(config_path)
OPERATION = _settings["operation"]
# strict or partial; read from the runner snapshot, never part of a request.
INDICATOR_EVALUATION = simulation_selection(config_path)[2].evaluation
if os.environ.get("CST_SIMULATION_OPERATION") != OPERATION or not os.environ.get("CST_SIMULATION_INVOCATION_ID"):
    raise ValueError("Use python scripts/simulate_system.py --config <project> (operations/indicator_sets infer the target; a legacy operation config takes --target all, or --target simulations_and_indicators for indicators-only); bare simulate_system.smk is unsupported")
if OPERATION == "indicators-only":
    include: "blueearth_cst/experiment/rules/indicators_only.smk"
else:
    include: "blueearth_cst/experiment/rules/simulate_and_indicators.smk"

from blueearth_cst.experiment.indicator_sets import calculation_records
from blueearth_cst.experiment.console_reporting import matching_sets
from blueearth_cst.shared.console_style import register_indicator_context

# Both included modules define INDICATOR_SETS (resolved named sets) and
# INDICATOR_ANCHOR. Sets with identical numerical requests share one plan and one
# immutable indicator set; set ids never enter a request or a wildcard.
INDICATOR_CALCULATIONS = []
for _item in INDICATOR_SETS:
    _records = calculation_records(_item.indicators)
    if _records not in INDICATOR_CALCULATIONS:
        INDICATOR_CALCULATIONS.append(_records)

# RULES comes from whichever module was included above.
# One plan and one indicator set per distinct request once the simulation
# checkpoint resolves.
PREPARE_INDICATOR_PLAN = RULES.banner_only("4.07", "prepare_indicator_plan", after_checkpoint=True, planned_jobs=max(len(INDICATOR_CALCULATIONS), 1), summary="check retained responses against each indicator request")
DERIVE_SYSTEM_INDICATORS = RULES.banner_only("4.08", "derive_system_indicators", after_checkpoint=True, planned_jobs=max(len(INDICATOR_CALCULATIONS), 1), planned_from="prepare_indicator_plan", summary="compute and publish each immutable indicator set from retained responses")


_INDICATOR_REQUESTS = None
_VERIFIED_INDICATOR_PLANS = {}


def _current_indicator_requests(wc=None):
    """Map each distinct request's short digest to the request itself."""
    from blueearth_cst.experiment.indicator_plan import current_indicator_requests

    global _INDICATOR_REQUESTS
    # Always ask Snakemake first: an incomplete simulation checkpoint must defer
    # expansion, and a failed resolution must never populate the scheduling cache.
    _frozen_simulation(wc)
    if _INDICATOR_REQUESTS is not None:
        return _INDICATOR_REQUESTS
    resolved = current_indicator_requests(exp_dir, INDICATOR_CALCULATIONS, INDICATOR_ANCHOR)
    requests = {}
    for records, request in zip(INDICATOR_CALCULATIONS, resolved, strict=True):
        identity = short_digest(content_sha256(request))
        requests[identity] = request
        register_indicator_context("indicator_request_id", identity, [item.set_id for item in matching_sets(INDICATOR_SETS, request)])
    # Per-process scheduling only. Rule execution/publication retain their fresh
    # code, response and environment checks; subsequent invocations start empty.
    _INDICATOR_REQUESTS = requests
    return _INDICATOR_REQUESTS


# 4.07  prepare_indicator_plan
checkpoint prepare_indicator_plan:
    message: PREPARE_INDICATOR_PLAN.banner(context="indicator_request {wildcards.indicator_request_id}")
    input:
        simulation=_frozen_simulation,
        responses=f"{engine_dir}/response_inventory.json",
    output:
        f"{engine_dir}/indicator_requests/{{indicator_request_id}}.json",
    wildcard_constraints:
        indicator_request_id=rf"[a-f0-9]{{{SHORT_DIGEST_CHARS}}}",
    params:
        request=lambda wc: _current_indicator_requests(wc)[wc.indicator_request_id],
    run:
        from blueearth_cst.experiment.console_reporting import prepare_indicator_plan as report_plan
        if wildcards.indicator_request_id != short_digest(content_sha256(params.request)):
            raise ValueError("indicator request differs from its exact checkpoint")
        report_plan(exp_dir, params.request, INDICATOR_SETS, INDICATOR_EVALUATION)


def _indicator_plan_paths(wc):
    """Every selected plan as a plain input, so all checkpoint jobs are scheduled
    together; ``_selected_indicator_outputs`` stops at the first incomplete one."""
    return [f"{engine_dir}/indicator_requests/{identity}.json" for identity in _current_indicator_requests(wc)]


def _selected_indicator_outputs(wc):
    """Declared outputs of every selected plan; tables are not rule outputs."""
    from blueearth_cst.experiment.indicator_plan import read_indicator_set, verify_indicator_plan
    outputs = []
    for identity, request in _current_indicator_requests(wc).items():
        path = checkpoints.prepare_indicator_plan.get(indicator_request_id=identity).output[0]
        # Rebuilt checkpoints invalidate validation even with preserved mtimes.
        # Cache only completed, successfully verified plans for these exact bytes.
        stored = read_canonical_json(Path(path))
        key = (identity, content_sha256(stored))
        if key not in _VERIFIED_INDICATOR_PLANS:
            _VERIFIED_INDICATOR_PLANS[key] = verify_indicator_plan(
                exp_dir, request, _observed_code=request["code_inventory"])
        plan = _VERIFIED_INDICATOR_PLANS[key]
        if Path(plan["targets"]["manifest"]).exists():
            read_indicator_set(exp_dir, Path(plan["targets"]["manifest"]))
        outputs.extend(plan["targets"].values())
    return outputs


def _indicator_set_plan(wc):
    for identity in _current_indicator_requests(wc):
        path = checkpoints.prepare_indicator_plan.get(indicator_request_id=identity).output[0]
        plan = read_canonical_json(Path(path))
        if short_digest(plan["indicator_set_id"]) == wc.indicator_set_id:
            request = plan["request"]
            register_indicator_context("indicator_set_id", wc.indicator_set_id, [item.set_id for item in matching_sets(INDICATOR_SETS, request)])
            return path
    raise ValueError("indicator set is not produced by any selected plan")


# 4.08  derive_system_indicators
rule derive_system_indicators:
    message: DERIVE_SYSTEM_INDICATORS.banner(context="indicator_set {wildcards.indicator_set_id}")
    input:
        plan=_indicator_set_plan,
    output:
        manifest=update(f"{engine_dir}/indicator_sets/{{indicator_set_id}}/indicators.json"),
        units=update(f"{engine_dir}/indicator_sets/{{indicator_set_id}}/data/group_members.csv"),
        group_metadata=update(f"{engine_dir}/indicator_sets/{{indicator_set_id}}/data/group_metadata.csv"),
        environment=update(f"{engine_dir}/indicator_sets/{{indicator_set_id}}/indicator_environment.json"),
        # derive_system_indicators copies the checked benchmark report into the set so
        # a reader never needs the installed asset. Declared here because an
        # undeclared file is one Snakemake neither tracks nor cleans: a
        # --forcerun or partial clean would leave a manifest whose
        # return_level_benchmark.sha256 binds a file that is gone.
        benchmark=update(f"{engine_dir}/indicator_sets/{{indicator_set_id}}/return_level_benchmark.json"),
        # The per-variable tables vary with each set's selection, so they are
        # written and inventoried by the job (staged, manifest last) rather
        # than declared here; the runner validates retained sets before any
        # forced producer runs.
    wildcard_constraints:
        indicator_set_id=rf"[a-f0-9]{{{SHORT_DIGEST_CHARS}}}",
    run:
        from blueearth_cst.experiment.indicator_plan import publish_indicator_set
        from blueearth_cst.experiment.console_reporting import report_indicator_set
        reused = Path(output.manifest).is_file()
        manifest = publish_indicator_set(exp_dir, read_canonical_json(Path(input.plan)))
        report_indicator_set(Path(exp_dir), manifest, INDICATOR_SETS, reused=reused)


# simulations_and_indicators -- target (unnumbered)
rule simulations_and_indicators:
    # The console counts resolved inputs after the indicator checkpoints.
    message: target_banner("simulations_and_indicators", None, compact=True)
    input:
        _indicator_plan_paths,
        _selected_indicator_outputs,


# --------------------------------------------------------------------------
# Console: the run's own opening and closing block, and the Snakemake restyle.
#
# Defined HERE, once, rather than in either included module: `simulate_system.smk`
# includes exactly one of `simulate_and_indicators.smk` and `indicators_only.smk`, so a
# copy in each would be two definitions of the same three handlers to keep in
# step. Both modules therefore define `project_dir`, `experiment`,
# `LOG_PARTS_DIR`, `WORKFLOW_LOG_NAME` and `BENCHMARKS_NAME` -- see the note in
# `indicators_only.smk`, which has no log or benchmark table to name.
#
# Restored 2026-09-16. The R12 split (868c4b7c) carried the rule bodies out of
# `run_stress_test.smk` but not the file scaffolding, so WF4 ran with no
# `_ConsoleHandler` at all.
# --------------------------------------------------------------------------

# Wall clock for the end-of-run summary. Taken at PARSE, not in `onstart`:
# Snakemake exposes no run duration to these handlers, and parse-to-finish is
# the interval a person actually waited. On WF4 that matters more than
# elsewhere, since the parse-time `_live_simulation_inputs()` above shells out
# to Julia for the simulator headers.
_RUN_STARTED = time.monotonic()

# One tally per run, opened at parse time and published to every job process
# through the environment: a rule prints its warnings from a process of its
# own, and the verdict below is written by this one. See
# `snake_utils.declare_warning_tally`.
declare_warning_tally(project_dir, WORKFLOW_LOG_NAME)


def _summary(failed):
    """Print the end-of-run block to STDERR, beside Snakemake's own output.

    stderr because that is where Snakemake writes its console, so a redirect
    that captures one captures both. Never raises: a summary that broke a
    successful run would be the worst possible trade for a convenience.
    """
    if os.environ.get("CST_SIMULATION_FINALIZE"):
        return  # The launcher publishes results and owns the final verdict.
    try:
        # One write, blank line before it -- see the note on `_header`.
        sys.stderr.write(
            "\n"
            + run_summary(
                "wf4 simulate_system",
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
    costs a line of console rather than the run -- which matters here, where
    `experiment` comes from whichever of the two modules was included.
    """
    try:
        # One write, carrying its own blank line on both sides: the block is the
        # first thing this toolbox puts on the console and it must not open
        # flush against Snakemake's preamble nor close flush against its
        # `Job stats:`. `open_run_header` owns that spacing.
        batching = globals().get("BATCH_PLAN_SUMMARY")
        from blueearth_cst.shared.run_log_core import plural
        from blueearth_cst.shared.snake_utils import listed, resolve_water_year_start
        details = {}
        if OPERATION == "indicators-only":
            details["simulation results"] = "Using saved simulations"
        else:
            details["scenario series"] = plural(len(RUN_IDS), "selected series", "selected series")
            details["simulation period"] = f"{SIM_WINDOW_START}–{SIM_WINDOW_END}"
            if globals().get("_simulation_complete"):
                details["simulation results"] = "Reusing saved simulations"
        if INDICATOR_SETS:
            details["indicator sets"] = listed([item.set_id for item in INDICATOR_SETS])
        details["water year"] = f"Starts in {resolve_water_year_start((_console_project.get('climate') or {}).get('water_year_start'))}"
        analysis = "Indicators from saved simulations"
        if OPERATION != "indicators-only":
            analysis = "Hydrological simulation and indicators" if INDICATOR_SETS else "Hydrological simulation"
        open_run_header(
            "wf4 simulate_system", project_dir, config_path,
            experiment=experiment, analysis=analysis, **details,
            **({"parallel simulations": batching} if batching and OPERATION != "indicators-only" else {}),
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
    # The owned runner distinguishes pre-execution refusals from partial job
    # failures, for which completed indicator sets can still be published.
    if os.environ.get("CST_SIMULATION_STARTED"):
        Path(os.environ["CST_SIMULATION_STARTED"]).write_text("started\n", encoding="utf-8")
    # Restyle Snakemake's own console output into this toolbox's grammar (one
    # line per job start and end). Here and not at parse time: the logging
    # stack does not exist yet then. Fail-open; see install_console_style.
    install_console_style()
    _header()
    if globals().get("_batch_resource_decision") is not None:
        from blueearth_cst.shared.resource_tracking import write_batch_decision
        write_batch_decision("plan", _batch_resource_decision,
            threads=int(_threads), run_ids=RUN_IDS)


onsuccess:
    _summary(failed=False)


onerror:
    _summary(failed=True)
