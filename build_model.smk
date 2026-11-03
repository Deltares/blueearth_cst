# Current guidance: dev/reference/toolbox-spatial-foundation.md
# Current guidance: dev/reference/wf1-model-construction.md
import sys
import time
import uuid
from pathlib import Path

# Shared helpers live in blueearth_cst/; make them importable regardless of the working
# directory by prepending this Snakefile's own directory to sys.path.
# See dev/records/milestones/r03/model-builder-design.md §3.
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.provenance import append_journal_line, configuration_inputs_digest, effective_config_digest, environment_file_hashes, file_sha256, journal_event, referenced_inputs_for_digest, toolbox_identity
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, DEFAULT_JULIA_THREADS, DEFAULT_WFLOW_OUTVARS, catalog_root, climate_store_rule, declare_path_tokens, declare_project_root, declare_warning_tally, get_config, historical_window_bounds, julia_prefix, listed, patch_psutil_windows_benchmark, region_rule, resolve_simulation_window, resolve_water_year_start, spatial_units_rule, validate_historical_window, warning_count
from blueearth_cst.shared.console_style import defer_warning, install_console_style, open_run_header, RuleRegistry, rule_banner, run_header, run_summary, target_banner, warn_if_project_dir_in_repo
from blueearth_cst.shared.console_style import header_config_path
from blueearth_cst.shared.config_composition import compose_config
from blueearth_cst.shared.workflow_archive_launch import captured_projection, require_capture
from blueearth_cst.spatial.config import parse_spatial_config
# The canonical climate figure set (rules 1.12 and 1.15 both draw it). Imported
# for figure_names() ONLY, so every figure is declared from the same list the
# plotter writes from and the two cannot drift.
from blueearth_cst.climate_analysis.climate_figures import figure_names

# The thematic map family rule 1.11 draws beside elevation_basin. Same reason as
# above: the output list comes from the registry the plotter iterates, so a
# figure cannot be added in one place and forgotten in the other.
from blueearth_cst.shared.plot_spatial_maps import figure_paths
from blueearth_cst.shared.plot_evaluation import STATION_PLOT_DIRNAME
# Recognises both "unset" spellings (YAML null and the legacy "None" string),
# so an unset observation key never becomes a declared input.
from blueearth_cst.shared.gauges import is_unset
from blueearth_cst.shared.wflow_outputs import CODES as WFLOW_OUTPUT_CODES
from blueearth_cst.shared.wflow_outputs import BASAVG_FIGURE_STEMS, SUBBASIN_PLOT_DIRNAME

# Windows: make Snakemake's benchmark memory/IO/CPU indicators work (else all NA).
patch_psutil_windows_benchmark()

# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention — keep it even though the Snakefile itself uses `config`.
config_path = workflow.configfiles[0]
CAPTURE_RECORD = require_capture("build_model", config_path)

# The consumed-key PROJECTION: the config paths this workflow actually reads.
# Digesting the projection rather than the whole file is what stops a WF3-only
# edit from re-firing WF1's record. A path this config lacks raises at parse
# time -- the declaration is a claim about what the workflow reads, so a typo
# must not quietly narrow the digest.
#
# HOISTED above the first config read (R13 D-8.2): the projection is a
# config-independent literal, and `compose_config` derives R(entry) from it, so
# it has to be known before any section is touched. Moving it changed no value.
CONFIG_PROJECTION = ("project", "basin", "climate", "model", "workflows.build_model")
# Presentation-only plotting settings are tracked by renderer params below,
# independently of scientific configuration identity.

# COMPOSE: the project file carries `{enabled, config_path}` stanzas and each
# workflow's settings live in its own file. This merges them back into exactly
# the mapping every reader below already expects (R13 D-8.1).
#
# The result is REBOUND to the Snakefile-global `config`, deliberately and not
# as a style choice: `check_project_consistency` takes its live config from
# `sm.config` -- Snakemake's `workflow.config` -- so binding elsewhere would
# leave WF3's drift guard comparing a two-key stanza against a full recorded
# section and failing rule 3.01 after WF1 and WF2 had already run.
config, WORKFLOW_CONFIG_PATHS = compose_config(
    config, config_path, entry="build_model", declared_sections=CONFIG_PROJECTION,
)
from blueearth_cst.shared.plot_config import resolve_annotation
from blueearth_cst.shared.plot_setup import setup_digest
plot_annotation = resolve_annotation(config)
plot_setup_digest = setup_digest()
CAPTURE_DIGESTS = captured_projection(CAPTURE_RECORD, config, ADVANCED_SETTINGS)
# Sorted so the declared input list below does not churn on dict order.
WF_CONFIG_PATHS = sorted(WORKFLOW_CONFIG_PATHS.values())

# Portable tee wrapper for the shell rules below: keeps live console output AND
# preserves the child's exit code (a bare `| tee` masks failures on cmd.exe --
# no pipefail there). See blueearth_cst/shared/run_logged.py / blueearth_cst.shared.snake_utils.run_and_tee.
run_logged = str(Path(workflow.basedir) / "blueearth_cst" / "shared" / "run_logged.py")

# R01 schema — three top-level sections. Read each into a local dict.
project_cfg = config["project"]
# R14 D-7.2: `shared:` dissolved into sections by KIND. `climate_cfg` is the
# only new binding -- `basin:` and `model:` are read at their use sites, which
# is where the v1 `shared_cfg` indirection was buying nothing.
climate_cfg = config.get("climate") or {}
# The water year the climate figures aggregate on, from the one shared key
# WF2 and WF3 also read. Figures are terminal artifacts, so this changes no
# number -- but a figure labelled 'annual' should mean the basin's year.
WATER_YEAR_START = resolve_water_year_start(get_config(climate_cfg, "water_year_start"))
my_cfg = config["workflows"]["build_model"]

# Project — paths and external resources
project_dir = get_config(project_cfg, "project_dir", optional=False)
# O-22: make the two-tier project_dir rule mechanical rather than
# documentary. Warns, never raises; test_case/ is the one exemption.
warn_if_project_dir_in_repo(project_dir, workflow.basedir)
DATA_SOURCES = get_config(project_cfg, "catalog", optional=False)  # C-40

# Shared — multi-workflow scientific knobs
basin_cfg = config["basin"]
spatial_cfg = parse_spatial_config(basin_cfg, my_cfg)
model_region = get_config(basin_cfg, "region", optional=False)
model_resolution = spatial_cfg.resolution
# Catalog ENTRY NAMES for the model-free basin delineation the shared climate
# store uses (R07 B1 / ext1-01). Both OPTIONAL: the defaults equal the shipped
# build template's setup_basemaps values, so a config that declares neither is
# unaffected and rule 3.01's guard digest stays byte-identical. Rule 1.05
# cross-checks them against the template and fails loud on a disagreement.
basin_hydrography = spatial_cfg.hydrography
basin_index = spatial_cfg.basin_index
historical_window = get_config(climate_cfg, "window", optional=False)
# ONE minimum window for the whole toolbox — snake_utils.MIN_HISTORICAL_YEARS
# (16), set by weathergenr's wavelet minimum and enforced identically here and
# at extraction. WF1 rejects it too, deliberately: a record too short for a
# stress test is a misconfigured project, and letting WF1 build a model on it
# only moves the failure to the workflow least able to explain it
# (pre-board record 45e3a564 R7-6, R3). Parse time, before any rule executes — same
# stance as the eobs rejection below. Whether the staged source actually COVERS
# the requested window is a different question, checked at rule 1.03.
validate_historical_window(historical_window)
# The period the MODEL IS RUN over, which is a different question from how much
# record to extract (above). Optional; absent means exact passthrough of
# historical_window, so every config predating this key is unaffected. Resolved
# and validated at parse time for the same reason the window above is.
#
# Editing it re-runs rule 1.09 through Snakemake's params trigger, which
# rebuilds the forcing and so re-runs 1.13 — and since 1.13's output.csv is
# temp(), that is the whole model. Correct, but not cheap.
# The two windows are authored in DIFFERENT files now, so the refusal names
# both -- a user with two files open should not have to work out which one
# the message means (R13 §12.5).
simulation_window = resolve_simulation_window(
    climate_cfg,
    my_cfg,
    shared_source=config_path,
    model_source=WORKFLOW_CONFIG_PATHS.get("build_model"),
)
SIMULATION_BOUNDS = historical_window_bounds(simulation_window)
clim_source = get_config(climate_cfg, "selected", optional=False)
# Wflow.jl thread count for rule 1.13. OPTIONAL — the default is P3-3's frozen
# baseline value, so an existing config is unaffected. Config-driven rather than
# inline so a deployment can tune it to its basin (a production basin has the
# cell-parallelism the 384-cell test fixture lacks) without a Snakefile edit.
# NOT Snakemake's `threads:`, which caps at --cores: see DEFAULT_JULIA_THREADS.
julia_threads = DEFAULT_JULIA_THREADS  # C-54: no project override; advanced_settings owns it
# Carries the juliaup version pin too; both are validated at parse time, before
# any rule can put them in a shell body.
wflow_julia = julia_prefix(julia_threads)

# P3-2a bounded support (design ext2-3): the wf1 raw-climate path (rule 1.03 +
# the model-parity plot re-source) supports era5, chirps and chirps_global
# only. eobs fails here at DAG-parse time — before any rule executes — so the
# rejection is early and loud on every run and dry-run.
if clim_source == "eobs":
    raise ValueError(
        "clim_historical: eobs is not supported by the P3-2a wf1 raw-climate "
        "path; supported sources: era5, chirps, chirps_global"
    )

# SHARED, not workflow-owned (R13 D-9.7). WF3 reads this key too -- to derive
# its indicator tables before the DAG is built -- and a key read by more than
# one workflow lives in the project file. It was the last such key, and
# `CROSS_WORKFLOW_READS` is retired with its move.
wflow_outvars = get_config(config.get("model") or {}, "outvars", DEFAULT_WFLOW_OUTVARS)
# `defaults/`, not `templates/` — these are read by rules 1.05/1.06/1.07, and the
# 2026-08-11 split moved them out of the copy-me directory. Rule 1.01 still routes
# them to a `templates/` bin inside the PROJECT: a different meaning of the word,
# and that one did not move. Under the R4 copy predicate a shipped default is
# normally recoverable from the toolbox and so is recorded rather than copied —
# the bin only receives one when the project points the key at its own file.
# C-22: both paths regrouped under `engine:`, the wflow-engine settings this
# workflow hands to hydromt. The group is optional and so are its leaves — the
# defaults are unchanged, so a config that declared neither key behaves as it
# always did.
engine_cfg = get_config(my_cfg, "engine", {}) or {}
model_build_config = get_config(engine_cfg, "build_config", "config/defaults/wflow_build_model.yml")
waterbodies_config = get_config(engine_cfg, "waterbodies_config", "config/defaults/wflow_update_waterbodies.yml")
output_locations = spatial_cfg.gauge_points_path

#: The one outvar rule 1.15 has an observation consumer for. Spelled as
#: `model.outvars` spells it (`shared/indicator_tables.py` is the registry that
#: fixes that vocabulary), because `observations:` keys are drawn from that list
#: verbatim and the loader compares them by string.
OBSERVED_DISCHARGE_VAR = "river discharge"

# C-56: `observations:` is a mapping KEYED BY VARIABLE (D-7.3), whose keys the
# loader has already checked against `model.outvars` — so by the time it is read
# here, every key names a variable the model was asked to output.
#
# Rule 1.15 is the only consumer and it plots discharge, so the discharge entry
# is the one that becomes an `input:`. An observed series for any OTHER variable
# parses, validates and is then read by nothing, which is a bounded coverage the
# repo's rules say must be stated rather than left to be discovered: hence the
# warning instead of a silent drop.
_observations = get_config(my_cfg, "observations", {}) or {}
observations_timeseries = _observations.get(OBSERVED_DISCHARGE_VAR)
_unconsumed_observations = sorted(
    name
    for name, source in _observations.items()
    if name != OBSERVED_DISCHARGE_VAR and not is_unset(source)
)
if _unconsumed_observations:
    # `defer_warning`, not `logger`: this is parse time, before the console
    # style is installed, so a logger call prints in Snakemake's own style and
    # a `warn_row` prints in ours but still above a header that has not been
    # written. Held instead, and printed under the RUN block. See the same fix
    # on WF2's resolution report.
    defer_warning(
        f"Observation variables not used: "
        f"{listed(_unconsumed_observations)}. Rule 1.15 plots "
        f"{OBSERVED_DISCHARGE_VAR} only; these entries will be ignored.", module="config"
    )

# The two OPTIONAL observation inputs, declared as real `input:` entries when
# configured and omitted entirely when not. `output_locations` is the internal
# compatibility name; its canonical config source is shared.basin.gauge_points.
#
# They were `params:` until 2026-08-02, carrying the PATH. Snakemake's params
# trigger compares the value, so editing the FILE changed nothing it could see:
# renumbering the gauge-point file left rule 1.08's model gauges on the old ids
# while the observations moved to the new ones, the join then matched nothing,
# and performance_indicators.csv emptied without a word. As `input:` the content
# mtime-triggers 1.08 (which writes the ids into the model) and 1.15 (which
# reads them back), so the rebuild happens by itself.
#
# Two consequences. A configured path that does not exist is a
# MissingInputException naming the file and the rule -- but ONLY once the rule
# actually needs to run. Verified: on a fresh project, or under --forcerun, the
# typo reds the dry-run; against a project whose outputs are already up to date
# Snakemake prunes the job before validating its inputs, so the typo sits latent
# until something else triggers a rebuild. Earlier and louder than the runtime
# check it replaces, but not a parse-time guarantee.
#
# And an unset key contributes no entry at all, so a config without observations
# declares nothing and cannot fail on it; the consumers read
# `getattr(sm.input, ..., None)`, the same shape rule 1.15 already uses for the
# chirps orography branch.
_locations_input = (
    {} if is_unset(output_locations) else {"output_locations": output_locations}
)
_observations_input = (
    {}
    if is_unset(observations_timeseries)
    else {"observations_timeseries": observations_timeseries}
)


# The content-addressed config bundle was removed here (config-snapshot
# redesign, 2026-08-13). It had no readers, and naming its directory after a
# digest over the WHOLE config meant an edit to any other workflow's section
# minted a fresh one. What replaces it is `run_record.yml`: current-only and
# one per workflow.
RUN_RECORD = f"{project_dir}/config/build_model/run_record.yml"

# Every external file this workflow's configuration points at. Hashed at parse
# time so the digest below moves when one is edited IN PLACE -- the recorded
# hash alone would move without re-firing anything.
#
# The per-workflow config files are in here for exactly that reason (R13
# D-10.5): after the split they hold the settings the project file used to, so
# leaving them out would let the most-edited config file in the project be
# edited in place without moving any digest. Derived from the one dict
# `compose_config` returned, so this set and `copy_config_files`' cannot drift.
CONFIG_REFERENCES = [
    *[(f"workflow_config_{name}", path)
      for name, path in sorted(WORKFLOW_CONFIG_PATHS.items())],
    ("engine.build_config", model_build_config),
    ("engine.waterbodies_config", waterbodies_config),
    *[("data_catalog", source) for source in
      (DATA_SOURCES if isinstance(DATA_SOURCES, (list, tuple)) else [DATA_SOURCES])],
    *[("output_locations", source) for source in _locations_input.values()],
    *[("observations_timeseries", source) for source in _observations_input.values()],
]

EFFECTIVE_CONFIG_DIGEST = CAPTURE_DIGESTS[0] if CAPTURE_DIGESTS else effective_config_digest(
    config, ADVANCED_SETTINGS, CONFIG_PROJECTION
)
# Full provenance includes Git state. The sidecar's refresh trigger below
# excludes it: a new commit alone does not produce new scientific outputs.
CONFIGURATION_INPUTS_DIGEST = CAPTURE_DIGESTS[1] if CAPTURE_DIGESTS else configuration_inputs_digest(
    EFFECTIVE_CONFIG_DIGEST,
    toolbox_identity(),
    environment_file_hashes(),
    referenced_inputs_for_digest(CONFIG_REFERENCES),
)

# R9 P2 commit 1: the live HydroMT model root moves under `models/`, the tree's
# home for engine-shaped model artifacts (design v10). ONE definition per
# workflow — every model-internal path below is built from it, and WF3 derives
# the same string from the same shape, so the two cannot drift.
#
# Not a config value and not part of rule 3.01's guard digest, which serializes
# `project`, `shared.basin`, `workflows.build_model` and
# `workflows.analyze_projections` only — so this move does not flip the digest
# and cannot make an existing project's drift guard refuse to run.
basin_dir = f"{project_dir}/models/hydrology/wflow"

# R9 P2 commit 2: engine-neutral geometry -- the region, the vector layers, the
# location registry and the generated catalog that DESCRIBES them -- lives under
# `data/`, reusable across engines and workflows (design v10). One binding, as
# with basin_dir above.
spatial_dir = f"{project_dir}/data/spatial"

# Rule 1.12 draws the model forcing; source diagnostics belong to WF0.
# Variable x kind is a fixed cross-product, fully knowable at parse time.
FORCING_PLOTS_DIR = f"{basin_dir}/forcing/plots"
_forcing_climate_pngs = [f"{FORCING_PLOTS_DIR}/{name}" for name in figure_names("forcing")]

# --- The shared historical-climate store (R07 B1) -----------------------------
# ONE producer contract, built here and identically in
# run_stress_test.smk, and splatted into rule 1.03 / rule 3.08 below.
# Everything content- or execution-determining (script, the single catalog
# input, outputs, params) comes from this object; only message/log/benchmark are
# workflow-local. tests/test_climate_store_contract.py parses both workflows and
# fails on ANY other difference.
CLIMATE_STORE = climate_store_rule(
    project_dir=project_dir,
    model_region=model_region,
    clim_source=clim_source,
    historical_window=historical_window,
    data_sources=DATA_SOURCES,
    hydrography=basin_hydrography,
    basin_index=basin_index,
)
store_dir = CLIMATE_STORE.store_dir

# --- The one project region artifact -------------------------------
# Same pattern, same reason: ONE producer contract, built here and identically
# in the other two workflows, splatted into rule 1.01 below. Only
# message/log/benchmark are workflow-local, and
# tests/test_region_rule.py parses all three workflows and fails on ANY other
# difference. `shared.basin.region` used to be delineated twice per project —
# here on the way to basins.geojson, and once per climate-store key — and WF2
# ran the whole climate-store producer just to obtain the polygon.
REGION = region_rule(
    project_dir=project_dir,
    model_region=model_region,
    data_sources=DATA_SOURCES,
    hydrography=basin_hydrography,
    basin_index=basin_index,
)

# --- The shared vector foundation -------------------------------
# Third and last of the shared producer contracts, same pattern again: built
# here and identically in the other two workflows, splatted into rule 1.02
# below, and tests/test_spatial_units_rule.py parses all three workflows and
# fails on ANY difference beyond message/log/benchmark.
#
# `parse_spatial_config(basin_cfg)` WITHOUT `my_cfg`, deliberately, and this is
# the one call site where that differs from the module-level `spatial_cfg`
# above: the shared rule's params must be a pure function of
# `project` + `shared.basin`, so the deprecated
# `workflows.build_model.output_locations` fallback cannot feed it.
#
# THE COST WAS REAL AND SILENT, AND IS NOW REFUSED AT THE SEAM. Rule 1.05 used
# to receive `my_cfg` and so honoured the legacy key on its way to the
# partition. A config that set ONLY `workflows.build_model.output_locations`
# then reached this rule with no gauge points at all, so its parent basins fell
# back to the AUTOMATIC partition: different subbasins, a different
# location_registry.csv, and different wflow_id values — with no error, because
# an absent gauge file is a legitimate configuration. Rule 1.08 would then add
# gauges to a model whose subbasins were derived without them.
#
# The shared-input contract forbids params drawn from workflow-local settings:
# projections-only configs do not carry the model section, so the producer would
# differ per invoking workflow. The asymmetry is refused one level up
# instead: `resolve_gauge_points_path` RAISES on a legacy-only config, at the
# module-level `parse_spatial_config` call above (line 47), before the DAG is
# built. Both keys naming the same path still parse. A migrating project is
# therefore stopped at parse time with the canonical key spelled out, rather
# than told the old one is deprecated and left to discover the partition moved.
#
# Fixed 2026-08-08 after a real project hit it: five automatic outlets in the
# registry, observations declaring the four station ids the gauge file pinned,
# and the failure surfacing at rule 1.15 a whole model build later. No tracked
# config was ever affected (all set `shared.basin.gauge_points`, the canonical
# key, checked 2026-08-06), which is why the suite stayed green throughout.
SPATIAL_UNITS = spatial_units_rule(
    project_dir=project_dir,
    spatial_config=parse_spatial_config(basin_cfg),
    data_sources=DATA_SOURCES,
)

from blueearth_cst.shared.map_reference import historical_map_reference_rule
from blueearth_cst.shared.wf3_science import water_year_start_number

MAP_REFERENCE = historical_map_reference_rule(
    project_dir=project_dir, model_region=model_region, climate=climate_cfg,
    data_sources=DATA_SOURCES, region_geojson=REGION.outputs["region_geojson"],
    basin_geojson=SPATIAL_UNITS.outputs["basins"],
    m0=water_year_start_number(WATER_YEAR_START),
    hydrography=basin_hydrography, basin_index=basin_index,
)
MAP_CANDIDATE_STORES = {
    source: climate_store_rule(
        project_dir=project_dir, model_region=model_region, clim_source=source,
        historical_window=historical_window, data_sources=DATA_SOURCES,
        hydrography=basin_hydrography, basin_index=basin_index,
        enforce_min_years=False,
        forcing_required=source not in {"chirps", "chirps_global"},
    )
    for source in dict.fromkeys(climate_cfg["sources"]) if source != clim_source
}

# Rule numbering (comment headers + log/benchmark filenames) uses `W.NN` = the
# rule's position in this workflow's LOGICAL order — data, then model build,
# then run, then records. Contiguous, and every dependency points from a lower
# number to a higher one, so a rule can never depend on something numbered after
# it. Positional since [R10-5] (2026-08-06); it used to be a stable identifier
# assigned at rule creation, which left it uncorrelated with anything.
#
# STILL A READING AID, NOT EXECUTION ORDER. Snakemake resolves execution from
# the DAG, so rules on separate branches run concurrently — 1.10-1.12 are
# parallel leaves off 1.09, and 1.03's climate branch runs beside the whole
# model build. Low-to-high says "cannot depend on", not "runs before".
#
# DEFINITION ORDER IN THIS FILE IS NOT THE NUMBER ORDER, and is not required to
# be: module-level code is interleaved between rules and depends on its position
# (`_basavg_pngs` before 1.15, `_source_plot_inputs` before 1.04), so reordering
# the blocks to match would be a behaviour risk taken for cosmetics. Read the
# numbers, not the file order. All three workflows share project_dir/logs, so
# the leading digit keeps them disambiguated and globally sortable.
#
# DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix (1.09b) until the next
# deliberate sweep — renumbering is a migration, not an edit.
# Convention: dev/reference/dev-naming-conventions.md §9; historical translations:
# dev/records/milestones/r10/rule-numbering-crosswalks.md.

# --- log layout ---------------------------------------------------------------
# EVERY WF1 rule that logs writes a PART under logs/_parts/, and rule 1.18 merges
# the parts into ONE logs/wf1_build_model.log, then deletes them. So the only
# WF1 file left in logs/ after a full run is that merged log — matching WF2 (2.08)
# and WF3 (3.18), and the deal benchmarks/wf1_benchmarks.md already had.
#
# LOG_RULES is the merge order: rule LABELS, not part paths. merge_logs lists each
# label's part dir to find its members, so a fan-out width lives only in the rule
# that owns it (WF1 has no fan-out today; WF3 does). Naming the labels rather than
# globbing `_parts/` keeps an orphan dir from a renamed rule out of the file.
#
# ORDER IS BY RULE NUMBER, and since [R10-5] that is also dependency order, so
# the merged log reads as the workflow does. tests/test_log_rules_contract.py
# asserts it — the assertion was deferred until the renumber precisely because
# before it, number order and execution order disagreed and nobody had ruled on
# which the list should follow.
#
# Neither gather rule declares a `log:`, so neither has a section. No entry for the merged-away forcing-recipe rule: [R10-1]
# folded it into 1.09, so its label has no producer, and an orphan label
# contributes an empty "no part from this run" section forever.
#
# The region rule was MISSING from this list until 2026-08-04, the first of four
# instances of one defect. It declares a `log:` part like every other rule here,
# but an unlisted label is not an error: merge_logs only looks up the labels it
# is given, so the section was silently absent from every merged log and its
# part was stranded under logs/_parts/ on every run. The benchmark gather does
# not share that failure mode -- wf1_benchmarks.md listed it all along -- which
# is why the two disagreed. That whole class is now mechanically checked, in
# both directions, for all three workflows.
#
# A `]` inside this block is fine again. It was not: test_model_reference.py
# sliced the list to the first one anywhere, so a bracketed followups reference
# truncated the label set it read. That parser is gone ([R10-10]), and the one
# that replaced it anchors the closing bracket at column 0.
from blueearth_cst.shared.execution_outputs import workflow_paths, declare_warning_tally
WORKFLOW_LOG_NAME = "wf1_build_model.log"

# One tally per run, opened at parse time and published to every job process
# through the environment: a rule prints its warnings from a process of its
# own, and the verdict below is written by this one. See
# `snake_utils.declare_warning_tally`.
declare_warning_tally(project_dir, WORKFLOW_LOG_NAME)
LOG_PARTS_DIR = f"{project_dir}/logs/_parts"
_paths = workflow_paths(project_dir, "wf1", WORKFLOW_LOG_NAME, "wf1_benchmarks.md", LOG_PARTS_DIR, f"{project_dir}/benchmarks/_parts")
WORKFLOW_LOG_PATH, BENCHMARK_OUTPUT_PATH = _paths.log, _paths.benchmark
LOG_PARTS_DIR, BENCH_PARTS_DIR = _paths.log_parts, _paths.benchmark_parts

# The run's key folders, stated ONCE. `run_header` prints them at the top of the
# console and every rule log repeats them in its own header; below that, any
# line mentioning one of these directories prints `<model>/staticmaps.nc`
# instead of twelve path components. `data` is the external tree, read out of
# the catalog because it is the one folder a project neither owns nor derives.
_console_catalogs = DATA_SOURCES if isinstance(DATA_SOURCES, (list, tuple)) else [DATA_SOURCES]
declare_path_tokens(
    console_only={
        "model-config": f"{basin_dir}/config",
        **{
            ("project-catalog" if len(_console_catalogs) == 1 else f"project-catalog-{i}"): path
            for i, path in enumerate(_console_catalogs, 1)
        },
    },
    data=catalog_root(DATA_SOURCES),
    model=basin_dir,
    climate=store_dir,
)
declare_project_root(project_dir)
RULES = RuleRegistry(LOG_PARTS_DIR, BENCH_PARTS_DIR)
LOG_RULES = RULES.log_rules
DELINEATE_REGION = RULES.logged("1.01", "delineate_region", summary="delineate the basin boundary")
DELINEATE_SUBBASINS_AND_RIVERS = RULES.logged("1.02", "delineate_subbasins_and_rivers", summary="derive subbasins, rivers, and output locations")
EXTRACT_CLIMATE_DATASETS = RULES.logged("1.03", "extract_climate_datasets", summary="extract a basin climate grid")
if MAP_CANDIDATE_STORES:
    EXTRACT_MAP_CANDIDATES = RULES.logged("1.03b", "extract_map_candidates", summary="extract native candidate climate for the shared map reference")
PRODUCE_MAP_REFERENCE = RULES.logged("1.03c", "produce_historical_map_reference", summary="prepare shared historical map display classes")
PREPARE_SOURCE_MAPS = RULES.logged("1.03d", "prepare_historical_map_fields", summary="prepare shared native-grid map fields")
PREPARE_LAND_AND_SOIL_MAPS = RULES.logged("1.05", "prepare_land_and_soil_maps", summary="prepare shared terrain, land-cover, and soil layers")
BUILD_WFLOW_MODEL = RULES.logged("1.06", "build_wflow_model", summary="parameterize from spatial foundation and global datasets")
ADD_RESERVOIRS_LAKES_GLACIERS = RULES.logged("1.07", "add_reservoirs_lakes_glaciers", summary="apply configured reservoir, lake, and glacier setup")
DECLARE_GAUGES_AND_OUTPUTS = RULES.logged("1.08", "declare_gauges_and_outputs", summary="configure discharge locations and output variables")
ADD_CLIMATE_FORCING = RULES.logged("1.09", "add_climate_forcing", summary="prepare historical forcing")
WRITE_GAUGE_INDEX = RULES.logged("1.10", "write_gauge_index", summary="write the outlet-to-location identity crosswalk")
PLOT_BASIN_MAP = RULES.logged("1.11", "plot_basin_map", summary="render basin and spatial-input maps")
PLOT_MODEL_FORCING = RULES.logged("1.12", "plot_model_forcing", summary="render model-forcing maps and time series")
RUN_HISTORICAL_SIMULATION = RULES.logged("1.13", "run_historical_simulation", summary="run Wflow-SBM with historical forcing")
EXPORT_SIMULATION_TABLES = RULES.logged("1.14", "export_simulation_tables", summary="export simulation time-series tables")
PLOT_MODEL_EVALUATION = RULES.logged("1.15", "plot_model_evaluation", summary="compare simulation outputs with available discharge observations")
WRITE_RUN_METADATA = RULES.banner_only("1.16", "write_run_metadata", summary="record configuration and input digests for these outputs")
GATHER_BENCHMARKS = RULES.banner_only("1.17", "gather_benchmarks", summary="merge rule timing and resource-use records")
GATHER_LOGS = RULES.banner_only("1.18", "gather_logs", summary="merge rule logs")

# WF1_TERMINALS — the artifacts with no WF1 consumer: every producing rule is
# upstream of them and none of them feeds another rule. That is exactly the
# input set the two gather rules (1.17 benchmarks, 1.18 logs) need, which
# schedules each one LAST without restating the targets in three places.
#
# NOT terminals, and so not in this list: the config snapshot (1.01 runs
# independently of the build and needs no gather waiting on it) and the two
# gather outputs themselves.
WF1_TERMINALS = [
    # The evaluation rule's terminal is its INDICATORS TABLE, not one of its
    # figures: since 2026-08-10 the figures are keyed by wflow_id and none of
    # their names is knowable at parse time. The table is written by the same
    # rule and is config-invariant, so the gather rules wait on exactly what
    # they waited on before.
    f"{basin_dir}/evaluation/performance_indicators.csv",
    # The Excel-ready tables (1.14). A product of the workflow rather than an
    # incidental by-product, so they are rule-`all` members like the figures —
    # nothing downstream consumes them, so without this they never build.
    # ONE representative is enough, as with the nine figures below: 1.14 writes
    # the whole set (WFLOW_TABLE_PATHS) in a single job, so requesting the
    # discharge table schedules every other table with it.
    f"{basin_dir}/run_default/output_q.csv",
    f"{spatial_dir}/plots/elevation_basin.png",
    # One representative schedules the full model-forcing figure set.
    f"{FORCING_PLOTS_DIR}/forcing_precip_map.png",
    f"{basin_dir}/staticgeoms/outlet_index.csv",
    # P1 spatial foundation. One representative output schedules the complete
    # multi-output rule and makes both gather rules wait for its log/benchmark.
    f"{spatial_dir}/spatial_catalog.yml",
    # The staleness sidecar (1.16). A terminal in its own right: it consumes
    # the indicators table and nothing consumes it, and listing it here is what
    # schedules it at all -- no rule reads a sidecar, so without this it would
    # never build.
    f"{basin_dir}/run_metadata.json",
]

# all — target aggregator: full historical build + performance plots
#
# Hoisted into a list so `message:` can print one target per line without
# restating them. Snakemake flattens a list argument, so `input: WF1_TARGETS`
# declares terminals, the optional launcher record and unowned diagnostic targets.
WF1_TARGETS = [
    *WF1_TERMINALS,
    *([RUN_RECORD] if os.environ.get("BLUEEARTH_WF012_CAPTURE_CONTEXT") else []),
    *_paths.gather_targets.values(),
]

# Figures are deliberately not targets: Snakemake's per-figure change checks
# made every run slow. Instead, request only the plotting rules' declared
# outputs that are missing now, so a deleted figure or plot folder is redrawn at
# the cost of one stat per path. A file deleted inside a declared directory,
# such as one station figure, is not detected.
_FIGURE_RULES = ("plot_basin_map", "plot_model_forcing", "plot_model_evaluation")


def _missing_figure_outputs(wildcards):
    return [
        path
        for name in _FIGURE_RULES
        for path in getattr(rules, name).output
        if not os.path.exists(path)
    ]


rule all:
    message: target_banner("all", WF1_TARGETS, project_dir, compact=True)
    input:
        WF1_TARGETS,
        missing_figures = _missing_figure_outputs,

# 1.01  delineate_region — the one project region artifact.
# Byte-identical to 2.01 and 3.03 except message/log/benchmark; everything
# else is splatted from REGION so the three cannot drift.
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

# 1.02  delineate_subbasins_and_rivers — the shared vector foundation.
# Byte-identical to 2.02 and 3.04 except message/log/benchmark; everything
# else is splatted from SPATIAL_UNITS so the three cannot drift.
rule delineate_subbasins_and_rivers:
    message: DELINEATE_SUBBASINS_AND_RIVERS.banner()
    input:
        **SPATIAL_UNITS.inputs,
    params:
        **SPATIAL_UNITS.params,
    output:
        **SPATIAL_UNITS.outputs,
    log:
        DELINEATE_SUBBASINS_AND_RIVERS.log(),
    benchmark:
        DELINEATE_SUBBASINS_AND_RIVERS.benchmark(),
    script: SPATIAL_UNITS.script

# 1.05  prepare_land_and_soil_maps — the thematic raster stack, WF1 ONLY
#
# The thematic raster stage. The vector layers and the registry come from
# 1.02 now; this rule folds LULC/LAI/soil onto the grid 1.02 handed it and
# writes the model-build interface. `hydrography` is the SEAM INTERMEDIATE
# carrying the whole hydrography grid stack that used to cross this boundary in
# memory, and re-deriving it here would mean two producers of one value.
rule prepare_land_and_soil_maps:
    message: PREPARE_LAND_AND_SOIL_MAPS.banner()
    input:
        # Consumed settings are tracked by basin_config/model_config params.
        # Raw config inputs would rebuild the model after a presentation edit.
        data_catalogs = DATA_SOURCES,
        hydrography = SPATIAL_UNITS.outputs["hydrography"],
        basins = SPATIAL_UNITS.outputs["basins"],
        subbasins = SPATIAL_UNITS.outputs["subbasins"],
        catchments = SPATIAL_UNITS.outputs["catchments"],
        rivers = SPATIAL_UNITS.outputs["rivers"],
        river_attributes = SPATIAL_UNITS.outputs["river_attributes"],
        locations = SPATIAL_UNITS.outputs["locations"],
        location_registry = SPATIAL_UNITS.outputs["location_registry"],
        scientific_code = [
            "blueearth_cst/spatial/config.py",
            "blueearth_cst/spatial/products.py",
        ],
    output:
        spatial_maps = f"{spatial_dir}/spatial_maps.nc",
        spatial_catalog = f"{spatial_dir}/spatial_catalog.yml",
        spatial_report = f"{spatial_dir}/spatial_report.yml",
    params:
        basin_config = basin_cfg,
        # Only this legacy field is consumed by parse_spatial_config. Evaluation,
        # observations and simulation settings must not rebuild thematic maps.
        model_config = {key: my_cfg[key] for key in ("output_locations",) if key in my_cfg},
    log:
        PREPARE_LAND_AND_SOIL_MAPS.log(),
    benchmark:
        PREPARE_LAND_AND_SOIL_MAPS.benchmark(),
    script: "blueearth_cst/spatial/prepare_spatial_maps.py"

# 1.06  build_wflow_model — parameterize Wflow-SBM on the P1 spatial foundation
rule build_wflow_model:
    message: BUILD_WFLOW_MODEL.banner()
    input:
        parameter_template = model_build_config,
        spatial_maps = f"{spatial_dir}/spatial_maps.nc",
        basins = f"{spatial_dir}/geoms/basins.geojson",
        subbasins = f"{spatial_dir}/geoms/subbasins.geojson",
        catchments = f"{spatial_dir}/geoms/catchments.geojson",
        rivers = f"{spatial_dir}/geoms/rivers.geojson",
        locations = f"{spatial_dir}/geoms/locations.geojson",
        location_registry = f"{spatial_dir}/location_registry.csv",
        spatial_catalog = f"{spatial_dir}/spatial_catalog.yml",
        spatial_report = f"{spatial_dir}/spatial_report.yml",
        data_catalogs = DATA_SOURCES,
    output:
        staticmaps = f"{basin_dir}/staticmaps.nc",
        wflow_toml = f"{basin_dir}/wflow_sbm.toml",
        region = f"{basin_dir}/staticgeoms/region.geojson",
        outlets = f"{basin_dir}/staticgeoms/outlets.geojson",
        # The post-normalization values hydromt was actually handed (R3). NOT
        # a copy of the parameter template: arguments are popped, P1 objects
        # replace them, and `lulc_mapping_fn` is derived at call time from the
        # grid's own source attribute, so it exists in no file on disk.
        values_used = f"{basin_dir}/hydromt_build_config.yml",
        # Completion sentinel (R7-1). `wflow_sbm.toml` is CREATED here and then
        # updated IN PLACE by rules 1.07-1.09, but only this rule declares it,
        # so a re-fire of the build rule alone used to leave the toml stripped of
        # every section the later rules add -- and the next wflow run died on
        # the missing key. The obvious fix (drop the ancient() on staticmaps in
        # those rules) is WRONG: they mutate staticmaps themselves via
        # mod.write()/mod.close(), so a plain input would make each one
        # re-trigger on its own execution forever. This sentinel is written by
        # nothing else, so it re-fires rule 1.07 without any self-trigger, and
        # the chain cascades from there (1.07 -> .txt -> 1.08 ->
        # .outputs_configured -> 1.09). The forcing-yml hop that used to sit in
        # the middle of that chain is gone: R10-1 merged 1.06 into 1.09, which
        # now writes the yml it used to consume.
        built = touch(f"{basin_dir}/.model_built"),
    log:
        BUILD_WFLOW_MODEL.log(),
    benchmark:
        BUILD_WFLOW_MODEL.benchmark(),
    script: "blueearth_cst/model/build_wflow_model.py"

# 1.07  add_reservoirs_lakes_glaciers — add waterbodies to the built model
# (temporary hydromt fix; can fold back into build_wflow_model when supported)
rule add_reservoirs_lakes_glaciers:
    message: ADD_RESERVOIRS_LAKES_GLACIERS.banner()
    input:
        # ancient(): this rule COMMITS writes back into staticmaps.nc, so a
        # plain input would re-trigger it on its own output. Ordering and
        # rebuild-detection come from the sentinel below instead (R7-1).
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"),
        model_built = f"{basin_dir}/.model_built",
        scientific_code = ["blueearth_cst/shared/wflow_write.py"],
    output:
        text_out = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt",
        # The values each method was handed, plus whether it ran (R3). The
        # status is half the provenance: a skipped method leaves no trace in
        # the model, so values alone would describe water bodies never added.
        values_used = f"{basin_dir}/hydromt_update_waterbodies.yml",
    params:
        data_catalog = DATA_SOURCES,
        config = waterbodies_config,
    log:
        ADD_RESERVOIRS_LAKES_GLACIERS.log(),
    benchmark:
        ADD_RESERVOIRS_LAKES_GLACIERS.benchmark(),
    script:
        "blueearth_cst/model/setup_reservoirs_lakes_glaciers.py"

# 1.08  declare_gauges_and_outputs — add gauges + output variables to the model
rule declare_gauges_and_outputs:
    message: DECLARE_GAUGES_AND_OUTPUTS.banner()
    input:
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"),
        text = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt",
        # P2 already created gauge/outlet maps from this registry. This rule
        # only adds output declarations and verifies those identities.
        location_registry = f"{spatial_dir}/location_registry.csv",
        scientific_code = [
            "blueearth_cst/shared/wflow_write.py",
            "blueearth_cst/shared/wflow_outputs.py",
        ],
    output:
        configured = touch(f"{basin_dir}/.outputs_configured"),
    params:
        outputs = wflow_outvars,
        data_catalog = DATA_SOURCES
    log:
        DECLARE_GAUGES_AND_OUTPUTS.log(),
    benchmark:
        DECLARE_GAUGES_AND_OUTPUTS.benchmark(),
    script:
        "blueearth_cst/model/setup_gauges_and_outputs.py"

# 1.10  write_gauge_index — write the outlet position -> subcatchment-ID mapping
rule write_gauge_index:
    message: WRITE_GAUGE_INDEX.banner()
    input:
        outlets_path = f"{basin_dir}/staticgeoms/outlets.geojson",
        location_registry = f"{spatial_dir}/location_registry.csv",
        # rule 1.09 rewrites every staticgeoms/ layer, so this read
        # needs the terminal anchor. Nothing in WF1 consumes outlet_index.csv,
        # so waiting for the model to be final costs only a schedule slot.
        model_final = ancient(f"{basin_dir}/.model_final"),
    output:
        outlet_index_path = f"{basin_dir}/staticgeoms/outlet_index.csv"
    log:
        WRITE_GAUGE_INDEX.log(),
    benchmark:
        WRITE_GAUGE_INDEX.benchmark(),
    script:
        "blueearth_cst/model/write_outlet_index.py"

# 1.09  add_climate_forcing — assemble the hydromt forcing recipe, then apply it
#
# [R10-1]: rule 1.06 `setup_runtime` was MERGED IN here. It wrote the `steps:`
# YAML below and nothing else read it — one job split across two rules, and the
# reason 1.06's R10 rename was withdrawn rather than replaced. 1.06 is now a gap
# in the numbering.
#
# Two details the merge folded in:
#  - 1.06's `gauges_path` input (staticgeoms/outlets.geojson) was DEAD — the
#    script never opened it — and is dropped rather than carried over.
#  - `.outputs_configured` is inherited from 1.06, and is load-bearing: it is
#    rule 1.08's completion sentinel, so the R7-1 rebuild cascade
#    (1.06 -> 1.07 -> 1.08 -> here) survives the merge.
#
# `script:` rather than `shell:` because Snakemake allows one per rule and the
# halves were one of each; the hydromt command issued is byte-identical to the
# one this rule ran before.
rule add_climate_forcing:
    message: ADD_CLIMATE_FORCING.banner()
    input:
        outputs_configured = f"{basin_dir}/.outputs_configured",
        # The climate store, as the forcing SOURCE (2026-08-10). This rule used
        # to re-read the global dataset from the catalog, a second full pass
        # over the source rule 1.03 had already clipped to this basin. Declaring
        # it makes the reuse a DAG edge rather than a convention, and is why
        # `simulation_window` must now sit inside `historical_window`.
        climate_nc = CLIMATE_STORE.outputs["climate_nc"],
        scientific_code = [
            "blueearth_cst/climate_analysis/prepare_climate_data_catalog.py",
            "blueearth_cst/shared/setup_time_horizon.py",
            "blueearth_cst/shared/climate_window.py",
        ],
    output:
        # Generated build YAML is provenance OF THE MODEL IT BUILT, so it lives
        # in the model's own config/ rather than a project-level generated/
        # (design v10). Still DECLARED after the merge — `semantic_tree_diff.py`
        # and the R9 path map both pin this path, and it is the record of what
        # the forcing was built from.
        forcing_yml = f"{basin_dir}/config/build_historical_forcing.yml",
        # The one-entry catalog pointing hydromt at the store. DECLARED, not
        # incidental: the script writes it, so an undeclared file here would be
        # invisible to Snakemake exactly as `output_gwr.csv` was.
        store_catalog = f"{basin_dir}/config/climate_store_catalog.yml",
        forcing_path = f"{basin_dir}/forcing/inmaps_historical.nc",
        # TERMINAL BUILD SENTINEL. `hydromt update wflow_sbm` calls
        # mod.write(), which rewrites the WHOLE model root -- measured on the R9
        # gate run, where staticmaps.nc, wflow_sbm.toml, hydromt.log and every
        # staticgeoms/ layer all carry this rule's timestamps. So THIS rule, not
        # 1.06, is the last writer of the model, and it is the only correct
        # ordering anchor for a reader.
        #
        # `.outputs_configured` (rule 1.08) was the anchor until now and is NOT
        # sufficient: it precedes this rule's write of staticmaps.nc by 17 s on
        # that same run. Rule 1.11 read staticmaps.nc and finished 9 s before
        # this rule rewrote it -- a timing accident, not a DAG edge, and one
        # that closes as the basin grows. The failure is silent:
        # HDF5_USE_FILE_LOCKING="FALSE" in the pixi env means a concurrent read
        # aborts below Python with no traceback (R9 P2 F5).
        #
        # RESIDUAL RISK, stated because no test can catch it: this sentinel is
        # correct only while this rule is the last writer of the model root. A
        # new rule that mutates the model after 1.09 must move the sentinel to
        # itself. `tests/test_model_root_ordering.py` checks that readers
        # declare it, not that it is attached to the right rule.
        #
        # [R10-1] MERGE NOTE: the original sentinel attached to the old 1.09, which then
        # consumed 1.06's YAML. The merge folded 1.06 in, so the rule that
        # carries the sentinel now also WRITES that YAML -- it is still the last
        # writer of the model root, so the anchor is unchanged.
        model_final = touch(f"{basin_dir}/.model_final"),
    params:
        # The SIMULATION window, not the extraction window. These two params set
        # both halves of the same span: the forcing hydromt builds, and the
        # wflow TOML's `[time]` starttime/endtime it runs over
        # (shared/setup_time_horizon.py). They are one key precisely because
        # forcing and run period cannot legitimately differ.
        # `climate.window` / `simulation_window` are inclusive YEARS since R14
        # (`C-70`, `C-71`); wflow's [time] block wants ISO datetimes, so they
        # are rendered here, through the one parser, rather than re-derived.
        starttime = SIMULATION_BOUNDS[0].isoformat(),
        endtime = SIMULATION_BOUNDS[1].isoformat(),
        clim_source = get_config(climate_cfg, "selected", optional=False),
        basin_dir = basin_dir,
        data_catalog = DATA_SOURCES,
    log:
        ADD_CLIMATE_FORCING.log(),
    benchmark:
        ADD_CLIMATE_FORCING.benchmark(),
    script: "blueearth_cst/model/add_climate_forcing.py"

# 1.13  run_historical_simulation — run the Wflow.jl model on historical forcing
rule run_historical_simulation:
    message: RUN_HISTORICAL_SIMULATION.banner()
    input:
        forcing_path = f"{basin_dir}/forcing/inmaps_historical.nc",
        # the model-root ordering contract. Already ordered after 1.09 through forcing_path; declared
        # anyway so the invariant is LOCAL rather than transitive -- a reader is
        # correct because of what it declares, not because of a chain someone
        # has to re-derive.
        model_final = ancient(f"{basin_dir}/.model_final"),
        scientific_code = [
            "blueearth_cst/model/run_wflow.jl",
            # This helper constructs Wflow.Config and invokes Wflow.run.
            "blueearth_cst/shared/wflow_progress.jl",
        ],
    output:
        # Retain full-precision native responses so evaluation and table exports
        # can rerun without another historical simulation. Rounded exports are
        # not a replacement for this input or for numerical baseline checks.
        csv_path = f"{basin_dir}/run_default/output.csv",
        # Wflow's own log. DECLARED but no longer temp(): it is now the ONLY
        # copy of Wflow's records.
        #
        # It was temp()'d from 2026-08-10 on the explicit ground that it was a
        # DUPLICATE -- the built TOML set `loglevel = "info"`, at which Wflow
        # sends the same records to the terminal and the file, and
        # run_logged.py captured the terminal into logs/wf1_build_model.log.
        # `_BASE_CONFIG` now also sets `[logging] silent = true`, which turns
        # off the terminal half to stop Julia's box-drawing blocks from
        # swamping the console. That removes the duplication the temp()
        # depended on, so keeping it would delete Wflow's log outright on every
        # successful run.
        #
        # This is the earlier ruling's own premise changing, not a reversal of
        # it: the ruling was "discard the copy", and after `silent` there is no
        # copy to discard. Relying on "Snakemake keeps a temp() output when the
        # job fails" instead was considered and rejected -- that would make the
        # log's survival depend on failure semantics for the one artifact you
        # need when diagnosing one.
        #
        # `run_default/log.txt` is already excluded by name from the tree
        # inventory and the baseline comparator (semantic_tree_diff
        # `_is_excluded`), as timestamp-laden and never value-comparable, so
        # its reappearance moves no gate.
        #
        # wf3 was never affected either way: there `logging.path_log` is keyed
        # PER MEMBER (downscale_climate_forcing.py) because rule 3.15 batches
        # members concurrently onto one model directory; dropping that keying
        # put twelve members on one path, each truncating it, and destroyed
        # eleven logs (R9 P2, measured 2026-08-05). Those files are the
        # attribution mechanism, and `silent` is what makes them the only
        # readable record of a batched run.
        wflow_log = f"{basin_dir}/run_default/log.txt",
    params:
        toml_path = f"{basin_dir}/wflow_sbm.toml",
        driver = str(Path(workflow.basedir) / "blueearth_cst" / "model" / "run_wflow.jl"),
    log:
        RUN_HISTORICAL_SIMULATION.log(),
    benchmark:
        RUN_HISTORICAL_SIMULATION.benchmark(),
    shell:
        # A DRIVER FILE rather than the `-e "using Wflow; Wflow.run()"` one-liner
        # Wflow's docstring suggests. Under `[logging] silent = true` that form
        # emits nothing at all for the whole run -- the longest single step in
        # WF1 -- so the driver adds the progress bar every other long step in the
        # toolbox animates. It is also the shape rule 3.15 already has.
        """python -u "{run_logged}" "{log}" -- {wflow_julia} "{params.driver}" "{params.toml_path}" """

# 1.03  extract_climate_datasets — the SHARED historical-climate store producer
# (R07 B1). This declaration and rule 3.08 in run_stress_test.smk are
# the same rule: identical name, script, single catalog input, outputs and
# params, all splatted from CLIMATE_STORE. Only message/log/benchmark are
# workflow-local. wf1's old `climate_historical/wf1_raw/` store and its
# ancient(staticmaps.nc) edge are retired — the extent is now a pure function of
# shared.basin + the catalog, so this rule needs no built model, and a region
# change re-extracts via Snakemake's params rerun-trigger.
#
# DO NOT add a workflow-local input or param here: an asymmetric input set
# re-creates the wf1<->wf3 re-extraction oscillation (design P2(b), ext1-02) and
# is forbidden. The catalog input is deliberately PLAIN, not ancient(): it is
# the store's freshness boundary (ext2-01).
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

# 1.15  plot_model_evaluation — analyse + plot the wflow RUN (parallel leaf
# with 1.11/1.12). Discharge only since the climate-plot consolidation; the climate figures it used
# to draw are the map/series family under forcing/plots/ now.

# O-24, basin-average half. The ONE plot_model_evaluation family that is a pure
# function of config, so it is declared rather than left undeclared.
#
# Two exclusions, both load-bearing — do not "simplify" either away:
#  - "river discharge" never gets a basavg column at all. Rule 1.08 covers it
#    via setup_outlets/setup_gauges and filters it out of the basin-average
#    setup (setup_gauges_and_outputs.py, `extras`).
#  - "precipitation" DOES get a column, and the rule drops it before plotting
#    (plot_results.py, `drop_vars("precipitation_basavg")`), so the figure is
#    never written. The SCRIPT keeps its name -- R10 renamed rule identifiers,
#    not modules.
# Canonical variable tokens identify the annual and seasonal basin products.
_evaluation_cfg = get_config(my_cfg, "evaluation", {})
if not isinstance(_evaluation_cfg, dict):
    raise ValueError("build_model.evaluation must be a mapping")
_subbasin_plots = get_config(_evaluation_cfg, "subbasin_plots", False)
if not isinstance(_subbasin_plots, bool):
    raise ValueError("build_model.evaluation.subbasin_plots must be a boolean")
_WFLOW_OUTVARS_WITHOUT_BASAVG_PLOT = ("river discharge", "precipitation")
_basavg_pngs = [
    f"{basin_dir}/evaluation/plots/wflow_{BASAVG_FIGURE_STEMS[WFLOW_OUTPUT_CODES[var]]}_{kind}_basin_avg.png"
    for var in wflow_outvars
    if var not in _WFLOW_OUTVARS_WITHOUT_BASAVG_PLOT
    for kind in ("annual_ts", "monthly_clim_band")
]
# Caption records exist only when the configured basin-average family is drawn.
_basavg_captions = (
    [
        f"{basin_dir}/evaluation/plots/figure_titles.md",
        f"{basin_dir}/evaluation/plots/_engine/figure_titles.json",
        f"{basin_dir}/evaluation/plots/figure_captions.md",
        f"{basin_dir}/evaluation/plots/_engine/figure_captions.json",
    ]
    if _basavg_pngs
    else []
)

# The COMPLETE set of tables rule 1.14 writes, derived from the same config
# key. Rule 1.08 declares exactly two kinds of `[output.csv]` column, and both
# reach a derived table because both carry a numeric id:
#
# * `Q` on the `outlets` map (plus the `gauges_locations` map when the project
#   has a location registry) -- declared UNCONDITIONALLY by
#   setup_gauges_and_outputs, i.e. even when `river discharge` is absent from
#   `wflow_outvars`, so `output_q.csv` is always produced;
# * one SHORT CODE per remaining `wflow_outvars` entry, on the `subcatchment`
#   map with a mean reducer -- `gwr_101`.
#
# That second kind is why this list has to exist. 1.14 used to declare
# `output_q.csv` alone, on the reasoning that extras were `<var>_basavg` and so
# carried no id to key on. True once; the short-code change (shared/
# wflow_outputs.CODES) renamed them to `<code>_<subcatchment_id>`, which
# `tidy_wflow_table.split_columns` matches exactly as it matches a gauge
# column. So `write_tidy_tables` wrote `output_gwr.csv` -- and PRUNED it on a
# later run -- while no rule declared it. Undeclared output is invisible to
# Snakemake: nothing rebuilds an undeclared table when it is deleted.
# Retained native responses now allow these declared exports to be regenerated.
WFLOW_TABLE_PATHS = [f"{basin_dir}/run_default/output_q.csv"] + [
    f"{basin_dir}/run_default/output_{WFLOW_OUTPUT_CODES[var]}.csv"
    for var in wflow_outvars
    if var != "river discharge"
]

# 1.14  export_simulation_tables — Excel-ready per-variable tables derived from 1.13
#
# DERIVED, not a rewrite. `output.csv` is read back by rule 1.15 through
# hydromt's own `WflowSbmModel.output_csv`, so its layout — ISO-8601 stamps,
# `Q_<id>` columns in wflow's internal order, full float precision — is an
# interface with that reader and stays untouched (AGENTS.md: do not
# re-engineer how hydromt handles data). The tables here are a product beside
# it: 5 decimal places, plain decimal (no `9.86e-5` for Excel to render as
# `9.87E-05`), date-only `YYYY-MM-DD` stamps, and bare station ids sorted
# numerically.
#
# Date-only rather than the `2000-01-02 00:00:00` this shipped with until
# 2026-08-10: that form parses as a DATETIME, which Excel then re-renders in
# the reader's locale, so one file read as `02-01-2000 00:00` on another
# machine. Dropping a time-of-day that is always midnight removes what the
# locale had to reformat.
#
# Being a product is also what makes the per-variable split safe. The variable
# moves into the FILENAME, which is what lets a column be `1010` rather than
# `Q_1010` without reviving the ambiguity `shared/gauges.py` exists to prevent.
#
# Sub-lettered like 3.01c: this belongs to 1.13's output, not after 1.15.
rule export_simulation_tables:
    message: EXPORT_SIMULATION_TABLES.banner()
    input:
        csv_path = f"{basin_dir}/run_default/output.csv",
        # the model-root ordering contract. Ordered after 1.09 transitively through csv_path, declared
        # anyway for the same reason 1.13 and 1.15 declare it: the invariant is
        # LOCAL, not something a reader has to re-derive from a chain.
        model_final = ancient(f"{basin_dir}/.model_final"),
        # Its [output.csv] columns and static maps label each table column
        # with the registry wflow_id (shared/native_locations).
        toml_path = ancient(f"{basin_dir}/wflow_sbm.toml"),
    output:
        # EVERY table, derived from the config -- not just discharge. Declared
        # as the individual files rather than a directory() so Snakemake tracks
        # each one like any other output.
        tables = WFLOW_TABLE_PATHS,
    log:
        EXPORT_SIMULATION_TABLES.log(),
    benchmark:
        EXPORT_SIMULATION_TABLES.benchmark(),
    script:
        "blueearth_cst/shared/tidy_wflow_table.py"

rule plot_model_evaluation:
   message: PLOT_MODEL_EVALUATION.banner()
   input:
       csv_path = f"{basin_dir}/run_default/output.csv",
       # the model-root ordering contract; transitively ordered after 1.09 via 1.13, declared for the
       # same reason as there.
       model_final = ancient(f"{basin_dir}/.model_final"),
       # The climate-store inputs went with the subcatchment climate figures
       # (the climate-plot consolidation). This rule evaluates DISCHARGE now, so it no longer waits
       # on rule 1.03 at all.
       # Both observation files, when configured: this rule reads the gauge
       # positions AND the observed series, so either changing must re-evaluate.
       **_locations_input,
       **_observations_input,
       location_registry = f"{spatial_dir}/location_registry.csv",
       script = "blueearth_cst/model/plot_results.py"
   # R07 B10 + O-24. The artifacts move into the engine subtree, split by KIND
   # (P1): figures and captions under evaluation/plots/, the indicators table one
   # level up in evaluation/.
   #
   # O-24, restated 2026-08-01 after checking what is actually derivable.
   #
   # DECLARED: the config-invariant subset below, plus `_basavg_pngs` — two PNGs
   # per basin-average entry in wflow_outvars, a pure function of config.
   #
   # AND the per-station bin as a DIRECTORY (t2608071206, 2026-08-18). Its
   # members cannot be enumerated at parse time and never could: their count is
   # the model's OUTLETS and SUBCATCHMENTS, a product of rule 1.06 read back
   # through Q_outlets / the subcatchment map, with output_locations
   # contributing only the extra gauge stations on top. The signatures sheets
   # are narrower still — they also need observations AND a run longer than a
   # year (`do_signatures`), so they are data-conditional, not merely
   # config-conditional. R7-5 assumed all of this was derivable "from
   # wflow_outvars / output_locations"; it is not, and no enumeration closes it.
   #
   # A directory() rather than a checkpoint, following WF0's `subbasins/` bin,
   # which solved the identical problem for its per-subbasin figures. A
   # checkpoint would re-plan the DAG to learn a set nothing downstream
   # consumes — no rule reads these PNGs (figures are terminal artifacts), so
   # the only thing the enumeration would buy is `--delete-all-output`
   # completeness, and declaring the bin buys that outright.
   #
   # `--delete-all-output` completeness now holds for configs WITH extra gauges
   # too, which is what R7-5 was actually about.
   #
   # Note the family list above is two, not the three R7-5 named:
   # `clim_{station}_{period}.png` no longer exists — the climate-plot consolidation removed it on
   # 2026-08-09 with the branch that drew it.
   #
   # The two clim_wflow_1_* figures were declared here until 2026-08-09 and are
   # gone with the branch that wrote them (the climate-plot consolidation). Their removal also
   # retires the failure mode declaring them introduced: a config with a
   # sub-year historical_window failed here with MissingOutputException where
   # it used to log a skip.
   #
   # No per-station FIGURE is declared any more. They are keyed by wflow_id
   # since 2026-08-10 — `hydrograph_1010.png` — and a wflow_id is a product of
   # the model build, so it is not knowable at parse time. `hydro_wflow_1.png`
   # only worked as a declared output because plot_results.py forced the first
   # outlet's label to that literal, which is the naming inconsistency the
   # rename removes. The indicators table is this rule's config-invariant artifact
   # and takes over as its terminal.
   output:
       indicators_csv = f"{basin_dir}/evaluation/performance_indicators.csv",
       basavg_pngs = _basavg_pngs,
       basavg_captions = _basavg_captions,
       subbasin_figures = (
           [directory(f"{basin_dir}/evaluation/plots/{SUBBASIN_PLOT_DIRNAME}")]
           if _subbasin_plots and _basavg_pngs else []
       ),
       station_plots = directory(
           f"{basin_dir}/evaluation/plots/{STATION_PLOT_DIRNAME}"
       ),
   params:
       annotation = plot_annotation,
       plot_setup_digest = plot_setup_digest,
       project_dir = f"{project_dir}",
       # The model root is OWNED BY THE RULE, not rebuilt inside the script.
       # plot_results.py used to derive `{project_dir}/hydrology_model` itself,
       # which made the model's location a fact spelled in two places; this move
       # is what forced the question, and passing it is the answer.
       model_dir = basin_dir,
       subbasin_plots = _subbasin_plots,
   log:
       PLOT_MODEL_EVALUATION.log(),
   benchmark:
       PLOT_MODEL_EVALUATION.benchmark(),
   script: "blueearth_cst/model/plot_results.py"

# 1.16 write_run_metadata — the staleness sidecar (design §5.8).
#
# `1.16`, not `1.17`: that number is gather_benchmarks and `dev/reference/
# dev-naming-conventions.md` §9 forbids renumbering to insert a rule. Placed BEFORE the two
# gather rules so its log part -- if it ever grows one -- is still gathered.
#
# It takes the indicators table as input rather than writing into it: that table
# and WF3's indicator CSV are baseline-fingerprinted, so embedding the digests
# in either would falsify the design's no-baseline-re-record claim.
rule write_run_metadata:
    message: WRITE_RUN_METADATA.banner()
    input:
        indicators_csv = f"{basin_dir}/evaluation/performance_indicators.csv",
        # a rule reading under the model root waits on the terminal
        # marker, because rule 1.07 rewrites the whole directory. The indicators
        # table already orders this rule after 1.15, so the marker adds no edge
        # in practice -- it is declared because the convention is mechanical,
        # and a reader exempted by an argument is how the next one gets missed.
        model_final = ancient(f"{basin_dir}/.model_final"),
    params:
        workflow_name = "build_model",
        config_path = config_path,
        effective_config_sha256 = EFFECTIVE_CONFIG_DIGEST,
        refresh_digest = configuration_inputs_digest(
            EFFECTIVE_CONFIG_DIGEST,
            {},
            CAPTURE_RECORD["environment"] if CAPTURE_RECORD else environment_file_hashes(),
            CAPTURE_RECORD["referenced_inputs"] if CAPTURE_RECORD else referenced_inputs_for_digest(CONFIG_REFERENCES),
        ),
    output:
        run_metadata = f"{basin_dir}/run_metadata.json",
    script:
        "blueearth_cst/shared/write_run_metadata.py"

# 1.11  plot_basin_map — every figure the spatial foundation supports: the
# basin/DEM map plus the thematic family beside it (parallel leaf).
#
# ONE rule for both, because they are one deliverable: the same overlay, the
# same style profile and the same folder. Splitting them would also duplicate
# the vector inputs and the plots directory across two rules for no scheduling
# gain — both halves are leaves.
rule plot_basin_map:
    message: PLOT_BASIN_MAP.banner()
    input:
        # elevation_basin depicts ELEVATION, which is data rather than a
        # model result, so it is drawn from the shared spatial foundation and
        # not from the wflow model. That also retires this rule's HDF5 race
        # workaround wholesale — it no longer opens staticmaps.nc, so it no
        # longer has to be ordered behind every writer of that file (the
        # sentinel edge, the ancient() staticmaps declaration, and the -c 3
        # abort-below-Python they existed to prevent, all gone).
        hydrography_nc = SPATIAL_UNITS.hydrography_nc,
        **{name: SPATIAL_UNITS.outputs[name] for name in
           ("basins", "subbasins", "rivers", "locations",
            "reservoirs", "lakes", "glaciers")},
        # The thematic half's raster stack. This is a NEW EDGE: 1.11 used to
        # depend on rule 1.02 alone and could run as soon as the vectors
        # existed; it now waits for 1.05 as well. That costs nothing in
        # practice — 1.05 is upstream of the model build, so nothing downstream
        # waits on this leaf — and it is the honest dependency: the family is
        # drawn FROM this file.
        spatial_maps = f"{spatial_dir}/spatial_maps.nc",
    output:
        # PNG only since 2026-08-10 (owner's call): nothing in the toolbox
        # or the platform read the vector deliverable, and 600 dpi at 180 mm
        # carries the figure everywhere it is used.
        titles_md = f"{spatial_dir}/plots/figure_titles.md",
        titles_json = f"{spatial_dir}/plots/_engine/figure_titles.json",
        captions_md = f"{spatial_dir}/plots/figure_captions.md",
        captions_json = f"{spatial_dir}/plots/_engine/figure_captions.json",
        basin_png = f"{spatial_dir}/plots/elevation_basin.png",
        # From the registry the plotter iterates, not restated here — the same
        # contract rule 1.12 has with climate_figures.figure_names(). Only the
        # figures whose source variable exists for EVERY shipped catalog source
        # are declared; `soil_depth_to_bedrock_basin` is drawn but left out, because
        # its variable is soilgrids v1.0's and a project on soilgrids_2020 would
        # fail this rule on "missing output files" rather than lose one figure.
        # See SpatialFigure.guaranteed.
        spatial_figures = figure_paths(f"{spatial_dir}/plots"),
    params:
        annotation = plot_annotation,
        plot_setup_digest = plot_setup_digest,
        spatial_dir = f"{spatial_dir}",
    log:
        PLOT_BASIN_MAP.log(),
    benchmark:
        PLOT_BASIN_MAP.benchmark(),
    script: "blueearth_cst/shared/plot_spatial_maps.py"

# 1.03b  extract_map_candidates — native candidates for shared map scales.
for _source, _spec in MAP_CANDIDATE_STORES.items():
    rule:
        name: EXTRACT_MAP_CANDIDATES.job_name(_source)
        message: EXTRACT_MAP_CANDIDATES.banner(part=_source)
        input: **_spec.inputs,
        output: **_spec.outputs,
        params: **_spec.params,
        log: EXTRACT_MAP_CANDIDATES.log(_source),
        benchmark: EXTRACT_MAP_CANDIDATES.benchmark(_source),
        script: _spec.script

# 1.03c  produce_historical_map_reference — also available with WF0 disabled.
include: "blueearth_cst/climate_analysis/rules/source_map_preparation.smk"

rule produce_historical_map_reference:
    message: PRODUCE_MAP_REFERENCE.banner()
    input: **MAP_REFERENCE.inputs,
    output: **MAP_REFERENCE.outputs,
    params: **MAP_REFERENCE.params,
    log: PRODUCE_MAP_REFERENCE.log(),
    benchmark: PRODUCE_MAP_REFERENCE.benchmark(),
    script: MAP_REFERENCE.script

# 1.12  plot_model_forcing — shared historical map display on model forcing.
rule plot_model_forcing:
    message: PLOT_MODEL_FORCING.banner()
    input:
        forcing_path = f"{basin_dir}/forcing/inmaps_historical.nc",
        map_reference = MAP_REFERENCE.outputs["reference"],
        **{key: value for key, value in MAP_REFERENCE.inputs.items() if key != "basins"},
        # the model-root ordering contract; see rule 1.13.
        model_final = ancient(f"{basin_dir}/.model_final"),
        **_locations_input,
        # Shared spatial context also used by WF0's source-grid diagnostics.
        **{name: SPATIAL_UNITS.outputs[name] for name in
           ("basins", "subbasins", "rivers", "locations")},
    # R07 B10 + O-24: the forcing/model-input QA figures sit beside the forcing
    # they describe, and ALL of them are declared — variable x kind is a fixed
    # cross-product, so the list comes from climate_figures.figure_names()
    # rather than being restated here.
    output:
        _forcing_climate_pngs,
        titles_md = f"{FORCING_PLOTS_DIR}/figure_titles.md",
        titles_json = f"{FORCING_PLOTS_DIR}/_engine/figure_titles.json",
        captions_md = f"{FORCING_PLOTS_DIR}/figure_captions.md",
        captions_json = f"{FORCING_PLOTS_DIR}/_engine/figure_captions.json",
    params:
        annotation = plot_annotation,
        plot_setup_digest = plot_setup_digest,
        project_dir = f"{project_dir}",
        # Rule-owned model root; see rule 1.15's note.
        model_dir = basin_dir,
        geoms_dir = SPATIAL_UNITS.spatial_dir + "/geoms",
        water_year_start = WATER_YEAR_START,
        map_reference_params = dict(MAP_REFERENCE.params),
    log:
        PLOT_MODEL_FORCING.log(),
    benchmark:
        PLOT_MODEL_FORCING.benchmark(),
    script: "blueearth_cst/model/plot_map_forcing.py"

# --- benchmark gather ---------------------------------------------------------
# Per-rule benchmarks land under benchmarks/_parts/. This runs once, after the
# terminal outputs (so all WF1 rules ran), and merges the WF1 parts into one
# benchmarks/wf1_benchmarks.md (rule column + TOTAL row), fresh each run.
rule gather_benchmarks:
    message: GATHER_BENCHMARKS.banner()
    input:
        WF1_TERMINALS,
    output:
        BENCHMARK_OUTPUT_PATH,
        BENCHMARK_OUTPUT_PATH.removesuffix(".md") + ".tsv",
    params:
        parts_dir = BENCH_PARTS_DIR,
        workflow_num = 1,
    script: "blueearth_cst/shared/merge_benchmarks.py"

# 1.18  gather_logs — merge every WF1 log part into ONE workflow log.
#
# Same rule as WF2's 2.08 and WF3's 3.18, against the same script; only the label
# list, the parts dir and the output name differ. `input:` is WF1_TERMINALS, the
# same set gather_benchmarks takes, which is what schedules it LAST: every
# logging rule is upstream of it. The parts stay in `params:` — they are
# `log:` files, and Snakemake does not track those in the DAG, so naming them as
# `input:` would demand them as buildable targets.
#
# The merge DELETES the parts it consumed and prunes the emptied dirs, so a clean
# full run leaves no logs/_parts/ at all. After a PARTIAL re-run only the re-run
# rules have parts, so the rewritten log marks the rest "no part from this run" —
# the same trade `merge_benchmarks` makes (pre-board record 45e3a564 R7-9).
rule gather_logs:
    message: GATHER_LOGS.banner()
    input:
        WF1_TERMINALS,
    output:
        WORKFLOW_LOG_PATH,
    params:
        rules = LOG_RULES,
        parts_dir = LOG_PARTS_DIR,
    script: "blueearth_cst/shared/merge_logs.py"


# --- Run journal (design §5.7) ------------------------------------------------
#
# Emitted from WORKFLOW-LEVEL HANDLERS, never from a rule: a rule that is up to
# date does not execute, so it cannot record an invocation, and a rule that
# DECLARED the journal would have it deleted before the job ran -- Snakemake
# removes declared outputs first, truncating the ledger to one line every run,
# silently. A one-line journal looks exactly like a young one.
#
# SCOPE (R5 as narrowed 2026-08-13, after the P0 probe): these handlers cover
# invocations in which at least one job executed. On Snakemake 9.6.2 a "Nothing
# to be done" no-op fires NONE of them -- workflow.py:1375-1377 returns before
# _onstart, guarded by no flag -- so the journal counts executed runs, and a gap
# in the dates means no work was done rather than that nobody looked. Do not add
# a parse-time or atexit emitter to reach past this; the owner ruled against it.
JOURNAL_PATH = f"{project_dir}/config/_engine/journal.jsonl"
INVOCATION_ID = uuid.uuid4().hex

# One toolbox read per invocation, shared by both handlers, so a line pair
# cannot straddle a commit.
_JOURNAL_TOOLBOX = toolbox_identity()


def _journal(event):
    append_journal_line(
        JOURNAL_PATH,
        journal_event(
            invocation_id=INVOCATION_ID,
            workflow="build_model",
            event=event,
            toolbox=_JOURNAL_TOOLBOX,
            effective_config_sha256=EFFECTIVE_CONFIG_DIGEST,
            configuration_inputs_sha256=CONFIGURATION_INPUTS_DIGEST,
            source_config_sha256=file_sha256(config_path),
        ),
    )


# Wall clock for the end-of-run summary. Taken at PARSE, not in `onstart`:
# Snakemake exposes no run duration to these handlers, and parse-to-finish is
# the interval a person actually waited. It therefore includes DAG
# construction -- a second or two here, and not worth a second clock.
_RUN_STARTED = time.monotonic()


def _summary(failed):
    """Print the end-of-run block to STDERR, beside Snakemake's own output.

    stderr because that is where Snakemake writes its console, so a redirect
    that captures one captures both -- and `run_logged` already tees rule
    output there. Never raises: a summary that broke a successful run would be
    the worst possible trade for a convenience.
    """
    try:
        # One write, blank line before it -- see the note on WF3's `_summary`.
        sys.stderr.write(
            "\n"
            + run_summary(
                "wf1 build_model",
                project_dir,
                WORKFLOW_LOG_NAME,
                "wf1_benchmarks.md",
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

    Same stream and same never-raises contract as `_summary`: these two are one
    block split across the run, and a header that could end a run would be the
    worst possible trade for a convenience.

    Takes no arguments on purpose. Every name it reports is read INSIDE the
    guard, so a value that turns out not to resolve in the `onstart` namespace
    costs a line of console rather than the run -- which an argument evaluated
    at the call site would not.
    """
    try:
        # One write, carrying a blank line on both sides -- see the note on
        # WF3's `_header`, which this mirrors.
        from blueearth_cst.experiment.batch_sizing import count_active_cells
        try:
            cells = count_active_cells(Path(basin_dir) / "staticmaps.nc")
            grid = f"{cells:,} active cells in existing model" if cells is not None else "Cell count available after model build"
        except (OSError, ValueError):
            grid = "Cell count unavailable from existing model"
        open_run_header(
            "wf1 build_model", project_dir, header_config_path(config_path, CAPTURE_RECORD),
            **{
                "hydrological model": "Wflow-SBM",
                "climate forcing": clim_source,
                "simulation period": f"{simulation_window['start']}–{simulation_window['end']}",
                "grid spacing": f"{model_resolution:g}°",
                "model grid": grid,
                "water year": f"Starts in {WATER_YEAR_START}",
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
    # Restyle Snakemake's own console output into this toolbox's grammar (one
    # line per job start and end). Here and not at parse time: the logging
    # stack does not exist yet then. Fail-open; see install_console_style.
    install_console_style()
    _header()
    _journal("started")


onsuccess:
    _journal("success")
    _summary(failed=False)


onerror:
    _journal("failed")
    _summary(failed=True)
