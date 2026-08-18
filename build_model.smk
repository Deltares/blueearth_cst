import sys
import time
import uuid
from pathlib import Path

# Shared helpers live in blueearth_cst/; make them importable regardless of the working
# directory by prepending this Snakefile's own directory to sys.path.
# See dev/milestones/r03/model-builder-design.md §3.
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.provenance import append_journal_line, configuration_inputs_digest, effective_config_digest, environment_file_hashes, file_sha256, journal_event, referenced_inputs_for_digest, toolbox_identity
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, catalog_root, declare_path_tokens, declare_project_root, DEFAULT_JULIA_THREADS, DEFAULT_WFLOW_OUTVARS, climate_store_rule, get_config, julia_prefix, patch_psutil_windows_benchmark, region_rule, resolve_simulation_window, resolve_water_year_start, rule_banner, run_summary, spatial_units_rule, target_banner, validate_historical_window, warn_if_project_dir_in_repo, install_console_style, run_header
from blueearth_cst.spatial.config import parse_spatial_config
# The canonical climate figure set (rules 1.13 and 1.15 both draw it). Imported
# for figure_names() ONLY, so every figure is declared from the same list the
# plotter writes from and the two cannot drift.
from blueearth_cst.climate_analysis.climate_figures import figure_names, source_climate_vars, source_figure_names

# The source family follows the WF0 filename grammar
# (`dev/working/wf0-figure-filename-rule.md`); rule 1.05 writes the same files
# to the same directory wf0's 0.05 does, so the two must name them alike. Kept
# beside wf0's declaration rather than imported from it — a Snakefile does not
# import another Snakefile — so any change is made in both places deliberately.
DECLARED_SPATIAL_SCOPES = ("basin_avg",)
SUBBASIN_PLOT_DIRNAME = "subbasins"
# The thematic map family rule 1.12 draws beside basin_area. Same reason as
# above: the output list comes from the registry the plotter iterates, so a
# figure cannot be added in one place and forgotten in the other.
from blueearth_cst.shared.plot_spatial_maps import figure_paths
from blueearth_cst.shared.plot_evaluation import STATION_PLOT_DIRNAME
# Recognises both "unset" spellings (YAML null and the legacy "None" string),
# so an unset observation key never becomes a declared input.
from blueearth_cst.shared.gauges import is_unset
from blueearth_cst.shared.wflow_outputs import CODES as WFLOW_OUTPUT_CODES

# Windows: make Snakemake's benchmark memory/IO/CPU metrics work (else all NA).
patch_psutil_windows_benchmark()

# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention — keep it even though the Snakefile itself uses `config`.
config_path = workflow.configfiles[0]

# Portable tee wrapper for the shell rules below: keeps live console output AND
# preserves the child's exit code (a bare `| tee` masks failures on cmd.exe --
# no pipefail there). See blueearth_cst/shared/run_logged.py / blueearth_cst.shared.snake_utils.run_and_tee.
run_logged = str(Path(workflow.basedir) / "blueearth_cst" / "shared" / "run_logged.py")

# R01 schema — three top-level sections. Read each into a local dict.
project_cfg = config["project"]
shared_cfg = config["shared"]
# The water year the climate figures aggregate on, from the one shared key
# WF2 and WF3 also read. Figures are terminal artifacts, so this changes no
# number -- but a figure labelled 'annual' should mean the basin's year.
WATER_YEAR_START = resolve_water_year_start(get_config(shared_cfg, "water_year_start"))
my_cfg = config["workflows"]["build_model"]

# Project — paths and external resources
project_dir = get_config(project_cfg, "project_dir", optional=False)
# O-22: make the two-tier project_dir rule mechanical rather than
# documentary. Warns, never raises; test_case/ is the one exemption.
warn_if_project_dir_in_repo(project_dir, workflow.basedir)
static_dir = get_config(project_cfg, "static_dir", optional=False)
DATA_SOURCES = get_config(project_cfg, "data_sources", optional=False)

# Shared — multi-workflow scientific knobs
basin_cfg = shared_cfg["basin"]
spatial_cfg = parse_spatial_config(basin_cfg, my_cfg)
model_region = get_config(basin_cfg, "region", optional=False)
model_resolution = spatial_cfg.resolution
# Catalog ENTRY NAMES for the model-free basin delineation the shared climate
# store uses (R07 B1 / ext1-01). Both OPTIONAL: the defaults equal the shipped
# build template's setup_basemaps values, so a config that declares neither is
# unaffected and rule 3.01's guard digest stays byte-identical. Rule 1.06
# cross-checks them against the template and fails loud on a disagreement.
basin_hydrography = spatial_cfg.hydrography
basin_index = spatial_cfg.basin_index
historical_window = get_config(shared_cfg, "historical_window", optional=False)
# ONE minimum window for the whole toolbox — snake_utils.MIN_HISTORICAL_YEARS
# (16), set by weathergenr's wavelet minimum and enforced identically here and
# at extraction. WF1 rejects it too, deliberately: a record too short for a
# stress test is a misconfigured project, and letting WF1 build a model on it
# only moves the failure to the workflow least able to explain it
# (dev/followups-archive.md R7-6, R3). Parse time, before any rule executes — same
# stance as the eobs rejection below. Whether the staged source actually COVERS
# the requested window is a different question, checked at rule 1.04.
validate_historical_window(historical_window)
# The period the MODEL IS RUN over, which is a different question from how much
# record to extract (above). Optional; absent means exact passthrough of
# historical_window, so every config predating this key is unaffected. Resolved
# and validated at parse time for the same reason the window above is.
#
# Editing it re-runs rule 1.10 through Snakemake's params trigger, which
# rebuilds the forcing and so re-runs 1.14 — and since 1.14's output.csv is
# temp(), that is the whole model. Correct, but not cheap.
simulation_window = resolve_simulation_window(shared_cfg, my_cfg)
clim_source = get_config(shared_cfg, "clim_historical", optional=False)
# Wflow.jl thread count for rule 1.14. OPTIONAL — the default is P3-3's frozen
# baseline value, so an existing config is unaffected. Config-driven rather than
# inline so a deployment can tune it to its basin (a production basin has the
# cell-parallelism the 384-cell test fixture lacks) without a Snakefile edit.
# NOT Snakemake's `threads:`, which caps at --cores: see DEFAULT_JULIA_THREADS.
julia_threads = get_config(shared_cfg, "julia_threads", DEFAULT_JULIA_THREADS)
# Carries the juliaup version pin too; both are validated at parse time, before
# any rule can put them in a shell body.
wflow_julia = julia_prefix(julia_threads)

# P3-2a bounded support (design ext2-3): the wf1 raw-climate path (rule 1.04 +
# the model-parity plot re-source) supports era5, chirps and chirps_global
# only. eobs fails here at DAG-parse time — before any rule executes — so the
# rejection is early and loud on every run and dry-run.
if clim_source == "eobs":
    raise ValueError(
        "clim_historical: eobs is not supported by the P3-2a wf1 raw-climate "
        "path; supported sources: era5, chirps, chirps_global"
    )

# Workflow-owned
wflow_outvars = get_config(my_cfg, "wflow_outvars", DEFAULT_WFLOW_OUTVARS)
# `defaults/`, not `templates/` — these are read by rules 1.06/1.07/1.08, and the
# 2026-08-11 split moved them out of the copy-me directory. Rule 1.01 still routes
# them to a `templates/` bin inside the PROJECT: a different meaning of the word,
# and that one did not move. Under the R4 copy predicate a shipped default is
# normally recoverable from the toolbox and so is recorded rather than copied —
# the bin only receives one when the project points the key at its own file.
model_build_config = get_config(my_cfg, "model_build_config", f"{static_dir}/defaults/wflow_build_model.yml")
waterbodies_config = get_config(my_cfg, "waterbodies_config", f"{static_dir}/defaults/wflow_update_waterbodies.yml")
output_locations = spatial_cfg.gauge_points_path
observations_timeseries = get_config(my_cfg, "observations_timeseries", None)

# The two OPTIONAL observation inputs, declared as real `input:` entries when
# configured and omitted entirely when not. `output_locations` is the internal
# compatibility name; its canonical config source is shared.basin.gauge_points.
#
# They were `params:` until 2026-08-02, carrying the PATH. Snakemake's params
# trigger compares the value, so editing the FILE changed nothing it could see:
# renumbering the gauge-point file left rule 1.09's model gauges on the old ids
# while the observations moved to the new ones, the join then matched nothing,
# and performance_metrics.csv emptied without a word. As `input:` the content
# mtime-triggers 1.09 (which writes the ids into the model) and 1.15 (which
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
RUN_RECORD = f"{project_dir}/config/runs/build_model/run_record.yml"

# The consumed-key PROJECTION: the config paths this workflow actually reads.
# Digesting the projection rather than the whole file is what stops a WF3-only
# edit from re-firing WF1's record. A path this config lacks raises at parse
# time -- the declaration is a claim about what the workflow reads, so a typo
# must not quietly narrow the digest.
CONFIG_PROJECTION = ("project", "shared", "workflows.build_model")

# Every external file this workflow's configuration points at. Hashed at parse
# time so the digest below moves when one is edited IN PLACE -- the recorded
# hash alone would move without re-firing anything.
CONFIG_REFERENCES = [
    ("model_build_config", model_build_config),
    ("waterbodies_config", waterbodies_config),
    *[("data_catalog", source) for source in
      (DATA_SOURCES if isinstance(DATA_SOURCES, (list, tuple)) else [DATA_SOURCES])],
    *[("output_locations", source) for source in _locations_input.values()],
    *[("observations_timeseries", source) for source in _observations_input.values()],
]

EFFECTIVE_CONFIG_DIGEST = effective_config_digest(
    config, ADVANCED_SETTINGS, CONFIG_PROJECTION
)
# Threaded through rule 1.01's params: below. Snakemake's params rerun-trigger
# then re-executes it whenever ANY component moves -- a new commit, a dirty
# flip, a lock-file change, an in-place catalog edit. Without this the record
# would keep the previous commit after a code-only change, which is the defect
# both design reviewers found independently.
CONFIGURATION_INPUTS_DIGEST = configuration_inputs_digest(
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

# The canonical climate figure set, one directory per dataset (rule 1.13 draws
# the forcing, 1.15 the source). variable x kind is a fixed cross-product, so
# unlike rule 1.15's per-station figures these are fully knowable at parse time
# and ALL of them are declared — see climate_figures' module docstring.
FORCING_PLOTS_DIR = f"{basin_dir}/forcing/plots"
_forcing_climate_pngs = [f"{FORCING_PLOTS_DIR}/{name}" for name in figure_names("forcing")]

# --- The shared historical-climate store (R07 B1) -----------------------------
# ONE producer contract, built here and identically in
# run_stress_test.smk, and splatted into rule 1.04 / rule 3.08 below.
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

# --- The one project region artifact (ADR 0006) -------------------------------
# Same pattern, same reason: ONE producer contract, built here and identically
# in the other two workflows, splatted into rule 1.02 below. Only
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

# --- The shared vector foundation (ADR 0006 §8) -------------------------------
# Third and last of the shared producer contracts, same pattern again: built
# here and identically in the other two workflows, splatted into rule 1.03
# below, and tests/test_spatial_units_rule.py parses all three workflows and
# fails on ANY difference beyond message/log/benchmark.
#
# `parse_spatial_config(basin_cfg)` WITHOUT `my_cfg`, deliberately, and this is
# the one call site where that differs from the module-level `spatial_cfg`
# above: §8b requires the shared rule's params to be a pure function of
# `project` + `shared.basin`, so the deprecated
# `workflows.build_model.output_locations` fallback cannot feed it.
#
# THE COST WAS REAL AND SILENT, AND IS NOW REFUSED AT THE SEAM. Rule 1.06 used
# to receive `my_cfg` and so honoured the legacy key on its way to the
# partition. A config that set ONLY `workflows.build_model.output_locations`
# then reached this rule with no gauge points at all, so its parent basins fell
# back to the AUTOMATIC partition: different subbasins, a different
# location_registry.csv, and different wflow_id values — with no error, because
# an absent gauge file is a legitimate configuration. Rule 1.09 would then add
# gauges to a model whose subbasins were derived without them.
#
# That is no longer reachable. §8b leaves no alternative on this side — a params
# payload drawn from a section the projections-only configs do not have would
# differ per invoking workflow — so the asymmetry is refused one level up
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

# Rule numbering (comment headers + log/benchmark filenames) uses `W.NN` = the
# rule's position in this workflow's LOGICAL order — data, then model build,
# then run, then records. Contiguous, and every dependency points from a lower
# number to a higher one, so a rule can never depend on something numbered after
# it. Positional since [R10-5] (2026-08-06); it used to be a stable identifier
# assigned at rule creation, which left it uncorrelated with anything.
#
# STILL A READING AID, NOT EXECUTION ORDER. Snakemake resolves execution from
# the DAG, so rules on separate branches run concurrently — 1.11-1.13 are
# parallel leaves off 1.10, and 1.04's climate branch runs beside the whole
# model build. Low-to-high says "cannot depend on", not "runs before".
#
# DEFINITION ORDER IN THIS FILE IS NOT THE NUMBER ORDER, and is not required to
# be: module-level code is interleaved between rules and depends on its position
# (`_basavg_pngs` before 1.15, `_source_plot_inputs` before 1.05), so reordering
# the blocks to match would be a behaviour risk taken for cosmetics. Read the
# numbers, not the file order. All three workflows share project_dir/logs, so
# the leading digit keeps them disambiguated and globally sortable.
#
# DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix (1.09b) until the next
# deliberate sweep — renumbering is a migration, not an edit.
# Convention: dev/reference/naming.md §9; full map:
# dev/reference/workflows/rule-index.md § What changed.

# --- log layout ---------------------------------------------------------------
# EVERY WF1 rule that logs writes a PART under logs/_parts/, and rule 1.17 merges
# the parts into ONE logs/wf1_build_model.log, then deletes them. So the only
# WF1 file left in logs/ after a full run is that merged log — matching WF2 (2.09)
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
# 1.01 snapshot_config declares no `log:` and so has no section, and neither
# gather rule does. No entry for the merged-away forcing-recipe rule: [R10-1]
# folded it into 1.10, so its label has no producer, and an orphan label
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
WORKFLOW_LOG_NAME = "wf1_build_model.log"
LOG_PARTS_DIR = f"{project_dir}/logs/_parts"

# The run's key folders, stated ONCE. `run_header` prints them at the top of the
# console and every rule log repeats them in its own header; below that, any
# line mentioning one of these directories prints `<model>/staticmaps.nc`
# instead of twelve path components. `data` is the external tree, read out of
# the catalog because it is the one folder a project neither owns nor derives.
declare_path_tokens(
    data=catalog_root(DATA_SOURCES),
    model=basin_dir,
    climate=store_dir,
)
declare_project_root(project_dir)
LOG_RULES = [
    "1.02_delineate_region",
    "1.03_delineate_spatial_units",
    "1.04_extract_historical_climate",
    "1.05_plot_climate_source",
    "1.06_prepare_spatial_maps",
    "1.07_build_wflow_model",
    "1.08_add_reservoirs_lakes_glaciers",
    "1.09_declare_wflow_outputs",
    "1.10_add_climate_forcing",
    "1.11_write_outlet_index",
    "1.12_plot_basin_map",
    "1.13_plot_forcing",
    "1.14_run_wflow",
    "1.14b_export_wflow_tables",
    "1.15_plot_wflow_evaluation",
]

# WF1_TERMINALS — the artifacts with no WF1 consumer: every producing rule is
# upstream of them and none of them feeds another rule. That is exactly the
# input set the two gather rules (1.16 benchmarks, 1.17 logs) need, which is
# what schedules each one LAST; both restated these seven literals before, so a
# new terminal had to be added in three places or a gather would run early and
# merge an incomplete set of parts.
#
# NOT terminals, and so not in this list: the config snapshot (1.01 runs
# independently of the build and needs no gather waiting on it) and the two
# gather outputs themselves.
WF1_TERMINALS = [
    # The evaluation rule's terminal is its METRICS TABLE, not one of its
    # figures: since 2026-08-10 the figures are keyed by wflow_id and none of
    # their names is knowable at parse time. The table is written by the same
    # rule and is config-invariant, so the gather rules wait on exactly what
    # they waited on before.
    f"{basin_dir}/evaluation/performance_metrics.csv",
    # The Excel-ready tables (1.14b). A product of the workflow rather than an
    # incidental by-product, so they are rule-`all` members like the figures —
    # nothing downstream consumes them, so without this they never build.
    # ONE representative is enough, as with the nine figures below: 1.14b writes
    # the whole set (WFLOW_TABLE_PATHS) in a single job, so requesting the
    # discharge table schedules every other table with it.
    f"{basin_dir}/run_default/output_q.csv",
    f"{spatial_dir}/plots/basin_area.png",
    # One representative of each canonical climate set: the rules produce their
    # nine figures as a unit, so naming every one here would add no edge.
    f"{FORCING_PLOTS_DIR}/forcing_precip_map.png",
    # R07 B4: the source-grid climate figures are rule-`all` members by
    # design — they are a product of the workflow, not an incidental
    # by-product, and their subgraph needs no built model. The representative is
    # the MAP, which every source draws, precipitation-only included.
    f"{store_dir}/plots/"
    + source_figure_names(
        clim_source,
        variables=source_climate_vars(clim_source),
        spatial_scopes=DECLARED_SPATIAL_SCOPES,
    )[0],
    f"{basin_dir}/staticgeoms/outlet_index.csv",
    # P1 spatial foundation. One representative output schedules the complete
    # multi-output rule and makes both gather rules wait for its log/benchmark.
    f"{spatial_dir}/spatial_catalog.yml",
    # The staleness sidecar (1.15b). A terminal in its own right: it consumes
    # the metrics table and nothing consumes it, and listing it here is what
    # schedules it at all -- no rule reads a sidecar, so without this it would
    # never build.
    f"{basin_dir}/evaluation/run_metadata.json",
]

# 1.00  all — target aggregator: full historical build + performance plots
#
# Hoisted into a list so `message:` can print one target per line without
# restating them. Snakemake flattens a list argument, so `input: WF1_TARGETS`
# declares exactly the same ten files the literals did — the terminals, plus
# the config snapshot and the two gathered artifacts.
WF1_TARGETS = [
    *WF1_TERMINALS,
    f"{project_dir}/config/runs/snake_config_build_model.yml",
    f"{project_dir}/logs/{WORKFLOW_LOG_NAME}",
    f"{project_dir}/benchmarks/wf1_benchmarks.md",
]

rule all:
    message: target_banner("1.00", "all", WF1_TARGETS, project_dir)
    input:
        WF1_TARGETS,

# 1.01  snapshot_config — current copies + immutable effective-config bundle
rule snapshot_config:
    message: rule_banner("1.01", "snapshot_config")
    input:
        config_build = model_build_config,
        config_snake = config_path,
        config_waterbodies = waterbodies_config,
        # Snapshotted into config/basin_data/ so the finished project can say
        # what it was evaluated against: both live outside the repo AND outside
        # project_dir, referenced by absolute path (R07 O-01). Declared inputs
        # since 2026-08-02, so the snapshot refreshes when the file changes.
        **_locations_input,
        **_observations_input,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "build_model",
        config_dir = f"{project_dir}/config",
        effective_config = config,
        advanced_settings = ADVANCED_SETTINGS,
        config_projection = CONFIG_PROJECTION,
        # A string digest, so the params trigger compares a value rather than
        # a structure. This is what keeps the record FRESH; see its definition.
        configuration_inputs_sha256 = CONFIGURATION_INPUTS_DIGEST,
    output:
        config_snake_out = f"{project_dir}/config/runs/snake_config_build_model.yml",
        run_record = RUN_RECORD,
    script:
        "blueearth_cst/model/copy_config_files.py"

# 1.02  delineate_region — the one project region artifact (ADR 0006).
# Byte-identical to 2.02 and 3.03 except message/log/benchmark; everything
# else is splatted from REGION so the three cannot drift.
rule delineate_region:
    message: rule_banner("1.02", "delineate_region")
    input:
        **REGION.inputs,
    params:
        **REGION.params,
    output:
        **REGION.outputs,
    log:
        f"{LOG_PARTS_DIR}/1.02_delineate_region.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.02_delineate_region.tsv",
    script: REGION.script

# 1.03  delineate_spatial_units — the shared vector foundation (ADR 0006 §8).
# Byte-identical to 2.03 and 3.04 except message/log/benchmark; everything
# else is splatted from SPATIAL_UNITS so the three cannot drift.
rule delineate_spatial_units:
    message: rule_banner("1.03", "delineate_spatial_units")
    input:
        **SPATIAL_UNITS.inputs,
    params:
        **SPATIAL_UNITS.params,
    output:
        **SPATIAL_UNITS.outputs,
    log:
        f"{LOG_PARTS_DIR}/1.03_delineate_spatial_units.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.03_delineate_spatial_units.tsv",
    script: SPATIAL_UNITS.script

# 1.06  prepare_spatial_maps — the thematic raster stack, WF1 ONLY
#
# The raster half of ADR 0006 §8. The vector layers and the registry come from
# 1.03 now; this rule folds LULC/LAI/soil onto the grid 1.03 handed it and
# writes the model-build interface. `hydrography` is the SEAM INTERMEDIATE
# (§8a): the whole hydrography grid stack used to cross this boundary in
# memory, and re-deriving it here would mean two producers of one value.
rule prepare_spatial_maps:
    message: rule_banner("1.06", "prepare_spatial_maps")
    input:
        config_snake = config_path,
        data_catalogs = DATA_SOURCES,
        hydrography = SPATIAL_UNITS.outputs["hydrography"],
        basins = SPATIAL_UNITS.outputs["basins"],
        subbasins = SPATIAL_UNITS.outputs["subbasins"],
        catchments = SPATIAL_UNITS.outputs["catchments"],
        rivers = SPATIAL_UNITS.outputs["rivers"],
        locations = SPATIAL_UNITS.outputs["locations"],
        location_registry = SPATIAL_UNITS.outputs["location_registry"],
    output:
        spatial_maps = f"{spatial_dir}/spatial_maps.nc",
        spatial_catalog = f"{spatial_dir}/spatial_catalog.yml",
        spatial_report = f"{spatial_dir}/spatial_report.yml",
    params:
        basin_config = basin_cfg,
        model_config = my_cfg,
    log:
        f"{LOG_PARTS_DIR}/1.06_prepare_spatial_maps.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.06_prepare_spatial_maps.tsv",
    script: "blueearth_cst/spatial/prepare_spatial_maps.py"

# 1.07  build_wflow_model — parameterize Wflow-SBM on the P1 spatial foundation
rule build_wflow_model:
    message: rule_banner("1.07", "build_wflow_model", summary="parameterize Wflow-SBM from global data")
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
        # updated IN PLACE by rules 1.08-1.10, but only this rule declares it,
        # so a re-fire of the build rule alone used to leave the toml stripped of
        # every section the later rules add -- and the next wflow run died on
        # the missing key. The obvious fix (drop the ancient() on staticmaps in
        # those rules) is WRONG: they mutate staticmaps themselves via
        # mod.write()/mod.close(), so a plain input would make each one
        # re-trigger on its own execution forever. This sentinel is written by
        # nothing else, so it re-fires rule 1.08 without any self-trigger, and
        # the chain cascades from there (1.08 -> .txt -> 1.09 ->
        # .outputs_configured -> 1.10). The forcing-yml hop that used to sit in
        # the middle of that chain is gone: R10-1 merged 1.07 into 1.10, which
        # now writes the yml it used to consume.
        built = touch(f"{basin_dir}/.model_built"),
    log:
        f"{LOG_PARTS_DIR}/1.07_build_wflow_model.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.07_build_wflow_model.tsv",
    script: "blueearth_cst/model/build_wflow_model.py"

# 1.08  add_reservoirs_lakes_glaciers — add waterbodies to the built model
# (temporary hydromt fix; can fold back into build_wflow_model when supported)
rule add_reservoirs_lakes_glaciers:
    message: rule_banner("1.08", "add_reservoirs_lakes_glaciers", summary="add waterbodies to the model")
    input:
        # ancient(): this rule COMMITS writes back into staticmaps.nc, so a
        # plain input would re-trigger it on its own output. Ordering and
        # rebuild-detection come from the sentinel below instead (R7-1).
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"),
        model_built = f"{basin_dir}/.model_built"
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
        f"{LOG_PARTS_DIR}/1.08_add_reservoirs_lakes_glaciers.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.08_add_reservoirs_lakes_glaciers.tsv",
    script:
        "blueearth_cst/model/setup_reservoirs_lakes_glaciers.py"

# 1.09  declare_wflow_outputs — add gauges + output variables to the model
rule declare_wflow_outputs:
    message: rule_banner("1.09", "declare_wflow_outputs")
    input:
        basin_nc = ancient(f"{basin_dir}/staticmaps.nc"),
        text = f"{basin_dir}/staticgeoms/reservoirs_lakes_glaciers.txt",
        # P2 already created gauge/outlet maps from this registry. This rule
        # only adds output declarations and verifies those identities.
        location_registry = f"{spatial_dir}/location_registry.csv",
    output:
        configured = touch(f"{basin_dir}/.outputs_configured"),
    params:
        outputs = wflow_outvars,
        data_catalog = DATA_SOURCES
    log:
        f"{LOG_PARTS_DIR}/1.09_declare_wflow_outputs.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.09_declare_wflow_outputs.tsv",
    script:
        "blueearth_cst/model/setup_gauges_and_outputs.py"

# 1.11  write_outlet_index — write the outlet position -> subcatchment-ID mapping
rule write_outlet_index:
    message: rule_banner("1.11", "write_outlet_index")
    input:
        outlets_path = f"{basin_dir}/staticgeoms/outlets.geojson",
        location_registry = f"{spatial_dir}/location_registry.csv",
        # ADR 0004: rule 1.10 rewrites every staticgeoms/ layer, so this read
        # needs the terminal anchor. Nothing in WF1 consumes outlet_index.csv,
        # so waiting for the model to be final costs only a schedule slot.
        model_final = ancient(f"{basin_dir}/.model_final"),
    output:
        outlet_index_path = f"{basin_dir}/staticgeoms/outlet_index.csv"
    log:
        f"{LOG_PARTS_DIR}/1.11_write_outlet_index.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.11_write_outlet_index.tsv",
    script:
        "blueearth_cst/model/write_outlet_index.py"

# 1.10  add_climate_forcing — assemble the hydromt forcing recipe, then apply it
#
# [R10-1]: rule 1.07 `setup_runtime` was MERGED IN here. It wrote the `steps:`
# YAML below and nothing else read it — one job split across two rules, and the
# reason 1.07's R10 rename was withdrawn rather than replaced. 1.07 is now a gap
# in the numbering.
#
# Two details the merge folded in:
#  - 1.07's `gauges_path` input (staticgeoms/outlets.geojson) was DEAD — the
#    script never opened it — and is dropped rather than carried over.
#  - `.outputs_configured` is inherited from 1.07, and is load-bearing: it is
#    rule 1.09's completion sentinel, so the R7-1 rebuild cascade
#    (1.07 -> 1.08 -> 1.09 -> here) survives the merge.
#
# `script:` rather than `shell:` because Snakemake allows one per rule and the
# halves were one of each; the hydromt command issued is byte-identical to the
# one this rule ran before.
rule add_climate_forcing:
    message: rule_banner("1.10", "add_climate_forcing", summary="build the forcing netCDF for the run period")
    input:
        outputs_configured = f"{basin_dir}/.outputs_configured",
        # The climate store, as the forcing SOURCE (2026-08-10). This rule used
        # to re-read the global dataset from the catalog, a second full pass
        # over the source rule 1.04 had already clipped to this basin. Declaring
        # it makes the reuse a DAG edge rather than a convention, and is why
        # `simulation_window` must now sit inside `historical_window`.
        climate_nc = CLIMATE_STORE.outputs["climate_nc"],
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
        # TERMINAL BUILD SENTINEL (ADR 0004). `hydromt update wflow_sbm` calls
        # mod.write(), which rewrites the WHOLE model root -- measured on the R9
        # gate run, where staticmaps.nc, wflow_sbm.toml, hydromt.log and every
        # staticgeoms/ layer all carry this rule's timestamps. So THIS rule, not
        # 1.07, is the last writer of the model, and it is the only correct
        # ordering anchor for a reader.
        #
        # `.outputs_configured` (rule 1.09) was the anchor until now and is NOT
        # sufficient: it precedes this rule's write of staticmaps.nc by 17 s on
        # that same run. Rule 1.12 read staticmaps.nc and finished 9 s before
        # this rule rewrote it -- a timing accident, not a DAG edge, and one
        # that closes as the basin grows. The failure is silent:
        # HDF5_USE_FILE_LOCKING="FALSE" in the pixi env means a concurrent read
        # aborts below Python with no traceback (R9 P2 F5).
        #
        # RESIDUAL RISK, stated because no test can catch it: this sentinel is
        # correct only while this rule is the last writer of the model root. A
        # new rule that mutates the model after 1.10 must move the sentinel to
        # itself. `tests/test_model_root_ordering.py` checks that readers
        # declare it, not that it is attached to the right rule.
        #
        # [R10-1] MERGE NOTE: ADR 0004 attached this to the old 1.10, which then
        # consumed 1.07's YAML. The merge folded 1.07 in, so the rule that
        # carries the sentinel now also WRITES that YAML -- it is still the last
        # writer of the model root, so the anchor is unchanged.
        model_final = touch(f"{basin_dir}/.model_final"),
    params:
        # The SIMULATION window, not the extraction window. These two params set
        # both halves of the same span: the forcing hydromt builds, and the
        # wflow TOML's `[time]` starttime/endtime it runs over
        # (shared/setup_time_horizon.py). They are one key precisely because
        # forcing and run period cannot legitimately differ.
        starttime = get_config(simulation_window, "starttime", optional=False),
        endtime = get_config(simulation_window, "endtime", optional=False),
        clim_source = get_config(shared_cfg, "clim_historical", optional=False),
        basin_dir = basin_dir,
        data_catalog = DATA_SOURCES,
    log:
        f"{LOG_PARTS_DIR}/1.10_add_climate_forcing.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.10_add_climate_forcing.tsv",
    script: "blueearth_cst/model/add_climate_forcing.py"

# 1.14  run_wflow — run the Wflow.jl model on historical forcing
rule run_wflow:
    message: rule_banner("1.14", "run_wflow", summary="run the model over the simulation window")
    input:
        forcing_path = f"{basin_dir}/forcing/inmaps_historical.nc",
        # ADR 0004. Already ordered after 1.10 through forcing_path; declared
        # anyway so the invariant is LOCAL rather than transitive -- a reader is
        # correct because of what it declares, not because of a chain someone
        # has to re-derive.
        model_final = ancient(f"{basin_dir}/.model_final"),
    output:
        # temp() since 2026-08-10 (owner ruling). Wflow's raw csv is an
        # INTERMEDIATE, not a deliverable: rule 1.14b derives the readable
        # per-variable tables from it and rule 1.15 reads it for the evaluation
        # metrics, but nothing downstream of those wants it and it left
        # `run_default/` showing a 0.9 MB file at 21-character precision beside
        # the 0.5 MB table meant to be read. Snakemake drops it once BOTH
        # consumers have run, which is the ordering the ask describes ("removed
        # after creating output_q.csv") expressed as a DAG property rather than
        # as a delete in someone's script.
        #
        # Two costs, both real and neither a blocker:
        #
        # * Re-running 1.15 alone re-runs 1.14 -- the whole Wflow model -- since
        #   its input no longer exists. Iterating on an evaluation FIGURE is
        #   exactly that case, so use `--notemp` when doing it.
        # * A baseline run needs `--notemp` too. `dev/baseline/manifest.json`
        #   pins this file as the wf1 discharge target at FULL precision, and
        #   check_baseline fails hard ("target missing on disk") without it.
        #   The derived table is not a substitute: 5 decimals on discharge
        #   running 1e-5..1e-1 is ~3 significant figures, which is coarser than
        #   the drift the tolerance comparator exists to catch.
        csv_path = temp(f"{basin_dir}/run_default/output.csv"),
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
        f"{LOG_PARTS_DIR}/1.14_run_wflow.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.14_run_wflow.tsv",
    shell:
        # A DRIVER FILE rather than the `-e "using Wflow; Wflow.run()"` one-liner
        # Wflow's docstring suggests. Under `[logging] silent = true` that form
        # emits nothing at all for the whole run -- the longest single step in
        # WF1 -- so the driver adds the progress bar every other long step in the
        # toolbox animates. It is also the shape rule 3.15 already has.
        """python -u "{run_logged}" "{log}" -- {wflow_julia} "{params.driver}" "{params.toml_path}" """

# 1.04  extract_historical_climate — the SHARED historical-climate store producer
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
rule extract_historical_climate:
    message: rule_banner("1.04", "extract_historical_climate", summary="clip the global climate dataset to the basin")
    input:
        **CLIMATE_STORE.inputs,
    params:
        **CLIMATE_STORE.params,
    output:
        **CLIMATE_STORE.outputs,
    log:
        f"{LOG_PARTS_DIR}/1.04_extract_historical_climate.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.04_extract_historical_climate.tsv",
    script:
        CLIMATE_STORE.script

# 1.15  plot_wflow_evaluation — analyse + plot the wflow RUN (parallel leaf
# with 1.12/1.13). Discharge only since ADR 0006; the climate figures it used
# to draw are the map/series family under forcing/plots/ now.

# O-24, basin-average half. The ONE plot_wflow_evaluation family that is a pure
# function of config, so it is declared rather than left undeclared.
#
# Two exclusions, both load-bearing — do not "simplify" either away:
#  - "river discharge" never gets a basavg column at all. Rule 1.09 covers it
#    via setup_outlets/setup_gauges and filters it out of the basin-average
#    setup (setup_gauges_and_outputs.py, `extras`).
#  - "precipitation" DOES get a column, and the rule drops it before plotting
#    (plot_results.py, `drop_vars("precipitation_basavg")`), so the figure is
#    never written. The SCRIPT keeps its name -- R10 renamed rule identifiers,
#    not modules.
# The filename is the CSV column name verbatim (func_plot_signature.plot_basavg
# writes f"{dvar}.png"), which is why it carries the config's spelling, spaces
# included.
_WFLOW_OUTVARS_WITHOUT_BASAVG_PLOT = ("river discharge", "precipitation")
_basavg_pngs = [
    f"{basin_dir}/evaluation/plots/{WFLOW_OUTPUT_CODES[var]}_subcatchment.png"
    for var in wflow_outvars
    if var not in _WFLOW_OUTVARS_WITHOUT_BASAVG_PLOT
]

# The COMPLETE set of tables rule 1.14b writes, derived from the same config
# key. Rule 1.09 declares exactly two kinds of `[output.csv]` column, and both
# reach a derived table because both carry a numeric id:
#
# * `Q` on the `outlets` map (plus the `gauges_locations` map when the project
#   has a location registry) -- declared UNCONDITIONALLY by
#   setup_gauges_and_outputs, i.e. even when `river discharge` is absent from
#   `wflow_outvars`, so `output_q.csv` is always produced;
# * one SHORT CODE per remaining `wflow_outvars` entry, on the `subcatchment`
#   map with a mean reducer -- `gwr_101`.
#
# That second kind is why this list has to exist. 1.14b used to declare
# `output_q.csv` alone, on the reasoning that extras were `<var>_basavg` and so
# carried no id to key on. True once; the short-code change (shared/
# wflow_outputs.CODES) renamed them to `<code>_<subcatchment_id>`, which
# `tidy_wflow_table.split_columns` matches exactly as it matches a gauge
# column. So `write_tidy_tables` wrote `output_gwr.csv` -- and PRUNED it on a
# later run -- while no rule declared it. Undeclared output is invisible to
# Snakemake: nothing rebuilds it when deleted, and since rule 1.14's
# `output.csv` became temp() the only way back was re-running the whole model.
WFLOW_TABLE_PATHS = [f"{basin_dir}/run_default/output_q.csv"] + [
    f"{basin_dir}/run_default/output_{WFLOW_OUTPUT_CODES[var]}.csv"
    for var in wflow_outvars
    if var != "river discharge"
]

# 1.14b  export_wflow_tables — Excel-ready per-variable tables derived from 1.14
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
# Sub-lettered like 3.01c: this belongs to 1.14's output, not after 1.15.
rule export_wflow_tables:
    message: rule_banner("1.14b", "export_wflow_tables")
    input:
        csv_path = f"{basin_dir}/run_default/output.csv",
        # ADR 0004. Ordered after 1.10 transitively through csv_path, declared
        # anyway for the same reason 1.14 and 1.15 declare it: the invariant is
        # LOCAL, not something a reader has to re-derive from a chain.
        model_final = ancient(f"{basin_dir}/.model_final"),
    output:
        # EVERY table, derived from the config -- not just discharge. Declared
        # as the individual files rather than a directory() so Snakemake tracks
        # each one like any other output.
        tables = WFLOW_TABLE_PATHS,
    log:
        f"{LOG_PARTS_DIR}/1.14b_export_wflow_tables.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.14b_export_wflow_tables.tsv",
    script:
        "blueearth_cst/shared/tidy_wflow_table.py"

rule plot_wflow_evaluation:
   message: rule_banner("1.15", "plot_wflow_evaluation")
   input:
       csv_path = f"{basin_dir}/run_default/output.csv",
       # ADR 0004; transitively ordered after 1.10 via 1.14, declared for the
       # same reason as there.
       model_final = ancient(f"{basin_dir}/.model_final"),
       # The climate-store inputs went with the subcatchment climate figures
       # (ADR 0006). This rule evaluates DISCHARGE now, so it no longer waits
       # on rule 1.04 at all.
       # Both observation files, when configured: this rule reads the gauge
       # positions AND the observed series, so either changing must re-evaluate.
       **_locations_input,
       **_observations_input,
       location_registry = f"{spatial_dir}/location_registry.csv",
       script = "blueearth_cst/model/plot_results.py"
   # R07 B10 + O-24. The artifacts move into the engine subtree, split by KIND
   # (P1): figures under evaluation/plots/, the metrics table one level up in
   # evaluation/, because plots/ holds figures only.
   #
   # O-24, restated 2026-08-01 after checking what is actually derivable.
   #
   # DECLARED: the config-invariant subset below, plus `_basavg_pngs` — one PNG
   # per basin-average entry in wflow_outvars, a pure function of config.
   #
   # AND the per-station bin as a DIRECTORY (t2608071206, 2026-08-18). Its
   # members cannot be enumerated at parse time and never could: their count is
   # the model's OUTLETS and SUBCATCHMENTS, a product of rule 1.07 read back
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
   # `clim_{station}_{period}.png` no longer exists — ADR 0006 removed it on
   # 2026-08-09 with the branch that drew it.
   #
   # The two clim_wflow_1_* figures were declared here until 2026-08-09 and are
   # gone with the branch that wrote them (ADR 0006). Their removal also
   # retires the failure mode declaring them introduced: a config with a
   # sub-year historical_window failed here with MissingOutputException where
   # it used to log a skip.
   #
   # No per-station FIGURE is declared any more. They are keyed by wflow_id
   # since 2026-08-10 — `hydrograph_1010.png` — and a wflow_id is a product of
   # the model build, so it is not knowable at parse time. `hydro_wflow_1.png`
   # only worked as a declared output because plot_results.py forced the first
   # outlet's label to that literal, which is the naming inconsistency the
   # rename removes. The metrics table is this rule's config-invariant artifact
   # and takes over as its terminal.
   output:
       metrics_csv = f"{basin_dir}/evaluation/performance_metrics.csv",
       basavg_pngs = _basavg_pngs,
       station_plots = directory(
           f"{basin_dir}/evaluation/plots/{STATION_PLOT_DIRNAME}"
       ),
   params:
       project_dir = f"{project_dir}",
       # The model root is OWNED BY THE RULE, not rebuilt inside the script.
       # plot_results.py used to derive `{project_dir}/hydrology_model` itself,
       # which made the model's location a fact spelled in two places; this move
       # is what forced the question, and passing it is the answer.
       model_dir = basin_dir,
   log:
       f"{LOG_PARTS_DIR}/1.15_plot_wflow_evaluation.log",
   benchmark:
       f"{project_dir}/benchmarks/_parts/1.15_plot_wflow_evaluation.tsv",
   script: "blueearth_cst/model/plot_results.py"

# 1.15b write_run_metadata — the staleness sidecar (design §5.8).
#
# `1.15b`, not `1.16`: that number is gather_benchmarks and `dev/reference/
# naming.md` §8b forbids renumbering to insert a rule. Placed BEFORE the two
# gather rules so its log part -- if it ever grows one -- is still gathered.
#
# It takes the metrics table as input rather than writing into it: that table
# and WF3's indicator CSV are baseline-fingerprinted, so embedding the digests
# in either would falsify the design's no-baseline-re-record claim.
rule write_run_metadata:
    message: rule_banner("1.15b", "write_run_metadata")
    input:
        metrics_csv = f"{basin_dir}/evaluation/performance_metrics.csv",
        # ADR 0004: a rule reading under the model root waits on the terminal
        # marker, because rule 1.08 rewrites the whole directory. The metrics
        # table already orders this rule after 1.15, so the marker adds no edge
        # in practice -- it is declared because the convention is mechanical,
        # and a reader exempted by an argument is how the next one gets missed.
        model_final = ancient(f"{basin_dir}/.model_final"),
    params:
        workflow_name = "build_model",
        effective_config_sha256 = EFFECTIVE_CONFIG_DIGEST,
        configuration_inputs_sha256 = CONFIGURATION_INPUTS_DIGEST,
    output:
        run_metadata = f"{basin_dir}/evaluation/run_metadata.json",
    script:
        "blueearth_cst/shared/write_run_metadata.py"

# 1.12  plot_basin_map — every figure the spatial foundation supports: the
# basin/DEM map plus the thematic family beside it (parallel leaf).
#
# ONE rule for both, because they are one deliverable. The thematic maps draw
# the same overlay and suppress its legend precisely BECAUSE basin_area carries
# it, so a run that produced one set without the other would ship ten maps whose
# linework nothing explains. Splitting them would also duplicate the vector
# inputs and the plots directory across two rules for no scheduling gain — both
# halves are leaves.
rule plot_basin_map:
    message: rule_banner("1.12", "plot_basin_map")
    input:
        # ADR 0007: basin_area depicts ELEVATION, which is data rather than a
        # model result, so it is drawn from the shared spatial foundation and
        # not from the wflow model. That also retires this rule's HDF5 race
        # workaround wholesale — it no longer opens staticmaps.nc, so it no
        # longer has to be ordered behind every writer of that file (the
        # sentinel edge, the ancient() staticmaps declaration, and the -c 3
        # abort-below-Python they existed to prevent, all gone).
        hydrography_nc = SPATIAL_UNITS.hydrography_nc,
        **{name: SPATIAL_UNITS.outputs[name] for name in
           ("basins", "subbasins", "rivers", "locations")},
        # The thematic half's raster stack. This is a NEW EDGE: 1.12 used to
        # depend on rule 1.03 alone and could run as soon as the vectors
        # existed; it now waits for 1.06 as well. That costs nothing in
        # practice — 1.06 is upstream of the model build, so nothing downstream
        # waits on this leaf — and it is the honest dependency: the family is
        # drawn FROM this file.
        spatial_maps = f"{spatial_dir}/spatial_maps.nc",
    output:
        # PNG only since 2026-08-10 (owner's call): nothing in the toolbox
        # or the platform read the vector deliverable, and 600 dpi at 180 mm
        # carries the figure everywhere it is used.
        basin_png = f"{spatial_dir}/plots/basin_area.png",
        # From the registry the plotter iterates, not restated here — the same
        # contract rule 1.13 has with climate_figures.figure_names(). Only the
        # figures whose source variable exists for EVERY shipped catalog source
        # are declared; `soil_depth_to_bedrock` is drawn but left out, because
        # its variable is soilgrids v1.0's and a project on soilgrids_2020 would
        # fail this rule on "missing output files" rather than lose one figure.
        # See SpatialFigure.guaranteed.
        spatial_figures = figure_paths(f"{spatial_dir}/plots"),
    params:
        spatial_dir = f"{spatial_dir}",
    log:
        f"{LOG_PARTS_DIR}/1.12_plot_basin_map.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.12_plot_basin_map.tsv",
    script: "blueearth_cst/shared/plot_spatial_maps.py"

# 1.13  plot_forcing — the canonical climate set for the wflow forcing
# (parallel leaf). Same figures as rule 1.15 draws for the source grid, so the
# two directories answer "what did the downscaling change?" side by side.
rule plot_forcing:
    message: rule_banner("1.13", "plot_forcing")
    input:
        forcing_path = f"{basin_dir}/forcing/inmaps_historical.nc",
        # ADR 0004; see rule 1.14.
        model_final = ancient(f"{basin_dir}/.model_final"),
        **_locations_input,
        # The SAME vector layers rule 1.05 draws, from rule 1.03's shared
        # foundation. Both climate map families read one layer set so the two
        # differ only in the raster underneath — which is what makes "what did
        # downscaling change?" answerable by putting the two directories side
        # by side. The model's own staticgeoms would reintroduce the divergence.
        **{name: SPATIAL_UNITS.outputs[name] for name in
           ("basins", "subbasins", "rivers", "locations")},
        # No `shared_levels` edge since 2026-08-16. These figures used to adopt
        # the colourbar rule 1.05 recorded, which is why that file was declared
        # here; the two families now classify their own footprints and this rule
        # no longer waits on 1.05 at all. See climate_figures.MAP_EXTENT.
    # R07 B10 + O-24: the forcing/model-input QA figures sit beside the forcing
    # they describe, and ALL of them are declared — variable x kind is a fixed
    # cross-product, so the list comes from climate_figures.figure_names()
    # rather than being restated here.
    output:
        _forcing_climate_pngs,
    params:
        project_dir = f"{project_dir}",
        # Rule-owned model root; see rule 1.15's note.
        model_dir = basin_dir,
        geoms_dir = SPATIAL_UNITS.spatial_dir + "/geoms",
        water_year_start = WATER_YEAR_START,
    log:
        f"{LOG_PARTS_DIR}/1.13_plot_forcing.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.13_plot_forcing.tsv",
    script: "blueearth_cst/model/plot_map_forcing.py"

# --- benchmark gather ---------------------------------------------------------
# Per-rule benchmarks land under benchmarks/_parts/. This runs once, after the
# terminal outputs (so all WF1 rules ran), and merges the WF1 parts into one
# benchmarks/wf1_benchmarks.md (rule column + TOTAL row), fresh each run.
rule gather_benchmarks:
    message: rule_banner("1.16", "gather_benchmarks")
    input:
        WF1_TERMINALS,
    output:
        f"{project_dir}/benchmarks/wf1_benchmarks.md",
    params:
        parts_dir = f"{project_dir}/benchmarks/_parts",
        workflow_num = 1,
    script: "blueearth_cst/shared/merge_benchmarks.py"

# 1.05  plot_climate_source — source-grid climate figures from the shared store
# (R07 B4 / P4). Declared HERE ONLY: unlike rule 1.04 this is a single
# declaration, so none of B1's two-DAG symmetry machinery applies.
#
# The whole subgraph of these figures is rule 1.04 (whose sole input is the
# tracked data catalog) plus this rule, so they build with NEITHER
# models/hydrology/wflow/ NOR config/defaults/wflow_build_model.yml on disk — the P4
# assertion, pinned by tests/test_plot_climate_source.py.
#
# The data catalog rides in `params`, not `input`: the era5 branch resolves
# `era5_orography` through it, but the store's freshness boundary is rule 1.04's
# catalog edge (ext2-01) and duplicating it here would re-plot on every catalog
# touch without the extraction having changed.
# What this source can honestly be drawn for. A precipitation-only source gets
# the precip figures and nothing else — its temperature and PET fields in the
# store are era5's, borrowed so the model can be forced, and drawing them under
# this source's name would report another dataset's values as this one's.
_source_plot_vars = source_climate_vars(clim_source)

_source_plot_inputs = {"climate_nc": CLIMATE_STORE.outputs["climate_nc"]}
if "oro_nc" in CLIMATE_STORE.outputs and _source_plot_vars != ("precip",):
    # On chirps the store carries an orography sidecar and it is a declared
    # input; on era5 there is none and the orography comes from the catalog
    # instead. It feeds the lapse correction behind the TEMPERATURE and PET
    # figures only, so where those are not drawn it is not an input either —
    # the store still produces it, for rule 3.08 and the forcing catalog.
    _source_plot_inputs["oro_nc"] = CLIMATE_STORE.outputs["oro_nc"]

# The vector layers the source maps are drawn over, from rule 1.03's shared
# foundation — NOT the wflow model's staticgeoms. That choice keeps this rule
# independent of the model build (1.07): the source grid is the climate BEFORE
# any model exists, and making it wait on one would invert that. Declared as
# real inputs so the edge is in the DAG rather than read behind Snakemake's
# back; the cost is that re-delineating the basin re-plots these figures, which
# is correct — the outline on them would otherwise be stale.
_source_plot_inputs.update(
    {name: SPATIAL_UNITS.outputs[name] for name in
     ("basins", "subbasins", "rivers", "locations")}
)

# The basin's cells on THIS source's grid — what the `basin_avg` series reduce
# over, and rule 1.04's own output. The same file weathergenr averages over, so
# the figures, the generator and the stress test share one definition of the
# basin; without it the series were a mean over the store's BUFFERED bbox.
_source_plot_inputs["basin_cells"] = CLIMATE_STORE.outputs["basin_cells"]

rule plot_climate_source:
    message: rule_banner("1.05", "plot_climate_source")
    input:
        **_source_plot_inputs,
    output:
        # The WF0 filename grammar (`dev/working/wf0-figure-filename-rule.md`),
        # NOT because this is WF0 but because it is the same FAMILY: this rule
        # and wf0's 0.05 write the same files to the same `<store>/plots/`, so a
        # rename in one is a rename in both or the two workflows disagree about
        # what they produce. The forcing family (1.13) is the separate one that
        # keeps its own names.
        [
            f"{store_dir}/plots/{name}"
            for name in source_figure_names(
                clim_source,
                variables=_source_plot_vars,
                spatial_scopes=DECLARED_SPATIAL_SCOPES,
            )
        ],
        # Per-subbasin figures are named for delineation ids, so their count is
        # a runtime fact — see the same note on wf0's rule 0.05.
        directory(f"{store_dir}/plots/{SUBBASIN_PLOT_DIRNAME}"),
    params:
        plot_dir = f"{store_dir}/plots",
        subbasin_plot_dir = f"{store_dir}/plots/{SUBBASIN_PLOT_DIRNAME}",
        data_sources = DATA_SOURCES,
        clim_source = clim_source,
        geoms_dir = SPATIAL_UNITS.spatial_dir + "/geoms",
        water_year_start = WATER_YEAR_START,
    log:
        f"{LOG_PARTS_DIR}/1.05_plot_climate_source.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/1.05_plot_climate_source.tsv",
    script: "blueearth_cst/climate_analysis/plot_climate_source.py"

# 1.17  gather_logs — merge every WF1 log part into ONE workflow log.
#
# Same rule as WF2's 2.09 and WF3's 3.18, against the same script; only the label
# list, the parts dir and the output name differ. `input:` is WF1_TERMINALS, the
# same set gather_benchmarks takes, which is what schedules it LAST: every
# logging rule is upstream of it. The parts stay in `params:` — they are
# `log:` files, and Snakemake does not track those in the DAG, so naming them as
# `input:` would demand them as buildable targets.
#
# The merge DELETES the parts it consumed and prunes the emptied dirs, so a clean
# full run leaves no logs/_parts/ at all. After a PARTIAL re-run only the re-run
# rules have parts, so the rewritten log marks the rest "no part from this run" —
# the same trade `merge_benchmarks` makes (dev/followups-archive.md R7-9).
rule gather_logs:
    message: rule_banner("1.17", "gather_logs")
    input:
        WF1_TERMINALS,
    output:
        f"{project_dir}/logs/{WORKFLOW_LOG_NAME}",
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
JOURNAL_PATH = f"{project_dir}/config/runs/journal.jsonl"
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
        sys.stderr.write(
            "\n" + run_header("wf1 build_model", project_dir, config_path) + "\n\n"
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
