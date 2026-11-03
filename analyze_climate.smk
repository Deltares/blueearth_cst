# Current guidance: dev/reference/toolbox-spatial-foundation.md
import os
import sys
import time
import uuid
from pathlib import Path

# Shared helpers live in blueearth_cst/; make them importable regardless of the working
# directory by prepending this Snakefile's own directory to sys.path.
# See dev/records/milestones/r03/model-builder-design.md §3.
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.provenance import append_journal_line, configuration_inputs_digest, effective_config_digest, environment_file_hashes, file_sha256, journal_event, referenced_inputs_for_digest, toolbox_identity
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, catalog_root, climate_store_rule, declare_path_tokens, declare_project_root, declare_warning_tally, get_config, patch_psutil_windows_benchmark, region_rule, resolve_water_year_start, spatial_units_rule, validate_historical_window, warning_count
from blueearth_cst.shared.console_style import install_console_style, open_run_header, RuleRegistry, rule_banner, run_header, run_summary, target_banner, warn_if_project_dir_in_repo, warn_row
from blueearth_cst.shared.console_style import header_config_path
from blueearth_cst.shared.config_composition import compose_config
from blueearth_cst.shared.workflow_archive_launch import captured_projection, require_capture
from blueearth_cst.spatial.config import parse_spatial_config
from blueearth_cst.climate_analysis.diagnostic_outputs import diagnostic_source_outputs, diagnostic_comparison_outputs
from blueearth_cst.climate_analysis.diagnostic_settings import parse_settings
from blueearth_cst.shared.map_reference import historical_map_reference_rule, require_wf0_completeness, source_map_inputs

# --- figure filenames ---------------------------------------------------------
# WF0's figures follow `dev/reference/wf0-figure-naming.md`:
#   <dataset_scope>_<variable>_<plot_context>_<spatial_scope>.png
# built by `climate_analysis.figure_naming`, never spelled here. The wflow
# FORCING family (rule 1.12) keeps its own names -- the rule stages WF0 first.
#
# Basin temporal figures use `basin_avg`; source maps use `basin_ext`.
# Optional `..._subbasin_<id>_avg.png` figures use IDs from rule 0.02's
# `subbasins.geojson`, which need not exist at DAG parse time. Their runtime
# inventory is covered by declared `directory()` outputs when enabled.
SUBBASIN_PLOT_DIRNAME = "subbasins"

# Windows: make Snakemake's benchmark memory/IO/CPU indicators work (else all NA).
patch_psutil_windows_benchmark()

# read path of the config file (Snakemake records it from --configfile) so
# downstream scripts can be handed the same path. Forwarding config_path is a
# repo convention -- keep it even though the Snakefile itself uses `config`.
config_path = workflow.configfiles[0]
CAPTURE_RECORD = require_capture("analyze_climate", config_path)

# The consumed-key PROJECTION: the config paths this workflow actually reads.
# Digesting the projection rather than the whole file is what stops another
# workflow's edit from re-firing this record.
#
# HOISTED above the first config read (R13 D-8.2): the projection is a
# config-independent literal, and `compose_config` derives R(entry) from it, so
# it has to be known before any section is touched. Moving it changed no value.
CONFIG_PROJECTION = ("project", "basin", "climate", "model", "workflows.analyze_climate")
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
    config, config_path, entry="analyze_climate", declared_sections=CONFIG_PROJECTION,
)
from blueearth_cst.shared.plot_config import resolve_annotation
from blueearth_cst.shared.plot_setup import setup_digest
plot_annotation = resolve_annotation(config)
plot_setup_digest = setup_digest()
CAPTURE_DIGESTS = captured_projection(CAPTURE_RECORD, config, ADVANCED_SETTINGS)
# Sorted so the declared input lists below do not churn on dict order.
WF_CONFIG_PATHS = sorted(WORKFLOW_CONFIG_PATHS.values())

# R01 schema — three top-level sections.
project_cfg = config["project"]
# R14 D-7.2: `shared:` dissolved into sections by KIND. `climate_cfg` is the
# only new binding -- `basin:` and `model:` are read at their use sites, which
# is where the v1 `shared_cfg` indirection was buying nothing.
climate_cfg = config.get("climate") or {}
my_cfg = config["workflows"]["analyze_climate"]

project_dir = get_config(project_cfg, "project_dir", optional=False)
# O-22: make the two-tier project_dir rule mechanical rather than documentary.
# Warns, never raises; test_case/ is the one exemption.
warn_if_project_dir_in_repo(project_dir, workflow.basedir)
DATA_SOURCES = get_config(project_cfg, "catalog", optional=False)  # C-40

basin_cfg = config["basin"]
spatial_cfg = parse_spatial_config(basin_cfg, my_cfg)
model_region = get_config(basin_cfg, "region", optional=False)
basin_hydrography = spatial_cfg.hydrography
basin_index = spatial_cfg.basin_index
historical_window = get_config(climate_cfg, "window", optional=False)
# ONE minimum window for the whole toolbox, enforced identically here and at
# extraction. Parse time, before any rule executes -- same stance as WF1.
validate_historical_window(historical_window)
# The water year the climate figures aggregate on, from the one shared key WF1,
# WF2 and WF3 also read. Figures are terminal artifacts, so this changes no
# number -- but a figure labelled 'annual' should mean the basin's year.
WATER_YEAR_START = resolve_water_year_start(get_config(climate_cfg, "water_year_start"))

# --- the candidate source set -------------------------------------------------
# THE PROJECT'S OWN SOURCE IS ALWAYS FIRST AND ALWAYS PRESENT. `candidate_sources`
# ADDS to it rather than replacing it, so a config that sets nothing gets exactly
# the figures WF1 already draws for `shared.clim_historical` and nothing else --
# this workflow is then a model-free entry point onto artifacts the project
# already has, not a new cost.
#
# Order is declaration order with duplicates dropped, NOT sorted: the primary
# source leads every figure set and every comparison table, which is the reading
# order a person wants when the question is "should I switch away from it?".
clim_source = get_config(climate_cfg, "selected", optional=False)
# `C-43`: the candidate set moved UP to `climate.sources` and WIDENED -- it is
# the full list with no privileged element, and `climate.selected` names one
# MEMBER of it. The v1 key held the OTHERS, beside a privileged
# `clim_historical`, which is why the migration unions the two rather than
# copying one.
#
# The loader already refuses a `selected` outside `sources`, so the only work
# left here is ORDER: the primary leads every figure set and comparison table,
# and `sources` is in declaration order.
_declared_sources = get_config(climate_cfg, "sources", []) or []
if isinstance(_declared_sources, str):
    raise ValueError(
        "climate.sources must be a LIST of source names, got the string "
        f"{_declared_sources!r}. A bare string would be iterated character "
        "by character and mint one store per letter."
    )
CANDIDATE_SOURCES = list(dict.fromkeys([clim_source, *_declared_sources]))

# P3-2a bounded support (design ext2-3): the raw-climate path supports era5,
# chirps and chirps_global only. Rejected HERE, at parse time, for every
# candidate rather than only for the project's own source -- an unsupported
# entry would otherwise fail deep inside a generated rule whose name does not
# say which config key put it there.
_SUPPORTED_SOURCES = ("era5", "chirps", "chirps_global")
for _src in CANDIDATE_SOURCES:
    if _src not in _SUPPORTED_SOURCES:
        _where = (
            "shared.clim_historical"
            if _src == clim_source
            else "workflows.analyze_climate.candidate_sources"
        )
        raise ValueError(
            f"{_where}: {_src!r} is not supported by the wf0 raw-climate path; "
            f"supported sources: {', '.join(_SUPPORTED_SOURCES)}"
        )

# The current-only run record, one per workflow (config-snapshot redesign,
# 2026-08-13).
RUN_RECORD = f"{project_dir}/config/analyze_climate/run_record.yml"

# Every external file this workflow's configuration points at. Hashed at parse
# time so the digest moves when one is edited IN PLACE.
CONFIG_REFERENCES = [
    # The per-workflow config files, so an in-place edit to the file that now
    # holds this workflow's settings moves the digest (R13 D-10.5). After the
    # split the project file no longer carries those settings, so leaving them
    # out would let the most-edited config in a project change with nothing
    # re-firing. Derived from the dict `compose_config` returned, so this set
    # and the one `copy_config_files` records cannot drift.
    *[(f"workflow_config_{name}", path)
      for name, path in sorted(WORKFLOW_CONFIG_PATHS.items())],
    *[("data_catalog", source) for source in
      (DATA_SOURCES if isinstance(DATA_SOURCES, (list, tuple)) else [DATA_SOURCES])],
]

EFFECTIVE_CONFIG_DIGEST = CAPTURE_DIGESTS[0] if CAPTURE_DIGESTS else effective_config_digest(
    config, ADVANCED_SETTINGS, CONFIG_PROJECTION
)
CONFIGURATION_INPUTS_DIGEST = CAPTURE_DIGESTS[1] if CAPTURE_DIGESTS else configuration_inputs_digest(
    EFFECTIVE_CONFIG_DIGEST,
    toolbox_identity(),
    environment_file_hashes(),
    referenced_inputs_for_digest(CONFIG_REFERENCES),
)

# --- The one project region artifact -------------------------------
# Splatted into rule 0.01 below, byte-identical to 1.01 / 2.01 / 3.03 except
# message/log/benchmark. tests/test_region_rule.py parses every workflow and
# fails on ANY other difference.
REGION = region_rule(
    project_dir=project_dir,
    model_region=model_region,
    data_sources=DATA_SOURCES,
    hydrography=basin_hydrography,
    basin_index=basin_index,
)

# --- The shared vector foundation -------------------------------
# `parse_spatial_config(basin_cfg)` WITHOUT `my_cfg`: the shared
# rule's params must depend only on shared project and basin settings, so a
# workflow-local key cannot feed it and the four declarations stay identical.
SPATIAL_UNITS = spatial_units_rule(
    project_dir=project_dir,
    spatial_config=parse_spatial_config(basin_cfg),
    data_sources=DATA_SOURCES,
)

# --- One climate store per candidate source -----------------------------------
# Built from the SAME factory the other three workflows splat, once per source.
# For `shared.clim_historical` the resulting spec is identical to the one WF1's
# rule 1.03 declares -- same script, inputs, params and outputs -- so the store
# this workflow writes is the store WF1 and WF3 read, at the same path, and
# whichever workflow runs first builds it.
#
# THE PATH CONVENTION IS DELIBERATE AND LOAD-BEARING. Candidate stores land in
# `data/climate/historical/<source>_<window>/` beside the project's own, not in
# a separate evaluation bin, so a candidate that WINS the comparison is already
# extracted: switching `shared.clim_historical` to it costs nothing and re-runs
# no extraction. `dev/scripts/prune_climate_store.py` already reports on exactly
# this directory family.
#
# `historical_window` IS A CEILING FOR A CANDIDATE, NOT A DEMAND. CHIRPS begins
# in 1981, a locally staged subset begins wherever the staging did, and a window
# one candidate cannot fill is the ordinary case of two datasets with different
# records -- not a misconfigured project. Each source is extracted over the
# widest span it holds inside the window, and the narrowing is logged
# (`shared/climate_window.py`).
#
# The MIN_HISTORICAL_YEARS floor splits with it, because its reason does:
# weathergenr's wavelet minimum binds the source that FEEDS the pipeline, and an
# extra candidate ends at a comparison figure. So the primary keeps the floor --
# its spec stays byte-identical to WF1's and WF3's, which is what
# tests/test_climate_store_contract.py pins -- and the extras relax it to a
# logged warning. A relaxed store cannot be promoted silently: WF1 and WF3
# declare the store WITHOUT the flag, so switching `shared.clim_historical` onto
# a candidate changes the params, re-extracts, and meets the floor there.
CLIMATE_STORES = {
    source: climate_store_rule(
        project_dir=project_dir,
        model_region=model_region,
        clim_source=source,
        historical_window=historical_window,
        data_sources=DATA_SOURCES,
        hydrography=basin_hydrography,
        basin_index=basin_index,
        enforce_min_years=(source == clim_source),
        forcing_required=(
            source == clim_source or source not in {"chirps", "chirps_global"}
        ),
    )
    for source in CANDIDATE_SOURCES
}

# --- rule numbering -----------------------------------------------------------
# `W.NN` = the rule's position in this workflow's LOGICAL order. `W` is a
# workflow ID, not a position (dev/reference/dev-naming-conventions.md §9), and IDs need not
# start at 1: this workflow is 0 because it precedes model creation, so
# `ls logs/` sorts wf0, wf1, wf2, wf3 in execution order without renumbering
# the three that already exist.
#
# No numbers are reserved: the reserved 0.07-0.09 band was dropped in the
# 2026-09-24 rule-naming migration. The evaluation rules (t2608181139) take
# numbers when they are built.
#
# DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix (0.04b).

# --- log layout ---------------------------------------------------------------
# Every logging rule writes a PART under logs/_parts/, and rule 0.07 merges the
# parts into ONE logs/wf0_analyze_climate.log, then deletes them.
#
# LOG_RULES is the merge order: rule LABELS, not part paths, and for a fan-out
# rule the label is SINGULAR. Rules 0.03 and 0.04 are generated once per
# candidate source, so each writes into a DIRECTORY named for the label
# (`0.03_extract_climate_datasets/<source>.log`) and merge_logs lists that
# directory to find its members -- the fan-out width lives only in the rule that
# owns it. Same shape as WF4's `4.05_run_wflow_simulations`, whose rule
# identifiers are `run_wflow_simulations_batch_<b>`.
#
# The list is built by the RuleRegistry below, one registration per rule, and
# tests/test_log_rules_contract.py asserts it in BOTH directions and in
# rule-number order.
from blueearth_cst.shared.execution_outputs import workflow_paths, declare_warning_tally
WORKFLOW_LOG_NAME = "wf0_analyze_climate.log"

# One tally per run, opened at parse time and published to every job process
# through the environment: a rule prints its warnings from a process of its
# own, and the verdict below is written by this one. See
# `snake_utils.declare_warning_tally`.
declare_warning_tally(project_dir, WORKFLOW_LOG_NAME)
LOG_PARTS_DIR = f"{project_dir}/logs/_parts"
_paths = workflow_paths(project_dir, "wf0", WORKFLOW_LOG_NAME, "wf0_benchmarks.md", LOG_PARTS_DIR, f"{project_dir}/benchmarks/_parts")
WORKFLOW_LOG_PATH, BENCHMARK_OUTPUT_PATH = _paths.log, _paths.benchmark
LOG_PARTS_DIR, BENCH_PARTS_DIR = _paths.log_parts, _paths.benchmark_parts

# The run's key folders, stated ONCE. `run_header` prints them at the top of the
# console and every rule log repeats them in its own header. No `model` row --
# this workflow builds none, and that is its whole point. `climate` is the
# PRIMARY source's store; a multi-source run prints the others in full, which is
# correct, since the comparison is about telling them apart.
declare_path_tokens(
    data=catalog_root(DATA_SOURCES),
    # The historical ROOT, not one store — wf0 is the workflow that reads several.
    # WF1 and WF3 declare their single store, so `<climate>/extract_historical.nc`
    # is the whole path there. Here that would tokenize the PRIMARY source and
    # leave every candidate at full length, so a comparison run prints its two
    # sources in two different shapes — the one thing this workflow exists to put
    # side by side. Rooted one level up, both read `<climate>/<key>/...` and the
    # store key stays visible, which is the part a reader is comparing.
    climate=os.path.dirname(CLIMATE_STORES[clim_source].store_dir),
)
declare_project_root(project_dir)
RULES = RuleRegistry(LOG_PARTS_DIR, BENCH_PARTS_DIR)
LOG_RULES = RULES.log_rules
DELINEATE_REGION = RULES.logged("0.01", "delineate_region", summary="delineate the basin boundary")
DELINEATE_SUBBASINS_AND_RIVERS = RULES.logged("0.02", "delineate_subbasins_and_rivers", summary="derive subbasins, rivers, and output locations")
EXTRACT_CLIMATE_DATASETS = RULES.logged("0.03", "extract_climate_datasets", summary="extract a basin climate grid")
PRODUCE_MAP_REFERENCE = RULES.logged("0.03b", "produce_historical_map_reference", summary="prepare the shared display reference")
PREPARE_SOURCE_MAPS = RULES.logged("0.03c", "prepare_historical_map_fields", summary="prepare shared native-grid map fields")
COMPUTE_DIAGNOSTICS = RULES.logged("0.04", "compute_climate_diagnostics", summary="compute source diagnostic tables")
PLOT_DIAGNOSTICS = RULES.logged("0.04b", "plot_climate_diagnostics", summary="render historical climate diagnostics")

# Rule 0.05 exists only when there is more than one source to compare, so it
# is registered -- and its label enters LOG_RULES -- only then.
if len(CANDIDATE_SOURCES) > 1:
    COMPARE_DIAGNOSTICS = RULES.logged("0.05", "plot_climate_comparison", summary="compare sources over the configured common period")
GATHER_BENCHMARKS = RULES.banner_only("0.06", "gather_benchmarks", summary="merge rule timing and resource-use records")
GATHER_LOGS = RULES.banner_only("0.07", "gather_logs", summary="merge rule logs")


DIAGNOSTIC_SETTINGS = parse_settings(my_cfg.get("diagnostics"), historical_window, CANDIDATE_SOURCES)
_subbasin_figures = my_cfg.get("subbasin_figures", False)
if type(_subbasin_figures) is not bool:
    raise ValueError("subbasin_figures must be a boolean")
DIAGNOSTIC_SETTINGS["subbasin_figures"] = _subbasin_figures
from blueearth_cst.shared.wf3_science import water_year_start_number
DIAGNOSTIC_M0 = water_year_start_number(WATER_YEAR_START)
MAP_REFERENCE = historical_map_reference_rule(
    project_dir=project_dir, model_region=model_region, climate=climate_cfg,
    data_sources=DATA_SOURCES, region_geojson=REGION.outputs["region_geojson"],
    basin_geojson=SPATIAL_UNITS.outputs["basins"], m0=DIAGNOSTIC_M0,
    hydrography=basin_hydrography, basin_index=basin_index,
)
require_wf0_completeness(DIAGNOSTIC_SETTINGS, MAP_REFERENCE.params["completeness"])
MAP_SOURCE_SPECS = {spec.source: spec for spec in MAP_REFERENCE.source_specs}
SOURCE_DIAGNOSTICS = {s: diagnostic_source_outputs(CLIMATE_STORES[s].store_dir, s, DIAGNOSTIC_SETTINGS) for s in CANDIDATE_SOURCES}
POOLED_ROOTS = [SOURCE_DIAGNOSTICS[s]["root"] for s in CANDIDATE_SOURCES]
COMPARISON_DIR = f"{project_dir}/data/climate/historical/comparison"
COMPARISON_DIAGNOSTICS = diagnostic_comparison_outputs(COMPARISON_DIR, CANDIDATE_SOURCES, DIAGNOSTIC_SETTINGS) if len(CANDIDATE_SOURCES) > 1 else None
WF0_TERMINALS = [p for spec in SOURCE_DIAGNOSTICS.values() for p in spec["render"].values()]
if COMPARISON_DIAGNOSTICS:
    WF0_TERMINALS += list(COMPARISON_DIAGNOSTICS["compute"].values()) + list(COMPARISON_DIAGNOSTICS["render"].values()) + list(COMPARISON_DIAGNOSTICS["source_summary"].values())
if DIAGNOSTIC_SETTINGS["subbasin_figures"]:
    WF0_TERMINALS += [f"{spec['root']}/plots/subbasins" for spec in SOURCE_DIAGNOSTICS.values()]
    if COMPARISON_DIAGNOSTICS:
        WF0_TERMINALS.append(f"{COMPARISON_DIAGNOSTICS['root']}/plots/subbasins")
WF0_TERMINALS.append(SPATIAL_UNITS.outputs["basins"])

WF0_TARGETS = [
    *WF0_TERMINALS,
    *([RUN_RECORD] if os.environ.get("BLUEEARTH_WF012_CAPTURE_CONTEXT") else []),
    *_paths.gather_targets.values(),
]

# all — target aggregator: the canonical climate figure set per source
rule all:
    message: target_banner("all", WF0_TARGETS, project_dir, compact=True)
    input:
        WF0_TARGETS,

# 0.01  delineate_region — the one project region artifact.
# Byte-identical to 1.01, 2.01 and 3.03 except message/log/benchmark; everything
# else is splatted from REGION so the four cannot drift.
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

# 0.02  delineate_subbasins_and_rivers — the shared vector foundation.
# Byte-identical to 1.02, 2.02 and 3.04 except message/log/benchmark.
#
# The VECTOR half only, as in WF2: the raster half (rule 1.05) stays WF1-only,
# so a climate-only run obtains basin and subbasin boundaries without reading
# `vito`, `modis_lai` or `soilgrids` at all.
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


# 0.03  extract_climate_datasets — ONE rule per candidate source.
# 0.04 / 0.04b — compute and render source-owned diagnostics.
#
# GENERATED IN A LOOP RATHER THAN WILDCARDED, and the reason is the store's
# output SET, not style: `climate_store_rule` returns `oro_nc` for a selected
# CHIRPS forcing store, but not for ERA5 or a comparison-only CHIRPS candidate.
# A Snakemake rule has a fixed output set, so one wildcard rule cannot cover
# these roles. Generating a concrete rule per source takes the shape from the
# spec instead.
#
# Not a new mechanism: WF4 declares its batch fan-out the same way
# (`simulate_and_indicators.smk`, `run_wflow_simulations_batch_<b>`).
#
# For `shared.clim_historical` the generated 0.03 is the shared producer
# contract with a different rule NAME -- same script, inputs, params, outputs.
# tests/test_climate_store_contract.py pins that equivalence rather than
# byte-identity of the declaration, which is the honest form of the claim.


for _source in CANDIDATE_SOURCES:
    _spec = CLIMATE_STORES[_source]

    rule:
        name: EXTRACT_CLIMATE_DATASETS.job_name(_source)
        message: EXTRACT_CLIMATE_DATASETS.banner(part=_source)
        input:
            **_spec.inputs,
        params:
            **_spec.params,
        output:
            **_spec.outputs,
        log:
            EXTRACT_CLIMATE_DATASETS.log(_source),
        benchmark:
            EXTRACT_CLIMATE_DATASETS.benchmark(_source),
        script:
            _spec.script

    _diag = SOURCE_DIAGNOSTICS[_source]
    _compute_inputs = source_map_inputs(MAP_SOURCE_SPECS[_source])
    # Orography is consumed by the separate native-grid PET preparation.
    _compute_inputs.pop("orography_inputs")
    _compute_inputs.pop("oro_nc", None)
    _compute_outputs = dict(_diag["compute"])
    _compute_outputs.pop("maps")
    if DIAGNOSTIC_SETTINGS["subbasin_figures"]:
        _compute_inputs["subbasins"] = SPATIAL_UNITS.outputs["subbasins"]
        _compute_outputs["subbasin_tables"] = directory(f"{_diag['root']}/tables/subbasins")

    rule:
        name: COMPUTE_DIAGNOSTICS.job_name(_source)
        message: COMPUTE_DIAGNOSTICS.banner(part=_source)
        input: **_compute_inputs,
        output: **_compute_outputs,
        params:
            settings=DIAGNOSTIC_SETTINGS, source=_source, store_dir=_spec.store_dir,
            m0=DIAGNOSTIC_M0, data_sources=DATA_SOURCES,
            preparation_code_digest=MAP_REFERENCE.params["preparation_code_digest"],
            map_environment=MAP_REFERENCE.params["environment"],
        log: COMPUTE_DIAGNOSTICS.log(_source),
        benchmark: COMPUTE_DIAGNOSTICS.benchmark(_source),
        script: "blueearth_cst/climate_analysis/compute_climate_diagnostics.py"

    _render_outputs = dict(_diag["render"])
    _render_inputs = {"tables": list(_diag["compute"].values()),
        "pooled": [p for spec in SOURCE_DIAGNOSTICS.values() for key, p in spec["compute"].items() if key in {"monthly_values", "metadata"}],
        "map_reference": MAP_REFERENCE.outputs["reference"],
        **MAP_REFERENCE.inputs,
        **{key: SPATIAL_UNITS.outputs[key] for key in ("basins", "subbasins", "rivers", "locations")}}
    if DIAGNOSTIC_SETTINGS["subbasin_figures"]:
        _render_inputs["subbasin_tables"] = f"{_diag['root']}/tables/subbasins"
        _render_outputs["subbasin_figures"] = directory(f"{_diag['root']}/plots/subbasins")

    rule:
        name: PLOT_DIAGNOSTICS.job_name(_source)
        message: PLOT_DIAGNOSTICS.banner(part=_source)
        input: **_render_inputs,
        output: **_render_outputs,
        params:
            annotation=plot_annotation,
            plot_setup_digest=plot_setup_digest,
            settings=DIAGNOSTIC_SETTINGS, source=_source, store_dir=_spec.store_dir,
            m0=DIAGNOSTIC_M0, pooled_roots=POOLED_ROOTS,
            subbasin_table_dirs=[f"{_diag['root']}/tables/subbasins"],
            map_reference_params=dict(MAP_REFERENCE.params),
        log: PLOT_DIAGNOSTICS.log(_source),
        benchmark: PLOT_DIAGNOSTICS.benchmark(_source),
        script: "blueearth_cst/climate_analysis/plot_climate_diagnostics.py"

# 0.03b  produce_historical_map_reference — shared native-grid display contract.
include: "blueearth_cst/climate_analysis/rules/source_map_preparation.smk"

rule produce_historical_map_reference:
    message: PRODUCE_MAP_REFERENCE.banner()
    input: **MAP_REFERENCE.inputs,
    output: **MAP_REFERENCE.outputs,
    params: **MAP_REFERENCE.params,
    log: PRODUCE_MAP_REFERENCE.log(),
    benchmark: PRODUCE_MAP_REFERENCE.benchmark(),
    script: MAP_REFERENCE.script

if COMPARISON_DIAGNOSTICS:
    _comparison_outputs = {**COMPARISON_DIAGNOSTICS["compute"], **COMPARISON_DIAGNOSTICS["render"], **COMPARISON_DIAGNOSTICS["source_summary"]}
    _comparison_inputs = {
        "source_stores": [CLIMATE_STORES[s].outputs["climate_nc"] for s in CANDIDATE_SOURCES],
        "daily": [SOURCE_DIAGNOSTICS[s]["compute"]["daily"] for s in CANDIDATE_SOURCES],
        "metadata": [SOURCE_DIAGNOSTICS[s]["compute"]["metadata"] for s in CANDIDATE_SOURCES],
        "pooled": [SOURCE_DIAGNOSTICS[s]["compute"]["monthly_values"] for s in CANDIDATE_SOURCES],
    }
    if DIAGNOSTIC_SETTINGS["subbasin_figures"]:
        _comparison_outputs["subbasin_figures"] = directory(f"{COMPARISON_DIAGNOSTICS['root']}/plots/subbasins")
        _comparison_outputs["subbasin_tables"] = directory(f"{COMPARISON_DIAGNOSTICS['root']}/tables/subbasins")
        _comparison_inputs["subbasin_tables"] = [f"{spec['root']}/tables/subbasins" for spec in SOURCE_DIAGNOSTICS.values()]

    rule plot_climate_comparison:
        message: COMPARE_DIAGNOSTICS.banner()
        input: **_comparison_inputs,
        output: **_comparison_outputs,
        params:
            annotation=plot_annotation,
            plot_setup_digest=plot_setup_digest,
            settings=DIAGNOSTIC_SETTINGS, sources=CANDIDATE_SOURCES,
            comparison_dir=COMPARISON_DIR, m0=DIAGNOSTIC_M0,
            pooled_roots=POOLED_ROOTS,
            subbasin_table_dirs=[f"{spec['root']}/tables/subbasins" for spec in SOURCE_DIAGNOSTICS.values()],
        log: COMPARE_DIAGNOSTICS.log(),
        benchmark: COMPARE_DIAGNOSTICS.benchmark(),
        script: "blueearth_cst/climate_analysis/compare_climate_diagnostics.py"


# --- benchmark gather ---------------------------------------------------------
# 0.06  gather_benchmarks — merge the WF0 parts into one benchmarks table.
rule gather_benchmarks:
    message: GATHER_BENCHMARKS.banner()
    input:
        WF0_TERMINALS,
    output:
        BENCHMARK_OUTPUT_PATH,
        BENCHMARK_OUTPUT_PATH.removesuffix(".md") + ".tsv",
    params:
        parts_dir = BENCH_PARTS_DIR,
        workflow_num = 0,
    script: "blueearth_cst/shared/merge_benchmarks.py"

# --- log gather ---------------------------------------------------------------
# 0.07  gather_logs — merge every WF0 log part into ONE workflow log.
#
# Same rule as WF1's 1.18, WF2's 2.08 and WF3's 3.18, against the same script;
# only the label list, the parts dir and the output name differ. `input:` is
# WF0_TERMINALS, which is what schedules it LAST. The parts stay in `params:` --
# they are `log:` files, which Snakemake does not track in the DAG, so naming
# them as `input:` would demand them as buildable targets.
rule gather_logs:
    message: GATHER_LOGS.banner()
    input:
        WF0_TERMINALS,
    output:
        WORKFLOW_LOG_PATH,
    params:
        rules = LOG_RULES,
        parts_dir = LOG_PARTS_DIR,
    script: "blueearth_cst/shared/merge_logs.py"


# --- Run journal --------------------------------------------------------------
#
# Emitted from WORKFLOW-LEVEL HANDLERS, never from a rule: a rule that is up to
# date does not execute, so it cannot record an invocation, and a rule that
# DECLARED the journal would have it deleted before the job ran, truncating the
# ledger to one line every run. See the same block in build_model.smk for the
# scope the P0 probe established -- these fire only when at least one job
# executed, so a gap in the dates means no work was done rather than that nobody
# looked.
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
            workflow="analyze_climate",
            event=event,
            toolbox=_JOURNAL_TOOLBOX,
            effective_config_sha256=EFFECTIVE_CONFIG_DIGEST,
            configuration_inputs_sha256=CONFIGURATION_INPUTS_DIGEST,
            source_config_sha256=file_sha256(config_path),
        ),
    )


# Wall clock for the end-of-run summary. Taken at PARSE, because parse-to-finish
# is the interval a person actually waited.
_RUN_STARTED = time.monotonic()


def _summary(failed):
    """Print the end-of-run block to STDERR, beside Snakemake's own output.

    Never raises: a summary that broke a successful run would be the worst
    possible trade for a convenience.
    """
    try:
        # One write, blank line before it -- see the note on WF3's `_summary`.
        sys.stderr.write(
            "\n"
            + run_summary(
                "wf0 analyze_climate",
                project_dir,
                WORKFLOW_LOG_NAME,
                "wf0_benchmarks.md",
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
    """Print the start-of-run block to STDERR, mirroring `_summary`."""
    try:
        # One write, carrying a blank line on both sides -- see the note on
        # WF3's `_header`, which this mirrors.
        from blueearth_cst.shared.snake_utils import listed
        reference = DIAGNOSTIC_SETTINGS["reference_period"]
        settings = {
            "climate data": listed(CANDIDATE_SOURCES),
            "analysis period": f"{historical_window['start']}–{historical_window['end']}",
        }
        if reference != historical_window:
            settings["reference period"] = f"{reference['start']}–{reference['end']}"
        settings["water year"] = f"Starts in {WATER_YEAR_START}"
        open_run_header(
            "wf0 analyze_climate", project_dir, header_config_path(config_path, CAPTURE_RECORD), **settings,
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
    # Restyle Snakemake's own console output into this toolbox's grammar. Here
    # and not at parse time: the logging stack does not exist yet then.
    install_console_style()
    _header()
    _journal("started")


onsuccess:
    _journal("success")
    _summary(failed=False)


onerror:
    _journal("failed")
    _summary(failed=True)
