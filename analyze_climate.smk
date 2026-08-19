import os
import sys
import time
import uuid
from pathlib import Path

# Shared helpers live in blueearth_cst/; make them importable regardless of the working
# directory by prepending this Snakefile's own directory to sys.path.
# See dev/milestones/r03/model-builder-design.md §3.
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.provenance import append_journal_line, configuration_inputs_digest, effective_config_digest, environment_file_hashes, file_sha256, journal_event, referenced_inputs_for_digest, toolbox_identity
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, catalog_root, climate_store_rule, declare_path_tokens, declare_project_root, get_config, patch_psutil_windows_benchmark, region_rule, resolve_water_year_start, rule_banner, run_summary, spatial_units_rule, target_banner, validate_historical_window, warn_if_project_dir_in_repo, install_console_style, run_header
from blueearth_cst.spatial.config import parse_spatial_config
# The canonical climate figure set. Imported for figure_names() ONLY, so every
# figure is declared from the same list the plotter writes from and the two
# cannot drift -- the same contract rule 1.05 and rule 1.13 rely on.
from blueearth_cst.climate_analysis.climate_figures import source_climate_vars, source_figure_names
# The cross-source comparison set (rule 0.06). Imported for the same reason:
# the module that WRITES the table and the figures is the one that names them.
from blueearth_cst.climate_analysis.compare_sources import comparison_outputs

# --- figure filenames ---------------------------------------------------------
# WF0's figures follow `dev/reference/wf0-figure-filename-rule.md`:
#   <dataset_scope>_<variable>_<plot_context>_<spatial_scope>.png
# built by `climate_analysis.figure_naming`, never spelled here. The wflow
# FORCING family (rule 1.13) keeps its own names -- the rule stages WF0 first.
#
# ONE spatial scope is declarable: `basin_avg`. The per-subbasin figures are
# named `..._subbasin_<id>_avg.png`, and the ids come from the DELINEATION --
# rule 0.03's `subbasins.geojson`, which need not exist when this file is
# parsed. Their count is therefore unknowable at DAG-construction time, exactly
# as rule 1.15's per-station figures are (see the O-24 note in build_model.smk).
# They land in a `subbasins/` bin declared as a `directory()`, which keeps
# `--delete-all-output` complete without the checkpoint that rule shape would
# otherwise need. Rule 1.15 took the same device on 2026-08-18 for the same
# reason (t2608071206), so `stations/` there and `subbasins/` here are one
# pattern rather than two coincidences.
DECLARED_SPATIAL_SCOPES = ("basin_avg",)
SUBBASIN_PLOT_DIRNAME = "subbasins"

# Windows: make Snakemake's benchmark memory/IO/CPU metrics work (else all NA).
patch_psutil_windows_benchmark()

# read path of the config file (Snakemake records it from --configfile) so
# downstream scripts can be handed the same path. Forwarding config_path is a
# repo convention -- keep it even though the Snakefile itself uses `config`.
config_path = workflow.configfiles[0]

# R01 schema — three top-level sections.
project_cfg = config["project"]
shared_cfg = config["shared"]
my_cfg = config["workflows"]["analyze_climate"]

project_dir = get_config(project_cfg, "project_dir", optional=False)
# O-22: make the two-tier project_dir rule mechanical rather than documentary.
# Warns, never raises; test_case/ is the one exemption.
warn_if_project_dir_in_repo(project_dir, workflow.basedir)
DATA_SOURCES = get_config(project_cfg, "data_sources", optional=False)

basin_cfg = shared_cfg["basin"]
spatial_cfg = parse_spatial_config(basin_cfg, my_cfg)
model_region = get_config(basin_cfg, "region", optional=False)
basin_hydrography = spatial_cfg.hydrography
basin_index = spatial_cfg.basin_index
historical_window = get_config(shared_cfg, "historical_window", optional=False)
# ONE minimum window for the whole toolbox, enforced identically here and at
# extraction. Parse time, before any rule executes -- same stance as WF1.
validate_historical_window(historical_window)
# The water year the climate figures aggregate on, from the one shared key WF1,
# WF2 and WF3 also read. Figures are terminal artifacts, so this changes no
# number -- but a figure labelled 'annual' should mean the basin's year.
WATER_YEAR_START = resolve_water_year_start(get_config(shared_cfg, "water_year_start"))

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
clim_source = get_config(shared_cfg, "clim_historical", optional=False)
_extra_sources = get_config(my_cfg, "candidate_sources", []) or []
if isinstance(_extra_sources, str):
    raise ValueError(
        "workflows.analyze_climate.candidate_sources must be a LIST of source "
        f"names, got the string {_extra_sources!r}. A bare string would be "
        "iterated character by character and mint one store per letter."
    )
CANDIDATE_SOURCES = list(dict.fromkeys([clim_source, *_extra_sources]))

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
RUN_RECORD = f"{project_dir}/config/runs/analyze_climate/run_record.yml"

# The consumed-key PROJECTION: the config paths this workflow actually reads.
# Digesting the projection rather than the whole file is what stops another
# workflow's edit from re-firing this record.
CONFIG_PROJECTION = ("project", "shared", "workflows.analyze_climate")

# Every external file this workflow's configuration points at. Hashed at parse
# time so the digest moves when one is edited IN PLACE.
CONFIG_REFERENCES = [
    ("data_catalog", source) for source in
    (DATA_SOURCES if isinstance(DATA_SOURCES, (list, tuple)) else [DATA_SOURCES])
]

EFFECTIVE_CONFIG_DIGEST = effective_config_digest(
    config, ADVANCED_SETTINGS, CONFIG_PROJECTION
)
CONFIGURATION_INPUTS_DIGEST = configuration_inputs_digest(
    EFFECTIVE_CONFIG_DIGEST,
    toolbox_identity(),
    environment_file_hashes(),
    referenced_inputs_for_digest(CONFIG_REFERENCES),
)

# --- The one project region artifact (ADR 0006) -------------------------------
# Splatted into rule 0.02 below, byte-identical to 1.02 / 2.02 / 3.03 except
# message/log/benchmark. tests/test_region_rule.py parses every workflow and
# fails on ANY other difference.
REGION = region_rule(
    project_dir=project_dir,
    model_region=model_region,
    data_sources=DATA_SOURCES,
    hydrography=basin_hydrography,
    basin_index=basin_index,
)

# --- The shared vector foundation (ADR 0006 §8) -------------------------------
# `parse_spatial_config(basin_cfg)` WITHOUT `my_cfg`: §8b requires the shared
# rule's params to be a pure function of `project` + `shared.basin`, so a
# workflow-local key cannot feed it and the four declarations stay identical.
SPATIAL_UNITS = spatial_units_rule(
    project_dir=project_dir,
    spatial_config=parse_spatial_config(basin_cfg),
    data_sources=DATA_SOURCES,
)

# --- One climate store per candidate source -----------------------------------
# Built from the SAME factory the other three workflows splat, once per source.
# For `shared.clim_historical` the resulting spec is identical to the one WF1's
# rule 1.04 declares -- same script, inputs, params and outputs -- so the store
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
    )
    for source in CANDIDATE_SOURCES
}

# --- rule numbering -----------------------------------------------------------
# `W.NN` = the rule's position in this workflow's LOGICAL order. `W` is a
# workflow ID, not a position (dev/reference/naming.md §9), and IDs need not
# start at 1: this workflow is 0 because it precedes model creation, so
# `ls logs/` sorts wf0, wf1, wf2, wf3 in execution order without renumbering
# the three that already exist.
#
# 0.07-0.09 ARE RESERVED, not missing: the station-sampling, observation
# comparison and Budyko rules land under them in this same milestone. A gap that
# closes within one landing is cheaper than renumbering the gathers afterwards,
# which naming.md calls a migration rather than an edit. 0.06 was part of that
# band until 2026-08-17, when the cross-source comparison took it.
#
# DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix (0.04b).

# --- log layout ---------------------------------------------------------------
# Every logging rule writes a PART under logs/_parts/, and rule 0.11 merges the
# parts into ONE logs/wf0_analyze_climate.log, then deletes them.
#
# LOG_RULES is the merge order: rule LABELS, not part paths, and for a fan-out
# rule the label is SINGULAR. Rules 0.04 and 0.05 are generated once per
# candidate source, so each writes into a DIRECTORY named for the label
# (`0.04_extract_historical_climate/<source>.log`) and merge_logs lists that
# directory to find its members -- the fan-out width lives only in the rule that
# owns it. Same shape as WF3's `3.15_run_wflow`, whose rule identifiers are
# `run_wflow_batch_<b>`.
#
# tests/test_log_rules_contract.py asserts this list in BOTH directions and in
# rule-number order, so an added logging rule must be registered here.
WORKFLOW_LOG_NAME = "wf0_analyze_climate.log"
LOG_PARTS_DIR = f"{project_dir}/logs/_parts"

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
LOG_RULES = [
    "0.02_delineate_region",
    "0.03_delineate_spatial_units",
    "0.04_extract_historical_climate",
    "0.04b_derive_climate_levels",
    "0.05_plot_climate_source",
]

# Rule 0.06 exists only when there is more than one source to compare, so its
# label is APPENDED rather than written into the literal above. Both halves are
# load-bearing: tests/test_log_rules_contract.py reads the literal textually and
# parses the workflow on the single-source fixture config, so a label sitting in
# the literal would be an orphan there. tests/test_compare_climate_sources.py
# covers the other direction, which that module's fixture cannot reach.
if len(CANDIDATE_SOURCES) > 1:
    LOG_RULES.append("0.06_compare_climate_sources")


def source_plot_dir(source):
    """Where one source's canonical climate figures land."""
    return f"{CLIMATE_STORES[source].store_dir}/plots"


def source_subbasin_dir(source):
    """The per-subbasin bin inside that source's plot directory."""
    return f"{source_plot_dir(source)}/{SUBBASIN_PLOT_DIRNAME}"


def source_figures(source):
    """The DECLARED figure filenames for one source, under the WF0 grammar."""
    return source_figure_names(
        source,
        variables=source_climate_vars(source),
        spatial_scopes=DECLARED_SPATIAL_SCOPES,
    )


# --- the cross-source comparison (rule 0.06) ----------------------------------
# ONE directory beside the per-source stores, not inside any of them: it belongs
# to no single source, and `data/climate/historical/` is already the root this
# workflow tokenizes its paths against (see declare_path_tokens above).
#
# DECLARED ONLY FOR A MULTI-SOURCE RUN. With no `candidate_sources` this
# workflow draws exactly what WF1 already draws and adds no cost -- the property
# AGENTS.md names as the reason wf0 is safe to run on any project -- and a
# "comparison" of one dataset would break it for a table with one row. The
# outputs come from `compare_sources.comparison_outputs`, so the module that
# writes the files is the one that names them, and the figure set narrows itself
# to the variables more than one source carries.
COMPARISON_DIR = f"{project_dir}/data/climate/historical/comparison"
COMPARISON_SUBBASIN_DIR = f"{COMPARISON_DIR}/{SUBBASIN_PLOT_DIRNAME}"
COMPARISON_OUTPUTS = (
    [f"{COMPARISON_DIR}/{name}" for name in comparison_outputs(CANDIDATE_SOURCES)]
    if len(CANDIDATE_SOURCES) > 1
    else []
)

# WF0_TERMINALS — the artifacts with no WF0 consumer. Every producing rule is
# upstream of them and none feeds another rule, which is exactly the input set
# the two gather rules need and what schedules each of them LAST.
#
# ONE representative figure per source is enough: rule 0.05 writes its whole set
# as a single job, so requesting one schedules the rest. The map is the
# representative because every source draws one, precipitation-only included.
WF0_TERMINALS = [
    *[f"{source_plot_dir(s)}/{source_figures(s)[0]}" for s in CANDIDATE_SOURCES],
    # Empty on a single-source run, where rule 0.06 is not declared either.
    *COMPARISON_OUTPUTS,
    # The vector foundation is a LEAF here -- nothing in this workflow consumes
    # `basins.geojson` downstream of the figures -- so without this edge rule
    # 0.03 could run in parallel with the merge and strand its log part under
    # `_parts/`. That is the defect WF1, WF2 and WF3 each recorded in turn.
    SPATIAL_UNITS.outputs["basins"],
]

WF0_TARGETS = [
    *WF0_TERMINALS,
    f"{project_dir}/config/runs/snake_config_analyze_climate.yml",
    f"{project_dir}/logs/{WORKFLOW_LOG_NAME}",
    f"{project_dir}/benchmarks/wf0_benchmarks.md",
]

# 0.00  all — target aggregator: the canonical climate figure set per source
rule all:
    message: target_banner("0.00", "all", WF0_TARGETS, project_dir)
    input:
        WF0_TARGETS,

# 0.01  snapshot_config — the current config copy + the run record
rule snapshot_config:
    message: rule_banner("0.01", "snapshot_config")
    input:
        config_snake = config_path,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "analyze_climate",
        config_dir = f"{project_dir}/config",
        effective_config = config,
        advanced_settings = ADVANCED_SETTINGS,
        config_projection = CONFIG_PROJECTION,
        # A string digest, so the params trigger compares a value rather than a
        # structure. This is what keeps the record FRESH when the checkout, the
        # lock files or a referenced catalog's bytes move.
        configuration_inputs_sha256 = CONFIGURATION_INPUTS_DIGEST,
    output:
        config_snake_out = f"{project_dir}/config/runs/snake_config_analyze_climate.yml",
        run_record = RUN_RECORD,
    script:
        "blueearth_cst/model/copy_config_files.py"

# 0.02  delineate_region — the one project region artifact (ADR 0006).
# Byte-identical to 1.02, 2.02 and 3.03 except message/log/benchmark; everything
# else is splatted from REGION so the four cannot drift.
rule delineate_region:
    message: rule_banner("0.02", "delineate_region")
    input:
        **REGION.inputs,
    params:
        **REGION.params,
    output:
        **REGION.outputs,
    log:
        f"{LOG_PARTS_DIR}/0.02_delineate_region.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/0.02_delineate_region.tsv",
    script: REGION.script

# 0.03  delineate_spatial_units — the shared vector foundation (ADR 0006 §8).
# Byte-identical to 1.03, 2.03 and 3.04 except message/log/benchmark.
#
# The VECTOR half only, as in WF2: the raster half (rule 1.06) stays WF1-only,
# so a climate-only run obtains basin and subbasin boundaries without reading
# `vito`, `modis_lai` or `soilgrids` at all.
rule delineate_spatial_units:
    message: rule_banner("0.03", "delineate_spatial_units")
    input:
        **SPATIAL_UNITS.inputs,
    params:
        **SPATIAL_UNITS.params,
    output:
        **SPATIAL_UNITS.outputs,
    log:
        f"{LOG_PARTS_DIR}/0.03_delineate_spatial_units.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/0.03_delineate_spatial_units.tsv",
    script: SPATIAL_UNITS.script


# 0.04  extract_historical_climate — ONE rule per candidate source.
# 0.05  plot_climate_source        — the canonical figure set for that source.
#
# GENERATED IN A LOOP RATHER THAN WILDCARDED, and the reason is the store's
# output SET, not style: `climate_store_rule` returns an `oro_nc` output for
# chirps and none for era5, and a Snakemake rule has a fixed output set, so one
# wildcard rule cannot cover both families. Wildcards would force a source
# taxonomy into this file that the factory already knows. Generating a concrete
# rule per source takes the shape from the spec instead, so adding a source is a
# config edit and nothing here changes.
#
# Not a new mechanism: WF3 declares its batch fan-out the same way
# (`run_stress_test.smk`, `run_wflow_batch_<b>`).
#
# For `shared.clim_historical` the generated 0.04 is the shared producer
# contract with a different rule NAME -- same script, inputs, params, outputs.
# tests/test_climate_store_contract.py pins that equivalence rather than
# byte-identity of the declaration, which is the honest form of the claim.
# --- one plotting scale per variable, shared by every source ------------------
# 0.04b derive_climate_levels — pool what each figure would plot, across every
# candidate, and write the boundaries all of them then draw against.
#
# WITHOUT this, each 0.05 job classifies its own figure from its own data, so
# two sources' precipitation maps carry two different colour ramps and a
# difference between them cannot be read off the figures — which is the one
# question this workflow exists to answer.
#
# The price is a BARRIER: no figure renders until every extraction has finished.
# Accepted deliberately (owner ruling 2026-08-16); the alternative that avoids
# it is a single plotting job over all sources, which trades the barrier for
# losing per-source parallelism and re-rendering every source when one changes.
#
# NOT the sidecar retired earlier the same day. That one shared a scale between
# the SOURCE and FORCING families and was retired because they frame different
# footprints; these are two source extractions over the same bbox. See
# climate_levels.py's module docstring.
CLIMATE_LEVELS = f"{project_dir}/data/climate/historical/climate_levels.json"

# Only variables at least two sources carry can be pooled meaningfully, but the
# module handles that itself — it pools each variable over the stores that carry
# it, and omits what nothing supplies.
LEVEL_VARS = sorted({var for s in CANDIDATE_SOURCES for var in source_climate_vars(s)})

rule derive_climate_levels:
    message: rule_banner("0.04b", "derive_climate_levels", summary="one plotting scale per variable, across sources")
    input:
        climate_ncs = [CLIMATE_STORES[s].outputs["climate_nc"] for s in CANDIDATE_SOURCES],
        # The SERIES scales are pooled over the basin's cells, because that is
        # what rule 0.05 draws. Pooling them over the buffered extraction and
        # applying the result to a basin mean would classify one quantity and
        # plot another -- the defect this module exists to prevent. The `map`
        # scale stays pooled over the full field, which is what a map shows.
        basin_cells = [CLIMATE_STORES[s].outputs["basin_cells"] for s in CANDIDATE_SOURCES],
    params:
        sources = CANDIDATE_SOURCES,
        variables = LEVEL_VARS,
        water_year_start = WATER_YEAR_START,
    output:
        levels = CLIMATE_LEVELS,
    log:
        f"{LOG_PARTS_DIR}/0.04b_derive_climate_levels.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/0.04b_derive_climate_levels.tsv",
    script:
        "blueearth_cst/climate_analysis/climate_levels.py"


for _source in CANDIDATE_SOURCES:
    _spec = CLIMATE_STORES[_source]

    rule:
        name: f"extract_historical_climate_{_source}"
        message: rule_banner("0.04", f"extract_historical_climate_{_source}", summary="clip global climate to the basin")
        input:
            **_spec.inputs,
        params:
            **_spec.params,
        output:
            **_spec.outputs,
        log:
            f"{LOG_PARTS_DIR}/0.04_extract_historical_climate/{_source}.log",
        benchmark:
            f"{project_dir}/benchmarks/_parts/0.04_extract_historical_climate/{_source}.tsv",
        script:
            _spec.script

    # The vector layers the source maps are drawn over come from rule 0.03's
    # shared foundation, NOT from any model's staticgeoms -- the source grid is
    # the climate BEFORE a model exists, and making these figures wait on one
    # would invert that. Declared as real inputs so the edge is in the DAG.
    #
    # The data catalog rides in `params`, not `input`: the era5 branch resolves
    # `era5_orography` through it, but the store's freshness boundary is 0.04's
    # catalog edge (ext2-01), and duplicating it here would re-plot on every
    # catalog touch without the extraction having changed.
    # What this source can honestly be drawn for. A precipitation-only source
    # gets the precip figures and nothing else -- its temperature and PET fields
    # in the store are era5's, borrowed so the model can be forced, and drawing
    # them here would answer the comparison this workflow exists for with a
    # panel that cannot differ.
    _plot_vars = source_climate_vars(_source)

    _plot_inputs = {"climate_nc": _spec.outputs["climate_nc"]}
    # The orography sidecar feeds the lapse correction behind the TEMPERATURE
    # and PET figures, so it is an input only where those are drawn. The store
    # still produces it either way -- rule 3.08 and the forcing catalog read it.
    if "oro_nc" in _spec.outputs and _plot_vars != ("precip",):
        _plot_inputs["oro_nc"] = _spec.outputs["oro_nc"]
    # The shared scale (rule 0.04b). A real input, so the DAG carries the barrier
    # rather than the script reading a file behind Snakemake's back.
    _plot_inputs["levels_json"] = CLIMATE_LEVELS
    # The cells the basin touches, on THIS source's grid. It is what the
    # `basin_avg` figures reduce over, and it is rule 0.04's own output -- the
    # same file weathergenr averages over, so the figures, the generator and the
    # stress test share one definition of the basin. Without it the series were
    # a mean over the store's BUFFERED bbox, which is a different area per grid.
    _plot_inputs["basin_cells"] = _spec.outputs["basin_cells"]
    _plot_inputs.update(
        {name: SPATIAL_UNITS.outputs[name] for name in
         ("basins", "subbasins", "rivers", "locations")}
    )

    rule:
        name: f"plot_climate_source_{_source}"
        message: rule_banner("0.05", f"plot_climate_source_{_source}")
        input:
            **_plot_inputs,
        output:
            # The basin-level set, declared file by file (O-24) ...
            [
                f"{source_plot_dir(_source)}/{name}"
                for name in source_figures(_source)
            ],
            # ... and the per-subbasin set as a DIRECTORY, because its members
            # are named for delineation ids this file cannot know at parse time.
            directory(source_subbasin_dir(_source)),
        params:
            plot_dir = source_plot_dir(_source),
            subbasin_plot_dir = source_subbasin_dir(_source),
            data_sources = DATA_SOURCES,
            clim_source = _source,
            geoms_dir = SPATIAL_UNITS.spatial_dir + "/geoms",
            water_year_start = WATER_YEAR_START,
        log:
            f"{LOG_PARTS_DIR}/0.05_plot_climate_source/{_source}.log",
        benchmark:
            f"{project_dir}/benchmarks/_parts/0.05_plot_climate_source/{_source}.tsv",
        script: "blueearth_cst/climate_analysis/plot_climate_source.py"


# 0.06  compare_climate_sources — every candidate on ONE axis, plus the table.
#
# Rules 0.04b and 0.05 already make the per-source figure sets comparable, by
# pinning them to a shared scale. This is the step that stops asking the reader
# to do the comparing: one annual figure and one monthly figure per variable
# with every source drawn on it, and a summary table saying what each source is
# (resolution, extracted window, reference) and what it delivers.
#
# NOT an input: `climate_levels.json`. The shared scale exists so SEPARATE
# figures can be read against each other; every figure here already carries
# every source on one axis, so the edge would buy nothing and would re-fire this
# rule whenever the scale moved. The stores are the only data this needs.
#
# One job, not a fan-out -- so its log part is a flat file, matching the label
# appended to LOG_RULES above.
if len(CANDIDATE_SOURCES) > 1:

    rule compare_climate_sources:
        message: rule_banner("0.06", "compare_climate_sources", summary="compare the candidate climate datasets")
        input:
            climate_ncs = [CLIMATE_STORES[s].outputs["climate_nc"] for s in CANDIDATE_SOURCES],
            # The DOMAIN the figures average over: the cells each source's own
            # grid contributes to the basin. Without it the means are taken over
            # each store's buffered bbox, and the buffer is counted in CELLS --
            # so a coarse grid reaches physically further out and the comparison
            # is partly of neighbouring climate. Rule 0.04 already writes this,
            # and weathergenr already averages over it.
            basin_cells = [CLIMATE_STORES[s].outputs["basin_cells"] for s in CANDIDATE_SOURCES],
            # The subbasin polygons the per-subbasin comparison figures reduce
            # over -- rule 0.03's shared foundation, the same layer rule 0.05
            # takes its subbasin set from, so both families cover the same
            # areas under the same ids.
            subbasins = SPATIAL_UNITS.outputs["subbasins"],
        params:
            sources = CANDIDATE_SOURCES,
            out_dir = COMPARISON_DIR,
            subbasin_plot_dir = COMPARISON_SUBBASIN_DIR,
            geoms_dir = SPATIAL_UNITS.spatial_dir + "/geoms",
            water_year_start = WATER_YEAR_START,
            # Read ONLY to fill provenance a store did not keep: the chirps
            # branch fetches a single variable and loses the entry's metadata,
            # so without this the Reference/Version/DOI columns are blank for
            # exactly the precipitation-only sources a comparison judges.
            data_sources = DATA_SOURCES,
        output:
            COMPARISON_OUTPUTS,
            # As in rule 0.05: the per-subbasin figures are named for
            # delineation ids, so their count is a runtime fact.
            directory(COMPARISON_SUBBASIN_DIR),
        log:
            f"{LOG_PARTS_DIR}/0.06_compare_climate_sources.log",
        benchmark:
            f"{project_dir}/benchmarks/_parts/0.06_compare_climate_sources.tsv",
        script:
            "blueearth_cst/climate_analysis/compare_sources.py"


# --- benchmark gather ---------------------------------------------------------
# 0.10  gather_benchmarks — merge the WF0 parts into one benchmarks table.
rule gather_benchmarks:
    message: rule_banner("0.10", "gather_benchmarks")
    input:
        WF0_TERMINALS,
    output:
        f"{project_dir}/benchmarks/wf0_benchmarks.md",
    params:
        parts_dir = f"{project_dir}/benchmarks/_parts",
        workflow_num = 0,
    script: "blueearth_cst/shared/merge_benchmarks.py"

# --- log gather ---------------------------------------------------------------
# 0.11  gather_logs — merge every WF0 log part into ONE workflow log.
#
# Same rule as WF1's 1.17, WF2's 2.09 and WF3's 3.18, against the same script;
# only the label list, the parts dir and the output name differ. `input:` is
# WF0_TERMINALS, which is what schedules it LAST. The parts stay in `params:` --
# they are `log:` files, which Snakemake does not track in the DAG, so naming
# them as `input:` would demand them as buildable targets.
rule gather_logs:
    message: rule_banner("0.11", "gather_logs")
    input:
        WF0_TERMINALS,
    output:
        f"{project_dir}/logs/{WORKFLOW_LOG_NAME}",
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
        sys.stderr.write(
            "\n" + run_header("wf0 analyze_climate", project_dir, config_path) + "\n\n"
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
