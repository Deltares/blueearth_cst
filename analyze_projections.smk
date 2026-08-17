import functools
import itertools
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

import yaml
from snakemake.exceptions import WorkflowError

# Shared helpers live in blueearth_cst/; make them importable regardless of the working
# directory by prepending this Snakefile's own directory to sys.path.
# See dev/milestones/r03/model-builder-design.md §3.
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.provenance import append_journal_line, configuration_inputs_digest, effective_config_digest, environment_file_hashes, file_sha256, journal_event, referenced_inputs_for_digest, toolbox_identity
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, catalog_root, declare_path_tokens, DEFAULT_BASIN_INDEX, DEFAULT_HYDROGRAPHY, get_config, patch_psutil_windows_benchmark, region_rule, resolve_water_year_start, rule_banner, run_summary, spatial_units_rule, target_banner, warn_if_project_dir_in_repo, install_console_style, run_header
from blueearth_cst.spatial.config import parse_spatial_config
from blueearth_cst.projections.gridded_outputs import RemovedGriddedOutputsError, validate_removed_gridded_options
# The figure family's CONTRACT only. `projection_figures` is deliberately
# matplotlib-free so a WF2 parse -- including the four dry-runs test_cli.py
# performs -- stays as cheap as it is today; the drawing lives in
# `projection_plots` and is imported by the producers, never here.
from blueearth_cst.projections import projection_figures

# Windows: make Snakemake's benchmark memory/IO/CPU metrics work (else all NA).
patch_psutil_windows_benchmark()

# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention.
config_path = workflow.configfiles[0]

# R01 schema
project_cfg = config["project"]
shared_cfg = config["shared"]
my_cfg = config["workflows"]["analyze_projections"]

project_dir = get_config(project_cfg, "project_dir", optional=False)
# O-22: make the two-tier project_dir rule mechanical rather than
# documentary. Warns, never raises; test_case/ is the one exemption.
warn_if_project_dir_in_repo(project_dir, workflow.basedir)
DATA_SOURCES = get_config(project_cfg, "data_sources_climate", optional=False)


# The content-addressed config bundle was removed here (config-snapshot
# redesign, 2026-08-13) -- no readers, and a digest over the whole config, so
# any other workflow's edit minted a fresh directory. `run_record.yml` replaces
# it: current-only and one per workflow.
RUN_RECORD = f"{project_dir}/config/runs/analyze_projections/run_record.yml"

# The consumed-key PROJECTION -- the config paths this workflow actually reads.
# Digesting it rather than the whole file is what stops a WF1- or WF3-only edit
# from re-firing WF2's record.
CONFIG_PROJECTION = ("project", "shared", "workflows.analyze_projections")

CONFIG_REFERENCES = [
    ("data_catalog", source) for source in
    (DATA_SOURCES if isinstance(DATA_SOURCES, (list, tuple)) else [DATA_SOURCES])
]

EFFECTIVE_CONFIG_DIGEST = effective_config_digest(
    config, ADVANCED_SETTINGS, CONFIG_PROJECTION
)
# Threaded through rule 2.01's params: so the record re-fires when the
# checkout, the lock files, or a referenced catalog's bytes move -- not only
# when the config is edited.
CONFIGURATION_INPUTS_DIGEST = configuration_inputs_digest(
    EFFECTIVE_CONFIG_DIGEST,
    toolbox_identity(),
    environment_file_hashes(),
    referenced_inputs_for_digest(CONFIG_REFERENCES),
)

# The region delineation reads project.data_sources (deltares), NOT the CMIP6
# catalog above: delineating a basin is a hydrography read, not a projections
# read. WF2 therefore reads BOTH catalogs — a divergence from the pre-v2.0
# workflow, which read only data_sources_climate. The name is historical: this
# fed the climate store until ADR 0003 replaced it with the region rule.
STORE_DATA_SOURCES = get_config(project_cfg, "data_sources", optional=False)

# Shared — the model-free basin delineation the climate store extracts against.
basin_cfg = shared_cfg["basin"]
model_region = get_config(basin_cfg, "region", optional=False)
basin_hydrography = get_config(basin_cfg, "hydrography", DEFAULT_HYDROGRAPHY)
basin_index = get_config(basin_cfg, "basin_index", DEFAULT_BASIN_INDEX)
historical_window = get_config(shared_cfg, "historical_window", optional=False)
# `shared.clim_historical` is deliberately NOT read here. WF2 has no climate
# store and no rule that consumes the observed source: it read the key
# `optional=False` and never used the value, so a config omitting it failed WF2
# for a setting WF2 does not have. Removed 2026-08-13 (defect E). The key stays
# required where it is actually used -- WF1 and the shared climate-store
# producer -- so this loosens nothing that matters.

clim_project = get_config(my_cfg, "clim_project", optional=False)
models = get_config(my_cfg, "models", optional=False)
scenarios = get_config(my_cfg, "scenarios", optional=False)
members = get_config(my_cfg, "members", optional=False)
# Step 5e-iii / §5.5: `variables` is a MAPPING declaring each variable's canonical
# quantity and change semantics. The bare list it replaces made stage B infer
# `relative` from the literal name "precip", so any other relative variable was
# silently differenced as though it were a temperature. Nothing infers anything
# from a name now.
from blueearth_cst.projections import variable_spec as _vs

VARIABLE_SPEC = _vs.parse(get_config(my_cfg, "variables", optional=False))
# The post-rename source names, for the catalog read. Sorted, because this list
# reaches the digest components.
variables = _vs.source_names(VARIABLE_SPEC)

# Water-year start, from the SHARED key so WF2, WF3 and the climate figures
# cannot disagree about what an annual value is.
#
# The old `workflows.analyze_projections.start_month_hyd_year` is REFUSED, not
# quietly honoured. It never reached the arithmetic: the Snakefile read it and
# passed it to rule 2.06, and derive_change_factors.py never read the param, so
# every change factor has always been computed Jan-Dec whatever the key said.
# Honouring it now would silently CHANGE results for any config carrying a
# non-Jan value -- the config would start meaning what it always claimed, which
# is right, but it must be a decision the owner makes rather than one a version
# bump makes for them.
_legacy_hyd = my_cfg.get("start_month_hyd_year")
if _legacy_hyd is not None:
    raise ValueError(
        "workflows.analyze_projections.start_month_hyd_year has moved to "
        f"shared.water_year_start (it is now honoured by WF2, WF3 and the "
        f"climate figures alike). Remove it and set:\n\n"
        f"    shared:\n      water_year_start: {_legacy_hyd}\n\n"
        "It never reached the change-factor arithmetic before, which always "
        "used Jan, so a non-Jan value will move every change factor once."
    )
water_year_start = resolve_water_year_start(get_config(shared_cfg, "water_year_start"))
time_horizon_hist = get_config(my_cfg, "historical_year_range", optional=False)
future_horizons = get_config(my_cfg, "future_horizons", optional=False)

# --- step 5e / D1: clip the reference window, never splice --------------------
# The change-factor reference is the GCM historical experiment, which ends
# 2014-12-31. A window reaching past that is CLIPPED; 2015+ exists only under the
# per-scenario ScenarioMIP entries and stitching them on would silently mix two
# experiments inside one reference (N8).
#
# Warnings are emitted PER CONDITION, at DAG build, on the principle that a signal
# firing on every run is a signal nobody reads. The alignment difference is
# deliberately silent by default -- the seed config's [1990, 2010] differs from
# shared.historical_window 2000-2020 on 100% of runs -- and is promoted only when
# the clip is what broke an alignment the config had asked for.
from blueearth_cst.projections import reference_window as _rw

REFERENCE_WINDOW = _rw.clip_reference_window(time_horizon_hist)
_SHARED_WINDOW = [historical_window["starttime"], historical_window["endtime"]]
for _line in _rw.window_warnings(REFERENCE_WINDOW, shared_window=_SHARED_WINDOW):
    print(f"WARNING analyze_projections: {_line}", file=sys.stderr)

# The durable record. D1 names provenance.json and report.md as its homes; neither
# exists yet (6a and 7 respectively), so stage B logs it and 6a moves it. Recorded
# rather than dropped because silence and absence are different: an alignment
# difference that does not warrant stderr must still be recoverable afterwards.
REFERENCE_RECORD = _rw.alignment_record(REFERENCE_WINDOW, shared_window=_SHARED_WINDOW)

# Everything downstream slices with the EFFECTIVE window. On the seed this is the
# requested window unchanged, which is why 5e is output-neutral here.
time_horizon_hist = list(REFERENCE_WINDOW.effective)

# S8-08(c) (owner ruling, 2026-07-31): the gridded branch is GONE, and with it
# both spellings of its config key.
#
# The option promised gridded products but `raw/{series_key}.nc` already IS the
# basin slice on the source grid -- and for an `Amon` source the monthly resample
# between them is the identity, so `grids/series/` would have been a near-copy of
# a file every run already writes. `grids/change/` was the only genuinely new
# artifact, and it was never declared to Snakemake by any rule.
#
# A config that asks for it RAISES rather than being ignored: silently dropping a
# `true` would hand back the `false` behaviour with no signal, which is the exact
# failure the 5e rename was written to avoid. `false` is accepted with a warning,
# because it requests precisely what the workflow now always does -- there is no
# point breaking every shipped config over a key that agreed with us.
try:
    _gridded_warnings = validate_removed_gridded_options(my_cfg)
except RemovedGriddedOutputsError as _error:
    raise WorkflowError(str(_error)) from None
for _warning in _gridded_warnings:
    print(_warning, file=sys.stderr)

# Step 5d: the statistic set. OPTIONAL and unset in every shipped config, so the
# config snapshot -- one of the 15 manifest targets, fingerprinted by sha256 --
# stays byte-identical and 5d does not touch a config contract. Adding and
# renaming config keys is 5e's job; batching the two would make neither
# attributable.
#
# `None` means the v2.0 default (mean, median, std). A config that opts into tail
# quantiles gets them labelled with the effective sample size, because `q_90` over
# a ~20-year window is the second-highest of 20 values and should say so.
stats = get_config(my_cfg, "stats", None, optional=True)

# --- step 6b / A2: the dry-month rule ----------------------------------------
# Resolved HERE, at DAG build, so a `change: relative` variable with no threshold
# fails before any job is scheduled -- same argument as 5e-i's save_grids. The
# shipped default (precip: 0.1 mm/day) lives in code; a variable outside the
# shipped set must supply its own, because borrowing precipitation's number would
# apply a rainfall threshold to an unrelated quantity in unrelated units.
from blueearth_cst.projections import dry_month as _dm

_relative_cfg = get_config(my_cfg, "relative_change", {}, optional=True) or {}
MIN_REFERENCE = _dm.resolve_thresholds(
    VARIABLE_SPEC, _relative_cfg.get("min_reference")
)
MAX_FLAGGED_MONTHS = _relative_cfg.get(
    "max_flagged_months", _dm.DEFAULT_MAX_FLAGGED_MONTHS
)

# R9 P2 commit 2: the projections overlay moves under `data/climate/`. ONE
# binding carries every WF2 output below it, and the two cache tiers keep their
# names -- `raw/` (as-fetched GCM slices) and `scalar/` (spatially reduced,
# S8-03) -- so `prune_series_cache.py`, which is keyed to that grammar, keeps
# working once repointed at the new root.
clim_project_dir = f"{project_dir}/data/climate/projections/{clim_project}"

# The figure set, derived ONCE from the contract module and never re-spelled.
# Every declaration below reads from these bindings: rule all's targets, rule
# 2.06's cloud output and its `figure_names` promise, rule 2.07's outputs, and
# the two gather rules' edges. The defect this replaces is eight figures written
# where three were declared -- five invisible to Snakemake, so not cleaned on
# failure, not remade when deleted, and unusable as a dependency.
WF2_FIGURE_RELATIVE_PATHS = projection_figures.figure_relative_paths(future_horizons)
WF2_FIGURE_PATHS = {
    relative: f"{clim_project_dir}/plots/{relative}"
    for relative in WF2_FIGURE_RELATIVE_PATHS
}
ANNUAL_PRECIPITATION_PLOT = WF2_FIGURE_PATHS["overview/annual-precipitation.png"]
ANNUAL_TEMPERATURE_PLOT = WF2_FIGURE_PATHS["overview/annual-temperature.png"]
# Rule 2.06 draws the cloud(s) from the stage-B merge; rule 2.07 draws the rest.
CHANGE_FACTOR_CLOUD_PLOTS = [
    WF2_FIGURE_PATHS[relative]
    for relative in WF2_FIGURE_RELATIVE_PATHS
    if relative.startswith("overview/change-factor-cloud")
]
CHANGE_FACTOR_CLOUD_PLOT = WF2_FIGURE_PATHS[projection_figures.CLOUD_FACETED_PATH]
MONTHLY_CHANGE_FACTOR_PLOTS = [
    WF2_FIGURE_PATHS[relative]
    for relative in WF2_FIGURE_RELATIVE_PATHS
    if relative.startswith("windows/")
]

# ONE producer contract, built here and identically in build_model.smk
# and run_stress_test.smk, and splatted into rule 1.02 / 2.02 / 3.03.
# Everything content- or execution-determining comes from this object; only
# message/log/benchmark are workflow-local. tests/test_region_rule.py parses all
# three workflows and fails on ANY other difference.
#
# WF2 needs the model-free delineated polygon and nothing else — that is what
# freed it from wf1's hydrology_model/staticgeoms/region.geojson (design D2/A1,
# goal G2). Until ADR 0003 the only model-free source was the climate-store
# producer, so WF2 declared the whole extraction to obtain a 3 kB outline and
# the design recorded that as "the accepted, stated price of A1". The region is
# now its own artifact, so WF2 declares this rule and no store: a
# projections-only run does no climate extraction at all.
REGION = region_rule(
    project_dir=project_dir,
    model_region=model_region,
    data_sources=STORE_DATA_SOURCES,
    hydrography=basin_hydrography,
    basin_index=basin_index,
)
region_path = REGION.region_geojson

# --- The shared vector foundation (ADR 0003 §8) -------------------------------
# Same pattern once more: ONE producer contract, built here and identically in
# the other two workflows, splatted into rule 2.03 below, and
# tests/test_spatial_units_rule.py parses all three and fails on ANY difference
# beyond message/log/benchmark.
#
# WF2 gains basin and subbasin boundaries — a context map beside the
# change-factor plots, and the option of subbasin-resolved indicators — WITHOUT
# a built model and without the thematic raster stack. Declaring the UNSPLIT
# rule 1.06 instead would have made a projections-only run resample `vito`,
# `modis_lai` and `soilgrids` to draw a subbasin outline; measured 2026-08-06,
# the split avoids ~71% of what that would add. `snakemake -n` on this file must
# therefore list `delineate_spatial_units` and no job whose inputs mention those
# three sources — that dry-run is §8's acceptance assertion, not this comment.
#
# WF2 does not yet CONSUME the layers (§10 leaves the consuming rules
# deliberately unnamed); making them reachable is this decision, using them is
# the next one.
#
# `parse_spatial_config(basin_cfg)` takes NO model section: §8b requires the
# shared rule's params to be a pure function of `project` + `shared.basin`, and
# this workflow has no `workflows.build_model` to offer anyway — which is
# exactly why the asymmetry would have been fatal.
SPATIAL_UNITS = spatial_units_rule(
    project_dir=project_dir,
    spatial_config=parse_spatial_config(basin_cfg),
    data_sources=STORE_DATA_SOURCES,
)

# --- series identity (migration step 2b; design §5.3, D9, D12) ----------------
# The series files below stop being temp() and become a persistent product, so
# they need an identity: "was this derived from exactly the inputs the current
# config implies?" Everything knowable at PARSE time goes into the rules' params,
# where Snakemake's params rerun-trigger can see it. The polygon fingerprint is
# deliberately NOT here — on a fresh project the polygon does not exist yet, and
# a param flipping from absent to a real value on the second invocation would
# re-derive every series once for nothing (design §6.14). The reduce job folds it
# in and stamps the result on what it writes.
from blueearth_cst.projections import series_identity as _si

# The store index sits beside the catalog that generated it; both come from one
# crawl and carry an equal crawled_on (D12). Derived rather than configured: a
# separately-configurable path would let the two drift.
STORE_INDEX = str(Path(DATA_SOURCES).parent / "cmip6_store_index.json")

# The reducer's identity for the cache key (risk-03): Snakemake's code trigger
# tracks the rule's script body, not the modules it imports, so a change to the
# reduction logic would otherwise reuse every cached series.
#
# Hashed by BEHAVIOUR of the enumerated reduction functions, not by file bytes.
# File-level hashing was the first implementation and it was too blunt: at step 4c
# an error-handling-only edit -- no formula touched -- invalidated all 9 series and
# cost a full network re-derivation. `kernel_hash` ignores comments, docstrings and
# formatting while catching a changed formula, a changed constant of any type
# (INCLUDING strings -- `resample(time="MS")` -> `("YS")` is a string-only edit),
# a changed default argument, and a swapped attribute lookup.
#
# The enumeration must name every function whose ARITHMETIC matters; a change in
# an unlisted callee is invisible, exactly as an unlisted file was before.
#
# `pixi.lock` is folded in as the environment fingerprint: the reduction's numbers
# depend on xarray/pandas behaviour, which no source hash can see. Coarse on
# purpose -- any dependency change re-derives, which is the safe direction for a
# cache whose failure mode is silently wrong numbers.
from blueearth_cst.projections.get_stats_climate_proj import get_stats_clim_projections
from blueearth_cst.projections.grid_weights import (
    cell_area_weights,
    latitude_weights,
    longitude_weights,
    midpoint_edges,
    weighted_spatial_mean,
)

# Step 5a: the spatial reduction is no longer a single `.mean()` inside
# get_stats_clim_projections -- it is a weighted mean whose weights come from the
# five functions below. They MUST be enumerated here. `kernel_hash` hashes the
# behaviour of the functions it is given and follows no call graph, so a changed
# edge derivation or area formula in an unlisted callee would be invisible and
# every cached series would be silently reused across a weighting change. That is
# the exact failure the enumeration exists to prevent.
REDUCER_KERNEL = [
    get_stats_clim_projections,
    weighted_spatial_mean,
    cell_area_weights,
    latitude_weights,
    longitude_weights,
    midpoint_edges,
]
REDUCER_HASH = _si.kernel_hash(
    REDUCER_KERNEL,
    env_fingerprint=_si.file_digest(Path(workflow.basedir) / "pixi.lock"),
)


# Stage B's arithmetic identity. Snakemake's code trigger tracks a rule's SCRIPT
# body, not the modules it imports, so an edit to the change arithmetic in
# `get_change_climate_proj.py` left `derive_change_factors` reporting "Nothing to
# be done" and its outputs silently stale. Stage A has `REDUCER_HASH` for exactly
# this; stage B had nothing, on the mistaken belief that it "re-runs every
# invocation" -- it re-runs when its script, inputs or params change, and an
# imported module is none of those.
#
# Found 2026-07-30 by the hydrological-year fix: the fix was correct, its unit
# tests passed, and the workflow would not have applied it.
#
# Same enumeration discipline as REDUCER_KERNEL: name every function whose
# ARITHMETIC matters, because kernel_hash follows no call graph.
from blueearth_cst.projections.get_change_climate_proj import (
    get_change_annual_clim_proj,
    hydrological_year_bounds,
    quantile_label,
)
from blueearth_cst.projections.calendar_weights import (
    days_in_month,
    month_length_weights,
)

STAGE_B_KERNEL = [
    get_change_annual_clim_proj,
    hydrological_year_bounds,
    quantile_label,
    month_length_weights,
    days_in_month,
]
STAGE_B_HASH = _si.kernel_hash(
    STAGE_B_KERNEL,
    env_fingerprint=_si.file_digest(Path(workflow.basedir) / "pixi.lock"),
)


#: WF2's spatial buffer around the region bbox, in degrees. Previously a bare
#: `buffer = 1` inside the script; named here because it is a digest component
#: (design §5.3 item 4) and a documented sampling choice, not an accident.
REGION_BUFFER_DEGREES = 1.0

# --- series keys (migration step 3; design §5.2, §5.3) ------------------------
# One stage-A rule fans out over {series_key} instead of two rules fanning out
# over {model} and {model}×{scenario}. The key is built here and looked up by the
# rule, NEVER parsed back out of the wildcard: `/` is sanitized to `_` and CMIP6
# model names contain both `-` and `_` (`NOAA-GFDL/GFDL-ESM4`), so splitting a key
# on `_` cannot recover (model, experiment) unambiguously.
#
# The key omits the member at this step. Under the step-3 scope decision the
# `members:` config list is still looped INSIDE one job, so a single series covers
# every member and a key naming one of them would be a lie; the members it covers
# are recorded in `cst_members`. Step 4 makes the member a wildcard alongside the
# §5.7 resolution ladder that gives per-member fan-out meaning, and appends
# `_{member}` to the grammar then. Two renames of an unmanifested intermediate is
# the price of not regressing multi-member configs for a step.
# --- source resolution (migration step 4a; design §5.7, D6, D7, D12) ----------
# Which requested combinations can actually be reduced, decided HERE at DAG build
# from a plain YAML read -- no hydromt import, no network. Unresolved combinations
# never become jobs, which is what lets the dummy-empty-netCDF pattern and
# `filter_nonempty` go away (step 4c).
#
# Absence and failure are different classes (design §4 criterion 7): almost every
# non-resolution is a NORMAL skip recorded for the composition record, because
# under R3' a model publishing no ssp370, or one member where another publishes
# three, is the expected shape of a correct run. Only two conditions stop the DAG
# build: a model absent from the catalog entirely, and a run where nothing
# resolves.
from blueearth_cst.projections import resolution as _res

_CATALOG = yaml.safe_load(Path(DATA_SOURCES).read_text(encoding="utf-8"))
_INDEX = (
    json.loads(Path(STORE_INDEX).read_text(encoding="utf-8"))
    if os.path.isfile(STORE_INDEX)
    else None
)

# D12/R14: the catalog and the index must be one observation, not two crawls.
_res.assert_index_matches_catalog(_CATALOG, _INDEX)

COMBINATIONS = _res.resolve(
    _CATALOG,
    clim_project=clim_project,
    models=models,
    scenarios=scenarios,
    members=members,
)

# The one model-level error (C7): absent from the generated catalog means absent
# from the store, so this is a typo or a stale config rather than thin data.
_unknown = _res.unknown_models(COMBINATIONS)
if _unknown:
    raise WorkflowError(
        "analyze_projections: model(s) not present in "
        f"{DATA_SOURCES} under any experiment: {', '.join(_unknown)}. "
        "The catalog is generated from a live crawl of the store, so a name "
        "absent from it is absent from the store. Check for a typo, or "
        "regenerate with dev/scripts/generate_cmip6_catalog.py."
    )

# D6: replaces the `ensemble.min_sources` key, which conflated a store property
# with a configuration failure. Non-configurable -- ensemble ADEQUACY is a
# downstream judgement (S6, N10), not a threshold this workflow encodes.
if not any(c.resolved for c in COMBINATIONS):
    raise WorkflowError(
        "analyze_projections: no requested combination resolved against "
        f"{DATA_SOURCES}.\n" + _res.format_status_report(COMBINATIONS)
    )

# Normal skips are reported, never silent -- this is what replaces the run-time
# `asymmetric hist/clim members` raise D7 supersedes (design D7 table).
_skip_report = _res.format_status_report(COMBINATIONS)
if _skip_report:
    logger.warning(_skip_report)

# D8/D12: a glob matching more than one {grid}/{version} means the read is not a
# single identifiable store. Measured at ~6% of pinned stores, so this is live.
_ambiguous = _res.ambiguous_pins(_INDEX, COMBINATIONS, clim_project)
if _ambiguous:
    raise WorkflowError(
        "analyze_projections: ambiguous store pins -- the catalog's "
        "/{variable}/*/* glob matches more than one {grid_label}/{version} for:\n  "
        + "\n  ".join(_ambiguous)
        + "\nThe read would not be a single identifiable store. Pin the version "
        "in the catalog or drop the affected combination."
    )

# A3: the certified/best-effort tier difference, warned once per variable.
_rename = {}
for _key, _entry in _CATALOG.items():
    if isinstance(_entry, dict) and _entry.get("data_adapter"):
        _rename = (_entry["data_adapter"] or {}).get("rename") or {}
        break
for _var in _res.best_effort_variables(variables, _rename):
    logger.warning(
        f"analyze_projections: variable {_var!r} is BEST-EFFORT, not "
        "catalog-certified. The crawl proved only pr/tas present, so a listed "
        f"member may not publish {_var!r} -- it will fail at read time rather "
        "than skip at resolution (design §5.5, ruling A3)."
    )

# The series set is DERIVED from resolution, not from the config cross-product.
# References are distinct (model, member) pairs: a reference is reduced once
# however many scenarios share it, which is why the seed config is 6 + 3 = 9
# rather than 12.
# Step 4b: the member is now part of the key, so the fan-out is one job per
# (model, experiment, member) and the `members:` loop leaves the script. This is
# what D7's strict same-member pairing makes meaningful -- a scenario point's
# reference is the SAME member label's historical series, never a substitute,
# because the label encodes forcing variant as well as realization.
#
# References are DISTINCT (model, member) pairs: one historical series serves
# every scenario that shares it, which is why the seed config is 6 + 3 = 9.
_needed = {(c.dataset, "historical", c.member) for c in COMBINATIONS if c.resolved}
_needed |= {(c.dataset, c.scenario, c.member) for c in COMBINATIONS if c.resolved}

SERIES = {
    f"{clim_project}_{model.replace('/', '_')}_{experiment}_{member}": (
        model,
        experiment,
        member,
    )
    for model, experiment, member in sorted(_needed)
}


def series_file(model, experiment, member):
    """Path of the series for one (model, experiment, member).

    S8-03: the directory is `scalar/`, not `series/`. `scalar` is this codebase's
    own word for the quantity -- `get_stats_climate_proj.py` writes
    `var_m_scalar = weighted_spatial_mean(...)` -- and it names the axis the
    shipped artifacts already use (scalar vs `monthly_change_mean_grid-*`). The
    FILENAME is unchanged and identical to the raw slice's: the directory carries
    the tier, the filename carries the identity.
    """
    key = f"{clim_project}_{model.replace('/', '_')}_{experiment}_{member}"
    return f"{clim_project_dir}/scalar/{key}.nc"


# The resolved data points: one per (model, scenario, member). Under N10 nothing
# aggregates across them, so each is its own change-factor row (design R3').
POINTS = sorted(
    (c.dataset, c.scenario, c.member) for c in COMBINATIONS if c.resolved
)


# Keyed like SERIES, and for the same reason (step 3's lesson): the sanitized
# model name cannot be parsed back out of a wildcard, because `/` becomes `_` and
# CMIP6 model names already contain both `-` and `_`. A first attempt at step 4b
# let {model} capture `INM_INM-CM4-8` and then looked up a catalog entry under
# that name, which does not exist. Look the key up; never reconstruct from it.
POINT_KEYS = {
    f"{model.replace('/', '_')}_{scenario}_{member}": (model, scenario, member)
    for (model, scenario, member) in POINTS
}


@functools.lru_cache(maxsize=None)
def point_tier(model, scenario):
    """`certified` | `best_effort` for one data point (ruling A3).

    The weakest tier among the variables this run reads. The crawl proved only
    `pr`/`tas` present per listed member, so a config naming `kin` or `press_msl`
    gets a point that is nameable but unverified — an honest tier difference the
    composition record carries rather than silently accepting.
    """
    entry = _si.load_catalog_entry(DATA_SOURCES, _res.entry_key(clim_project, model, scenario))
    rename = (entry.get("data_adapter", {}) or {}).get("rename", {}) or {}
    return "best_effort" if _res.best_effort_variables(variables, rename) else "certified"


@functools.lru_cache(maxsize=None)
def series_digest_components(model, experiment, member):
    """Parse-time digest components for one (model, experiment) series.

    Memoized: the generated catalog is ~3 900 lines and Snakemake evaluates a
    params callable once per job per param, so an unmemoized parse would re-read
    it on every lookup. Keyed on (model, experiment) — every other input is
    module-level and fixed for the run.

    Step 4b: the member is a wildcard, so the sequence has exactly one element.
    The recipe is unchanged from step 2b -- it always took a member SEQUENCE, so
    turning the loop into a fan-out did not change the digest shape.
    """
    catalog_entry = f"{clim_project}_{model}_{experiment}_{{member}}"
    entry = _si.load_catalog_entry(DATA_SOURCES, catalog_entry)
    return _si.digest_components(
        catalog_entry=catalog_entry,
        entry=entry,
        members=[member],
        pins_by_member={member: _si.load_pins(STORE_INDEX, catalog_entry, member)},
        buffer_degrees=REGION_BUFFER_DEGREES,
        # The digest carries the SOURCE NAMES only -- what was actually fetched
        # and reduced. `canonical`, `units` and `change` are deliberately NOT in
        # the cache key: they cannot change a cached byte, they are read by stage
        # B, and stage B has no cache. Folding them in re-fetched all 9 slices
        # over the network for a `change: relative -> absolute` edit that touches
        # no stored value -- the same over-invalidation the design faulted
        # file-level hashing for at 4c.
        variable_spec=variables,
        experiment=experiment,
        reducer_module_hash=REDUCER_HASH,
    )

### Dictionary elements from the config based on wildcards
def get_horizon(wildcards):
    return config["workflows"]["analyze_projections"]["future_horizons"][wildcards.horizon]

# Rule numbering (comment headers + log/benchmark filenames) uses `W.NN` = the
# rule's position in this workflow's LOGICAL order — data, then the product,
# then figures and records. Contiguous, and every dependency points from a lower
# number to a higher one. Positional since [R10-5] (2026-08-06); this file is
# where the old scheme showed worst, defining its rules 2.00, 2.03b, 2.03, 2.01,
# 2.02, 2.04, 2.06, 2.07 — numbers out of order against the DAG.
#
# STILL A READING AID, NOT EXECUTION ORDER — stage A fans out over {series_key},
# so low-to-high says "cannot depend on", not "runs before". Definition order in
# this file is not the number order and is not required to be.
#
# DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix (2.05b) until the next
# deliberate sweep. Convention: dev/reference/naming.md §9; full map:
# dev/reference/workflows/rule-index.md § What changed.

# --- log layout ---------------------------------------------------------------
# EVERY WF2 rule that logs writes a PART under logs/_parts/, and rule 2.09 merges
# the parts into ONE logs/wf2_analyze_projections.log, then deletes them. So the
# only WF2 file left in logs/ after a full run is that merged log — the same deal
# benchmarks/wf2_benchmarks.md already gets from `gather_benchmarks`.
#
# LOG_RULES is the merge order: rule LABELS, not part paths. merge_logs lists
# each label's part dir to find its members, so the fan-out width lives in one
# place — the rule that owns it. Naming the labels rather than globbing `_parts/`
# is what keeps an orphan dir from a renamed rule out of the file (test_local
# still holds `2.02_monthly_stats_hist`, `2.03_monthly_stats_fut`,
# `2.04_monthly_change` from the pre-step-4d names, and now the whole pre-R10
# generation besides): not a label, never read.
#
# ORDER IS BY RULE NUMBER, and since [R10-5] that is also dependency order, so
# this list needed no reordering to satisfy the assertion
# tests/test_log_rules_contract.py now makes — it was already in EXECUTION
# order, and the renumber is what made the two agree. Before it this file's
# comment claimed "order is by rule number" while the list opened 2.03b, 2.01,
# 2.02; which of the two was wrong was a ruling nobody had made, and it is why
# the assertion was deferred to this sweep.
#
# Each section carries its own timestamps, so chronology stays readable inside a
# section. 2.01 snapshot_config declares no `log:`, and neither gather rule does.
#
# TWO INSTANCES OF ONE DEFECT ARE FIXED HERE. The region rule was missing from
# this list -- it is splatted into all three workflows from ONE producer
# contract, but each workflow owns its own `log:` label and its own LOG_RULES,
# so registering it is a per-file obligation the shared definition does not
# carry, and it was missed in all three files in turn. And a store label
# outlived the rule ADR 0003 §5 deleted, contributing a phantom "no part from
# this run" section to every merged WF2 log. WF2 builds no climate store,
# standalone or otherwise: it reads the region polygon from `delineate_region`.
# Both directions are now mechanically checked for all three workflows.
WORKFLOW_LOG_NAME = "wf2_analyze_projections.log"
LOG_PARTS_DIR = f"{project_dir}/logs/_parts"

# The run's key folders, stated once -- see the same block in
# build_model.smk. `data` is the DELTARES catalog's root, not the CMIP6
# one: `data_sources_climate` points at a `gs://` store, which has no local
# prefix to shorten. No `model` row -- WF2 builds none.
declare_path_tokens(
    data=catalog_root(STORE_DATA_SOURCES),
    projections=clim_project_dir,
)
LOG_RULES = [
    "2.02_delineate_region",
    "2.03_delineate_spatial_units",
    "2.04_fetch_gcm_slice",
    "2.05_reduce_gcm_series",
    "2.06_derive_change_factors",
    "2.07_plot_gcm_timeseries",
]

# 2.00  all — target aggregator: change-factor summaries + projection plots
#
# Hoisted into a dict so `message:` can print one target per line without
# restating them. A dict rather than a list because five of these were already
# NAMED inputs and the names document what each file is; `**` splats them back
# as the same keyword arguments. The three that were positional gain a name,
# which changes nothing -- no script reads `rule all`'s inputs, by name or index.
WF2_TARGETS = {
    # S8-04/05: the tidy tables are the result surface now; the three wide
    # `annual_change_scalar_stats_summary*` files they replaced are gone.
    "change_factors_annual": (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
    "change_factors_monthly": (clim_project_dir + f"/summary/{clim_project}_change_factors_monthly.csv"),
    # R5 deliverable 3 — demanded explicitly, not left as an incidental
    # by-product of derive_change_factors' other outputs.
    "composition": (clim_project_dir + "/summary/composition.csv"),
    # The whole figure set, so `rule all` demands every figure rather than the
    # three that happened to be named. `--delete-all-output` reaches them for
    # the same reason.
    **{
        f"figure_{index}": path
        for index, path in enumerate(WF2_FIGURE_PATHS.values(), start=1)
    },
    # ADR 0003 §8. The vector rule has NO WF2 consumer yet (§10), so a target
    # entry is what makes it reachable at all — an undeclared leaf is simply
    # never scheduled. One representative schedules the whole multi-output rule,
    # the same deal WF1's `spatial_catalog.yml` terminal gets. It is also a
    # gather input below: a leaf running parallel to the merge would strand its
    # log part under `_parts/`, which is the defect the LOG_RULES block above
    # documents three times over.
    "spatial_basins": SPATIAL_UNITS.outputs["basins"],
    "snake_config": f"{project_dir}/config/runs/snake_config_analyze_projections.yml",
    # ONE merged log for the whole workflow (was: one per fan-out stage,
    # alongside four rules writing logs/2.NN_*.log directly -- five files a
    # reader had to open in the right order to follow one run). Every rule
    # now logs into logs/_parts/ and rule 2.09 merges the lot.
    "workflow_log": f"{project_dir}/logs/{WORKFLOW_LOG_NAME}",
    "benchmarks": f"{project_dir}/benchmarks/wf2_benchmarks.md",
}

rule all:
    message: target_banner("2.00", "all", WF2_TARGETS.values(), project_dir)
    input:
        **WF2_TARGETS,

# The `ruleorder:` directive is GONE as of step 4d, and not by choice: it named
# `monthly_change` and `monthly_change_scalar_merge`, which 4d merges into
# `derive_change_factors`, and an unknown rule name is a parse error. A directive
# naming only `reduce_gcm_series` would be a no-op (ruleorder needs two rules to
# order).
#
# This settles a deferral rather than skipping it. AGENTS.md and
# dev/milestones/r04/climate-projections-design.md §3 recorded it as "stale insurance,
# removal deferred to a task that first encodes the ambiguity-sensitive config
# shapes as regression tests". The 2026-07 dry-run evidence stands (it constrained
# nothing on the tests fixture or a reduced config), and the merge removes the
# ambiguity it was insuring against: there is no longer a second stage-B rule that
# could claim the same output.

# 2.02  delineate_region — the one project region artifact (ADR 0003).
# Byte-identical to 1.02 and 3.03 except message/log/benchmark.
#
# This REPLACES rule 2.11 extract_climate_grid -- the name that rule answered to
# when it was deleted; WF1/WF3's surviving copy is now extract_historical_climate
# (R10). WF2 declared the whole shared
# climate-store producer to obtain the delineated polygon and never read the
# gridded extraction it also wrote (design N7); a projections-only run paid for
# a multi-decade extraction to learn a basin outline. The region is now its own
# artifact, so WF2 declares the delineation alone.
rule delineate_region:
    message: rule_banner("2.02", "delineate_region")
    input:
        **REGION.inputs,
    params:
        **REGION.params,
    output:
        **REGION.outputs,
    log:
        f"{LOG_PARTS_DIR}/2.02_delineate_region.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/2.02_delineate_region.tsv",
    script: REGION.script

# 2.03  delineate_spatial_units — the shared vector foundation (ADR 0003 §8).
# Byte-identical to 1.03 and 3.04 except message/log/benchmark.
#
# WF2 declares the VECTOR half only. The raster half (rule 1.06) stays WF1-only,
# so a projections-only run obtains basin and subbasin boundaries without
# reading `vito`, `modis_lai` or `soilgrids` at all — see the SPATIAL_UNITS
# comment block above for why that is the whole point.
rule delineate_spatial_units:
    message: rule_banner("2.03", "delineate_spatial_units")
    input:
        **SPATIAL_UNITS.inputs,
    params:
        **SPATIAL_UNITS.params,
    output:
        **SPATIAL_UNITS.outputs,
    log:
        f"{LOG_PARTS_DIR}/2.03_delineate_spatial_units.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/2.03_delineate_spatial_units.tsv",
    script: SPATIAL_UNITS.script

# 2.01  snapshot_config — current copies + immutable effective-config bundle
#
# S8-08(d): this rule and the fetch rule both answered to `2.01` until S8-08(d)
# separated them -- read as of that date; both have since moved again under
# [R10-5]. The numbers are a reference aid, so two rules answering to one is
# exactly the confusion they exist to prevent. No path moves:
# this rule names its output explicitly and has no banner-derived log or
# benchmark, so the collision only ever showed in the console message.
rule snapshot_config:
    message: rule_banner("2.01", "snapshot_config")
    input:
        config_snake = config_path,
    params:
        data_catalogs = DATA_SOURCES,
        workflow_name = "analyze_projections",
        config_dir = f"{project_dir}/config",
        effective_config = config,
        advanced_settings = ADVANCED_SETTINGS,
        config_projection = CONFIG_PROJECTION,
        # A string digest, so the params trigger compares a value. This is what
        # keeps the record fresh when the CHECKOUT moves; see its definition.
        configuration_inputs_sha256 = CONFIGURATION_INPUTS_DIGEST,
    output:
        config_snake_out = f"{project_dir}/config/runs/snake_config_analyze_projections.yml",
        run_record = RUN_RECORD,
    script:
        "blueearth_cst/model/copy_config_files.py"

# 2.04  fetch_gcm_slice — acquire ONE raw slice; the only rule that reads the store.
#
# Stage A splits into fetch -> reduce (design revision 6) because the two have
# wildly different costs and different invalidation causes. Measured 2026-07-30
# (dev/milestones/r08/2026-07-30_wf2-fetch-reduce-benchmark.md): opening one remote source
# ~1142 s, transferring its data ~19 s, reducing it ~0.2 s, raw slice 0.07 MB. So a
# reduction edit used to cost a full re-download of nine sources; now it re-reads
# local disk.
#
# THE LOAD-BEARING DETAIL: these params must NOT include the reducer hash. They
# carry `raw_digest_components` (series_identity.raw_components -- the component set
# minus reducer_module_hash), so Snakemake's params trigger cannot re-download on a
# formula edit. Passing `digest_components` here instead would silently undo the
# entire split while every test still passed.
rule fetch_gcm_slice:
    message: rule_banner("2.04", "fetch_gcm_slice", "{wildcards.series_key}", summary="download one CMIP6 slice")
    wildcard_constraints:
        series_key = "|".join(re.escape(k) for k in SERIES),
    input:
        region_path = region_path,
    output:
        # PERSISTENT + update(), for the same reason as the series below: Snakemake
        # removes outputs in Job.prepare(), so a revalidate-and-skip cache silently
        # never fires without the flag.
        raw_nc = update(clim_project_dir + "/raw/{series_key}.nc"),
    params:
        catalog_path = DATA_SOURCES,
        catalog_entry = lambda wildcards: f"{clim_project}_{SERIES[wildcards.series_key][0]}_{SERIES[wildcards.series_key][1]}_{{member}}",
        member = lambda wildcards: SERIES[wildcards.series_key][2],
        variables = variables,
        # S8-08(a): the units the STORED values are in, keyed by post-rename source
        # name. The catalog's data_adapter converts on read but leaves the `units`
        # attribute describing the pre-conversion quantity, so every array claimed
        # `kg m-2 s-1` over mm/day. Not a digest component -- it labels the values,
        # it does not change them.
        variable_units = {v.source: v.units for v in VARIABLE_SPEC.values()},
        buffer_degrees = REGION_BUFFER_DEGREES,
        acquisition_window = lambda wildcards: list(_si.acquisition_window(SERIES[wildcards.series_key][1])),
        raw_digest_components = lambda wildcards: _si.raw_components(
            series_digest_components(*SERIES[wildcards.series_key])
        ),
    log:
        LOG_PARTS_DIR + "/2.04_fetch_gcm_slice/{series_key}.log",
    benchmark:
        project_dir + "/benchmarks/_parts/2.04_fetch_gcm_slice/{series_key}.tsv",
    script: "blueearth_cst/projections/fetch_gcm_raw.py"

# 2.05  reduce_gcm_series — ONE stage-A rule over {series_key} (step 3).
# Collapses monthly_stats_hist + monthly_stats_fut: they ran the same script with
# different params, and the only structural difference was their output naming
# (`historical_stats_time_{model}` vs `stats_time-{model}_{scenario}`) -- which is
# why the collapse needed the naming unified first.
#
# The hist->fut ordering edge is gone (step 2b) and there is now no edge between
# series at all: every series is independent, so the stage fans out at full width.
# Since revision 6 it reads the local raw slice above and makes NO network call.
rule reduce_gcm_series:
    message: rule_banner("2.05", "reduce_gcm_series", "{wildcards.series_key}", summary="reduce the slice to a basin-average series")
    wildcard_constraints:
        # Anchor to the keys actually built at parse time. Without this the
        # wildcard would also match paths that merely look like keys.
        series_key = "|".join(re.escape(k) for k in SERIES),
    input:
        region_path = region_path,
        # revision 6: the reduction's input is a LOCAL slice, not the remote store.
        raw_nc = clim_project_dir + "/raw/{series_key}.nc",
    output:
        # PERSISTENT + update(): the flag is load-bearing. Snakemake's
        # Job.prepare() removes existing outputs before every job, which would
        # delete the file D9's revalidation inspects -- so without it a scheduled
        # job can never cache-hit and always re-downloads (measured; design D9
        # item 3's implementation note).
        series_nc = update(clim_project_dir + "/scalar/{series_key}.nc"),
    params:
        catalog_path = DATA_SOURCES,
        project_dir = f"{project_dir}",
        name_scenario = lambda wildcards: SERIES[wildcards.series_key][1],
        # ONE member per job now (step 4b), not the config list.
        name_members = lambda wildcards: [SERIES[wildcards.series_key][2]],
        name_model = lambda wildcards: SERIES[wildcards.series_key][0],
        name_clim_project = clim_project,
        variables = variables,
        # S8-08(a): see rule 2.04.
        variable_units = {v.source: v.units for v in VARIABLE_SPEC.values()},
        series_nc_out = lambda wildcards: f"{clim_project_dir}/scalar/{wildcards.series_key}.nc",
        digest_components = lambda wildcards: series_digest_components(*SERIES[wildcards.series_key]),
        acquisition_window = lambda wildcards: list(_si.acquisition_window(SERIES[wildcards.series_key][1])),
        store_index = STORE_INDEX,
        buffer_degrees = REGION_BUFFER_DEGREES,
    log:
        LOG_PARTS_DIR + "/2.05_reduce_gcm_series/{series_key}.log",
    benchmark:
        project_dir + "/benchmarks/_parts/2.05_reduce_gcm_series/{series_key}.tsv",
    script: "blueearth_cst/projections/get_stats_climate_proj.py"

# 2.06  derive_change_factors — stage B, ONE job (step 4d, design §5 "B. Derive")
# Replaces monthly_change (fanned out per point_key x horizon) + its aggregator
# monthly_change_scalar_merge. The design gives stage B one job with no fan-out.
# The per-point change netCDFs were temp() outputs of the fan-out; they are now
# job-internal intermediates with the same lifetime (the script uses a
# TemporaryDirectory), so nothing durable changed shape.
rule derive_change_factors:
    message: rule_banner("2.06", "derive_change_factors", summary="compare each horizon against the reference period")
    input:
        # EXPLICIT expanded list (risk-06 / revision 4), never a glob and never a
        # config cross-product: exactly the series the resolved combination set
        # names. The script asserts the set it opens equals this list, so a model
        # dropped from the config cannot rejoin through a leftover file.
        #
        # ancient() DROPPED (step 2b, design D9): with a persistent series store,
        # suppressing the mtime trigger is exactly the hole ext2-01 found -- a
        # re-derived series would not re-trigger its consumer. The in-job digest
        # assertion is the backstop that holds regardless of invocation.
        series_nc = sorted(
            {series_file(m, "historical", mem) for (m, sc, mem) in POINTS}
            | {series_file(m, sc, mem) for (m, sc, mem) in POINTS}
        ),
        # D9: stage B recomputes every expected digest INCLUDING the current
        # polygon fingerprint, so it needs the polygon itself.
        region_path = region_path,
    output:
        # S8-05: the three wide `annual_change_scalar_stats_summary*` files are
        # GONE. The tidy tables below supersede them -- same numbers, long format,
        # per-row provenance, plus the future level the wide form never carried.
        # Verified before removal: run_stress_test.smk and
        # blueearth_cst/experiment/ referenced them zero times, and rule 2.07
        # declared the `.nc` as an input it never opened. The `.nc` survives as a
        # job-internal intermediate, because the tidy reshape reads it back.
        # Both cloud views, drawn from this rule's own stage-B merge. The
        # combined one is present only for a multi-horizon config, which is what
        # `figure_relative_paths` decides -- so this list is exactly what the
        # producer writes, and never a promise it declines to keep.
        stats_change_plt = CHANGE_FACTOR_CLOUD_PLOTS,
        # R5 deliverable 3. A stage-B output, so it describes a COMPLETED run
        # (ext2-08): a failed run leaves the DAG-build stderr summary and the job
        # logs, and no composition artifact.
        composition_csv = (clim_project_dir + "/summary/composition.csv"),
        # Step 6a-i / S8-04: the tidy long-format table. Now the PRIMARY result
        # rather than an addition beside the wide files, so it moves into summary/
        # and its name says what it is when detached from the tree.
        change_factors_annual = (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
        # Step 6a-ii: change per calendar month -- the seasonal shift the
        # annual figure averages away.
        change_factors_monthly = (clim_project_dir + f"/summary/{clim_project}_change_factors_monthly.csv"),
        # Step 6a-iii: everything needed to reconstruct the run (design 5.9).
        # S8-06: beside composition.csv -- both are run-level records rather than
        # results, and report.md stays at the root as the one human entry point.
        provenance_json = (clim_project_dir + "/summary/provenance.json"),
        # Step 7-ii: the run told in the order a reader needs it, with the
        # 5.9 disclaimer block. Reads provenance.json; derives nothing.
        report_md = (clim_project_dir + "/report.md"),
    params:
        clim_project_dir = f"{clim_project_dir}",
        horizons = future_horizons,
        water_year_start = water_year_start,
        time_horizon_hist = time_horizon_hist,
        stats = stats,
        stage_b_hash = STAGE_B_HASH,
        effective_config_sha256 = EFFECTIVE_CONFIG_DIGEST,
        # WF2 needs no new sidecar file: summary/provenance.json already
        # carries the narrow digest, so it gains the WIDE one beside it. That
        # is the field the staleness test uses -- the narrow one cannot see a
        # toolbox move or an in-place catalog edit. Verified not to be in
        # dev/baseline/manifest.json, so this does not touch the baseline.
        configuration_inputs_sha256 = CONFIGURATION_INPUTS_DIGEST,
        min_reference = MIN_REFERENCE,
        # The figure set, by NAME. Stage B runs before the plot rule, so it
        # cannot take them as inputs -- but they are declared, so naming them
        # is a promise the plot rule is required to keep.
        figure_names = WF2_FIGURE_RELATIVE_PATHS,
        max_flagged_months = MAX_FLAGGED_MONTHS,
        variable_spec = {k: list(v) for k, v in VARIABLE_SPEC.items()},
        reference_record = REFERENCE_RECORD,
        catalog_crawled_on = _res._as_date_string((_INDEX or {}).get("crawled_on", "")),
        # EVERY requested combination with its ladder status -- not just the
        # resolved ones. The skips are what make the record auditable, and they
        # exist only here: an unresolved combination never becomes a job.
        combinations = [
            {
                "point_key": f"{c.dataset.replace('/', '_')}_{c.scenario}_{c.member}",
                "dataset": c.dataset,
                "institution": c.institution,
                "source_id": c.source_id,
                "scenario": c.scenario,
                "member": c.member,
                "status": c.status,
                "detail": c.detail,
                "catalog_entry": _res.entry_key(clim_project, c.dataset, c.scenario),
            }
            for c in COMBINATIONS
        ],
        # One record per resolved data point. Built HERE because the sanitized
        # model name cannot be parsed back out of a key (step 4b's lesson), so the
        # job is handed the tuple rather than reconstructing it.
        points = [
            {
                "point_key": key,
                "model": model,
                "scenario": scenario,
                "member": member,
                "series_path_hist": series_file(model, "historical", member),
                "series_path": series_file(model, scenario, member),
                "digest_components_hist": series_digest_components(model, "historical", member),
                "digest_components_fut": series_digest_components(model, scenario, member),
                "series_key": f"{clim_project}_{model.replace('/', '_')}_{scenario}_{member}",
                "reference_series_key": f"{clim_project}_{model.replace('/', '_')}_historical_{member}",
                # A3: the WEAKEST tier among the variables read. `pr`/`tas` are the
                # only two the crawl proved present, so anything else is nameable
                # but unverified for a listed member.
                "tier": point_tier(model, scenario),
            }
            for key, (model, scenario, member) in POINT_KEYS.items()
        ],
    log:
        f"{LOG_PARTS_DIR}/2.06_derive_change_factors.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/2.06_derive_change_factors.tsv",
    script: "blueearth_cst/projections/derive_change_factors.py"

# 2.07  plot_gcm_timeseries — plot projected anomaly time series
rule plot_gcm_timeseries:
    message: rule_banner("2.07", "plot_gcm_timeseries")
    input:
        # Ordering edge only -- 2.07 never opens this. Repointed at S8-05 from the
        # retired wide summary to the tidy annual table, which is the stage-B
        # artifact that now plays the same "stage B finished" role.
        stats_change_summary = (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
        # READ, not an ordering edge. The monthly figures render this table's own
        # numbers rather than recomputing them, which is what stops the picture
        # and the table describing different quantities under one name -- the
        # defect this rule shipped until 2026-08-17. It also states the reference
        # window the annual overviews are captioned with.
        change_factors_monthly = (clim_project_dir + f"/summary/{clim_project}_change_factors_monthly.csv"),
        stats_time_nc_hist = sorted({series_file(m, "historical", mem) for (m, _sc, mem) in POINTS}),
        stats_time_nc = sorted({series_file(m, sc, mem) for (m, sc, mem) in POINTS}),
    params:
        clim_project_dir = f"{clim_project_dir}",
        scenarios = scenarios,
        horizons = future_horizons,
    output:
        # The compact figure contract: two full-period overviews plus one monthly
        # change-factor figure per configured horizon. Both cloud views are rule
        # 2.06's outputs -- they come off its stage-B merge -- so they are not
        # duplicated here.
        #
        # Every path comes from `figure_relative_paths`, so what is declared and
        # what is written cannot drift. The predecessor declared three of the
        # eight it wrote; the five undeclared ones were not cleaned on failure,
        # not remade when deleted, and unusable as a dependency.
        annual_precipitation = ANNUAL_PRECIPITATION_PLOT,
        annual_temperature = ANNUAL_TEMPERATURE_PLOT,
        monthly_change_factors = MONTHLY_CHANGE_FACTOR_PLOTS,
        # S8-02: `timeseries/gcm_timeseries.nc` is GONE. It merged the nine series
        # into one cube that nothing read -- not `all`, not WF3, not any script --
        # while rounding to 2 dp (the quantisation step 5c removed from the series)
        # and stripping every `cst_*` attr, so it was a lossier, untraceable copy
        # of the tier it duplicated. The durable timeseries tier is `scalar/`;
        # `summary/*_change_factors_*.csv` is the analysis-ready form.
    log:
        f"{LOG_PARTS_DIR}/2.07_plot_gcm_timeseries.log",
    benchmark:
        f"{project_dir}/benchmarks/_parts/2.07_plot_gcm_timeseries.tsv",
    script: "blueearth_cst/projections/plot_proj_timeseries.py"

# --- log gather ---------------------------------------------------------------
# 2.09  gather_logs — merge every WF2 log part into ONE workflow log.
#
# Replaces the two per-stage gathers (`gather_series_logs` 2.09 /
# `gather_raw_logs` 2.08). Those merged only the fan-out rules, each into its own
# logs/2.NN_<rule>.log, while the four single-job rules wrote their logs straight
# to logs/ — so following one run meant opening five files and knowing their
# order. One workflow log with `==` rule banners is the same information in
# reading order.
#
# `input:` is the terminal artifact set (identical to `gather_benchmarks`), which
# is what schedules this LAST: every logging rule is upstream of it. The parts
# themselves stay in `params:` — they are `log:` files, and Snakemake does not
# track those in the DAG, so naming them as `input:` would demand them as
# buildable targets.
#
# The merge DELETES the parts it consumed and prunes the emptied dirs, so a clean
# full run leaves no logs/_parts/ at all. Consequence to know: after a PARTIAL
# re-run only the re-run rules have parts, so the merged log rewrites with the
# untouched sections marked "no part from this run" rather than resurrecting text
# it no longer has. That is the same trade `merge_benchmarks` already makes
# (dev/followups-archive.md R7-9) — the artifact describes the run that produced it, not
# an accumulated history.
#
# WF1 (1.17) and WF3 (3.18) declare the same rule against the same script; only
# the label list, the parts dir and the output name differ.
rule gather_logs:
    message: rule_banner("2.09", "gather_logs")
    input:
        (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
        CHANGE_FACTOR_CLOUD_PLOT,
        ANNUAL_PRECIPITATION_PLOT,
        # ADR 0003 §8: rule 2.03 is a LEAF here — nothing in WF2 consumes the
        # vector layers yet (§10) — so it is not upstream of the three terminals
        # above the way 2.02 is. Without this edge it would run in parallel
        # with the merge and its part would be stranded under `_parts/`, the
        # exact defect the LOG_RULES block records for 1.02, 2.02 and 3.03.
        SPATIAL_UNITS.outputs["basins"],
    output:
        f"{project_dir}/logs/{WORKFLOW_LOG_NAME}",
    params:
        rules = LOG_RULES,
        parts_dir = LOG_PARTS_DIR,
    script: "blueearth_cst/shared/merge_logs.py"

# --- benchmark gather ---------------------------------------------------------
# Merge WF2's per-rule benchmark parts (benchmarks/_parts/2.*) into one
# benchmarks/wf2_benchmarks.md (rule column + TOTAL row) once the summary +
# plots are built (all WF2 rules ran).
rule gather_benchmarks:
    message: rule_banner("2.08", "gather_benchmarks")
    input:
        (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
        CHANGE_FACTOR_CLOUD_PLOT,
        ANNUAL_PRECIPITATION_PLOT,
        # Same reason as gather_logs above: 2.03 is a leaf, and the benchmark
        # gather has to wait for it or its `_parts/` row misses this run.
        SPATIAL_UNITS.outputs["basins"],
    output:
        f"{project_dir}/benchmarks/wf2_benchmarks.md",
    params:
        parts_dir = f"{project_dir}/benchmarks/_parts",
        workflow_num = 2,
    script: "blueearth_cst/shared/merge_benchmarks.py"


# --- Run journal (design §5.7) ------------------------------------------------
#
# Workflow-level handlers, never a rule: a rule that is up to date does not
# execute, and a rule DECLARING the journal would have it deleted before the
# job ran, truncating the ledger to one line every run. See the same block in
# build_model.smk for the scope the P0 probe established -- these fire
# only when at least one job executed, which is what R5 was narrowed to.
JOURNAL_PATH = f"{project_dir}/config/runs/journal.jsonl"
INVOCATION_ID = uuid.uuid4().hex
_JOURNAL_TOOLBOX = toolbox_identity()


def _journal(event):
    append_journal_line(
        JOURNAL_PATH,
        journal_event(
            invocation_id=INVOCATION_ID,
            workflow="analyze_projections",
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
                "wf2 analyze_projections",
                project_dir,
                WORKFLOW_LOG_NAME,
                "wf2_benchmarks.md",
                elapsed_seconds=time.monotonic() - _RUN_STARTED,
                failed=failed,
                log_parts_dir=LOG_PARTS_DIR,
            )
            + "\n"
        )
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        print(f"(run summary unavailable: {exc})", file=sys.stderr)


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
            "\n"
            + run_header("wf2 analyze_projections", project_dir, config_path)
            + "\n\n"
        )
    except Exception as exc:  # noqa: BLE001 -- never break a run over a banner
        print(f"(run header unavailable: {exc})", file=sys.stderr)


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
