# Current guidance: dev/reference/toolbox-spatial-foundation.md
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
# See dev/records/milestones/r03/model-builder-design.md §3.
sys.path.insert(0, str(Path(workflow.basedir)))
from blueearth_cst.shared.provenance import append_journal_line, configuration_inputs_digest, effective_config_digest, environment_file_hashes, file_sha256, journal_event, referenced_inputs_for_digest, toolbox_identity
from blueearth_cst.shared.snake_utils import ADVANCED_SETTINGS, DEFAULT_BASIN_INDEX, DEFAULT_HYDROGRAPHY, catalog_root, declare_path_tokens, declare_project_root, declare_warning_tally, get_config, listed, patch_psutil_windows_benchmark, region_rule, resolve_water_year_start, spatial_units_rule, warning_count, window_year_pair
from blueearth_cst.shared.console_style import defer_warning, install_console_style, open_run_header, RuleRegistry, rule_banner, run_header, run_summary, target_banner, warn_if_project_dir_in_repo
from blueearth_cst.shared.console_style import header_config_path
from blueearth_cst.shared.config_composition import compose_config
from blueearth_cst.shared.workflow_archive_launch import captured_projection, require_capture
from blueearth_cst.spatial.config import parse_spatial_config
from blueearth_cst.projections.gridded_outputs import RemovedGriddedOutputsError, validate_removed_gridded_options
# The figure family's CONTRACT only. `projection_figures` is deliberately
# matplotlib-free so a WF2 parse -- including the four dry-runs test_cli.py
# performs -- stays as cheap as it is today; the drawing lives in
# `projection_suite_figures` and is imported by the renderer, never here.
from blueearth_cst.projections import projection_figures
from blueearth_cst.projections.get_change_climate_proj import analysis_windows

# Windows: make Snakemake's benchmark memory/IO/CPU indicators work (else all NA).
patch_psutil_windows_benchmark()

# read path of the config file (Snakemake records it from --configfile) so
# downstream R scripts can be handed the same path. Forwarding config_path is
# a repo convention.
config_path = workflow.configfiles[0]
CAPTURE_RECORD = require_capture("analyze_projections", config_path)

# The consumed-key PROJECTION: the config paths this workflow actually reads.
# Digesting the projection rather than the whole file is what stops another
# workflow's edit from re-firing this record.
#
# HOISTED above the first config read (R13 D-8.2): the projection is a
# config-independent literal, and `compose_config` derives R(entry) from it, so
# it has to be known before any section is touched. Moving it changed no value.
CONFIG_PROJECTION = ("project", "basin", "climate", "model", "workflows.analyze_projections")
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
    config, config_path, entry="analyze_projections", declared_sections=CONFIG_PROJECTION,
)
from blueearth_cst.shared.plot_config import resolve_annotation
from blueearth_cst.shared.plot_setup import setup_digest
plot_annotation = resolve_annotation(config)
plot_setup_digest = setup_digest()
CAPTURE_DIGESTS = captured_projection(CAPTURE_RECORD, config, ADVANCED_SETTINGS)
# Sorted so the declared input lists below do not churn on dict order.
WF_CONFIG_PATHS = sorted(WORKFLOW_CONFIG_PATHS.values())

# R01 schema
project_cfg = config["project"]
# R14 D-7.2: `shared:` dissolved into sections by KIND. `climate_cfg` is the
# only new binding -- `basin:` and `model:` are read at their use sites, which
# is where the v1 `shared_cfg` indirection was buying nothing.
climate_cfg = config.get("climate") or {}
my_cfg = config["workflows"]["analyze_projections"]

project_dir = get_config(project_cfg, "project_dir", optional=False)
# O-22: make the two-tier project_dir rule mechanical rather than
# documentary. Warns, never raises; test_case/ is the one exemption.
warn_if_project_dir_in_repo(project_dir, workflow.basedir)
DATA_SOURCES = get_config(my_cfg, "catalog", optional=False)  # C-39


# The content-addressed config bundle was removed here (config-snapshot
# redesign, 2026-08-13) -- no readers, and a digest over the whole config, so
# any other workflow's edit minted a fresh directory. `run_record.yml` replaces
# it: current-only and one per workflow.
RUN_RECORD = f"{project_dir}/config/analyze_projections/run_record.yml"

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
# Threaded through rule 2.01's params: so the record re-fires when the
# checkout, the lock files, or a referenced catalog's bytes move -- not only
# when the config is edited.
CONFIGURATION_INPUTS_DIGEST = CAPTURE_DIGESTS[1] if CAPTURE_DIGESTS else configuration_inputs_digest(
    EFFECTIVE_CONFIG_DIGEST,
    toolbox_identity(),
    environment_file_hashes(),
    referenced_inputs_for_digest(CONFIG_REFERENCES),
)

# The region delineation reads project.catalog (deltares), NOT the CMIP6
# catalog above: delineating a basin is a hydrography read, not a projections
# read. WF2 therefore reads BOTH catalogs — a divergence from the pre-v2.0
# workflow, which read only the climate catalog. The name is historical: this
# fed the climate store until the shared-region refactor replaced it with the region rule.
STORE_DATA_SOURCES = get_config(project_cfg, "catalog", optional=False)  # C-40

# Shared — the model-free basin delineation the climate store extracts against.
basin_cfg = config["basin"]
model_region = get_config(basin_cfg, "region", optional=False)
# `C-15`: every spatial input lives under `basin.sources:`. Read through the
# same parse WF0 and WF1 use, so one section cannot mean two things depending
# on which entry point read it.
_basin_sources = get_config(basin_cfg, "sources", {}) or {}
basin_hydrography = get_config(_basin_sources, "hydrography", DEFAULT_HYDROGRAPHY)
basin_index = get_config(_basin_sources, "basin_index", DEFAULT_BASIN_INDEX)
# `shared.clim_historical` is deliberately NOT read here. WF2 has no climate
# store and no rule that consumes the observed source: it read the key
# `optional=False` and never used the value, so a config omitting it failed WF2
# for a setting WF2 does not have. Removed 2026-08-13 (defect E). The key stays
# required where it is actually used -- WF1 and the shared climate-store
# producer -- so this loosens nothing that matters.

def _window_pair(window, key):
    """Thin alias so this file reads the same for both retyped windows."""
    return window_year_pair(window, f"workflows.analyze_projections.{key}")


def _named_windows(windows):
    """`C-60`: the `future_windows:` LIST, back to `{name: [start, end]}`.

    Every consumer downstream -- `suite_folders`, rule 2.05's `params:`,
    `get_horizon` -- takes the mapping this returns, so the retype stops here.

    `name` is optional (D-7.3's sibling in 7.3). An unnamed window is keyed
    `<start>-<end>`, which is stable and self-describing. Figure directories use
    the distinct analysis-window years, recorded separately from these names.

    Insertion order is preserved: the list is ordered by the author now, which
    is what the v1 mapping only got by accident of YAML insertion order.
    """
    if not isinstance(windows, (list, tuple)):
        raise WorkflowError(
            "workflows.analyze_projections.future_windows is a mapping, which is "
            "the v1 `future_horizons` shape. It is a LIST now (`C-60`):\n\n"
            "    future_windows:\n      - {start: 2046, end: 2054, name: mid}\n"
        )
    named = {}
    for entry in windows:
        pair = _window_pair(entry, "future_windows[]")
        name = entry.get("name") or f"{pair[0]}-{pair[1]}"
        if name in named:
            raise WorkflowError(
                f"workflows.analyze_projections.future_windows names {name!r} "
                "twice; each window becomes a figure directory, so the names "
                "must be distinct."
            )
        named[str(name)] = pair
    if not named:
        raise WorkflowError(
            "workflows.analyze_projections.future_windows is empty; declare at "
            "least one window."
        )
    return named


# C-25: the key is `ensemble:` now. The VALUE is unchanged and still names the
# projections subdirectory and every series-cache filename, so this renames what
# the user writes and moves no output path.
clim_project = get_config(my_cfg, "ensemble", optional=False)
models = get_config(my_cfg, "models", optional=False)
scenarios = get_config(my_cfg, "scenarios", optional=False)

# C-63: `members:` is a GROUP now -- the bare preference list it used to be,
# plus the two `member_*` keys that were flat beside it. Read as one block here;
# `selection` and `overrides` are unpacked further down, where `_res` is in scope.
_members_cfg = get_config(my_cfg, "members", optional=False)
if not isinstance(_members_cfg, dict):
    raise WorkflowError(
        "workflows.analyze_projections.members is a bare list, which is the v1 "
        "shape. It is a mapping now (`C-63`):\n\n"
        "    members:\n      preference: [r1i1p1f1]\n"
        "      selection: first_available\n      overrides: {}\n\n"
        "`member_selection` and `member_overrides` fold into it as `selection` "
        "and `overrides`."
    )
members = get_config(_members_cfg, "preference", optional=False)
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

# Water-year start, from the project climate key so WF2, WF3 and the figures
# cannot disagree about what an annual value is.
#
# The old `workflows.analyze_projections.start_month_hyd_year` is REFUSED, not
# quietly honoured. It never reached the arithmetic: the Snakefile read it and
# passed it to rule 2.05, and derive_change_factors.py never read the param, so
# every change factor has always been computed Jan-Dec whatever the key said.
# Honouring it now would silently CHANGE results for any config carrying a
# non-Jan value -- the config would start meaning what it always claimed, which
# is right, but it must be a decision the owner makes rather than one a version
# bump makes for them.
_legacy_hyd = my_cfg.get("start_month_hyd_year")
if _legacy_hyd is not None:
    raise ValueError(
        "workflows.analyze_projections.start_month_hyd_year has moved to "
        "climate.water_year_start (it is now honoured by WF2, WF3 and the "
        f"climate figures alike). Remove it and set:\n\n"
        f"    climate:\n      water_year_start: {_legacy_hyd}\n\n"
        "It never reached the change-factor arithmetic before, which always "
        "used Jan, so a non-Jan value will move every change factor once."
    )
water_year_start = resolve_water_year_start(get_config(climate_cfg, "water_year_start"))
# C-59: `reference_window: {start, end}`, CALENDAR years and deliberately so
# (`C-74`, D-7.4). It is NOT routed through `resolve_water_year_start` /
# `hydrological_year_bounds`: those trim to complete water years one layer down,
# so applying the offset here would apply it twice. Flattened to the [start, end]
# pair the reference-window helpers already take, so nothing downstream retypes.
_reference_window = get_config(my_cfg, "reference_window", optional=False)
time_horizon_hist = _window_pair(_reference_window, "reference_window")

# C-60: `future_windows:` is a LIST of `{start, end, name}` -- ordered by the
# author rather than by mapping insertion, which is what the v1 mapping relied on.
# Converted back to the `{name: [start, end]}` shape every consumer already takes
# (`suite_folders`, rule 2.05's params, `get_horizon`), so C-60 is a
# config-surface change only.
_future_windows = get_config(my_cfg, "future_windows", optional=False)
future_horizons = _named_windows(_future_windows)

# --- step 5e / D1: clip the reference window, never splice --------------------
# The change-factor reference is the GCM historical experiment, which ends
# 2014-12-31. A window reaching past that is CLIPPED; 2015+ exists only under the
# per-scenario ScenarioMIP entries and stitching them on would silently mix two
# experiments inside one reference (N8).
#
# The GCM reference is independent of the observed historical climate window.
from blueearth_cst.projections import reference_window as _rw

REFERENCE_WINDOW = _rw.clip_reference_window(time_horizon_hist)

# Record the GCM reference alone in provenance and the report.
REFERENCE_RECORD = _rw.alignment_record(REFERENCE_WINDOW)

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
    defer_warning(_warning, module="config")

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

# `C-66`: `relative_change:` is gone. Both of its keys have homes that fit them
# better -- `min_denominator` is per-variable metadata on the variable itself
# (`C-64`) and `max_flagged_months` is a toolbox constraint (`C-65`) -- so the
# section had nothing left to hold. The threshold now arrives INSIDE the spec,
# resolved at parse: a declared `variables.<name>.min_denominator` first, the
# registry second.
#
# The LOCAL stays `MIN_REFERENCE`: P1b's tier rule renames what names a key back
# to the user, not what a variable happens to be called here.
MIN_REFERENCE = _dm.resolve_thresholds(VARIABLE_SPEC)
# `C-65`: a toolbox CONSTRAINT, not a project setting. The threshold at which a
# monthly relative product stops being reportable is a property of the method,
# so no project may relax it -- which is why this reads from
# `advanced_settings.constraints` and takes no `_relative_cfg` fallback.
MAX_FLAGGED_MONTHS = ADVANCED_SETTINGS["constraints"]["max_flagged_months"]

# R9 P2 commit 2: the projections overlay moves under `data/climate/`. ONE
# binding carries every WF2 output below it, and the two cache tiers keep their
# names -- `raw/` (as-fetched GCM slices) and `scalar/` (spatially reduced,
# S8-03) -- so `prune_series_cache.py`, which is keyed to that grammar, keeps
# working once repointed at the new root.
clim_project_dir = f"{project_dir}/data/climate/projections/{clim_project}"

# ONE producer contract, built here and identically in build_model.smk
# and run_stress_test.smk, and splatted into rule 1.01 / 2.01 / 3.03.
# Everything content- or execution-determining comes from this object; only
# message/log/benchmark are workflow-local. tests/test_region_rule.py parses all
# three workflows and fails on ANY other difference.
#
# WF2 needs the model-free delineated polygon and nothing else — that is what
# freed it from wf1's hydrology_model/staticgeoms/region.geojson (design D2/A1,
# goal G2). Until the shared-region refactor the only model-free source was the climate-store
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

# --- The shared vector foundation -------------------------------
# Same pattern once more: ONE producer contract, built here and identically in
# the other two workflows, splatted into rule 2.02 below, and
# tests/test_spatial_units_rule.py parses all three and fails on ANY difference
# beyond message/log/benchmark.
#
# WF2 gains basin and subbasin boundaries — a context map beside the
# change-factor plots, and the option of subbasin-resolved indicators — WITHOUT
# a built model and without the thematic raster stack. Declaring the UNSPLIT
# rule 1.05 instead would have made a projections-only run resample `vito`,
# `modis_lai` and `soilgrids` to draw a subbasin outline; measured 2026-08-06,
# the split avoids ~71% of what that would add. `snakemake -n` on this file must
# therefore list `delineate_subbasins_and_rivers` and no job whose inputs mention those
# three sources — the dry-run checks that dependency boundary.
#
# WF2 requests these layers as explicit terminal products.
#
# `parse_spatial_config(basin_cfg)` takes NO model section: the
# shared rule's params must depend only on shared project and basin settings, and
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
# The list names the ROOTS. Since t2608071218 `kernel_hash` follows their calls
# into blueearth_cst (and hashes the plain module constants they read), because
# hand enumeration had missed `_spatial_dim`, `check_axis` and `intersection`.
# Code in a script's `__main__` block is still never hashed -- keep arithmetic in
# a rooted function (`merge_member_series` moved out for that reason).
#
# `pixi.lock` is folded in as the environment fingerprint: the reduction's numbers
# depend on xarray/pandas behaviour, which no source hash can see. Coarse on
# purpose -- any dependency change re-derives, which is the safe direction for a
# cache whose failure mode is silently wrong numbers.
from blueearth_cst.projections.get_stats_climate_proj import get_stats_clim_projections, merge_member_series
from blueearth_cst.projections.grid_weights import (
    cell_area_weights,
    latitude_weights,
    longitude_weights,
    midpoint_edges,
    weighted_spatial_mean,
)

# Step 5a: the spatial reduction is no longer a single `.mean()` inside
# get_stats_clim_projections -- it is a weighted mean whose weights come from the
# five functions below. The closure would reach them through the root anyway;
# they stay listed so the kernel reads as the reduction it hashes.
REDUCER_KERNEL = [
    get_stats_clim_projections,
    merge_member_series,
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
# Same roots-plus-closure rule as REDUCER_KERNEL; the closure is what reaches
# `intersection` and `_to_datetime_index`.
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


#: WF2's spatial buffer around the region bbox, in GRID CELLS. Previously a bare
#: `buffer = 1` inside the script; named here because it is a digest component
#: (design §5.3 item 4) and a documented sampling choice, not an accident.
#:
#: It read `REGION_BUFFER_DEGREES = 1.0` until t2608182238, which was a misnomer
#: the whole time: hydromt spends `buffer` as resolution multiplicity, so one
#: unit is 0.70 deg on EC-Earth3 and 2.77 deg on CanESM5. The bare `1` this
#: replaced was already a cell count. The rename changed no footprint and no
#: number -- see `series_identity.SCHEMA_VERSION`, bumped 4->5 so the slices
#: cached under the old key refuse loudly rather than re-deriving in silence.
REGION_BUFFER_CELLS = 1

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

# t2608192107: `members` is an ORDERED PREFERENCE and `member_selection` says
# what to do with it. Both keys are OPTIONAL, so no existing config has to move
# -- and a single-member list resolves identically under either policy, which is
# what makes the new default safe for every config that exists today.
#
# Parsed HERE rather than beside `members` at the top of the file because the
# policy names live on `_res`, which is imported a few lines up.
# C-63: both were flat `member_selection` / `member_overrides`; they are leaves of
# the `members:` group now, read from the block captured at the top of the file.
member_selection = get_config(_members_cfg, "selection", default=_res.FIRST_AVAILABLE)
# A model may name its own preference list, REPLACING the global one. Coalesced
# because `get_config` returns a key written-but-empty as None rather than {}.
member_overrides = get_config(_members_cfg, "overrides", default={}) or {}

COMBINATIONS = _res.resolve(
    _CATALOG,
    clim_project=clim_project,
    models=models,
    scenarios=scenarios,
    members=members,
    selection=member_selection,
    overrides=member_overrides,
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

# An override names a SPECIFIC realisation the operator wants, so its failure is
# a configuration error rather than the thin-data skip the same status means for
# the global preference list -- falling back silently would defeat the point of
# writing it. Also catches a key naming a model the run does not request, which
# nothing else would see: `unknown_models` only looks at models that ARE
# requested, so a typo there would otherwise be a silent no-op.
_bad_overrides = _res.unresolved_overrides(COMBINATIONS, member_overrides)
if _bad_overrides:
    raise WorkflowError(
        "analyze_projections: member_overrides resolved nothing for "
        f"{', '.join(_bad_overrides)}. An override asserts a specific "
        "realisation, so it is not allowed to fall back to the global "
        "`members:` list.\n" + _res.format_status_report(COMBINATIONS)
    )

# D6: replaces the `ensemble.min_sources` key, which conflated a store property
# with a configuration failure. Non-configurable -- ensemble ADEQUACY is a
# downstream judgement (S6, N10), not a threshold this workflow encodes.
if not any(c.resolved for c in COMBINATIONS):
    raise WorkflowError(
        "analyze_projections: no requested combination resolved against "
        f"{DATA_SOURCES}.\n" + _res.format_status_report(COMBINATIONS)
    )

# A partially resolved preference list is normal for CMIP6 and does not earn a
# console warning. Fatal resolution paths above embed the complete status report
# in their WorkflowError, so configuration failures retain the diagnostics.

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
_best_effort = _res.best_effort_variables(variables, _rename)
if _best_effort:
    # ONE row, not one per variable: the sentence is the same each time and
    # only the name changes, so the names are what the row lists. Through
    # `defer_warning` for the reason given on the skip report above.
    defer_warning(
        f"Availability unverified: {listed(sorted(_best_effort))}; catalog checks only pr/tas. "
        "Missing variables will fail during fetching.", module="resolution"
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


def _series_label(series_key):
    """What a fanned-out line PRINTS for one series: the part that varies.

    The key itself is `cmip6_<vendor>_<model>_<experiment>_<member>` -- 52
    characters on the RUN and the DONE of every member, of which the catalog
    name is constant and the vendor is a function of the model. What is left is
    unique by CMIP6's own naming (a model name identifies its institution), and
    it is static, so one grep still finds a series across the console, the log
    and the benchmark table.

    Carried as a PARAM rather than a wildcard because `message:` is formatted
    per job against params as well as wildcards, and the key is what names the
    files -- it cannot be shortened itself.
    """
    model, experiment, member = SERIES[series_key]
    return f"{model.rsplit('/', 1)[-1]} {experiment} {member}"


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
        buffer_cells=REGION_BUFFER_CELLS,
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
    # Reads the CONVERTED mapping, not the raw config: `future_windows` is a list
    # and the wildcard is a horizon NAME, so the raw document cannot be indexed by
    # it any more.
    return future_horizons[wildcards.horizon]

# Rule numbering (comment headers + log/benchmark filenames) uses `W.NN` = the
# rule's position in this workflow's LOGICAL order — data, then the product,
# then figures and records. Contiguous, and every dependency points from a lower
# number to a higher one. Positional since [R10-5] (2026-08-06); this file is
# where the old scheme showed worst, defining its rules 2.00, 2.03b, 2.02, 2.01,
# 2.01, 2.03, 2.05, 2.06 — numbers out of order against the DAG.
#
# STILL A READING AID, NOT EXECUTION ORDER — stage A fans out over {series_key},
# so low-to-high says "cannot depend on", not "runs before". Definition order in
# this file is not the number order and is not required to be.
#
# DO NOT RENUMBER TO INSERT A RULE. Use a letter suffix (2.05b) until the next
# deliberate sweep. Convention: dev/reference/dev-naming-conventions.md §9; historical translations:
# dev/records/milestones/r10/rule-numbering-crosswalks.md.

# --- log layout ---------------------------------------------------------------
# EVERY WF2 rule that logs writes a PART under logs/_parts/, and rule 2.08 merges
# the parts into ONE logs/wf2_analyze_projections.log, then deletes them. So the
# only WF2 file left in logs/ after a full run is that merged log — the same deal
# benchmarks/wf2_benchmarks.md already gets from `gather_benchmarks`.
#
# LOG_RULES is the merge order: rule LABELS, not part paths. merge_logs lists
# each label's part dir to find its members, so the fan-out width lives in one
# place — the rule that owns it. Naming the labels rather than globbing `_parts/`
# is what keeps an orphan dir from a renamed rule out of the file (test_local
# still holds `2.01_monthly_stats_hist`, `2.02_monthly_stats_fut`,
# `2.03_monthly_change` from the pre-step-4d names, and now the whole pre-R10
# generation besides): not a label, never read.
#
# ORDER IS BY RULE NUMBER, and since [R10-5] that is also dependency order, so
# this list needed no reordering to satisfy the assertion
# tests/test_log_rules_contract.py now makes — it was already in EXECUTION
# order, and the renumber is what made the two agree. Before it this file's
# comment claimed "order is by rule number" while the list opened 2.03b, 2.01,
# 2.01; which of the two was wrong was a ruling nobody had made, and it is why
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
# outlived the rule the shared-region refactor deleted, contributing a phantom "no part from
# this run" section to every merged WF2 log. WF2 builds no climate store,
# standalone or otherwise: it reads the region polygon from `delineate_region`.
# Both directions are now mechanically checked for all three workflows.
from blueearth_cst.shared.execution_outputs import workflow_paths, declare_warning_tally
WORKFLOW_LOG_NAME = "wf2_analyze_projections.log"

# One tally per run, opened at parse time and published to every job process
# through the environment: a rule prints its warnings from a process of its
# own, and the verdict below is written by this one. See
# `snake_utils.declare_warning_tally`.
declare_warning_tally(project_dir, WORKFLOW_LOG_NAME)
LOG_PARTS_DIR = f"{project_dir}/logs/_parts"
_paths = workflow_paths(project_dir, "wf2", WORKFLOW_LOG_NAME, "wf2_benchmarks.md", LOG_PARTS_DIR, f"{project_dir}/benchmarks/_parts")
WORKFLOW_LOG_PATH, BENCHMARK_OUTPUT_PATH = _paths.log, _paths.benchmark
LOG_PARTS_DIR, BENCH_PARTS_DIR = _paths.log_parts, _paths.benchmark_parts

# The run's key folders, stated once -- see the same block in
# build_model.smk. `data` is the DELTARES catalog's root, not the CMIP6
# one: WF2's `catalog:` points at a `gs://` store, which has no local
# prefix to shorten. No `model` row -- WF2 builds none.
declare_path_tokens(
    data=catalog_root(STORE_DATA_SOURCES),
    projections=clim_project_dir,
)
declare_project_root(project_dir)
RULES = RuleRegistry(LOG_PARTS_DIR, BENCH_PARTS_DIR)
LOG_RULES = RULES.log_rules
DELINEATE_REGION = RULES.logged("2.01", "delineate_region", summary="delineate the basin boundary")
DELINEATE_SUBBASINS_AND_RIVERS = RULES.logged("2.02", "delineate_subbasins_and_rivers", summary="derive subbasins, rivers, and output locations")
FETCH_CMIP6_PROJECTIONS = RULES.logged("2.03", "fetch_cmip6_projections", summary="acquire one basin slice")
REDUCE_TO_BASIN_AVERAGES = RULES.logged("2.04", "reduce_to_basin_averages", summary="compute basin-average climate series")
DERIVE_CHANGE_FACTORS = RULES.logged("2.05", "derive_change_factors", summary="compare each future window with the reference period")
PREPARE_PROJECTION_ANALYSIS = RULES.logged("2.05b", "prepare_projection_analysis", summary="compute projection diagnostics and figure eligibility")
PLOT_CLIMATE_PROJECTIONS = RULES.logged("2.06", "plot_climate_projections", summary="render eligible projection figures")
GATHER_BENCHMARKS = RULES.banner_only("2.07", "gather_benchmarks", summary="merge rule timing and resource-use records")
GATHER_LOGS = RULES.banner_only("2.08", "gather_logs", summary="merge rule logs")

# all — target aggregator: change-factor summaries + projection plots
#
# Hoisted into a dict so `message:` can print one target per line without
# restating them. A dict rather than a list because five of these were already
# NAMED inputs and the names document what each file is; `**` splats them back
# as the same keyword arguments. The three that were positional gain a name,
# which changes nothing -- no script reads `rule all`'s inputs, by name or index.
ANALYSIS_WINDOWS = analysis_windows(future_horizons, water_year_start, time_horizon_hist)
ANALYSIS_TABLE_NAMES = (
    "coverage", "availability", "changes", "model_changes", "change_summary",
    "seasonal_values", "seasonal_values_summary",
    "sign_agreement", "climatology", "climatology_summary", "monthly_change_summary",
    "member_anomalies", "trajectory_range", "variability_member_ratios",
    "variability_model_ratios", "variability_summary",
)
ANALYSIS_OUTPUTS = {
    name: f"{clim_project_dir}/analysis/{name}.csv" for name in ANALYSIS_TABLE_NAMES
}
ANALYSIS_OUTPUTS["analysis_record"] = f"{clim_project_dir}/_engine/analysis_record.json"
WF2_SUITE_PATHS = {
    folder: f"{clim_project_dir}/plots/{folder}"
    for folder in projection_figures.suite_folders(future_horizons, water_year_start, time_horizon_hist)
}
FIGURE_CAPTIONS_MD = f"{clim_project_dir}/plots/figure_captions.md"
FIGURE_CAPTIONS_JSON = f"{clim_project_dir}/_engine/figure_captions.json"

WF2_TARGETS = {
    **{f"analysis_{name}": path for name, path in ANALYSIS_OUTPUTS.items()},
    # S8-04/05: the tidy tables are the result surface now; the three wide
    # `annual_change_scalar_stats_summary*` files they replaced are gone.
    "change_factors_annual": (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
    "change_factors_monthly": (clim_project_dir + f"/summary/{clim_project}_change_factors_monthly.csv"),
    # R5 deliverable 3 — demanded explicitly, not left as an incidental
    # by-product of derive_change_factors' other outputs.
    "composition": (clim_project_dir + "/summary/composition.csv"),
    # Runtime availability decides which PNGs exist within these owned folders.
    **{f"figure_folder_{folder}": path for folder, path in WF2_SUITE_PATHS.items()},
    "figure_titles": f"{clim_project_dir}/plots/figure_titles.md",
    "figure_title_manifest": f"{clim_project_dir}/_engine/figure_titles.json",
    "figure_captions": FIGURE_CAPTIONS_MD,
    "figure_manifest": FIGURE_CAPTIONS_JSON,
    # The vector rule has NO WF2 consumer yet, so a target
    # entry is what makes it reachable at all — an undeclared leaf is simply
    # never scheduled. One representative schedules the whole multi-output rule,
    # the same deal WF1's `spatial_catalog.yml` terminal gets. It is also a
    # gather input below: a leaf running parallel to the merge would strand its
    # log part under `_parts/`, which is the defect the LOG_RULES block above
    # documents three times over.
    "spatial_basins": SPATIAL_UNITS.outputs["basins"],
    **({"project_config": RUN_RECORD} if os.environ.get("BLUEEARTH_WF012_CAPTURE_CONTEXT") else {}),
    # ONE merged log for the whole workflow (was: one per fan-out stage,
    # alongside four rules writing logs/2.NN_*.log directly -- five files a
    # reader had to open in the right order to follow one run). Every rule
    # now logs into logs/_parts/ and rule 2.08 merges the lot.
    **_paths.gather_targets,
}

rule all:
    message: target_banner("all", WF2_TARGETS.values(), project_dir, compact=True)
    input:
        **WF2_TARGETS,

# The `ruleorder:` directive is GONE as of step 4d, and not by choice: it named
# `monthly_change` and `monthly_change_scalar_merge`, which 4d merges into
# `derive_change_factors`, and an unknown rule name is a parse error. A directive
# naming only `reduce_to_basin_averages` would be a no-op (ruleorder needs two rules to
# order).
#
# This settles a deferral rather than skipping it. AGENTS.md and
# dev/records/milestones/r04/climate-projections-design.md §3 recorded it as "stale insurance,
# removal deferred to a task that first encodes the ambiguity-sensitive config
# shapes as regression tests". The 2026-07 dry-run evidence stands (it constrained
# nothing on the tests fixture or a reduced config), and the merge removes the
# ambiguity it was insuring against: there is no longer a second stage-B rule that
# could claim the same output.

# 2.01  delineate_region — the one project region artifact.
# Byte-identical to 1.01 and 3.03 except message/log/benchmark.
#
# This REPLACES rule 2.11 extract_climate_grid -- the name that rule answered to
# when it was deleted; WF1/WF3's surviving copy is now extract_climate_datasets
# (R10). WF2 declared the whole shared
# climate-store producer to obtain the delineated polygon and never read the
# gridded extraction it also wrote (design N7); a projections-only run paid for
# a multi-decade extraction to learn a basin outline. The region is now its own
# artifact, so WF2 declares the delineation alone.
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

# 2.02  delineate_subbasins_and_rivers — the shared vector foundation.
# Byte-identical to 1.02 and 3.04 except message/log/benchmark.
#
# WF2 declares the VECTOR half only. The raster half (rule 1.05) stays WF1-only,
# so a projections-only run obtains basin and subbasin boundaries without
# reading `vito`, `modis_lai` or `soilgrids` at all — see the SPATIAL_UNITS
# comment block above for why that is the whole point.
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

# 2.03  fetch_cmip6_projections — acquire ONE raw slice; the only rule that reads the store.
#
# Stage A splits into fetch -> reduce (design revision 6) because the two have
# wildly different costs and different invalidation causes. Measured 2026-07-30
# (dev/records/milestones/r08/2026-07-30_wf2-fetch-reduce-benchmark.md): opening one remote source
# ~1142 s, transferring its data ~19 s, reducing it ~0.2 s, raw slice 0.07 MB. So a
# reduction edit used to cost a full re-download of nine sources; now it re-reads
# local disk.
#
# THE LOAD-BEARING DETAIL: these params must NOT include the reducer hash. They
# carry `raw_digest_components` (series_identity.raw_components -- the component set
# minus reducer_module_hash), so Snakemake's params trigger cannot re-download on a
# formula edit. Passing `digest_components` here instead would silently undo the
# entire split while every test still passed.
rule fetch_cmip6_projections:
    message: FETCH_CMIP6_PROJECTIONS.banner(context="{params.series_label}")
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
        # DISPLAY ONLY -- `message:` reads it, nothing else does. It is
        # deliberately not a digest component: see `_series_label`.
        series_label = lambda wildcards: _series_label(wildcards.series_key),
        catalog_entry = lambda wildcards: f"{clim_project}_{SERIES[wildcards.series_key][0]}_{SERIES[wildcards.series_key][1]}_{{member}}",
        member = lambda wildcards: SERIES[wildcards.series_key][2],
        variables = variables,
        # S8-08(a): the units the STORED values are in, keyed by post-rename source
        # name. The catalog's data_adapter converts on read but leaves the `units`
        # attribute describing the pre-conversion quantity, so every array claimed
        # `kg m-2 s-1` over mm/day. Not a digest component -- it labels the values,
        # it does not change them.
        variable_units = {v.source: v.units for v in VARIABLE_SPEC.values()},
        buffer_cells = REGION_BUFFER_CELLS,
        acquisition_window = lambda wildcards: list(_si.acquisition_window(SERIES[wildcards.series_key][1])),
        raw_digest_components = lambda wildcards: _si.raw_components(
            series_digest_components(*SERIES[wildcards.series_key])
        ),
    threads: 1
    resources:
        mem_mb = 1024,
    log:
        FETCH_CMIP6_PROJECTIONS.log("{series_key}"),
    benchmark:
        FETCH_CMIP6_PROJECTIONS.benchmark("{series_key}"),
    script: "blueearth_cst/projections/fetch_gcm_raw.py"

# 2.04  reduce_to_basin_averages — ONE stage-A rule over {series_key} (step 3).
# Collapses monthly_stats_hist + monthly_stats_fut: they ran the same script with
# different params, and the only structural difference was their output naming
# (`historical_stats_time_{model}` vs `stats_time-{model}_{scenario}`) -- which is
# why the collapse needed the naming unified first.
#
# The hist->fut ordering edge is gone (step 2b) and there is now no edge between
# series at all: every series is independent, so the stage fans out at full width.
# Since revision 6 it reads the local raw slice above and makes NO network call.
rule reduce_to_basin_averages:
    message: REDUCE_TO_BASIN_AVERAGES.banner(context="{params.series_label}")
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
        # DISPLAY ONLY -- `message:` reads it, nothing else does. It is
        # deliberately not a digest component: see `_series_label`.
        series_label = lambda wildcards: _series_label(wildcards.series_key),
        name_scenario = lambda wildcards: SERIES[wildcards.series_key][1],
        # ONE member per job now (step 4b), not the config list.
        name_members = lambda wildcards: [SERIES[wildcards.series_key][2]],
        name_model = lambda wildcards: SERIES[wildcards.series_key][0],
        name_clim_project = clim_project,
        variables = variables,
        # S8-08(a): see rule 2.03.
        variable_units = {v.source: v.units for v in VARIABLE_SPEC.values()},
        series_nc_out = lambda wildcards: f"{clim_project_dir}/scalar/{wildcards.series_key}.nc",
        digest_components = lambda wildcards: series_digest_components(*SERIES[wildcards.series_key]),
        acquisition_window = lambda wildcards: list(_si.acquisition_window(SERIES[wildcards.series_key][1])),
        store_index = STORE_INDEX,
        buffer_cells = REGION_BUFFER_CELLS,
    threads: 1
    resources:
        mem_mb = 1024,
    log:
        REDUCE_TO_BASIN_AVERAGES.log("{series_key}"),
    benchmark:
        REDUCE_TO_BASIN_AVERAGES.benchmark("{series_key}"),
    script: "blueearth_cst/projections/get_stats_climate_proj.py"

# Stage B's settings as an INPUT FILE, rewritten only when they change
# (t2608112051). The params rerun-trigger needs Snakemake metadata from the run
# that built summary/, and a .snakemake that never saw that build -- another
# worktree, a fresh clone -- has none, so a changed reference window, horizon
# or model set fell through to mtime and left the OLD change factors in place
# while reporting success. An input's mtime needs no metadata.
_STAGE_B_STAMP = f"{clim_project_dir}/_engine/stage_b_settings.json"
_stage_b_settings = json.dumps(
    {
        "horizons": future_horizons,
        "time_horizon_hist": time_horizon_hist,
        "water_year_start": water_year_start,
        "stats": stats,
        "stage_b_hash": STAGE_B_HASH,
        "min_reference": MIN_REFERENCE,
        "max_flagged_months": MAX_FLAGGED_MONTHS,
        "variable_spec": {k: list(v) for k, v in VARIABLE_SPEC.items()},
        "points": sorted(POINT_KEYS),
        # Policy reporting must refresh even when an excluded source was already
        # unresolved, and when this checkout has no prior Snakemake metadata.
        "source_quality_exclusions": [
            c._asdict() for c in COMBINATIONS if c.status == _res.SOURCE_QUALITY_EXCLUDED
        ],
    },
    sort_keys=True,
    indent=1,
    default=str,
)
if (
    not os.path.isfile(_STAGE_B_STAMP)
    or Path(_STAGE_B_STAMP).read_text(encoding="utf-8") != _stage_b_settings
):
    Path(_STAGE_B_STAMP).parent.mkdir(parents=True, exist_ok=True)
    Path(_STAGE_B_STAMP).write_text(_stage_b_settings, encoding="utf-8")

# 2.05  derive_change_factors — stage B, ONE job (step 4d, design §5 "B. Derive")
# Replaces monthly_change (fanned out per point_key x horizon) + its aggregator
# monthly_change_scalar_merge. The design gives stage B one job with no fan-out.
# The per-point change netCDFs were temp() outputs of the fan-out; they are now
# in-memory datasets; no temporary NetCDFs are written. Durable outputs retain
# their existing shape.
rule derive_change_factors:
    message: DERIVE_CHANGE_FACTORS.banner()
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
        # Re-derives on a settings change without Snakemake metadata; above.
        stage_b_settings = _STAGE_B_STAMP,
    output:
        # S8-05: the three wide `annual_change_scalar_stats_summary*` files are
        # GONE. The tidy tables below supersede them -- same numbers, long format,
        # per-row provenance, plus the future level the wide form never carried.
        # Verified before removal: run_stress_test.smk and
        # blueearth_cst/experiment/ referenced them zero times, and rule 2.06
        # declared the `.nc` as an input it never opened. The `.nc` survives as a
        # in-memory dataset shared by the tidy tables. Figures use 2.05b tables.
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
    threads: 1
    resources:
        mem_mb = 1024,
    log:
        DERIVE_CHANGE_FACTORS.log(),
    benchmark:
        DERIVE_CHANGE_FACTORS.benchmark(),
    script: "blueearth_cst/projections/derive_change_factors.py"

# 2.05b reads the resolved scalar inputs independently of aligned change factors.
# The polygon is needed to assert identities against today's region fingerprint.
rule prepare_projection_analysis:
    message: PREPARE_PROJECTION_ANALYSIS.banner()
    input:
        series_nc = sorted(
            {series_file(m, "historical", mem) for (m, sc, mem) in POINTS}
            | {series_file(m, sc, mem) for (m, sc, mem) in POINTS}
        ),
        region_path = region_path,
        stage_b_settings = _STAGE_B_STAMP,
    output:
        **ANALYSIS_OUTPUTS,
    params:
        horizons = future_horizons,
        water_year_start = water_year_start,
        reference_window = time_horizon_hist,
        variable_spec = {k: list(v) for k, v in VARIABLE_SPEC.items()},
        min_reference = MIN_REFERENCE,
        windows = ANALYSIS_WINDOWS,
        table_names = ANALYSIS_TABLE_NAMES,
        combinations = [
            {
                "dataset": c.dataset, "institution": c.institution,
                "source_id": c.source_id, "scenario": c.scenario,
                "member": c.member, "status": c.status, "detail": c.detail,
            }
            for c in COMBINATIONS
        ],
        points = [
            {
                "point_key": key, "model": model, "scenario": scenario,
                "member": member,
                "series_path_hist": series_file(model, "historical", member),
                "series_path": series_file(model, scenario, member),
                "digest_components_hist": series_digest_components(model, "historical", member),
                "digest_components_fut": series_digest_components(model, scenario, member),
                "series_key": f"{clim_project}_{model.replace('/', '_')}_{scenario}_{member}",
                "reference_series_key": f"{clim_project}_{model.replace('/', '_')}_historical_{member}",
            }
            for key, (model, scenario, member) in POINT_KEYS.items()
        ],
    threads: 1
    resources:
        mem_mb = 1024,
    log:
        PREPARE_PROJECTION_ANALYSIS.log(),
    benchmark:
        PREPARE_PROJECTION_ANALYSIS.benchmark(),
    script: "blueearth_cst/projections/prepare_projection_analysis.py"

# 2.06 is a pure renderer of analysis tables; its owned directories contain only
# products marked available, and the manifest names the PNGs actually written.
rule plot_climate_projections:
    message: PLOT_CLIMATE_PROJECTIONS.banner()
    input:
        tables = [ANALYSIS_OUTPUTS[name] for name in ANALYSIS_TABLE_NAMES],
        record = ANALYSIS_OUTPUTS["analysis_record"],
    params:
        scenarios = scenarios,
        annotation = plot_annotation,
        plot_setup_digest = plot_setup_digest,
    output:
        folders = [directory(path) for path in WF2_SUITE_PATHS.values()],
        titles_md = f"{clim_project_dir}/plots/figure_titles.md",
        titles_json = f"{clim_project_dir}/_engine/figure_titles.json",
        captions_md = FIGURE_CAPTIONS_MD,
        captions_json = FIGURE_CAPTIONS_JSON,
    threads: 1
    resources:
        mem_mb = 1024,
    log:
        PLOT_CLIMATE_PROJECTIONS.log(),
    benchmark:
        PLOT_CLIMATE_PROJECTIONS.benchmark(),
    script: "blueearth_cst/projections/plot_proj_timeseries.py"

# --- log gather ---------------------------------------------------------------
# 2.08  gather_logs — merge every WF2 log part into ONE workflow log.
#
# Replaces the two per-stage gathers (`gather_series_logs` 2.08 /
# `gather_raw_logs` 2.07). Those merged only the fan-out rules, each into its own
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
# (pre-board record 45e3a564 R7-9) — the artifact describes the run that produced it, not
# an accumulated history.
#
# WF1 (1.18) and WF3 (3.18) declare the same rule against the same script; only
# the label list, the parts dir and the output name differ.
rule gather_logs:
    message: GATHER_LOGS.banner()
    input:
        (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
        FIGURE_CAPTIONS_JSON,
        ANALYSIS_OUTPUTS["analysis_record"],
        # rule 2.02 is a LEAF here — nothing in WF2 consumes the
        # vector layers yet — so it is not upstream of the three terminals
        # above the way 2.01 is. Without this edge it would run in parallel
        # with the merge and its part would be stranded under `_parts/`, the
        # exact defect the LOG_RULES block records for 1.01, 2.01 and 3.03.
        SPATIAL_UNITS.outputs["basins"],
    output:
        WORKFLOW_LOG_PATH,
    params:
        rules = LOG_RULES,
        parts_dir = LOG_PARTS_DIR,
    script: "blueearth_cst/shared/merge_logs.py"

# --- benchmark gather ---------------------------------------------------------
# Merge WF2's per-rule benchmark parts (benchmarks/_parts/2.*) into one
# benchmarks/wf2_benchmarks.md (rule column + TOTAL row) once the summary +
# plots are built (all WF2 rules ran).
rule gather_benchmarks:
    message: GATHER_BENCHMARKS.banner()
    input:
        (clim_project_dir + f"/summary/{clim_project}_change_factors_annual.csv"),
        FIGURE_CAPTIONS_JSON,
        ANALYSIS_OUTPUTS["analysis_record"],
        # Same reason as gather_logs above: 2.02 is a leaf, and the benchmark
        # gather has to wait for it or its `_parts/` row misses this run.
        SPATIAL_UNITS.outputs["basins"],
    output:
        BENCHMARK_OUTPUT_PATH,
        BENCHMARK_OUTPUT_PATH.removesuffix(".md") + ".tsv",
    params:
        parts_dir = BENCH_PARTS_DIR,
        workflow_num = 2,
    script: "blueearth_cst/shared/merge_benchmarks.py"


# --- Run journal (design §5.7) ------------------------------------------------
#
# Workflow-level handlers, never a rule: a rule that is up to date does not
# execute, and a rule DECLARING the journal would have it deleted before the
# job ran, truncating the ledger to one line every run. See the same block in
# build_model.smk for the scope the P0 probe established -- these fire
# only when at least one job executed, which is what R5 was narrowed to.
JOURNAL_PATH = f"{project_dir}/config/_engine/journal.jsonl"
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
        open_run_header(
            "wf2 analyze_projections", project_dir, header_config_path(config_path, CAPTURE_RECORD),
            **{
                "reference period": REFERENCE_WINDOW.effective_label,
                "future periods": listed([f"{start}–{end}" for start, end in future_horizons.values()]),
                "emissions scenarios": listed(scenarios),
                "climate models": listed(models),
                "water year": f"Starts in {resolve_water_year_start(climate_cfg.get('water_year_start'))}",
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
    # A reused analysis needs no repeated reference-period warning.
    for _line in _rw.window_warnings(REFERENCE_WINDOW):
        defer_warning(_line, module="reference_window")
    _header()
    _journal("started")


onsuccess:
    _journal("success")
    _summary(failed=False)


onerror:
    _journal("failed")
    _summary(failed=True)
