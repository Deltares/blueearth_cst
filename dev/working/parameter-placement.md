# Configuration surface — problem statement and inventory (DRAFT)

> **DRAFT / PROPOSAL — not agreed. Do not cite this as a rule.**
>
> A working document. §4 is a **statement of the problem**; §5 is a *tentative*
> direction, deliberately subordinate and not to be applied. The misfits in §6
> are recommendations awaiting an owner ruling.
>
> AGENTS.md carries no reference to it, by design. It moves to `dev/reference/`
> and gains an AGENTS.md pointer only once the problem statement is accepted and
> a direction is chosen.
>
> One thing in the repo does point here, and this paragraph used to deny it:
> `run_stress_test.smk` cites **M1** (§6) for why `project.static_dir` was not
> deleted outright. That citation is to a *proposal*, not a rule — but it is why
> this file cannot be drained from `working/` without being promoted first (the
> promotion rule, `dev/README.md`).
>
> The INVENTORY (§1–§3, and the appendix) is **measured** and stands on its own
> whatever is decided.

Written 2026-08-12, after three parameters (`shared.seed`,
`shared.water_year_start`, the two spell factors) were placed in one session by
reasoning from precedent rather than from a rule.

---

## 1. The three tiers

Configuration in this toolbox is not one surface, it is three, and they differ
in who owns them, who writes them, and what a change means.

| Tier | Owner | Written by | A change means |
|---|---|---|---|
| **T1 Toolbox** | the toolbox | a maintainer, in-repo | every project's behaviour changes |
| **T2 Project** | the project | the user, per basin | this project's results change |
| **T3 Generated** | the run | a rule, into `project_dir` | nothing — it is a RECORD of what ran |

Most of the confusion this document exists to resolve is T1-vs-T2. T3 is
well-defined already and is included because it is a third of the surface and
is routinely mistaken for input.

## 2. T1 — toolbox configuration (tracked, in-repo)

36 tracked config files. Grouped by what they actually are:

| Group | Files | Role |
|---|---|---|
| **Toolbox knobs** | `config/advanced_settings.yml` | 5 keys: `constraints` (1) · `defaults` (3) · `runtime` (1). Closed schema. |
| **Engine templates** | `config/defaults/` × 3 — `wflow_build_model.yml`, `wflow_update_waterbodies.yml`, `weathergen_config.yml` | Rule INPUTS in the engines' own vocabulary. Changing one changes a run. |
| **Data catalogs** | `config/catalogs/` × 3 live + 2 archived; `tests/data/tests_data_catalog.yml` | hydromt `-d` targets. `cmip6_data.yml` (3919 lines) is **generated** by a crawl. |
| **Scaffolds** | `config/templates/snake_config.template.yml`, `wflow_sbm.reference.toml`, 5 archived single-workflow configs | Copied, never read by a rule. |
| **Shipped example projects** | `test_case/snake_config_*.yml` × 4, `tests/snake_config_fixture.yml` | T2 documents that happen to live in-repo. |
| **Process / build** | `pixi.toml`, `pyproject.toml`, `Project.toml`, `Manifest.toml`, `.github/workflows/ci.yml`, `profiles/default/config.yaml`, `.testing-policy.yml`, `.git-workflow.yml`, `dev/reference/sealed-records.yml`, `dev/scripts/{stage_data,scaffold_extras}.yml` | Not pipeline parameters. Listed so the count is honest. |

**A sixth, invisible group: Python `DEFAULT_*` constants — 14 of them.** Three
re-export `advanced_settings`; the rest are defaults a user cannot see without
reading source. This group is the problem's centre of gravity.

## 3. T2 — project configuration

One `--configfile` YAML. **55 leaf keys**, 17 required:

| Section | Leaves | Contents |
|---|---|---|
| `project` | 4 | `project_dir`, `static_dir`, two catalog paths |
| `shared` | 13 | basin definition, catalog bindings, delineation tolerances, window, `clim_historical` |
| `workflows.*` | 38 | per-workflow; `run_stress_test` alone holds the stress-test grid |

Full listing: appendix.

## 4. The problem, stated

Not "some parameters are in the wrong place". Four distinct failures, each
observed:

**P1 — A parameter's DEFAULT and its KEY live in different tiers, inconsistently.**
`seed`, `water_year_start`, `julia_threads` publish their defaults in T1
`advanced_settings`. `max_per_basin`, `gauge_snap_tolerance_m`, `hydrography`,
`basin_index`, `stats` and the spell factors keep theirs in Python. Same class
of value, two conventions, decided by whichever session added it. A user
reading the config cannot discover what a key defaults to.

**P2 — There is no test that a declared parameter reaches anything.**
Twice in one session a key was found that a config could set and nothing read:
WF2's `start_month_hyd_year` (read, passed to rule 2.06, never used — every
change factor was Jan–Dec regardless) and `relax_priority` (the wrapper does not
forward it). Both were found by hand. `static_dir` is a live third: WF3 reads it
as required and never uses it. The C34 finding was a fourth. **Four inert or
partly-inert parameters, four manual discoveries, zero mechanical detection.**

**P3 — One concept, several spellings, because nothing forces convergence.**
The water year was `start_month_hyd_year` (T2, string, inert), `year_start_month`
(T1 engine template, integer, live), and hardcoded calendar years in two more
places. Nothing detected the divergence; it was found by reading.

**P4 — Grouping is by history, not by kind.** `shared.basin` holds the basin's
definition (`region`, `resolution`), catalog bindings (`spatial_sources.*`) and
delineation tolerances (`max_per_basin`, `gauge_snap_tolerance_m`,
`river_uparea_km2`) under one heading. That is why `max_per_basin` reads as
misplaced: it is not in the wrong tier, it is grouped with things unlike it.

**What "convenient, future-looking, efficient" would mean here**

- *Convenient* — a user can find every knob that affects their basin, and see
  its default, without reading Python.
- *Future-looking* — adding a parameter has one obvious home, and adding a
  second stochastic step or a third stress axis does not require re-litigating.
- *Efficient* — the same value is declared once. P3 is what its absence costs.

**The sharpest test of any proposal: would it have caught P2 mechanically?**
That is the failure with real consequences — results computed under settings
nobody chose.

## 5. Tentative direction — NOT agreed

Recorded so the next discussion starts somewhere, not as a decision.

Two questions place a parameter: **does changing it change the NUMBERS**, and
**is the right value project-specific or universal**. Engine-native vocabulary
is a forced case (hydromt/wflow/weathergenr schemas are used verbatim per
AGENTS.md's hard constraint), so those stay in `config/defaults/`.

The clause that settles the live disputes:

> A Python `DEFAULT_*` is correct **only** when the value has no config surface
> at all. Once a config key can set it, the default belongs where the key is
> documented.

Open, and not answered here: whether that home should be `advanced_settings`
(one file, closed schema, already tested) or a per-key default declared beside
the key in the template (closer to the user, no second file to consult). The
first is the existing precedent; the second is arguably more *convenient* by the
definition above. **This is the decision to make.**

Also unaddressed: a mechanical answer to P2. A "declared keys ⊆ read keys"
check, in the spirit of `interchange_contracts`, would have caught all four
inert parameters. Whether that is feasible against Snakemake's `params:`
indirection is unknown and worth a probe before promising it.

## 6. Misfits found

Proposals only; none applied.

| # | Misfit | Proposal | Cost |
|---|---|---|---|
| M1 | `project.static_dir` — required by WF1 and WF3, used by WF1 only, ignored by WF3, and can only ever be `config` because the fallbacks it feeds resolve to in-repo toolbox files | delete; use literal `config/defaults/…` | 2 Snakefiles, 5 configs, template, `test_guard_invalidation.py:241` (uses it as its `_WF1_GUARDED` example). Breaking for every config. |
| M2 | `DEFAULT_ANCHOR = "YE-DEC"` defined **twice** (`metrics_definition.py:18`, `climate_figures.py:120`) — introduced by this session's water-year work, the exact drift P3 describes | single-source from `water_year_end_anchor(DEFAULT_WATER_YEAR_START)` | trivial, non-breaking |
| M3 | Five config-key defaults in Python (`DEFAULT_SPELL_FACTOR`, `DEFAULT_MAX_SUBBASINS_PER_BASIN`, `DEFAULT_GAUGE_SNAP_TOLERANCE_M`, `DEFAULT_HYDROGRAPHY`/`DEFAULT_BASIN_INDEX`, `DEFAULT_STATS`) | move the DEFAULTS; keys stay put | 5 schema entries + tests; non-breaking |
| M4 | `shared.basin` mixes three kinds (P4) | regroup | breaking for every config; legibility only — pair with a schema version bump, not standalone |

M2 is the only one that is unambiguously correct regardless of how §5 resolves.

---

## Appendix — parameter sets per file

> Reproduced verbatim in
> `dev/working/2026-08-12_config-parameter-review.task.md`, so that brief is
> self-contained. Both are the same 2026-08-12 measurement; the review is
> instructed to re-measure rather than trust either. If one is corrected,
> correct both or delete this one — two copies of an inventory is the exact
> duplication that review's Q3 exists to find.

### `config/advanced_settings.yml` (5)
`constraints.min_historical_years` · `defaults.julia_threads` ·
`defaults.seed` · `defaults.water_year_start` · `runtime.julia_version`

### Project config — `project` (4)
`project_dir`* · `static_dir`* · `data_sources`* · `data_sources_climate`*

### Project config — `shared` (13)
`basin.region`* · `basin.resolution` · `basin.gauge_points` ·
`basin.automatic_subbasins.max_per_basin` · `basin.gauge_snap_tolerance_m` ·
`basin.river_uparea_km2` · `basin.spatial_sources.{rivers,lulc,lai,soil}` ·
`historical_window.{starttime,endtime}`* · `clim_historical`*
· *(optional, undocumented in the template: `basin.hydrography`,
`basin.basin_index`)*

### Project config — `workflows.build_model`
`enabled` · `model_build_config` · `waterbodies_config` · `wflow_outvars` ·
`observations_timeseries` · `simulation_window.{starttime,endtime}`*

### Project config — `workflows.analyze_projections`
`enabled` · `clim_project`* · `models`* · `scenarios`* · `members`* ·
`variables`* · `historical_year_range`* · `future_horizons`* · `stats` ·
`save_grids` *(`start_month_hyd_year` retired 2026-08-12 — now refused)*

### Project config — `workflows.run_stress_test`
`enabled` · `experiment_name` · `realizations_num` · `horizontime_climate`* ·
`run_length` · `run_historical` · `stress_test.{temp,precip}.{step_num,
transient_change,mean.{min,max},variance.{min,max}}` ·
`stress_test.{dry,wet}_spell_factor`

### `config/defaults/weathergen_config.yml` (4 sections)
`run_weather_generator` (2) · `generate_weather` (16 + 6 injected) ·
`apply_climate_perturbations` (15) · `write_netcdf` (5). Sections are
weathergenr 1.2.0 function names; keys are their argument names.

### `config/defaults/wflow_build_model.yml` / `wflow_update_waterbodies.yml`
hydromt `setup_*` blocks, verbatim in hydromt_wflow's schema.

### Python `DEFAULT_*` (14)
Re-exports T1: `DEFAULT_JULIA_THREADS`, `DEFAULT_SEED`,
`DEFAULT_WATER_YEAR_START`.
Back a config key (**M3**): `DEFAULT_SPELL_FACTOR`,
`DEFAULT_MAX_SUBBASINS_PER_BASIN`, `DEFAULT_GAUGE_SNAP_TOLERANCE_M`,
`DEFAULT_HYDROGRAPHY`, `DEFAULT_BASIN_INDEX`, `DEFAULT_STATS`.
Duplicated (**M2**): `DEFAULT_ANCHOR` ×2.
No config surface, correctly constants: `DEFAULT_DECIMALS`,
`DEFAULT_MIN_REFERENCE`, `DEFAULT_MAX_FLAGGED_MONTHS`.

### T3 — generated, per `project_dir` (record, never input)
`config/runs/<workflow>/<digest>/{source.yml, effective.yml,
referenced-files.json, files/**}` — content-addressed run snapshots ·
`config/catalogs/*` · `config/templates/*` ·
`experiments/<id>/{experiment.yml, model_reference.yml,
snake_config_run_stress_test.yml, catalogs/*, runs/**}` ·
`experiments/<id>/climate/weathergenr/config/weathergen_config.yml` ·
`models/hydrology/wflow/{wflow_sbm.toml, config/*}`

`*` = required (`optional=False`).
