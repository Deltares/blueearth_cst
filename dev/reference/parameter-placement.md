# Where a parameter lives

Assessment and placement rule for every configurable value in the toolbox.
Written 2026-08-12, after three parameters (`shared.seed`,
`shared.water_year_start`, the two spell factors) were placed in one session by
reasoning from precedent rather than from a rule — which works until two
precedents disagree, which is exactly what `max_per_basin` exposed.

## The five homes

| # | Home | What it holds today |
|---|---|---|
| 1 | `config/advanced_settings.yml` | 5 keys: `constraints` (1), `defaults` (3), `runtime` (1). Closed schema — an unknown key is rejected at parse time. |
| 2 | project config `shared.*` | 13 leaf keys. Read by two or more workflows. |
| 3 | project config `workflows.<wf>.*` | 38 leaf keys, plus 4 under `project`. |
| 4 | `config/defaults/*.yml` | 3 engine-native templates (wflow build, waterbodies, weathergen). |
| 5 | Python `DEFAULT_*` constants | 14, of which 3 merely re-export home 1. |

55 leaf keys in the project config; 17 of them required (`optional=False`).

## The rule

Two questions, in order.

**Q1 — does changing the value change the NUMBERS?**

**Q2 — is the right value basin- or project-specific, or the same everywhere?**

| Q1 | Q2 | Home |
|---|---|---|
| yes | project-specific | **Project config.** `shared.*` if two or more workflows read it, `workflows.<wf>.*` if one does. |
| yes | universal (a method constant) | **`advanced_settings.defaults`**, overridable per project by a named key. |
| no (speed, verbosity, paths) | — | **`advanced_settings.defaults`**, or `runtime` for an external toolchain pin. |
| — | must never be relaxed | **`advanced_settings.constraints`**. |
| — | expressed in an engine's own vocabulary | **`config/defaults/<engine>.yml`.** Forced: AGENTS.md's hard constraint says hydromt / wflow / weathergenr schemas are used verbatim. |

And the rule that decides the cases actually in dispute:

> **A `DEFAULT_*` constant in Python is only correct when the value has NO
> config surface at all.** The moment a config key can set it, its default
> belongs in `advanced_settings.defaults`, beside the key's documentation,
> where a user can read it without opening the source.

That is what separates `max_per_basin` (a config key whose default hides in
`spatial/config.py`) from `DEFAULT_DECIMALS` (an internal formatting choice no
config exposes, correctly a constant).

Q1 is the load-bearing question, and it is what makes `max_per_basin` a project
parameter rather than a tool knob despite feeling like one: it changes **what
the results are reported over**. `julia_threads` changes only how fast the same
numbers arrive. Both are "knobs"; only one moves a number.

## Misfits

Ordered by how much they can bite. Each is a proposal, not an applied change.

### M1 — `project.static_dir` is required by two workflows, used by one, and can only ever be `config`

`Snakefile_model_creation:54` reads it (required) and uses it at 115–116 as the
fallback base for `model_build_config` / `waterbodies_config`.
`Snakefile_climate_experiment:41` reads it (required) and **never uses it** —
WF3 fails without a key it ignores. WF2 does not read it.

The fallbacks resolve to `{static_dir}/defaults/wflow_build_model.yml`, which is
a **toolbox file inside the repo**. So the only value that works is `config`; any
other points at nothing. Every shipped config also sets both keys explicitly, so
the fallback is rarely taken at all.

**Proposal:** delete the key. Replace the two fallbacks with the literal
`config/defaults/…` paths. A project wanting its own build config already has
the documented route — set `model_build_config:` directly.
**Cost:** 2 Snakefile reads, 5 configs, the template, and
`tests/test_guard_invalidation.py:241`, which uses `static_dir` as its example
of a `_WF1_GUARDED` key and needs a different one. Breaking for any config that
sets it — which is all of them — but the removal is mechanical.

### M2 — `DEFAULT_ANCHOR = "YE-DEC"` is defined twice

`shared/metrics_definition.py:18` and `climate_analysis/climate_figures.py:120`.
One value, two sources of truth, introduced in this session's water-year work —
the exact drift the rule exists to prevent, committed while writing the rule.

**Proposal:** single-source it. The value derives from
`advanced_settings.defaults.water_year_start`, so
`snake_utils.water_year_end_anchor(DEFAULT_WATER_YEAR_START)` is the one
definition; both modules import it. **Cost:** trivial, non-breaking.

### M3 — defaults for config keys that live in Python instead of `advanced_settings`

| Constant | Backs |
|---|---|
| `DEFAULT_SPELL_FACTOR` | `stress_test.{dry,wet}_spell_factor` |
| `DEFAULT_MAX_SUBBASINS_PER_BASIN` (11) | `shared.basin.automatic_subbasins.max_per_basin` |
| `DEFAULT_GAUGE_SNAP_TOLERANCE_M` | `shared.basin.gauge_snap_tolerance_m` |
| `DEFAULT_HYDROGRAPHY`, `DEFAULT_BASIN_INDEX` | `shared.basin.{hydrography,basin_index}` |
| `DEFAULT_STATS` | `workflows.climate_projections.stats` |

Each is a config key whose default a user cannot see without reading source,
while `seed`, `water_year_start` and `julia_threads` publish theirs. Same class
of value, two conventions — decided by which session added it.

**Proposal:** move all five defaults into `advanced_settings.defaults`, keeping
every override key exactly where it is. This is the answer to "should
`max_per_basin` move?" — the *key* stays under `shared.basin` (it changes
results and is basin-specific, Q1+Q2), the *default* moves.
**Cost:** five schema entries and their tests; non-breaking for project configs.

### M4 — `shared.basin` mixes three kinds of value

`region` and `resolution` are the basin's definition. `spatial_sources.*` name
catalog entries — data-source bindings. `gauge_snap_tolerance_m`,
`river_uparea_km2` and `max_per_basin` are delineation tolerances. All three
kinds sit flat under one heading, which is why `max_per_basin` reads as
misplaced: it is not misplaced *relative to the config*, it is grouped with
things it is unlike.

**Proposal:** left open deliberately. Regrouping is breaking for every project
config and buys legibility only — worth doing with a schema version bump if one
happens for another reason, not on its own.

## Applying it

The rule earns its keep only if the next parameter is placed by citing it.
Suggested: add one line to AGENTS.md's Conventions pointing here, and treat a
new `DEFAULT_*` backing a config key as a review smell.
